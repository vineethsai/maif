"""
MAIF Checkpointer for LangGraph

Implements the LangGraph BaseCheckpointSaver interface using MAIF
for persistent state storage with cryptographic provenance.

This checkpointer stores all graph state transitions in a MAIF artifact,
providing:
- Tamper-evident audit trails via hash chains
- Ed25519 signatures on all state changes
- Full state history for debugging and compliance
- Thread-safe checkpoint management
"""

import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from contextlib import contextmanager

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object
    Checkpoint = dict
    CheckpointMetadata = dict
    CheckpointTuple = tuple

from maif.integrations._base import EventType, MAIFProvenanceTracker
from maif.integrations._utils import safe_serialize, format_timestamp


class MAIFCheckpointer(BaseCheckpointSaver if LANGGRAPH_AVAILABLE else object):
    """MAIF-backed checkpointer for LangGraph state persistence.
    
    This checkpointer stores LangGraph state checkpoints in a MAIF artifact,
    providing cryptographic provenance tracking for all state transitions.
    Each checkpoint is stored as a separate block with:
    - Full state serialization
    - Metadata (thread_id, checkpoint_id, timestamp)
    - Parent checkpoint reference
    - Cryptographic signature
    
    Usage:
        from langgraph.graph import StateGraph
        from maif.integrations.langgraph import MAIFCheckpointer
        
        # Create checkpointer
        checkpointer = MAIFCheckpointer("graph_state.maif")
        
        # Compile graph with checkpointer
        app = graph.compile(checkpointer=checkpointer)
        
        # Run with thread_id for checkpoint tracking
        result = app.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config={"configurable": {"thread_id": "conversation-1"}}
        )
        
        # Finalize when done (optional but recommended)
        checkpointer.finalize()
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier for the checkpointer agent
        
    Attributes:
        artifact_path: Path to the MAIF artifact
        tracker: The underlying MAIFProvenanceTracker
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required for MAIFCheckpointer. "
                "Install it with: pip install langgraph"
            )
        
        # Initialize the base class with the serializer
        super().__init__(serde=JsonPlusSerializer())
        
        self.artifact_path = Path(artifact_path)
        self._agent_id = agent_id or "langgraph_checkpointer"
        
        # Initialize tracker (lazy)
        self._tracker: Optional[MAIFProvenanceTracker] = None
        
        # In-memory checkpoint index for fast lookups
        # Maps (thread_id, checkpoint_ns, checkpoint_id) -> checkpoint data
        self._checkpoints: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing checkpoints if artifact exists
        self._load_existing_checkpoints()
    
    def _ensure_tracker(self) -> MAIFProvenanceTracker:
        """Ensure the provenance tracker is initialized."""
        if self._tracker is None:
            self._tracker = MAIFProvenanceTracker(
                artifact_path=self.artifact_path,
                agent_id=self._agent_id,
                auto_finalize=False,
            )
        return self._tracker
    
    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from MAIF artifact if it exists."""
        if not self.artifact_path.exists():
            return
        
        try:
            from maif import MAIFDecoder
            
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            
            for block in decoder.blocks:
                block_metadata = block.metadata or {}
                if block_metadata.get("type") == EventType.STATE_CHECKPOINT.value:
                    # Deserialize the event data
                    try:
                        raw_data = block.data
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode("utf-8")
                        event_data = json.loads(raw_data)
                        
                        # The checkpoint data is nested under 'data' key
                        checkpoint_data = event_data.get("data", {})
                        
                        # Extract checkpoint identifiers from the nested data
                        thread_id = checkpoint_data.get("thread_id", "")
                        checkpoint_ns = checkpoint_data.get("checkpoint_ns", "")
                        checkpoint_id = checkpoint_data.get("checkpoint_id", "")
                        
                        if thread_id and checkpoint_id:
                            key = (thread_id, checkpoint_ns, checkpoint_id)
                            self._checkpoints[key] = checkpoint_data
                            
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                        pass  # Skip malformed checkpoints
                            
        except Exception:
            # If loading fails, start fresh
            pass
    
    def _get_checkpoint_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> Tuple[str, str, str]:
        """Create a unique key for a checkpoint."""
        return (thread_id, checkpoint_ns or "", checkpoint_id)
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config.
        
        Args:
            config: Configuration dict with 'configurable' containing
                   'thread_id' and optionally 'checkpoint_ns', 'checkpoint_id'
                   
        Returns:
            CheckpointTuple if found, None otherwise
        """
        with self._lock:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "")
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            checkpoint_id = configurable.get("checkpoint_id")
            
            if not thread_id:
                return None
            
            # If no specific checkpoint_id, get the latest
            if checkpoint_id is None:
                checkpoint_id = self._get_latest_checkpoint_id(thread_id, checkpoint_ns)
                if checkpoint_id is None:
                    return None
            
            key = self._get_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
            checkpoint_data = self._checkpoints.get(key)
            
            if checkpoint_data is None:
                return None
            
            # Reconstruct the CheckpointTuple
            checkpoint = checkpoint_data.get("checkpoint", {})
            metadata = checkpoint_data.get("metadata", {})
            parent_config = checkpoint_data.get("parent_config")
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )
    
    def _get_latest_checkpoint_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> Optional[str]:
        """Get the latest checkpoint ID for a thread."""
        latest_id = None
        latest_ts = 0
        
        for key, data in self._checkpoints.items():
            key_thread_id, key_checkpoint_ns, key_checkpoint_id = key
            if key_thread_id == thread_id and key_checkpoint_ns == checkpoint_ns:
                ts = data.get("timestamp", 0)
                if ts > latest_ts:
                    latest_ts = ts
                    latest_id = key_checkpoint_id
        
        return latest_id
    
    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the given criteria.
        
        Args:
            config: Configuration to filter by (thread_id, checkpoint_ns)
            filter: Additional metadata filters
            before: Return checkpoints before this config
            limit: Maximum number of checkpoints to return
            
        Yields:
            CheckpointTuple for each matching checkpoint
        """
        with self._lock:
            configurable = (config or {}).get("configurable", {})
            thread_id = configurable.get("thread_id")
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            
            # Collect matching checkpoints
            matches = []
            for key, data in self._checkpoints.items():
                key_thread_id, key_checkpoint_ns, key_checkpoint_id = key
                
                # Filter by thread_id if specified
                if thread_id and key_thread_id != thread_id:
                    continue
                
                # Filter by checkpoint_ns
                if checkpoint_ns and key_checkpoint_ns != checkpoint_ns:
                    continue
                
                # Apply before filter
                if before:
                    before_config = before.get("configurable", {})
                    before_id = before_config.get("checkpoint_id")
                    if before_id and key_checkpoint_id >= before_id:
                        continue
                
                matches.append((key, data))
            
            # Sort by timestamp (newest first)
            matches.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            
            # Apply limit
            if limit:
                matches = matches[:limit]
            
            # Yield CheckpointTuples
            for key, data in matches:
                key_thread_id, key_checkpoint_ns, key_checkpoint_id = key
                
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": key_thread_id,
                            "checkpoint_ns": key_checkpoint_ns,
                            "checkpoint_id": key_checkpoint_id,
                        }
                    },
                    checkpoint=data.get("checkpoint", {}),
                    metadata=data.get("metadata", {}),
                    parent_config=data.get("parent_config"),
                )
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a checkpoint.
        
        Args:
            config: Configuration with thread_id and checkpoint info
            checkpoint: The checkpoint data to store
            metadata: Checkpoint metadata
            new_versions: Channel version updates (not used in MAIF storage)
            
        Returns:
            Updated config with the new checkpoint_id
        """
        with self._lock:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "")
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            
            # Generate checkpoint_id from the checkpoint's id field
            checkpoint_id = checkpoint.get("id", str(int(time.time() * 1000)))
            
            # Get parent config if this is a continuation
            parent_checkpoint_id = configurable.get("checkpoint_id")
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
            
            # Create checkpoint data package
            timestamp = time.time()
            checkpoint_data = {
                "checkpoint": checkpoint,
                "metadata": metadata,
                "parent_config": parent_config,
                "timestamp": timestamp,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
            
            # Store in memory index
            key = self._get_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
            self._checkpoints[key] = checkpoint_data
            
            # Log to MAIF artifact
            tracker = self._ensure_tracker()
            tracker.log_event(
                event_type=EventType.STATE_CHECKPOINT,
                data=checkpoint_data,
                metadata={
                    "type": EventType.STATE_CHECKPOINT.value,
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "timestamp": timestamp,
                },
            )
            
            # Return updated config
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
    
    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes (pending state updates).
        
        This method stores writes that haven't been committed to a checkpoint yet.
        In MAIF, we log these as NODE_END events for debugging and audit purposes.
        
        Args:
            config: Configuration with thread_id
            writes: Sequence of (channel, value) tuples
            task_id: The task/node that produced the writes
        """
        with self._lock:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "")
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            checkpoint_id = configurable.get("checkpoint_id", "")
            
            # Log node completion with writes
            tracker = self._ensure_tracker()
            
            # Serialize writes safely
            serialized_writes = []
            for channel, value in writes:
                try:
                    serialized_writes.append({
                        "channel": channel,
                        "value": value,
                    })
                except Exception:
                    serialized_writes.append({
                        "channel": channel,
                        "value": str(value)[:1000],
                    })
            
            tracker.log_event(
                event_type=EventType.NODE_END,
                data={
                    "task_id": task_id,
                    "writes": serialized_writes,
                    "num_writes": len(writes),
                },
                metadata={
                    "type": EventType.NODE_END.value,
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "timestamp": time.time(),
                },
            )
    
    # Async methods - delegate to sync versions
    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async version of get_tuple."""
        return self.get_tuple(config)
    
    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ):
        """Async version of list."""
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item
    
    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of put."""
        return self.put(config, checkpoint, metadata, new_versions)
    
    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async version of put_writes."""
        self.put_writes(config, writes, task_id)
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact.
        
        Call this when you're done using the checkpointer to properly
        seal the artifact with cryptographic signatures.
        """
        if self._tracker is not None:
            self._tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the path to the MAIF artifact."""
        return str(self.artifact_path)
    
    def __enter__(self) -> "MAIFCheckpointer":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - finalize the artifact."""
        self.finalize()

