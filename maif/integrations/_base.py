"""
Base classes and protocols for MAIF framework integrations.

This module provides the foundational abstractions that all framework-specific
integrations build upon. It defines:

- MAIFProvenanceTracker: Core class for logging events to MAIF artifacts
- Event type constants for standardized logging
- Common interfaces for callbacks and handlers

All framework integrations (LangChain, CrewAI, LangGraph, Strands) inherit
from or use these base classes to ensure consistent provenance tracking.
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standardized event types for MAIF provenance logging.
    
    These event types provide consistent categorization across all
    framework integrations, making audit trails queryable and comparable.
    """
    
    # LLM Events
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    
    # Chain/Workflow Events
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    CHAIN_ERROR = "chain_error"
    
    # Tool Events
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    
    # Retrieval Events
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_END = "retrieval_end"
    
    # Agent Events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ACTION = "agent_action"
    AGENT_ERROR = "agent_error"
    
    # Task Events (CrewAI)
    TASK_START = "task_start"
    TASK_END = "task_end"
    TASK_ERROR = "task_error"
    
    # State Events (LangGraph)
    STATE_CHECKPOINT = "state_checkpoint"
    STATE_RESTORE = "state_restore"
    NODE_START = "node_start"
    NODE_END = "node_end"
    
    # Memory Events
    MEMORY_SAVE = "memory_save"
    MEMORY_LOAD = "memory_load"
    
    # Session Events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Custom Events
    CUSTOM = "custom"


@dataclass
class ProvenanceEvent:
    """A single provenance event to be logged to a MAIF artifact.
    
    Attributes:
        event_type: The type of event (from EventType enum)
        data: The primary event data (will be serialized to JSON)
        metadata: Additional metadata about the event
        timestamp: Unix timestamp of when the event occurred
        run_id: Optional identifier linking related events
        parent_run_id: Optional identifier for parent event (for nesting)
    """
    
    event_type: EventType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
        }


class MAIFProvenanceTracker:
    """Core class for tracking provenance in MAIF artifacts.
    
    This class provides the foundation for all framework integrations.
    It handles:
    - Creating and managing MAIF artifacts
    - Logging events with proper serialization
    - Maintaining event relationships (parent/child)
    - Finalizing artifacts with cryptographic signatures
    
    Usage:
        tracker = MAIFProvenanceTracker("session.maif", agent_id="my-agent")
        block_id = tracker.log_event(
            EventType.LLM_START,
            {"model": "gpt-4", "prompt": "Hello"}
        )
        tracker.finalize()
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Identifier for the agent/system creating events
        auto_finalize: If True, finalize on context manager exit
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
        auto_finalize: bool = True,
    ):
        self.artifact_path = Path(artifact_path)
        self.agent_id = agent_id or "maif_integration"
        self.auto_finalize = auto_finalize
        self._encoder = None
        self._initialized = False
        self._block_ids: List[str] = []
        self._finalized = False
        
    def _ensure_initialized(self) -> None:
        """Lazily initialize the MAIF encoder."""
        if self._initialized:
            return
            
        # Import here to avoid circular imports
        from maif import MAIFEncoder
        
        # Ensure parent directory exists
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._encoder = MAIFEncoder(
            str(self.artifact_path),
            agent_id=self.agent_id
        )
        self._initialized = True
        
        # Log session start
        self._log_session_start()
    
    def _log_session_start(self) -> None:
        """Log the session initialization event."""
        from maif.integrations._utils import safe_serialize
        
        session_data = {
            "event_type": EventType.SESSION_START.value,
            "agent_id": self.agent_id,
            "artifact_path": str(self.artifact_path),
            "timestamp": time.time(),
        }
        
        self._encoder.add_text_block(
            safe_serialize(session_data),
            metadata={
                "type": "session_start",
                "agent_id": self.agent_id,
            }
        )
    
    def log_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """Log an event to the MAIF artifact.
        
        Args:
            event_type: The type of event to log
            data: The primary event data
            metadata: Additional metadata
            run_id: Optional run identifier
            parent_run_id: Optional parent run identifier
            
        Returns:
            The block ID of the logged event
        """
        self._ensure_initialized()
        
        if self._finalized:
            raise RuntimeError("Cannot log events to a finalized artifact")
        
        from maif.integrations._utils import safe_serialize
        
        event = ProvenanceEvent(
            event_type=event_type,
            data=data,
            metadata=metadata or {},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        
        # Add framework-specific metadata
        event.metadata["agent_id"] = self.agent_id
        event.metadata["event_type"] = event_type.value
        
        # Serialize and add to artifact
        content = safe_serialize(event.to_dict())
        
        self._encoder.add_text_block(
            content,
            metadata={
                "type": event_type.value,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "timestamp": event.timestamp,
            }
        )
        
        # Generate a block ID (simplified - using timestamp)
        block_id = f"{event_type.value}_{int(event.timestamp * 1000)}"
        self._block_ids.append(block_id)
        
        return block_id
    
    def log_binary_event(
        self,
        event_type: EventType,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a binary event to the MAIF artifact.
        
        Use this for embeddings, model weights, or other binary data.
        
        Args:
            event_type: The type of event to log
            data: Binary data to store
            metadata: Additional metadata
            
        Returns:
            The block ID of the logged event
        """
        self._ensure_initialized()
        
        if self._finalized:
            raise RuntimeError("Cannot log events to a finalized artifact")
        
        from maif.core.secure_format import SecureBlockType
        
        meta = metadata or {}
        meta["type"] = event_type.value
        meta["timestamp"] = time.time()
        meta["agent_id"] = self.agent_id
        
        self._encoder.add_binary_block(
            data=data,
            block_type=SecureBlockType.BINARY,
            metadata=meta,
        )
        
        block_id = f"{event_type.value}_{int(time.time() * 1000)}"
        self._block_ids.append(block_id)
        
        return block_id
    
    def get_artifact_path(self) -> str:
        """Get the path to the MAIF artifact."""
        return str(self.artifact_path)
    
    def get_block_ids(self) -> List[str]:
        """Get all block IDs logged in this session."""
        return self._block_ids.copy()
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact.
        
        This signs and seals the artifact. No more events can be logged
        after finalization.
        """
        if self._finalized:
            return
            
        if not self._initialized:
            # Nothing to finalize
            return
        
        # Log session end
        from maif.integrations._utils import safe_serialize
        
        session_end_data = {
            "event_type": EventType.SESSION_END.value,
            "agent_id": self.agent_id,
            "total_events": len(self._block_ids),
            "timestamp": time.time(),
        }
        
        self._encoder.add_text_block(
            safe_serialize(session_end_data),
            metadata={
                "type": "session_end",
                "agent_id": self.agent_id,
            }
        )
        
        self._encoder.finalize()
        self._finalized = True
        
        logger.info(
            f"Finalized MAIF artifact at {self.artifact_path} "
            f"with {len(self._block_ids)} events"
        )
    
    def __enter__(self) -> "MAIFProvenanceTracker":
        """Context manager entry."""
        self._ensure_initialized()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - finalize if auto_finalize is True."""
        if self.auto_finalize:
            self.finalize()


class BaseMAIFCallback(ABC):
    """Abstract base class for framework-specific callback handlers.
    
    This class provides the common structure that all framework callbacks
    should follow. Subclasses implement framework-specific event handling
    while inheriting consistent provenance tracking.
    
    Attributes:
        tracker: The MAIFProvenanceTracker instance for logging
        run_id_map: Maps framework run IDs to our internal IDs
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
    ):
        """Initialize the callback handler.
        
        Args:
            artifact_path: Path to the MAIF artifact
            agent_id: Identifier for this callback handler
        """
        self.tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id,
            auto_finalize=False,  # Manual finalization for callbacks
        )
        self.run_id_map: Dict[str, str] = {}
    
    def _convert_run_id(self, run_id: Any) -> Optional[str]:
        """Convert a framework run ID to a string.
        
        Args:
            run_id: The run ID from the framework (may be UUID, str, etc.)
            
        Returns:
            String representation of the run ID, or None
        """
        if run_id is None:
            return None
        return str(run_id)
    
    def finalize(self) -> None:
        """Finalize the underlying MAIF artifact."""
        self.tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the path to the MAIF artifact."""
        return self.tracker.get_artifact_path()
    
    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the name of the framework this callback supports."""
        pass

