"""
MAIF Memory Backend for CrewAI

Provides MAIF-backed memory storage for CrewAI agents, enabling:
- Long-term memory persistence with cryptographic provenance
- Searchable memory via MAIF's semantic capabilities
- Cross-session memory retrieval
- Tamper-evident memory audit trails

This module implements custom memory components that can be used
with CrewAI's memory system.
"""

import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from maif.integrations._base import EventType, MAIFProvenanceTracker
from maif.integrations._utils import (
    safe_serialize,
    truncate_string,
    generate_run_id,
    safe_get_attr,
)

logger = logging.getLogger(__name__)


class MAIFCrewMemory:
    """MAIF-backed memory storage for CrewAI agents.
    
    This class provides a memory backend that stores all agent memories
    in a MAIF artifact with cryptographic provenance. It can be used to:
    
    - Persist agent knowledge across sessions
    - Search and retrieve relevant memories
    - Maintain tamper-evident audit trails of memory operations
    - Share memories between agents in a crew
    
    Usage:
        from maif.integrations.crewai import MAIFCrewMemory
        
        memory = MAIFCrewMemory("agent_memory.maif")
        
        # Store a memory
        memory.save(
            content="The user prefers concise responses",
            agent="researcher",
            tags=["preference", "user"],
        )
        
        # Search memories
        results = memory.search("user preferences", limit=5)
        
        # Get all memories for an agent
        agent_memories = memory.get_by_agent("researcher")
        
        memory.finalize()
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier for the memory system
        auto_finalize: Whether to finalize on context manager exit
        
    Attributes:
        tracker: The underlying MAIFProvenanceTracker
        memories: In-memory index of stored memories
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
        auto_finalize: bool = True,
    ):
        self.artifact_path = Path(artifact_path)
        self._agent_id = agent_id or "crewai_memory"
        self._auto_finalize = auto_finalize
        
        # Initialize tracker
        self._tracker = MAIFProvenanceTracker(
            artifact_path=self.artifact_path,
            agent_id=self._agent_id,
            auto_finalize=False,
        )
        
        # In-memory index of memories
        self._memories: List[Dict[str, Any]] = []
        self._memory_count = 0
        
        # Load existing memories if artifact exists
        self._load_existing_memories()
    
    def _load_existing_memories(self) -> None:
        """Load existing memories from MAIF artifact if it exists."""
        if not self.artifact_path.exists():
            return
        
        try:
            from maif import MAIFDecoder
            
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            
            for block in decoder.blocks:
                metadata = block.metadata or {}
                # Check both 'type' and 'event_type' keys for compatibility
                block_type = metadata.get("type") or metadata.get("event_type", "")
                if block_type in [
                    EventType.MEMORY_SAVE.value,
                    "memory_save",
                ]:
                    try:
                        data = block.data
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        memory_data = json.loads(data)
                        
                        # Extract the memory content from the event data
                        # The actual memory is nested in the 'data' field of the event
                        if "data" in memory_data and isinstance(memory_data["data"], dict):
                            mem = memory_data["data"]
                            # Only add if it has required fields
                            if "content" in mem or "id" in mem:
                                self._memories.append(mem)
                                self._memory_count += 1
                        elif "content" in memory_data:
                            # Direct memory format
                            self._memories.append(memory_data)
                            self._memory_count += 1
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(f"Failed to parse memory block: {e}")
                        pass  # Skip malformed memories
                        
        except Exception as e:
            logger.warning(f"Failed to load existing memories: {e}")
    
    def save(
        self,
        content: str,
        agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """Save a memory to the MAIF artifact.
        
        Args:
            content: The memory content to store
            agent: Agent that created/owns this memory
            tags: Optional tags for categorization
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            The memory ID
        """
        self._memory_count += 1
        memory_id = f"mem_{generate_run_id()[:8]}_{self._memory_count}"
        
        memory_data = {
            "id": memory_id,
            "content": content,
            "agent": agent,
            "tags": tags or [],
            "importance": min(1.0, max(0.0, importance)),
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        
        # Add to in-memory index
        self._memories.append(memory_data)
        
        # Log to MAIF artifact
        self._tracker.log_event(
            event_type=EventType.MEMORY_SAVE,
            data=memory_data,
            metadata={
                "memory_id": memory_id,
                "agent": agent,
                "tags": tags or [],
                "framework": "crewai",
            },
        )
        
        logger.debug(f"Saved memory {memory_id} for agent {agent}")
        return memory_id
    
    def search(
        self,
        query: str,
        limit: int = 10,
        agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search memories by content.
        
        This performs a simple keyword-based search. For semantic search,
        use MAIF's semantic capabilities directly.
        
        Args:
            query: Search query
            limit: Maximum number of results
            agent: Filter by agent
            tags: Filter by tags (any match)
            min_importance: Minimum importance score
            
        Returns:
            List of matching memories sorted by relevance
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for memory in self._memories:
            # Apply filters
            if agent and memory.get("agent") != agent:
                continue
            
            if tags:
                memory_tags = set(memory.get("tags", []))
                if not memory_tags.intersection(tags):
                    continue
            
            if memory.get("importance", 0.5) < min_importance:
                continue
            
            # Score by keyword match
            content_lower = memory.get("content", "").lower()
            content_words = set(content_lower.split())
            
            matching_words = query_words.intersection(content_words)
            if matching_words:
                score = len(matching_words) / len(query_words)
                results.append((memory, score))
            elif query_lower in content_lower:
                # Substring match
                results.append((memory, 0.5))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]
    
    def get_by_agent(
        self,
        agent: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific agent.
        
        Args:
            agent: Agent role/name
            limit: Maximum number of memories to return
            
        Returns:
            List of memories for the agent, sorted by creation time (newest first)
        """
        agent_memories = [
            m for m in self._memories
            if m.get("agent") == agent
        ]
        
        # Sort by creation time (newest first)
        agent_memories.sort(
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )
        
        if limit:
            agent_memories = agent_memories[:limit]
        
        return agent_memories
    
    def get_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get memories by tags.
        
        Args:
            tags: Tags to search for
            match_all: If True, require all tags; if False, any tag matches
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        tag_set = set(tags)
        results = []
        
        for memory in self._memories:
            memory_tags = set(memory.get("tags", []))
            
            if match_all:
                if tag_set.issubset(memory_tags):
                    results.append(memory)
            else:
                if tag_set.intersection(memory_tags):
                    results.append(memory)
        
        # Sort by importance then creation time
        results.sort(
            key=lambda x: (x.get("importance", 0.5), x.get("created_at", 0)),
            reverse=True,
        )
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_recent(
        self,
        limit: int = 10,
        agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get the most recent memories.
        
        Args:
            limit: Number of memories to return
            agent: Optional agent filter
            
        Returns:
            List of recent memories
        """
        memories = self._memories
        if agent:
            memories = [m for m in memories if m.get("agent") == agent]
        
        # Sort by creation time (newest first)
        sorted_memories = sorted(
            memories,
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )
        
        return sorted_memories[:limit]
    
    def get_important(
        self,
        limit: int = 10,
        min_importance: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Get high-importance memories.
        
        Args:
            limit: Number of memories to return
            min_importance: Minimum importance threshold
            
        Returns:
            List of important memories
        """
        important = [
            m for m in self._memories
            if m.get("importance", 0.5) >= min_importance
        ]
        
        # Sort by importance (highest first)
        important.sort(
            key=lambda x: x.get("importance", 0.5),
            reverse=True,
        )
        
        return important[:limit]
    
    def update_importance(
        self,
        memory_id: str,
        importance: float,
    ) -> bool:
        """Update the importance of a memory.
        
        Args:
            memory_id: The memory ID
            importance: New importance score (0.0 to 1.0)
            
        Returns:
            True if memory was found and updated
        """
        for memory in self._memories:
            if memory.get("id") == memory_id:
                old_importance = memory.get("importance", 0.5)
                memory["importance"] = min(1.0, max(0.0, importance))
                
                # Log the update
                self._tracker.log_event(
                    event_type=EventType.CUSTOM,
                    data={
                        "memory_id": memory_id,
                        "old_importance": old_importance,
                        "new_importance": memory["importance"],
                    },
                    metadata={
                        "framework": "crewai",
                        "event_subtype": "memory_importance_update",
                    },
                )
                return True
        
        return False
    
    def delete(self, memory_id: str) -> bool:
        """Mark a memory as deleted.
        
        Note: In MAIF, we don't actually delete data for audit purposes.
        The memory is marked as deleted in the index and logged.
        
        Args:
            memory_id: The memory ID to delete
            
        Returns:
            True if memory was found and deleted
        """
        for i, memory in enumerate(self._memories):
            if memory.get("id") == memory_id:
                deleted_memory = self._memories.pop(i)
                
                # Log deletion event
                self._tracker.log_event(
                    event_type=EventType.CUSTOM,
                    data={
                        "memory_id": memory_id,
                        "deleted_content": truncate_string(
                            deleted_memory.get("content", ""), 500
                        ),
                        "agent": deleted_memory.get("agent"),
                    },
                    metadata={
                        "framework": "crewai",
                        "event_subtype": "memory_delete",
                    },
                )
                return True
        
        return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories.
        
        Returns:
            List of all memories
        """
        return list(self._memories)
    
    def count(
        self,
        agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Count memories matching filters.
        
        Args:
            agent: Optional agent filter
            tags: Optional tags filter
            
        Returns:
            Number of matching memories
        """
        count = 0
        for memory in self._memories:
            if agent and memory.get("agent") != agent:
                continue
            if tags:
                memory_tags = set(memory.get("tags", []))
                if not memory_tags.intersection(tags):
                    continue
            count += 1
        return count
    
    def clear_agent_memories(self, agent: str) -> int:
        """Clear all memories for an agent.
        
        Args:
            agent: Agent whose memories to clear
            
        Returns:
            Number of memories cleared
        """
        original_count = len(self._memories)
        self._memories = [
            m for m in self._memories
            if m.get("agent") != agent
        ]
        cleared = original_count - len(self._memories)
        
        if cleared > 0:
            self._tracker.log_event(
                event_type=EventType.CUSTOM,
                data={
                    "agent": agent,
                    "memories_cleared": cleared,
                },
                metadata={
                    "framework": "crewai",
                    "event_subtype": "memory_clear_agent",
                },
            )
        
        return cleared
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact.
        
        This seals the artifact with cryptographic signatures.
        """
        self._tracker.finalize()
        logger.info(
            f"Finalized CrewAI memory artifact at {self.artifact_path} "
            f"with {len(self._memories)} memories"
        )
    
    def get_artifact_path(self) -> str:
        """Get the path to the MAIF artifact."""
        return str(self.artifact_path)
    
    def __enter__(self) -> "MAIFCrewMemory":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._auto_finalize:
            self.finalize()
    
    def __len__(self) -> int:
        """Return number of memories."""
        return len(self._memories)

