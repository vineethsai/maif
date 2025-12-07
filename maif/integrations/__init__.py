"""
MAIF Framework Integrations

This package provides drop-in integrations for popular AI agent frameworks,
enabling cryptographic provenance tracking with minimal code changes.

Supported Frameworks:
    - LangChain: Callbacks, VectorStore, ChatMessageHistory
    - CrewAI: Crew/Agent callbacks, Memory
    - LangGraph: Checkpointer for state persistence
    - Strands (AWS): Agent callbacks

Installation:
    pip install maif[integrations]

Quick Start:
    # LangGraph example
    from maif.integrations.langgraph import MAIFCheckpointer
    
    checkpointer = MAIFCheckpointer("graph_state.maif")
    app = graph.compile(checkpointer=checkpointer)

Each subpackage provides framework-specific implementations that automatically
log all agent actions to MAIF artifacts with cryptographic provenance.
"""

from maif.integrations._base import (
    EventType,
    ProvenanceEvent,
    MAIFProvenanceTracker,
    BaseMAIFCallback,
)

from maif.integrations._utils import (
    safe_serialize,
    format_timestamp,
    generate_run_id,
    truncate_string,
    extract_error_info,
    SerializationError,
)

# Lazy imports for framework-specific modules
# These are imported only when accessed to avoid requiring all dependencies

__all__ = [
    # Base classes
    "EventType",
    "ProvenanceEvent",
    "MAIFProvenanceTracker",
    "BaseMAIFCallback",
    # Utilities
    "safe_serialize",
    "format_timestamp",
    "generate_run_id",
    "truncate_string",
    "extract_error_info",
    "SerializationError",
]


def __getattr__(name: str):
    """Lazy loading of framework-specific modules."""
    
    # LangGraph
    if name == "langgraph":
        from maif.integrations import langgraph as _langgraph
        return _langgraph
    
    # LangChain
    if name == "langchain":
        from maif.integrations import langchain as _langchain
        return _langchain
    
    # CrewAI
    if name == "crewai":
        from maif.integrations import crewai as _crewai
        return _crewai
    
    # Strands
    if name == "strands":
        from maif.integrations import strands as _strands
        return _strands
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

