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
    import importlib

    _submodules = {
        "langgraph": "maif.integrations.langgraph",
        "langchain": "maif.integrations.langchain",
        "crewai": "maif.integrations.crewai",
        "strands": "maif.integrations.strands",
    }

    if name in _submodules:
        module = importlib.import_module(_submodules[name])
        globals()[name] = module  # Cache it
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

