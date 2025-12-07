"""
MAIF LangGraph Integration

Provides a MAIF-backed checkpointer for LangGraph state persistence
with cryptographic provenance tracking.

Quick Start:
    from langgraph.graph import StateGraph
    from maif.integrations.langgraph import MAIFCheckpointer
    
    checkpointer = MAIFCheckpointer("graph_state.maif")
    app = graph.compile(checkpointer=checkpointer)
    
    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "my-thread"}}
    )

Migration from SqliteSaver:
    from maif.integrations.langgraph import migrate_from_sqlite
    
    migrate_from_sqlite("checkpoints.db", "checkpoints.maif")

CLI Tools:
    python -m maif.integrations.langgraph.cli inspect state.maif
    python -m maif.integrations.langgraph.cli verify state.maif
    python -m maif.integrations.langgraph.cli export state.maif --format json
    python -m maif.integrations.langgraph.cli migrate checkpoints.db state.maif

All state transitions are automatically logged to the MAIF artifact
with cryptographic signatures for tamper-evident audit trails.
"""

from maif.integrations.langgraph.checkpointer import MAIFCheckpointer
from maif.integrations.langgraph.migration import (
    migrate_from_sqlite,
    compare_checkpointers,
)
from maif.integrations.langgraph.patterns import (
    create_chat_graph,
    create_rag_graph,
    create_multi_agent_graph,
    finalize_graph,
    get_artifact_path,
    # State types for type hints
    ChatState,
    RAGState,
    MultiAgentState,
    ChatMessage,
)

__all__ = [
    # Core
    "MAIFCheckpointer",
    # Migration
    "migrate_from_sqlite",
    "compare_checkpointers",
    # Patterns
    "create_chat_graph",
    "create_rag_graph",
    "create_multi_agent_graph",
    "finalize_graph",
    "get_artifact_path",
    # Types
    "ChatState",
    "RAGState",
    "MultiAgentState",
    "ChatMessage",
]

