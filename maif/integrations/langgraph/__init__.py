"""
MAIF LangGraph Integration

Provides a MAIF-backed checkpointer for LangGraph state persistence
with cryptographic provenance tracking.

Usage:
    from langgraph.graph import StateGraph
    from maif.integrations.langgraph import MAIFCheckpointer
    
    checkpointer = MAIFCheckpointer("graph_state.maif")
    app = graph.compile(checkpointer=checkpointer)
    
    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "my-thread"}}
    )

All state transitions are automatically logged to the MAIF artifact
with cryptographic signatures for tamper-evident audit trails.
"""

from maif.integrations.langgraph.checkpointer import MAIFCheckpointer

__all__ = ["MAIFCheckpointer"]

