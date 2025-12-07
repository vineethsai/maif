"""
MAIF LangChain Integration

Provides MAIF-backed components for LangChain:
- MAIFCallbackHandler: Callback handler for provenance tracking
- MAIFVectorStore: Vector store with embedded provenance
- MAIFChatMessageHistory: Chat history with cryptographic audit

Status: Not yet implemented

Usage (when implemented):
    from maif.integrations.langchain import MAIFCallbackHandler
    
    handler = MAIFCallbackHandler("session.maif")
    llm.invoke("Hello", config={"callbacks": [handler]})
    handler.finalize()

See maif/integrations/INTEGRATION_PLAN.md for implementation details.
"""

# Placeholder - to be implemented
# from maif.integrations.langchain.callback import MAIFCallbackHandler
# from maif.integrations.langchain.vectorstore import MAIFVectorStore
# from maif.integrations.langchain.memory import MAIFChatMessageHistory

__all__ = [
    # "MAIFCallbackHandler",
    # "MAIFVectorStore",
    # "MAIFChatMessageHistory",
]


def __getattr__(name: str):
    """Raise informative error for unimplemented components."""
    if name in ("MAIFCallbackHandler", "MAIFVectorStore", "MAIFChatMessageHistory"):
        raise NotImplementedError(
            f"{name} is not yet implemented. "
            "See maif/integrations/INTEGRATION_PLAN.md for implementation details."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

