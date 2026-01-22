"""
MAIF LangChain Integration

Status: Beta

Provides MAIF-backed components for LangChain with cryptographic provenance:

- MAIFCallbackHandler: Track all LLM/chain calls
  Status: Production-ready - Full implementation of all LangChain callbacks

- MAIFChatMessageHistory: Chat memory with audit trail
  Status: Production-ready - Full BaseChatMessageHistory implementation

- MAIFVectorStore: Store embeddings with provenance
  Status: Experimental - In-memory only, suitable for small datasets (<10k docs)
  Limitations: No persistence of full embeddings, no delete/update, basic cosine similarity

- MAIFLoader: Document loader for MAIF files
  Status: Production-ready - Full BaseLoader implementation
  Features: Extracts text blocks as Documents, preserves signatures and provenance

- MAIFRetriever: RAG retriever for semantic search over MAIF blocks
  Status: Production-ready - Full BaseRetriever implementation
  Features: Similarity and MMR search, provenance tracking, score thresholds

Usage:
    # Callback Handler
    from langchain_openai import ChatOpenAI
    from maif.integrations.langchain import MAIFCallbackHandler

    handler = MAIFCallbackHandler("session.maif")
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("Hello!")
    handler.finalize()

    # Document Loader
    from maif.integrations.langchain import MAIFLoader

    loader = MAIFLoader("documents.maif")
    docs = loader.load()
    for doc in docs:
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Signature: {doc.metadata.get('signature')}")

    # Retriever for RAG
    from langchain_openai import OpenAIEmbeddings
    from maif.integrations.langchain import MAIFRetriever

    retriever = MAIFRetriever(
        file_paths=["docs.maif"],
        embeddings=OpenAIEmbeddings(),
        k=4,
    )
    relevant_docs = retriever.invoke("What is MAIF?")

Note: All classes work without LangChain installed (they gracefully fail at
instantiation with an ImportError explaining the missing dependency).
"""

# Check if LangChain is available
try:
    from langchain_core.document_loaders.base import BaseLoader
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import classes - they handle missing LangChain gracefully
from maif.integrations.langchain.callback import MAIFCallbackHandler
from maif.integrations.langchain.vectorstore import MAIFVectorStore
from maif.integrations.langchain.memory import MAIFChatMessageHistory
from maif.integrations.langchain.loader import MAIFLoader
from maif.integrations.langchain.retriever import MAIFRetriever

__all__ = [
    # Availability flag
    "LANGCHAIN_AVAILABLE",
    # Core components
    "MAIFCallbackHandler",
    "MAIFVectorStore",
    "MAIFChatMessageHistory",
    # Document loading
    "MAIFLoader",
    # RAG retrieval
    "MAIFRetriever",
]
