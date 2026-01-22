"""
MAIF Retriever for LangChain RAG Applications.

Provides a LangChain-compatible retriever that searches MAIF blocks
by semantic similarity for use in RAG (Retrieval-Augmented Generation)
applications.
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path

try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.embeddings import Embeddings
    from pydantic import Field, PrivateAttr
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseRetriever = object
    Document = dict
    Embeddings = object
    CallbackManagerForRetrieverRun = None
    AsyncCallbackManagerForRetrieverRun = None

    # Minimal Field implementation for when pydantic isn't available
    def Field(*args, **kwargs):
        return kwargs.get("default", None)

    def PrivateAttr(*args, **kwargs):
        return kwargs.get("default", None)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class MAIFRetriever(BaseRetriever if LANGCHAIN_AVAILABLE else object):
    """MAIF Retriever for LangChain RAG applications.

    Searches MAIF blocks by semantic similarity using embeddings.
    Supports both in-memory indexing and lazy loading for large files.
    Preserves full provenance metadata for audit trails.

    Usage:
        from langchain_openai import OpenAIEmbeddings
        from maif.integrations.langchain import MAIFRetriever

        embeddings = OpenAIEmbeddings()
        retriever = MAIFRetriever(
            file_paths=["docs.maif"],
            embeddings=embeddings,
            k=4,
        )

        # Search for relevant documents
        docs = retriever.invoke("What is MAIF?")

        # Use in a RAG chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_template(
            "Answer based on context: {context}\\n\\nQuestion: {question}"
        )
        llm = ChatOpenAI()

        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
        )

        answer = chain.invoke("What is MAIF?")

    With provenance tracking:
        from maif.integrations.langchain import MAIFRetriever, MAIFCallbackHandler

        handler = MAIFCallbackHandler("rag_session.maif")
        retriever = MAIFRetriever(
            file_paths=["docs.maif"],
            embeddings=embeddings,
            track_provenance=True,
            provenance_handler=handler,
        )

    Metadata in Retrieved Documents:
        - source: Path to the source MAIF file
        - block_id: Unique identifier for the block
        - similarity_score: Cosine similarity score
        - signature: Block signature (if available)
        - provenance: Provenance chain information
    """

    # Pydantic model fields
    k: int = Field(default=4, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score threshold"
    )
    search_type: str = Field(
        default="similarity",
        description="Search type: 'similarity' or 'mmr'"
    )
    fetch_k: int = Field(
        default=20,
        description="Number of documents to fetch for MMR"
    )
    lambda_mult: float = Field(
        default=0.5,
        description="Lambda multiplier for MMR diversity (0=max diversity, 1=max relevance)"
    )

    # Private attributes (not part of the pydantic model)
    _file_paths: List[Path] = PrivateAttr(default_factory=list)
    _embeddings: Any = PrivateAttr(default=None)
    _documents: List[Document] = PrivateAttr(default_factory=list)
    _doc_embeddings: List[List[float]] = PrivateAttr(default_factory=list)
    _indexed: bool = PrivateAttr(default=False)
    _track_provenance: bool = PrivateAttr(default=False)
    _provenance_handler: Any = PrivateAttr(default=None)
    _text_blocks_only: bool = PrivateAttr(default=True)
    _verify_signatures: bool = PrivateAttr(default=True)

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        embeddings: Embeddings,
        k: int = 4,
        score_threshold: Optional[float] = None,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        track_provenance: bool = False,
        provenance_handler: Any = None,
        text_blocks_only: bool = True,
        verify_signatures: bool = True,
        **kwargs: Any,
    ):
        """Initialize the MAIF retriever.

        Args:
            file_paths: Path(s) to MAIF file(s) to search
            embeddings: Embeddings model for semantic similarity
            k: Number of documents to retrieve (default: 4)
            score_threshold: Minimum similarity score (optional)
            search_type: "similarity" or "mmr" (maximal marginal relevance)
            fetch_k: Number of docs to fetch for MMR reranking
            lambda_mult: MMR diversity parameter (0=diverse, 1=relevant)
            track_provenance: If True, log retrieval operations
            provenance_handler: MAIFCallbackHandler for provenance tracking
            text_blocks_only: If True, only index text blocks
            verify_signatures: If True, verify block signatures
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MAIFRetriever. "
                "Install with: pip install langchain-core"
            )

        if not NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for MAIFRetriever. "
                "Install with: pip install numpy"
            )

        # Initialize parent class with pydantic fields
        super().__init__(
            k=k,
            score_threshold=score_threshold,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

        # Set private attributes
        if isinstance(file_paths, (str, Path)):
            self._file_paths = [Path(file_paths)]
        else:
            self._file_paths = [Path(p) for p in file_paths]

        self._embeddings = embeddings
        self._documents = []
        self._doc_embeddings = []
        self._indexed = False
        self._track_provenance = track_provenance
        self._provenance_handler = provenance_handler
        self._text_blocks_only = text_blocks_only
        self._verify_signatures = verify_signatures

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get documents relevant to a query.

        This is the core retrieval method called by invoke().

        Args:
            query: The search query
            run_manager: Callback manager for tracking

        Returns:
            List of relevant documents
        """
        # Ensure documents are indexed
        if not self._indexed:
            self._build_index()

        if not self._documents:
            return []

        # Get query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Perform search based on search type
        if self.search_type == "mmr":
            results = self._mmr_search(query_embedding)
        else:
            results = self._similarity_search(query_embedding)

        # Log provenance if enabled
        if self._track_provenance and self._provenance_handler:
            self._log_retrieval(query, results)

        return results

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version of get_relevant_documents.

        Args:
            query: The search query
            run_manager: Async callback manager

        Returns:
            List of relevant documents
        """
        # For now, just call the sync version
        # In a full implementation, this would use async embeddings
        return self._get_relevant_documents(query)

    def _build_index(self) -> None:
        """Build the document index from MAIF files."""
        from maif.integrations.langchain.loader import MAIFLoader

        all_documents = []

        for file_path in self._file_paths:
            if not file_path.exists():
                continue

            loader = MAIFLoader(
                file_path=file_path,
                text_blocks_only=self._text_blocks_only,
                verify_signatures=self._verify_signatures,
                include_provenance=True,
            )

            all_documents.extend(loader.load())

        if not all_documents:
            self._indexed = True
            return

        # Generate embeddings for all documents
        texts = [doc.page_content for doc in all_documents]
        self._doc_embeddings = self._embeddings.embed_documents(texts)
        self._documents = all_documents
        self._indexed = True

    def _similarity_search(
        self,
        query_embedding: List[float],
    ) -> List[Document]:
        """Perform similarity search.

        Args:
            query_embedding: Query vector

        Returns:
            Top-k similar documents
        """
        if not self._doc_embeddings:
            return []

        # Calculate cosine similarities
        query_vec = np.array(query_embedding)
        similarities = []

        for i, doc_embedding in enumerate(self._doc_embeddings):
            doc_vec = np.array(doc_embedding)
            similarity = self._cosine_similarity(query_vec, doc_vec)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold and limit
        results = []
        for idx, score in similarities[:self.k]:
            if self.score_threshold is not None and score < self.score_threshold:
                continue

            doc = self._documents[idx]
            # Add similarity score to metadata
            doc_with_score = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "similarity_score": float(score),
                },
            )
            results.append(doc_with_score)

        return results

    def _mmr_search(
        self,
        query_embedding: List[float],
    ) -> List[Document]:
        """Perform MMR (Maximal Marginal Relevance) search.

        MMR balances relevance with diversity in the results.

        Args:
            query_embedding: Query vector

        Returns:
            Top-k diverse and relevant documents
        """
        if not self._doc_embeddings:
            return []

        query_vec = np.array(query_embedding)

        # Calculate initial similarities
        similarities = []
        for i, doc_embedding in enumerate(self._doc_embeddings):
            doc_vec = np.array(doc_embedding)
            similarity = self._cosine_similarity(query_vec, doc_vec)
            similarities.append((i, similarity))

        # Sort and get top fetch_k candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = similarities[:self.fetch_k]

        # MMR selection
        selected_indices = []
        selected_embeddings = []

        while len(selected_indices) < self.k and candidates:
            best_score = -1
            best_idx = -1
            best_candidate_idx = -1

            for i, (doc_idx, relevance) in enumerate(candidates):
                if doc_idx in selected_indices:
                    continue

                # Calculate maximum similarity to already selected docs
                max_sim_to_selected = 0
                if selected_embeddings:
                    doc_vec = np.array(self._doc_embeddings[doc_idx])
                    for sel_vec in selected_embeddings:
                        sim = self._cosine_similarity(doc_vec, np.array(sel_vec))
                        max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
                mmr_score = (
                    self.lambda_mult * relevance -
                    (1 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = doc_idx
                    best_candidate_idx = i

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            selected_embeddings.append(self._doc_embeddings[best_idx])
            candidates.pop(best_candidate_idx)

        # Build result documents
        results = []
        for idx in selected_indices:
            doc = self._documents[idx]
            # Find original similarity score
            original_score = next(
                (score for i, score in similarities if i == idx),
                0.0
            )

            if self.score_threshold is not None and original_score < self.score_threshold:
                continue

            doc_with_score = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "similarity_score": float(original_score),
                    "search_type": "mmr",
                },
            )
            results.append(doc_with_score)

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot / norm) if norm > 0 else 0.0

    def _log_retrieval(self, query: str, results: List[Document]) -> None:
        """Log retrieval operation for provenance tracking.

        Args:
            query: The search query
            results: Retrieved documents
        """
        if not self._provenance_handler:
            return

        from maif.integrations._base import EventType

        tracker = getattr(self._provenance_handler, "_tracker", None)
        if tracker is None:
            return

        result_info = []
        for doc in results:
            result_info.append({
                "block_id": doc.metadata.get("block_id"),
                "source": doc.metadata.get("source"),
                "similarity_score": doc.metadata.get("similarity_score"),
                "content_preview": doc.page_content[:200],
            })

        tracker.log_event(
            event_type=EventType.RETRIEVAL_END,
            data={
                "query": query[:500],
                "k": self.k,
                "search_type": self.search_type,
                "num_results": len(results),
                "results": result_info[:10],  # Limit for size
            },
            metadata={
                "type": "maif_retrieval",
                "timestamp": time.time(),
            },
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add additional documents to the index.

        Args:
            documents: Documents to add
        """
        if not documents:
            return

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        new_embeddings = self._embeddings.embed_documents(texts)

        # Add to index
        self._documents.extend(documents)
        self._doc_embeddings.extend(new_embeddings)

    def add_maif_files(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> None:
        """Add additional MAIF files to the index.

        Args:
            file_paths: Path(s) to MAIF file(s) to add
        """
        if isinstance(file_paths, (str, Path)):
            new_paths = [Path(file_paths)]
        else:
            new_paths = [Path(p) for p in file_paths]

        self._file_paths.extend(new_paths)

        # Re-index if already indexed
        if self._indexed:
            from maif.integrations.langchain.loader import MAIFLoader

            for file_path in new_paths:
                if not file_path.exists():
                    continue

                loader = MAIFLoader(
                    file_path=file_path,
                    text_blocks_only=self._text_blocks_only,
                    verify_signatures=self._verify_signatures,
                    include_provenance=True,
                )

                new_docs = loader.load()
                self.add_documents(new_docs)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[tuple[Document, float]]:
        """Search with explicit score return.

        Args:
            query: Search query
            k: Number of results (defaults to self.k)

        Returns:
            List of (document, score) tuples
        """
        # Temporarily override k if specified
        original_k = self.k
        if k is not None:
            self.k = k

        try:
            docs = self._get_relevant_documents(query)
            return [
                (doc, doc.metadata.get("similarity_score", 0.0))
                for doc in docs
            ]
        finally:
            self.k = original_k

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        if not self._indexed:
            self._build_index()

        return {
            "num_documents": len(self._documents),
            "num_files": len(self._file_paths),
            "file_paths": [str(p) for p in self._file_paths],
            "indexed": self._indexed,
            "search_type": self.search_type,
            "k": self.k,
            "score_threshold": self.score_threshold,
        }

    def clear_index(self) -> None:
        """Clear the document index."""
        self._documents = []
        self._doc_embeddings = []
        self._indexed = False

    def rebuild_index(self) -> None:
        """Force rebuild of the document index."""
        self.clear_index()
        self._build_index()
