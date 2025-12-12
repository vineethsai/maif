"""
Vector DB wrapper for CrewAI enhanced demo (Chroma + sentence-transformers).
"""

from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDB:
    """Persistent ChromaDB with sentence-transformers embeddings."""

    def __init__(
        self,
        persist_directory: str = "examples/crewai_enhanced/data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.collection_name = "knowledge_base"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "MAIF CrewAI knowledge base"},
            )
        return collection

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> list:
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,  # ensures ndarray
        )
        return embeddings.tolist()

    def add_documents(
        self, doc_id: str, chunks: List[Dict], document_metadata: Optional[Dict] = None
    ):
        texts = [c["text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)
        # Ensure embeddings are a list of list[float]
        embeddings = [
            emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings
        ]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {"doc_id": doc_id, "chunk_index": i, "chunk_id": f"{doc_id}_{i}"}
            if "metadata" in chunk:
                meta.update(chunk["metadata"])
            if document_metadata:
                for k, v in document_metadata.items():
                    meta.setdefault(f"doc_{k}", str(v))
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self, query: str, top_k: int = 5, filter_doc_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        where = {"doc_id": {"$in": filter_doc_ids}} if filter_doc_ids else None
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - (distance / 2.0))
            chunks.append(
                {
                    "doc_id": results["metadatas"][0][i].get("doc_id", "unknown"),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "text": results["documents"][0][i],
                    "score": float(score),
                    "block_id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        return chunks

    def get_stats(self) -> Dict:
        count = self.collection.count()
        num_documents = 0
        if count > 0:
            results = self.collection.get(limit=count, include=["metadatas"])
            doc_ids = {meta.get("doc_id", "unknown") for meta in results["metadatas"]}
            num_documents = len(doc_ids)

        return {
            "total_chunks": count,
            "num_documents": num_documents,
            "embedding_dimension": self.embedding_dim,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }


_vector_db_instance: Optional[VectorDB] = None


def get_vector_db() -> VectorDB:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDB()
    return _vector_db_instance


