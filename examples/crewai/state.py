"""
Shared state for CrewAI enhanced RAG demo.
"""

from typing import List, Dict, Optional, TypedDict, Any


class RAGState(TypedDict, total=False):
    question: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    verification: Dict[str, Any]
    citations: List[Dict[str, Any]]
    session_artifact_path: str
    kb_artifact_paths: Dict[str, str]
    messages: List[Dict[str, str]]
    confidence: float
    needs_revision: bool
    iteration: int
    max_iterations: int


