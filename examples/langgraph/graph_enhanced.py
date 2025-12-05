"""
Enhanced LangGraph with real vector DB and LLM fact-checking.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from examples.langgraph.state import RAGState
from examples.langgraph.nodes.init_session import init_session_node
from examples.langgraph.nodes.retrieve_enhanced import retrieve_node_enhanced
from examples.langgraph.nodes.synthesize import synthesize_node
from examples.langgraph.nodes.fact_check_enhanced import fact_check_node_enhanced
from examples.langgraph.nodes.cite import citation_node


def build_enhanced_graph() -> StateGraph:
    """
    Build the ENHANCED LangGraph with:
    - Real ChromaDB vector search
    - LLM-based fact-checking
    - Real embeddings

    Flow:
        START
          ↓
        init_session → Create/load session artifact
          ↓
        retrieve_enhanced → ChromaDB semantic search + MAIF logging
          ↓
        synthesize → Generate answer with Gemini
          ↓
        fact_check_enhanced → LLM verification + MAIF logging
          ↓
          ├─ verified? → cite → END
          └─ needs_revision? → synthesize (loop)

    Returns:
        StateGraph ready for compilation
    """
    # Create graph
    graph = StateGraph(RAGState)

    # Add nodes (with enhanced versions!)
    graph.add_node("init_session", init_session_node)
    graph.add_node("retrieve", retrieve_node_enhanced)  # Enhanced with ChromaDB
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("fact_check", fact_check_node_enhanced)  # Enhanced with LLM
    graph.add_node("cite", citation_node)

    # Add edges
    graph.add_edge(START, "init_session")
    graph.add_edge("init_session", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "fact_check")

    # Conditional routing after fact_check
    def route_after_fact_check(state: RAGState) -> str:
        """Route based on verification status."""
        verification_status = state.get("verification_status", "unknown")
        needs_revision = state.get("needs_revision", False)

        if needs_revision and verification_status == "needs_revision":
            return "synthesize"  # Revise
        else:
            return "cite"  # Finish

    graph.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {"synthesize": "synthesize", "cite": "cite"},
    )

    graph.add_edge("cite", END)

    return graph


def create_enhanced_app(
    checkpoints_db: str = "examples/langgraph/data/checkpoints_enhanced.db",
):
    """
    Create the ENHANCED LangGraph application.

    Args:
        checkpoints_db: Path to SQLite checkpoint database

    Returns:
        Compiled LangGraph app with all enhancements
    """
    # Ensure directory exists
    Path(checkpoints_db).parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint database
    conn = sqlite3.connect(checkpoints_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Build and compile graph
    graph = build_enhanced_graph()
    app = graph.compile(checkpointer=checkpointer)

    return app


if __name__ == "__main__":
    print("🏗️  Building ENHANCED LangGraph...")
    print("  Real ChromaDB vector search")
    print("  Real embeddings (sentence-transformers)")
    print("  LLM fact-checking (Gemini)")
    print("  Multi-turn conversations")

    app = create_enhanced_app()
    print("\n Enhanced graph built successfully!")

    print("\n Enhanced Features:")
    print(" 1. ChromaDB for semantic search")
    print(" 2. Sentence-transformers embeddings")
    print(" 3. Gemini-powered fact-checking")
    print(" 4. MAIF provenance for everything")
