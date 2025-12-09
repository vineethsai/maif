#!/usr/bin/env python3
"""
Smoke test for CrewAI enhanced demo.

Runs KB creation (if needed) and a single QA pipeline in offline mode.
"""

import os
from pathlib import Path

from create_kb_enhanced import build_kb
from maif_utils import SessionManager
from nodes.retrieve import run_retrieve
from nodes.synthesize import run_synthesize
from nodes.fact_check import run_fact_check
from nodes.cite import run_citations
from state import RAGState


def ensure_kb():
    kb_dir = Path("examples/crewai_enhanced/data/kb")
    if not any(kb_dir.glob("*.maif")):
        build_kb()


def run_smoke():
    # Force offline path for LLM calls if key not present
    os.environ.setdefault("GEMINI_API_KEY", "")

    ensure_kb()
    session_mgr = SessionManager("examples/crewai_enhanced/data/sessions")
    session_path = session_mgr.create_session("smoke-test")

    state: RAGState = {
        "question": "What are key climate change mitigation strategies?",
        "session_artifact_path": session_path,
        "retrieved_chunks": [],
        "messages": [],
    }

    session_mgr.log_user_message(session_path, state["question"])
    state = run_retrieve(state, session_mgr)
    state = run_synthesize(state, session_mgr)
    state = run_fact_check(state, session_mgr)
    state = run_citations(state, session_mgr)

    print("Answer:", state.get("answer"))
    print("Confidence:", state.get("confidence"))
    print("Citations:", state.get("citations"))
    print("Session artifact:", session_path)


if __name__ == "__main__":
    run_smoke()


