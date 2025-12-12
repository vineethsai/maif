#!/usr/bin/env python3
"""
CrewAI Enhanced RAG Demo (Gemini + Chroma + MAIF) with LangGraph-parity menu.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from demo_agents import run_demo as run_agents_pipeline
from vector_db import get_vector_db
from maif import MAIFDecoder
from maif_utils import SessionManager

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
SESSIONS_DIR = DATA_DIR / "sessions"


def ensure_paths():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def kb_exists() -> bool:
    return any(KB_DIR.glob("*.maif"))


def latest_session() -> Optional[Path]:
    files = sorted(SESSIONS_DIR.glob("*.maif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_decoder(path: Path) -> Optional[MAIFDecoder]:
    try:
        dec = MAIFDecoder(str(path))
        dec.load()
        return dec
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def show_vector_db_stats():
    print("\n" + "=" * 80)
    print("VECTOR DATABASE STATISTICS")
    print("=" * 80 + "\n")
    try:
        stats = get_vector_db().get_stats()
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Documents: {stats['num_documents']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Storage: {stats['persist_directory']}")
    except Exception as e:
        print(f"Error getting stats: {e}")


def show_session_history(session_path: str):
    if not session_path or not Path(session_path).exists():
        print("\nNo session artifact found. Ask a question first.")
        return
    mgr = SessionManager()
    history = mgr.get_session_history(session_path)

    def _pp(val: Any) -> str:
        try:
            return json.dumps(val, indent=2, ensure_ascii=False)
        except Exception:
            return str(val)

    print("\n" + "=" * 80)
    print("SESSION HISTORY")
    print("=" * 80)
    for i, entry in enumerate(history, 1):
        meta = entry.get("metadata", {}) or {}
        et = meta.get("type", "unknown")
        content = entry.get("content", "")
        print(f"[{i}] {et}")
        print(_pp({"metadata": meta, "content": content}))
        print("-" * 40)


def show_maif_artifact_details(session_path: str):
    if not session_path or not Path(session_path).exists():
        print("\nNo session artifact found. Ask a question first.")
        return
    dec = load_decoder(Path(session_path))
    if not dec:
        return
    print("\n" + "=" * 80)
    print("MAIF ARTIFACT DETAILS")
    print("=" * 80)
    print(f"Path: {session_path}")
    if Path(session_path).exists():
        print(f"Size: {Path(session_path).stat().st_size:,} bytes")
    block_types: Dict[str, int] = {}
    for blk in dec.blocks:
        et = (blk.metadata or {}).get("type", "unknown")
        block_types[et] = block_types.get(et, 0) + 1
    print(f"Total Blocks: {len(dec.blocks)}")
    print("Block Distribution:")
    for k, v in sorted(block_types.items()):
        print(f"- {k}: {v}")


def show_agent_stats(history: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("MULTI-AGENT STATISTICS")
    print("=" * 80)
    if not history:
        print("No runs yet.")
        return
    total_questions = len(history)
    answers = sum(1 for h in history if h.get("answer"))
    verifications = [h.get("verification", {}) for h in history if h.get("verification")]
    avg_conf = (
        sum(v.get("confidence", 0) for v in verifications) / len(verifications)
        if verifications
        else 0
    )
    print(f"Questions: {total_questions}")
    print(f"Answers produced: {answers}")
    print(f"Avg verification confidence: {avg_conf:.2f}")


def multi_turn(session_path: str) -> str:
    current_path = session_path
    while True:
        try:
            q = input("\n[Multi-turn] Your question (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"done", "exit", "quit"}:
            break
        result = run_agents_pipeline(q, session_path=current_path)
        if result.get("session_artifact_path"):
            current_path = result["session_artifact_path"]
        print(f"Turn logged to: {current_path}")
    return current_path


def run_pipeline(question: str, session_path: Optional[str]) -> Dict[str, Any]:
    ensure_paths()
    return run_agents_pipeline(question, session_path=session_path)


def show_menu():
    print("\n" + "-" * 80)
    print("MENU:")
    print("1. Ask a question")
    print("2. View session history")
    print("3. Inspect MAIF artifact")
    print("4. Show multi-agent stats")
    print("5. Show vector DB stats")
    print("6. Start new session")
    print("7. Multi-turn conversation mode")
    print("8. Exit")
    print("-" * 80)


def main():
    ensure_paths()
    session_path: Optional[str] = None
    state_history: List[Dict[str, Any]] = []

    while True:
        show_menu()
        choice = input("Choice: ").strip()
        if choice == "1":
            if not kb_exists():
                print("KB not found. Run create_kb_enhanced.py first.")
                continue
            question = input("Enter your question: ").strip()
            if not question:
                print("Please enter a question.")
                continue
            result = run_pipeline(question=question, session_path=session_path)
            if result:
                session_path = result.get("session_artifact_path", session_path)
                state_history.append(result)
        elif choice == "2":
            if not session_path:
                session_path = str(latest_session()) if latest_session() else None
            show_session_history(session_path or "")
        elif choice == "3":
            if not session_path:
                session_path = str(latest_session()) if latest_session() else None
            show_maif_artifact_details(session_path or "")
        elif choice == "4":
            show_agent_stats(state_history)
        elif choice == "5":
            show_vector_db_stats()
        elif choice == "6":
            # start new session
            session_path = None
            print("Started a new session (will create on next question).")
        elif choice == "7":
            if not kb_exists():
                print("KB not found. Run create_kb_enhanced.py first.")
                continue
            session_path = multi_turn(session_path or (str(latest_session()) if latest_session() else None))
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
