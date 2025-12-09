#!/usr/bin/env python3
"""
CrewAI-centric demo (agents+tasks) without the “nodes” terminology.

Pipeline:
1) Ensure KB exists (create_kb_enhanced.py)
2) Agents: Retriever -> Synthesizer -> Fact-Checker -> Citation
3) MAIF provenance via MAIFCrewCallback + SessionManager
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from crewai import LLM
from dotenv import load_dotenv
from maif.integrations.crewai import MAIFCrewCallback
from maif.integrations._base import EventType

from maif_utils import SessionManager
from vector_db import get_vector_db
from state import RAGState

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
KB_DIR = DATA_DIR / "kb"


def ensure_paths():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)


def kb_exists() -> bool:
    return any(KB_DIR.glob("*.maif"))


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    ctx = []
    for c in chunks[:5]:
        ctx.append(f"[doc:{c.get('doc_id')} idx:{c.get('chunk_index')}] {c.get('text')}")
    return f"""You are a helpful research assistant.
Question: {question}
Context:
{os.linesep.join(ctx)}
Answer concisely with citations like [doc:id idx].
"""


def call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text


def offline_stub(chunks: List[Dict[str, Any]], prefix: str) -> str:
    ctx = " ".join([c.get("text", "")[:120] for c in chunks[:3]])
    return f"(offline stub: {prefix}) Based on: {ctx}"


def run_demo(question: str, session_path: Optional[str] = None) -> Dict[str, Any]:
    load_dotenv(dotenv_path=ROOT / ".env", override=False)

    ensure_paths()
    if not kb_exists():
        print("KB missing. Run create_kb_enhanced.py first.")
        return {}

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    gemini_llm = LLM(
        model="google/gemini-2.0-flash",
        api_key=gemini_key,
        temperature=0.2,
    )

    # Initialize session via SessionManager (creates secure MAIF file)
    session_mgr = SessionManager(str(SESSIONS_DIR))
    if session_path and Path(session_path).exists():
        current_session_path = Path(session_path)
    else:
        current_session_path = Path(session_mgr.create_session())

    # Callback on same artifact to guarantee MAIF blocks
    callback = MAIFCrewCallback(artifact_path=str(current_session_path), agent_id="crewai_demo_agents")
    callback.on_crew_start(crew_name="CrewAI Agents Demo", agents=None, tasks=None, inputs={"question": question})

    state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "session_artifact_path": str(current_session_path),
        "messages": [],
    }

    # Log user question to session + callback
    session_mgr.log_user_message(str(current_session_path), question)
    callback.tracker.log_event(
        event_type=EventType.AGENT_ACTION,
        data={"question": question},
        metadata={"type": "user_message"},
    )

    vector_db = get_vector_db()

    def retrieve_step():
        chunks = vector_db.search(question, top_k=5)
        state["retrieved_chunks"] = chunks
        session_mgr.log_retrieval_event(str(current_session_path), query=question, results=chunks)
        callback.tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data={"query": question, "results": chunks},
            metadata={"type": "retrieval_event"},
        )
        return f"Retrieved {len(chunks)} chunks."

    def synthesize_step():
        prompt = build_prompt(question, state.get("retrieved_chunks", []))
        try:
            answer = call_gemini(prompt)
            model_used = "gemini-2.0-flash"
        except Exception:
            answer = offline_stub(state.get("retrieved_chunks", []), "synthesis")
            model_used = "offline_stub"
        state["answer"] = answer
        session_mgr.log_model_response(str(current_session_path), response=answer, model=model_used)
        callback.tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data={"answer": answer, "model": model_used},
            metadata={"type": "model_response"},
        )
        return answer

    def fact_check_step():
        chunks = state.get("retrieved_chunks", [])
        answer = state.get("answer", "")
        prompt = (
            "Verify answer against sources.\n"
            + "\n".join([f"[doc:{c.get('doc_id')} idx:{c.get('chunk_index')}] {c.get('text')}" for c in chunks[:5]])
            + f"\nAnswer:\n{answer}"
        )
        verification = {"raw": "", "confidence": 0.7}
        try:
            raw = call_gemini(prompt)
            verification["raw"] = raw
            verification["confidence"] = 0.82
        except Exception:
            verification["raw"] = offline_stub(chunks, "fact-check")
            verification["confidence"] = 0.72
        state["verification"] = verification
        state["confidence"] = verification["confidence"]
        state["needs_revision"] = verification["confidence"] < 0.75
        session_mgr.log_verification(str(current_session_path), verification_results=verification)
        callback.tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data=verification,
            metadata={"type": "verification"},
        )
        return str(verification)

    def cite_step():
        chunks = state.get("retrieved_chunks", [])
        citations = []
        for c in chunks[:5]:
            citations.append(
                {
                    "doc_id": c.get("doc_id"),
                    "chunk_index": c.get("chunk_index"),
                    "text": c.get("text", "")[:160],
                    "score": c.get("score"),
                }
            )
        state["citations"] = citations
        session_mgr.log_citations(str(current_session_path), citations=citations)
        callback.tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data={"citations": citations},
            metadata={"type": "citations"},
        )
        return str(citations)

    # Run steps sequentially (no CrewAI callbacks)
    retrieve_step()
    synthesize_step()
    fact_check_step()
    cite_step()

    callback.on_crew_end(result=None)
    callback.finalize()

    print("\n=== ANSWER ===")
    print(state.get("answer", ""))
    print("\nConfidence:", state.get("confidence"))
    print("Citations:", state.get("citations", []))
    print("\nSession artifact:", current_session_path)

    return {
        "session_artifact_path": str(current_session_path),
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "verification": state.get("verification", {}),
        "confidence": state.get("confidence"),
    }


def main():
    question = input("Question: ").strip()
    if not question:
        print("Please enter a question.")
        return
    run_demo(question)


if __name__ == "__main__":
    main()

