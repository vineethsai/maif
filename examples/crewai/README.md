# MAIF + CrewAI Enhanced RAG Demo

This mirrors the LangGraph enhanced example using CrewAI: multi-agent RAG with Chroma retrieval, Gemini synthesis + fact-check + citations, and full MAIF provenance (sessions + KB artifacts).

## Quickstart
```
cd examples/crewai_enhanced
pip install -r requirements_enhanced.txt
cp env.example .env   # keep .env untracked
python create_kb_enhanced.py
python demo_enhanced.py
```

Artifacts and storage:
- `data/kb/` MAIF KB artifacts
- `data/sessions/` MAIF session artifacts
- `data/chroma_db/` Chroma persistence

Menu (demo_enhanced.py):
- Create knowledge base
- Ask a question (retrieve → synthesize → fact-check → cite)
- Vector DB stats

The pipeline logs to MAIF via `SessionManager` and `MAIFCrewCallback` (task/step-level provenance).

## What’s inside
- `create_kb_enhanced.py` – build KB with embeddings + MAIF artifacts
- `vector_db.py` – Chroma + sentence-transformers wrapper
- `maif_utils.py` – MAIF session/KB managers and logging helpers
- `demo_enhanced.py` – menu/CLI to run the CrewAI RAG flow
- `state.py` – shared state schema for the crew
- `nodes/` – agent task implementations (retrieve, synthesize, fact-check, cite)
- `requirements_enhanced.txt` – deps
- `env.example` – sample env (set `GEMINI_API_KEY`)

## Warnings
- Keep `.env` and generated artifacts out of git (ignored by default).
- Gemini calls require the API key in your environment.
- If `GEMINI_API_KEY` is missing, the demo falls back to offline stubs for synthesis/fact-checking.


