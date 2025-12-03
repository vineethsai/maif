# LangGraph + MAIF Multi-Agent Research Assistant

Multi-agent RAG system that searches a **local knowledge base** with full cryptographic provenance.

## What This Does

- **Searches local documents** (NOT Google/web) using semantic similarity
- **Generates answers** using Gemini API
- **Verifies facts** with LLM-based checking
- **Logs everything** to MAIF artifacts (cryptographic audit trail)
- **5 agents** collaborating via LangGraph

## Quick Start

### 1. Set API Key
```bash
cd examples/langgraph
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 2. Install
```bash
pip3 install --user chromadb sentence-transformers langgraph \
             langgraph-checkpoint-sqlite google-generativeai python-dotenv tqdm
```

### 3. Create KB
```bash
python3 create_kb_enhanced.py  # Creates 3 climate change documents
```

### 4. Run
```bash
python3 demo_enhanced.py
```

## Architecture

5 agents coordinated by LangGraph, all logging to MAIF:
```
init_session → retrieve (ChromaDB) → synthesize (Gemini) → 
fact_check (LLM) → cite → Done
              ↑                              ↓
              └──────── (revise if needed) ──┘
```

## Important Notes

- **This searches a LOCAL knowledge base** (3 climate change documents)
- **NOT a web search tool** - it's for RAG over your own documents
- **Ask climate change questions** to see it work (that's what's in the KB)
- **Options 2 & 3** now fixed - they show MAIF provenance

## Features

- ✅ Real ChromaDB vector search (semantic similarity)
- ✅ Real embeddings (sentence-transformers, 384-dim)
- ✅ LLM fact-checking (Gemini verifies each claim)
- ✅ Multi-turn conversations
- ✅ MAIF cryptographic provenance (every action logged)
- ✅ Interactive console UI

## What Gets Logged to MAIF

Every agent action creates a hash-chained block:
- User questions
- ChromaDB search results
- Gemini responses
- Fact-check results
- Citations

All cryptographically linked - any tampering breaks the chain!

## Files

### Main Scripts
- `demo_enhanced.py` - Interactive demo with all features
- `create_kb_enhanced.py` - Creates knowledge base with embeddings
- `demo_hybrid.py` - Version with web search fallback (NEW)

### Core Components
- `graph_enhanced.py` - LangGraph construction
- `vector_db.py` - ChromaDB integration
- `maif_utils.py` - MAIF logging helpers
- `state.py` - State definition

### Agent Nodes (in `nodes/`)
- `init_session.py` - Session management
- `retrieve_enhanced.py` - ChromaDB semantic search
- `synthesize.py` - Gemini answer generation
- `fact_check_enhanced.py` - LLM verification
- `cite.py` - Citation formatting

## Project Structure

```
examples/langgraph/
├── README.md                      # This file
├── .env                           # API keys (create from .env.example)
├── requirements_enhanced.txt      # Dependencies
├── demo_enhanced.py              # Main demo ⭐
├── create_kb_enhanced.py         # KB creation ⭐
├── graph_enhanced.py             # LangGraph
├── vector_db.py                  # ChromaDB
├── maif_utils.py                 # MAIF helpers
├── nodes/                        # 5 agent nodes
└── data/
    ├── sessions/                 # MAIF session artifacts
    ├── kb/                       # KB MAIF artifacts
    └── chroma_db/               # Vector DB
```

## How It Works

1. **User asks question** about climate change
2. **ChromaDB searches** local KB with semantic similarity
3. **Gemini generates** answer from retrieved chunks
4. **LLM verifies** each claim against sources
5. **Citations added** with source attribution
6. **Everything logged** to MAIF with hash chains

## Troubleshooting

**"Vector DB is empty"**
```bash
python3 create_kb_enhanced.py
```

**"GEMINI_API_KEY not found"**
```bash
echo "GEMINI_API_KEY=your_key" > .env
```

**"Can only ask about climate change?"**
- Yes, that's what's in the local KB
- Add your own documents to KB
- OR use `demo_hybrid.py` for web search fallback

## 📊 Example Session Artifact

After running a query, the session MAIF contains:

```
Block 1: Session initialization (METADATA)
Block 2: User question (TEXT)
Block 3: Retrieval event - 5 chunks found (METADATA)
Block 4: Draft answer from Gemini (TEXT)
Block 5: Fact-check results - 80% confidence (METADATA)
Block 6: Final answer with citations (TEXT)
Block 7: Citation list (METADATA)
```

All blocks linked via `previous_hash` → cryptographic provenance!

## 🎯 Key Features

### Multi-Agent Collaboration
- **Retriever**: Finds relevant information
- **Synthesizer**: Generates coherent answers
- **Fact-Checker**: Verifies accuracy
- **Citation Agent**: Adds proper attribution

### Iterative Refinement
- If fact-checking fails, answer is revised
- Maximum 3 iterations to prevent infinite loops
- Confidence scores tracked at each step

### Cryptographic Provenance
- Every operation logged to MAIF
- Hash-chained blocks (tamper-evident)
- Can prove: "This answer came from these sources at this time"

### Resumability
- LangGraph checkpoints enable pausing/resuming
- Multi-turn conversations supported
- Thread-based session management

## 🔧 Configuration

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your_key_here

# Optional: Increase iteration limit
MAX_ITERATIONS=5
```

### Customization

#### Change the LLM

Edit `nodes/synthesize.py`:
```python
# Replace Gemini with OpenAI, Claude, etc.
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

#### Change Vector DB

Edit `nodes/retrieve.py`:
```python
# Replace mock with real vector DB
import chromadb
client = chromadb.Client()
collection = client.get_collection("knowledge_base")
results = collection.query(query_texts=[question], n_results=5)
```

#### Adjust Fact-Checking Threshold

Edit `nodes/fact_check.py`:
```python
# Line ~67
if confidence >= 0.8:  # Change this threshold
    verification_status = 'verified'
```

## 📚 Advanced Usage

### Create Your Own KB Artifacts

```python
from examples.langgraph.maif_utils import KBManager

kb_manager = KBManager()

# Prepare chunks
chunks = [
    {
        "text": "Your document text here...",
        "embedding": [0.1, 0.2, ...],  # 384-dim vector
        "metadata": {"source": "doc.pdf", "page": 1}
    },
    # ... more chunks
]

# Create KB artifact
kb_path = kb_manager.create_kb_artifact(
    doc_id="my_document",
    chunks=chunks,
    document_metadata={"title": "My Document", "author": "Me"}
)
```

### Multi-Turn Conversations

```python
from examples.langgraph.graph import create_app

app = create_app()
session_id = "user_123"
config = {"configurable": {"thread_id": session_id}}

# Turn 1
result1 = app.invoke({"question": "What is X?", ...}, config)

# Turn 2 (continues same session)
result2 = app.invoke({"question": "Tell me more about Y", ...}, config)

# Both turns logged to same session MAIF artifact!
```

### Inspect Provenance

```python
from examples.langgraph.maif_utils import SessionManager

session_manager = SessionManager()
history = session_manager.get_session_history("data/sessions/abc123.maif")

for entry in history:
    print(f"{entry['block_type']}: {entry['metadata'].get('type')}")
```

## 🐛 Troubleshooting

### "Gemini API Error"
- Check your API key in `.env`
- Verify you have quota remaining
- Check network connectivity

### "No module named langgraph"
```bash
pip install langgraph langgraph-checkpoint-sqlite
```

### "Session artifact not found"
- Ensure `data/sessions/` directory exists
- Check session_id matches between calls

### "Import error for MAIF"
- Make sure you're running from project root
- Check `sys.path` includes MAIF library

## 🤝 Contributing

This is a reference implementation. To extend:

1. Add new nodes in `nodes/`
2. Update graph in `graph.py`
3. Add MAIF logging in your nodes
4. Test end-to-end

## 📄 License

Part of the MAIF project - see main LICENSE

## 🎉 Credits

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [MAIF](https://github.com/vineethsai/maifscratch-1)
- [Gemini API](https://ai.google.dev/)

