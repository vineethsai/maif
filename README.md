<p align="center">
  <img src="docs/assets/maif-logo.svg" alt="MAIF Logo" width="200"/>
</p>

<h1 align="center">MAIF</h1>
<h3 align="center">Multimodal Artifact File Format for Trustworthy AI Agents</h3>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/vineethsai/maif/actions"><img src="https://img.shields.io/badge/tests-431%20passed-brightgreen.svg" alt="Tests"></a>
  <a href="https://vineethsai.github.io/maif/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation"></a>
</p>

<p align="center">
  Cryptographically-secure, auditable file format for AI agent memory with provenance tracking
</p>

---

## Overview

MAIF is a file format and SDK designed for AI agents that need **trustworthy memory**. Every piece of data is cryptographically linked, creating tamper-evident audit trails that prove exactly what happened, when, and by which agent.

**Key Capabilities:**

- **Cryptographic Provenance** - Hash-chained blocks for tamper-evident audit trails
- **Multi-Agent Coordination** - Shared artifacts with agent-specific logging
- **Multimodal Storage** - Text, embeddings, images, video, knowledge graphs
- **Privacy-by-Design** - Encryption, anonymization, access control built-in
- **High Performance** - Memory-mapped I/O, streaming, semantic compression

## Use Cases

- **Multi-Agent Systems** - Shared memory with full provenance (see LangGraph example)
- **RAG Pipelines** - Document storage with embeddings, search, and citation tracking
- **Compliance & Audit** - Immutable audit trails for regulated industries
- **Research** - Reproducible experiments with complete data lineage
- **Enterprise AI** - Secure, auditable AI workflows with access control

---

## Quick Start

**Prerequisites:** Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/vineethsai/maif.git
cd maif

# Install MAIF
pip install -e .

# With ML features (embeddings, semantic search)
pip install -e ".[ml]"
```

### Your First MAIF Artifact

```python
from maif_api import create_maif, load_maif

# Create an agent memory artifact
maif = create_maif("my_agent", enable_privacy=True)

# Add content with automatic provenance tracking
maif.add_text("User asked about weather in NYC", title="Query")
maif.add_text("Temperature is 72°F, sunny", title="Response")

# Save with cryptographic signing
maif.save("agent_memory.maif", sign=True)

# Later: Load and verify integrity
loaded = load_maif("agent_memory.maif")
assert loaded.verify_integrity()  # Cryptographic hash chain verification
print(f"Loaded {len(loaded.get_content_list())} blocks")
```

---

## Featured Example: Multi-Agent RAG System

A production-ready multi-agent system with **LangGraph orchestration** and **MAIF provenance tracking**.

```bash
cd examples/langgraph

# Configure API key
echo "GEMINI_API_KEY=your_key" > .env

# Install dependencies
pip install -r requirements_enhanced.txt

# Create knowledge base with embeddings
python3 create_kb_enhanced.py

# Run the interactive demo
python3 demo_enhanced.py
```

**What's Included:**
- 5 specialized agents (Retriever, Synthesizer, Fact-Checker, Citation, Web Search)
- ChromaDB vector store with semantic search
- Gemini API integration for LLM reasoning
- Complete audit trail of every agent action
- Multi-turn conversation support

See [`examples/langgraph/README.md`](examples/langgraph/README.md) for full documentation.

---

## Features

### Cryptographic Provenance

Every block is linked via `previous_hash` - any tampering breaks the chain.

```python
# Automatic hash chaining
maif.add_text("First message")   # Block 1: hash = abc123
maif.add_text("Second message")  # Block 2: previous_hash = abc123, hash = def456
maif.add_text("Third message")   # Block 3: previous_hash = def456, hash = ghi789

# Verify the entire chain
assert maif.verify_integrity()  # Validates all hash links
```

### Privacy & Security

Built-in encryption, anonymization, and access control.

```python
from maif import PrivacyLevel, EncryptionMode

# Add encrypted content
maif.add_text(
    "Sensitive data",
    encrypt=True,
    anonymize=True,  # Auto-redact PII
    privacy_level=PrivacyLevel.CONFIDENTIAL
)

# Access control
maif.add_access_rule(AccessRule(
    role="analyst",
    permissions=[Permission.READ],
    resources=["reports"]
))
```

### Multimodal Support

Store and search across text, images, video, embeddings, and knowledge graphs.

```python
# Text with metadata
maif.add_text("Analysis results", title="Report", language="en")

# Images with feature extraction
maif.add_image("chart.png", title="Sales Chart")

# Semantic embeddings
maif.add_embeddings([[0.1, 0.2, ...]], model_name="all-MiniLM-L6-v2")

# Multimodal content
maif.add_multimodal({
    "text": "Product description",
    "image_path": "product.jpg",
    "metadata": {"category": "electronics"}
})
```

### Novel Algorithms

Advanced semantic processing capabilities:

- **ACAM** - Adaptive Cross-Modal Attention for multimodal fusion
- **HSC** - Hierarchical Semantic Compression (up to 64× compression)
- **CSB** - Cryptographic Semantic Binding for embedding authenticity

---

## Performance

| Metric | Performance |
|--------|-------------|
| Semantic Search | ~30ms for 1M+ vectors |
| Compression Ratio | Up to 64× (HSC) |
| Hash Verification | >500 MB/s throughput |
| Memory Efficiency | 64KB minimum buffer |

---

## Project Structure

```
maif/
├── maif/                  # Core library
│   ├── core.py           # MAIFEncoder, MAIFDecoder
│   ├── security.py       # Signing, verification
│   ├── privacy.py        # Encryption, anonymization
│   └── semantic*.py      # Embeddings, compression
├── maif_api.py           # High-level API
├── examples/
│   ├── langgraph/        # Multi-agent RAG system ⭐
│   ├── basic/            # Getting started
│   ├── security/         # Privacy & encryption
│   └── advanced/         # Agent framework, lifecycle
├── tests/                # 431 tests
├── docs/                 # VitePress documentation
└── benchmarks/           # Performance tests
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Online Docs](https://vineethsai.github.io/maif/) | Full documentation site |
| [API Reference](docs/api/) | Detailed API documentation |
| [User Guides](docs/guide/) | Step-by-step tutorials |
| [Examples](examples/) | Working code examples |

---

## Examples

### Basic Usage
```bash
python examples/basic/simple_api_demo.py
python examples/basic/basic_usage.py
```

### Privacy & Security
```bash
python examples/security/privacy_demo.py
python examples/security/classified_api_simple_demo.py
```

### Advanced Features
```bash
python examples/advanced/maif_agent_demo.py          # Agent framework
python examples/advanced/lifecycle_management_demo.py # Lifecycle management
python examples/advanced/video_demo.py               # Video processing
```

---

## Contributing

We welcome contributions! Please ensure:

1. All tests pass (`pytest tests/`)
2. Code follows PEP 8 style
3. New features include tests and documentation
4. Security-sensitive changes include impact analysis

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## References

- [FIPS 140-2 Standards](https://csrc.nist.gov/publications/detail/fips/140/2/final) - Cryptographic module requirements
- [NIST 800-53](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) - Security and privacy controls
- [ISO BMFF](https://www.iso.org/standard/68960.html) - Binary format inspiration

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Build trustworthy AI agents with cryptographic provenance</b>
</p>
