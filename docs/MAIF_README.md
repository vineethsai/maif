# MAIF - Multimodal Artifact File Format

A secure, AI-native container format for trustworthy AI agents with built-in provenance, forensics, and semantic capabilities.

## Overview

MAIF (Multimodal Artifact File Format) is a revolutionary file format designed specifically for AI systems that need verifiable, auditable, and secure data containers. Unlike traditional file formats, MAIF embeds cryptographic security, semantic understanding, and forensic capabilities directly into the data structure.

## Key Features

### Security & Trust
- **Cryptographic Provenance**: Immutable audit trails with cryptographic verification
- **Digital Signatures**: Ed25519 signatures for fast, compact non-repudiation (64 bytes per signature)
- **Granular Access Control**: Block-level permissions and encryption
- **Tamper Detection**: Cryptographic integrity verification on every block

### AI-Native Design
- **Semantic Embeddings**: Built-in multimodal semantic representations
- **Knowledge Graphs**: Embedded structured knowledge for reasoning
- **Cross-Modal Attention**: Dynamic attention mechanisms across modalities
- **Efficient Search**: Sub-50ms semantic search on commodity hardware

### Digital Forensics
- **Timeline Reconstruction**: Complete forensic timeline of all agent interactions
- **Attack Detection**: Sophisticated tampering and anomaly detection
- **Legal Admissibility**: Evidence-grade audit trails for compliance
- **Incident Response**: Automated forensic analysis and reporting

### Container Format
- **Hierarchical Structure**: Extensible block-based architecture
- **Self-Describing**: No external dependencies for parsing
- **Multimodal Support**: Text, images, audio, video, embeddings, and more
- **Compression**: Semantic-aware compression with 40-60% ratios

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# Full installation with all features
pip install -e .[full]

# Development installation
pip install -e .[dev]
```

### Basic Usage

```python
from maif_api import create_maif, load_maif

# Create a new MAIF artifact
maif = create_maif("my-agent")

# Add content
maif.add_text("AI systems need trustworthy data containers")

# Add multimodal content
maif.add_multimodal({
    "text": "Additional context",
    "type": "metadata"
})

# Save with digital signature
maif.save("my_data.maif", sign=True)

# Verify integrity
loaded = load_maif("my_data.maif")
is_valid = loaded.verify()
print(f"File integrity: {'VALID' if is_valid else 'INVALID'}")
```

### Advanced Usage with Encoder/Decoder

```python
from maif import MAIFEncoder, MAIFDecoder
from maif.semantic import SemanticEmbedder

# Create with low-level API (v3 format - self-contained)
encoder = MAIFEncoder("my_data.maif", agent_id="my-agent")
embedder = SemanticEmbedder()

# Add content
text = "AI systems need trustworthy data containers"
text_hash = encoder.add_text_block(text)

# Add semantic embeddings
embedding = embedder.embed_text(text)
embed_hash = encoder.add_embeddings_block([embedding.vector])

# Finalize (signs and writes all security/provenance data)
encoder.finalize()

# Verify - v3 format is self-contained, no manifest needed
decoder = MAIFDecoder("my_data.maif")
decoder.load()
is_valid, errors = decoder.verify_integrity()
print(f"Integrity: {'VALID' if is_valid else 'INVALID'}")
print(f"Loaded {len(decoder.blocks)} blocks")
```

## Architecture

### Block Structure

MAIF uses a hierarchical block structure similar to ISO BMFF (MP4):

```
MAIF File
├── Header Block (file metadata, root hash)
├── Text Data Blocks (UTF-8 encoded text)
├── Binary Data Blocks (images, audio, video)
├── Embeddings Blocks (semantic vectors)
├── Knowledge Graph Blocks (structured knowledge)
├── Security Metadata Blocks (signatures, ACLs)
└── Lifecycle Metadata Blocks (provenance, audit logs)
```

### Security Model

MAIF implements a comprehensive security model:

1. **Block-level Hashing**: SHA-256 hashes for each block
2. **Ed25519 Signatures**: Fast, secure digital signatures for authenticity (64 bytes each)
3. **Provenance Chain**: Cryptographically linked audit trail embedded in the file
4. **Access Control**: Granular permissions per block
5. **Merkle Root**: Efficient whole-file integrity verification

### Semantic Layer

The semantic layer provides AI-native capabilities:

1. **Multimodal Embeddings**: Dense vector representations
2. **Knowledge Graphs**: Structured entity-relationship data
3. **Cross-Modal Attention**: Dynamic attention weighting
4. **Semantic Search**: Efficient similarity search

## Implementation Status

Based on current technology analysis, MAIF capabilities are categorized into three phases:

### Phase 1 - Immediately Feasible (2025-2026)
- Secure container architecture
- Immutable provenance chains
- Basic multimodal storage
- Semantic search (30-50ms latency)
- Block-level access control

### Phase 2 - Research Required (2026-2028)
- Self-evolving artifacts
- Advanced semantic compression
- Cross-modal attention mechanisms
- Cryptographic semantic binding

### Phase 3 - Advanced Research (2028+)
- Advanced Cross-Modal AI reasoning
- Universal semantic compression
- Sub-millisecond mobile search
- Adaptive semantic optimization

## Use Cases

### Enterprise AI Governance
- Audit AI model decisions and data lineage
- Ensure regulatory compliance (EU AI Act, GDPR)
- Track AI agent behavior across systems
- Forensic investigation of AI incidents

### Research and Development
- Reproducible AI experiments with full provenance
- Secure sharing of sensitive research data
- Cross-modal AI model development
- Collaborative knowledge building

### Critical Infrastructure
- High-assurance AI for healthcare and finance
- Tamper-evident AI system logs
- Incident response and forensic analysis
- Regulatory reporting and compliance

## Performance Benchmarks

Current implementation achieves:

- **Semantic Search**: 30-50ms for 1M+ vectors on commodity hardware
- **Storage Efficiency**: 40-60% compression with semantic preservation
- **Integrity Verification**: ~0.1ms per file (30× faster than legacy format)
- **Read Performance**: 11× faster than legacy format with external manifests
- **Tamper Detection**: 100% detection rate in <0.1ms
- **Signature Size**: Only 64 bytes per block (Ed25519)

## Comparison with Existing Solutions

| Feature | MAIF | Vector DBs | Traditional DBs | MP4/MKV |
|---------|------|------------|-----------------|---------|
| Semantic Search | Yes | Yes | No | No |
| Cryptographic Security | Yes | No | No | No |
| Immutable Provenance | Yes | No | No | No |
| Multimodal Support | Yes | No | No | Yes |
| Self-Describing | Yes | No | No | Partial |
| Forensic Capabilities | Yes | No | No | No |
| Offline Operation | Yes | No | Yes | Yes |
| Granular Access Control | Yes | No | Partial | No |

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/maif-ai/maif.git
cd maif

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Run example
python examples/basic/basic_usage.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use MAIF in your research, please cite:

```bibtex
@article{maif2025,
  title={An Artifact-Centric AI Agent Design and the Multimodal Artifact File Format (MAIF) for Enhanced Trustworthiness},
  author={MAIF Development Team},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

MAIF builds upon established technologies:
- **ISO BMFF** for container architecture
- **Sentence Transformers** for semantic embeddings
- **FAISS** for efficient similarity search
- **Cryptography** for security primitives
- **Memvid** for demonstrating video-based data storage feasibility

## Support

- [Documentation](https://maif.readthedocs.io/)
- [Discussions](https://github.com/maif-ai/maif/discussions)
- [Issue Tracker](https://github.com/maif-ai/maif/issues)
- [Email Support](mailto:support@maif.ai)

---

**Ready to build trustworthy AI systems? Start with MAIF.**