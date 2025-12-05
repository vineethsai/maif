# Basic MAIF Examples

Simple examples demonstrating core MAIF functionality without external dependencies.

## Overview

These examples show fundamental MAIF operations:
- Creating and saving artifacts
- Adding different content types
- Loading and verifying artifacts
- Basic integrity checking

No AWS, no vector databases, no complex dependencies. Just core MAIF.

## Examples

### basic_usage.py

Demonstrates essential MAIF operations using the secure format:

```python
from maif import MAIFEncoder, MAIFDecoder

# Create encoder (secure format with Ed25519)
encoder = MAIFEncoder("data.maif", agent_id="agent-001")

# Add text block (signed immediately)
encoder.add_text_block(
    "Agent conversation data",
    metadata={"agent_id": "agent-001"}
)

# Finalize (self-contained, no manifest needed)
encoder.finalize()

# Load and read
decoder = MAIFDecoder("data.maif")
decoder.load()
for block in decoder.blocks:
    print(f"Block: {block.header.block_type}, Size: {block.header.size}")
```

**Run:**
```bash
python3 basic_usage.py
```

**What it demonstrates:**
- MAIFEncoder for creating artifacts
- Adding text blocks with metadata
- Saving to disk with manifest
- MAIFDecoder for reading artifacts
- Iterating through blocks
- Basic error handling

### simple_api_demo.py

Shows the simplified API for common tasks:

```python
from maif_api import create_maif, load_maif

# Create artifact
maif = create_maif("demo_agent")

# Add content
maif.add_text("Sample document", title="Document 1")
maif.add_multimodal({
    "text": "Description",
    "metadata": {"category": "example"}
}, title="Multimodal Content")

# Save with signing
maif.save("demo.maif", sign=True)

# Load and verify
loaded = load_maif("demo.maif")
assert loaded.verify_integrity()
```

**Run:**
```bash
python3 simple_api_demo.py
```

**What it demonstrates:**
- Simple API (easier than core)
- Multiple content types
- Cryptographic signing
- Integrity verification
- Privacy features (encryption, anonymization)
- Search functionality

## Key Concepts

### MAIF Artifacts
Self-contained files with everything built in:
- File header with Ed25519 public key and Merkle root
- Signed content blocks (text, binary, embeddings, etc.)
- Hash chains (cryptographic links between blocks)
- Embedded provenance chain for full audit trail

No external manifest files needed — everything is in the `.maif` file.

### Block Structure
Each block contains:
- Type identifier (4-character FourCC)
- Size in bytes
- Unique ID (UUID)
- Timestamp
- Previous block hash (for chain integrity)
- **Ed25519 signature (64 bytes)**
- Metadata dictionary
- Raw data bytes

### Integrity Verification
MAIF uses Ed25519 signatures and SHA-256 hash chains:
```
Block 1 (hash: abc123, signed with Ed25519)
    ↓ previous_hash
Block 2 (hash: def456, signed with Ed25519)
    ↓ previous_hash
Block 3 (hash: ghi789, signed with Ed25519)
```

Any tampering breaks both the hash chain and the signatures — instantly detectable.

## Usage Patterns

### Creating Artifacts

```python
from maif_api import create_maif

# Initialize
maif = create_maif("agent_id")

# Add content
text_id = maif.add_text("Content here")
image_id = maif.add_image("photo.jpg")

# Save
maif.save("output.maif")
```

### Loading Artifacts

```python
from maif_api import load_maif

# Load
maif = load_maif("output.maif")

# Verify integrity
if maif.verify_integrity():
    print("Artifact is valid")

# Access content
content_list = maif.get_content_list()
for item in content_list:
    print(f"{item['type']}: {item['title']}")
```

### Privacy Features

```python
# Create with privacy enabled
maif = create_maif("agent", enable_privacy=True)

# Add encrypted text
maif.add_text(
    "Sensitive information",
    title="Confidential",
    encrypt=True,
    anonymize=True
)

# Get privacy report
report = maif.get_privacy_report()
```

## Requirements

### Core Dependencies
- Python 3.8+
- numpy
- cryptography
- pydantic

### Optional Dependencies
- sentence-transformers (for semantic features)
- opencv-python (for image processing)
- pillow (for image handling)

Install with:
```bash
pip install -e ../../  # Install from repository root
```

## Next Steps

After understanding these basics:

1. **Security Examples** - See `examples/security/` for privacy and encryption
2. **Advanced Examples** - See `examples/advanced/` for multi-agent systems
3. **LangGraph RAG** - See `examples/langgraph/` for production system
4. **Documentation** - See `docs/` for complete API reference

## Common Issues

### Import Errors
Ensure MAIF is installed:
```bash
cd ../..  # Go to repository root
pip install -e .
```

### File Not Found
Check paths are relative to script location or use absolute paths.

### Verification Failures
Ensure you're not modifying MAIF files directly. They must be append-only.

## Additional Resources

- Main README: `../../README.md`
- API Documentation: `../../docs/SIMPLE_API_GUIDE.md`
- Core API Reference: `../../docs/api/core/`
- User Guides: `../../docs/guide/`

