# MAIFEncoder & MAIFDecoder

The `MAIFEncoder` and `MAIFDecoder` classes provide low-level access to MAIF v3 binary format operations. The v3 format is **self-contained** with all security and provenance embedded directly in the `.maif` file.

::: tip When to Use
Use these classes when you need:
- Direct control over block structure
- Immutable, signed data containers
- Custom block types
- Integration with existing systems

For most use cases, prefer the high-level `MAIFClient` and `Artifact` classes.
:::

## MAIFEncoder

The encoder creates self-contained MAIF v3 files with embedded security and provenance.

### Quick Start

```python
from maif import MAIFEncoder

# Create encoder with output path (v3 format)
encoder = MAIFEncoder("output.maif", agent_id="my-agent")

# Add blocks
encoder.add_text_block("Hello, World!")
encoder.add_binary_block(b"binary data", block_type=BlockType.BINARY)

# Finalize - signs and writes all security data
encoder.finalize()
```

### Constructor

```python
class MAIFEncoder:
    def __init__(
        self,
        file_path: str,
        agent_id: str = "default-agent",
        enable_privacy: bool = False
    ):
        """
        Initialize MAIF v3 encoder.

        Args:
            file_path: Output path for the self-contained MAIF file
            agent_id: Unique agent identifier
            enable_privacy: Enable privacy features (encryption, anonymization)
        """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `blocks` | `List[SecureBlock]` | List of all blocks in the encoder |
| `agent_id` | `str` | Agent identifier |
| `provenance` | `List[ProvenanceEntry]` | Embedded provenance chain |

### Methods

#### add_text_block

Add a text block with automatic signing.

```python
def add_text_block(
    self,
    text: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Add a text block (automatically signed with Ed25519).

    Args:
        text: Text content to add
        metadata: Optional metadata dictionary

    Returns:
        Block ID
    """
```

**Example:**

```python
from maif import MAIFEncoder

encoder = MAIFEncoder("output.maif", agent_id="agent-1")

# Simple text block
block_id = encoder.add_text_block("Hello, World!")

# Text with metadata
block_id = encoder.add_text_block(
    "Important information",
    metadata={"category": "research", "priority": "high"}
)

encoder.finalize()
```

#### add_binary_block

Add a binary data block.

```python
def add_binary_block(
    self,
    data: bytes,
    block_type: BlockType,
    metadata: Optional[Dict] = None
) -> str:
    """
    Add a binary block (automatically signed with Ed25519).

    Args:
        data: Binary data to add
        block_type: Type of block (BlockType enum)
        metadata: Optional metadata dictionary

    Returns:
        Block ID
    """
```

**Example:**

```python
from maif import MAIFEncoder, BlockType

encoder = MAIFEncoder("output.maif", agent_id="agent-1")

# Add image data
with open("image.png", "rb") as f:
    block_id = encoder.add_binary_block(
        f.read(),
        BlockType.IMAGE,
        metadata={"format": "png", "width": 1920, "height": 1080}
    )

# Add JSON data as binary
import json
block_id = encoder.add_binary_block(
    json.dumps({"key": "value"}).encode(),
    BlockType.BINARY,
    metadata={"format": "json"}
)

encoder.finalize()
```

#### add_embeddings_block

Add semantic embeddings.

```python
def add_embeddings_block(
    self,
    embeddings: List[List[float]],
    metadata: Optional[Dict] = None
) -> str:
    """
    Add an embeddings block (automatically signed with Ed25519).

    Args:
        embeddings: List of embedding vectors
        metadata: Optional metadata dictionary

    Returns:
        Block ID
    """
```

**Example:**

```python
# Add embeddings from your model
embeddings = [
    [0.1, 0.2, 0.3, ...],  # 768-dimensional vector
    [0.4, 0.5, 0.6, ...],
]

block_id = encoder.add_embeddings_block(
    embeddings,
    metadata={"model": "sentence-transformers", "dimensions": 768}
)

encoder.finalize()
```

#### finalize

Finalize the MAIF file with signatures and Merkle root.

```python
def finalize(self) -> None:
    """
    Finalize the MAIF file.
    
    This method:
    - Calculates the Merkle root
    - Signs the file with Ed25519
    - Writes the provenance chain
    - Updates file header with final checksums
    
    After calling finalize(), no more blocks can be added.
    """
```

**Example:**

```python
encoder = MAIFEncoder("output.maif", agent_id="agent-1")
encoder.add_text_block("Hello")
encoder.add_text_block("World")

# Finalize - signs and completes the file
encoder.finalize()
# Creates: output.maif (self-contained, no manifest needed)
```

## MAIFDecoder

The decoder reads self-contained MAIF v3 files and provides access to blocks, provenance, and security info.

### Quick Start

```python
from maif import MAIFDecoder

# Load a MAIF v3 file (no manifest needed)
decoder = MAIFDecoder("data.maif")
decoder.load()

# Access blocks
for block in decoder.blocks:
    print(f"Block {block.header.block_id}: {block.header.block_type}")
    
# Verify integrity
is_valid, errors = decoder.verify_integrity()
print(f"Integrity: {'VALID' if is_valid else 'INVALID'}")
```

### Constructor

```python
class MAIFDecoder:
    def __init__(self, file_path: str):
        """
        Initialize MAIF v3 decoder.

        Args:
            file_path: Path to the self-contained MAIF file

        Raises:
            FileNotFoundError: If MAIF file doesn't exist
        """
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `blocks` | `List[SecureBlock]` | List of all blocks in the file |
| `file_header` | `SecureFileHeader` | File header with metadata |
| `provenance` | `List[ProvenanceEntry]` | Embedded provenance chain |

### Methods

#### load

Load and parse the MAIF file.

```python
def load(self) -> None:
    """
    Load and parse the MAIF file.
    
    Must be called before accessing blocks or provenance.
    """
```

#### verify_integrity

Verify the integrity of the MAIF file.

```python
def verify_integrity(self) -> Tuple[bool, List[str]]:
    """
    Verify file integrity including:
    - File header checksum
    - Merkle root verification
    - Block content hashes
    - Block signatures (Ed25519)
    - Provenance chain integrity

    Returns:
        Tuple of (is_valid, error_messages)
    """
```

**Example:**

```python
decoder = MAIFDecoder("data.maif")
decoder.load()

is_valid, errors = decoder.verify_integrity()
if is_valid:
    print("✓ File integrity verified")
else:
    print("✗ Integrity check failed:")
    for error in errors:
        print(f"  - {error}")
```

#### get_file_info

Get file information summary.

```python
def get_file_info(self) -> Dict[str, Any]:
    """
    Get file information.

    Returns:
        Dictionary with version, block_count, is_signed, is_finalized, etc.
    """
```

#### get_security_info

Get security information.

```python
def get_security_info(self) -> Dict[str, Any]:
    """
    Get security information.

    Returns:
        Dictionary with key_algorithm, public_key, merkle_root, etc.
    """
```

#### get_provenance

Get the embedded provenance chain.

```python
def get_provenance(self) -> List[ProvenanceEntry]:
    """
    Get the embedded provenance chain.

    Returns:
        List of ProvenanceEntry objects in chronological order
    """
```

#### export_manifest

Export manifest data (for compatibility with tools expecting JSON).

```python
def export_manifest(self) -> Dict[str, Any]:
    """
    Export manifest-like data from the self-contained file.

    Returns:
        Dictionary with blocks, provenance, file_info, security
    """
```

## SecureBlock

The `SecureBlock` dataclass represents a single block in a MAIF v3 file.

```python
@dataclass
class SecureBlock:
    header: SecureBlockHeader
    data: bytes
    metadata: Optional[Dict] = None
    
    def get_content_hash(self) -> bytes:
        """Calculate SHA-256 hash of block data."""
```

### SecureBlockHeader Properties

| Property | Type | Description |
|----------|------|-------------|
| `block_type` | `BlockType` | Type of block (TEXT, BINARY, IMAGE, etc.) |
| `size` | `int` | Size of the block data in bytes |
| `content_hash` | `bytes` | SHA-256 hash of block data (32 bytes) |
| `block_signature` | `bytes` | Ed25519 signature (64 bytes) |
| `block_id` | `str` | Unique block identifier (UUID) |
| `timestamp` | `int` | Block creation timestamp |
| `version` | `int` | Block version number |
| `flags` | `int` | Block flags (signed, encrypted, etc.) |

## Block Types

MAIF v3 supports the following standard block types:

```python
from maif import BlockType

BlockType.TEXT         # Text content (UTF-8)
BlockType.BINARY       # Binary data
BlockType.IMAGE        # Image data
BlockType.VIDEO        # Video data
BlockType.AUDIO        # Audio data
BlockType.EMBEDDINGS   # Vector embeddings
BlockType.METADATA     # Metadata block
BlockType.SECURITY     # Security information
BlockType.PROVENANCE   # Provenance chain
```

## Complete Example

```python
from maif import MAIFEncoder, MAIFDecoder, BlockType

# Create a self-contained MAIF v3 file
encoder = MAIFEncoder("research.maif", agent_id="demo-agent")

# Add text content
doc_id = encoder.add_text_block(
    "This is a document about AI safety.",
    metadata={"title": "AI Safety", "category": "research"}
)

# Add another text block
encoder.add_text_block(
    "Additional research notes on alignment.",
    metadata={"category": "notes"}
)

# Add embeddings
encoder.add_embeddings_block(
    [[0.1, 0.2, 0.3] * 256],  # 768-dimensional embedding
    metadata={"model": "ada-002"}
)

# Finalize (signs everything with Ed25519)
encoder.finalize()

# Load and verify
decoder = MAIFDecoder("research.maif")
decoder.load()

# Check integrity
is_valid, errors = decoder.verify_integrity()
print(f"Integrity: {'✓ VALID' if is_valid else '✗ INVALID'}")

# Get file info
file_info = decoder.get_file_info()
print(f"Blocks: {file_info['block_count']}")
print(f"Signed: {file_info['is_signed']}")

# Get security info
security = decoder.get_security_info()
print(f"Algorithm: {security['key_algorithm']}")

# Get provenance
provenance = decoder.get_provenance()
print(f"Provenance entries: {len(provenance)}")
for entry in provenance:
    print(f"  - {entry.action} by {entry.agent_id}")

# Read blocks
for block in decoder.blocks:
    print(f"\nBlock ID: {block.header.block_id}")
    print(f"  Type: {block.header.block_type.name}")
    print(f"  Size: {block.header.size} bytes")
    if block.header.block_type == BlockType.TEXT:
        print(f"  Content: {block.data.decode('utf-8')[:50]}...")
```
