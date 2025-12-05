"""
MAIF Core Module

Contains the core encoding/decoding functionality:
- MAIFEncoder: Create MAIF files with Ed25519 signatures
- MAIFDecoder: Read and verify MAIF files
- Block types and format structures
"""

from .core import (
    MAIFEncoder,
    MAIFDecoder,
    MAIFParser,
    MAIFBlock,
    MAIFVersion,
    MAIFHeader,
    create_maif,
    verify_maif,
    quick_create,
    quick_verify,
    quick_read,
)

from .secure_format import (
    SecureBlock,
    SecureBlockHeader,
    SecureFileHeader,
    ProvenanceEntry,
    FileFooter,
    BlockFlags,
    FileFlags,
    SecureBlockType,
    BlockType,  # This is an alias for SecureBlockType
    MAGIC_HEADER,
    MAGIC_FOOTER,
    FORMAT_VERSION_MAJOR,
    FORMAT_VERSION_MINOR,
)

__all__ = [
    # Primary classes
    "MAIFEncoder",
    "MAIFDecoder",
    "MAIFParser",
    "BlockType",
    # Data classes
    "MAIFBlock",
    "MAIFVersion",
    "MAIFHeader",
    # Secure format
    "SecureBlock",
    "SecureBlockHeader",
    "SecureFileHeader",
    "ProvenanceEntry",
    "FileFooter",
    "BlockFlags",
    "FileFlags",
    "SecureBlockType",
    # Convenience functions
    "create_maif",
    "verify_maif",
    "quick_create",
    "quick_verify",
    "quick_read",
    # Constants
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
]
