"""
MAIF Compression Module

Contains compression features:
- Multiple compression algorithms (gzip, zstd, brotli, lz4)
- Hierarchical Semantic Compression (HSC)
"""

try:
    from .compression_manager import CompressionManager
except ImportError:
    CompressionManager = None

try:
    from .compression import (
        CompressionAlgorithm,
        CompressionConfig,
        CompressionMetadata,
        CompressionResult,
        MAIFCompressor,
        SemanticAwareCompressor,
    )
except ImportError:
    CompressionAlgorithm = None
    CompressionConfig = None
    CompressionMetadata = None
    CompressionResult = None
    MAIFCompressor = None
    SemanticAwareCompressor = None

__all__ = [
    "CompressionAlgorithm",
    "CompressionConfig",
    "CompressionManager",
    "CompressionMetadata",
    "CompressionResult",
    "MAIFCompressor",
    "SemanticAwareCompressor",
]

