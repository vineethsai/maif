"""
MAIF Streaming Module

Contains streaming support:
- Memory-mapped I/O for large files
- Streaming read/write operations
"""

try:
    from .streaming import (
        MAIFStreamReader,
        MAIFStreamWriter,
        StreamingConfig,
        PerformanceProfiler,
        StreamingMAIFProcessor,
        MAIFStreamer,
    )
except ImportError:
    MAIFStreamReader = None
    MAIFStreamWriter = None
    StreamingConfig = None
    PerformanceProfiler = None
    StreamingMAIFProcessor = None
    MAIFStreamer = None

try:
    from .streaming_ultra import UltraHighThroughputReader
except ImportError:
    UltraHighThroughputReader = None

__all__ = [
    "MAIFStreamReader",
    "MAIFStreamWriter",
    "MAIFStreamer",
    "PerformanceProfiler",
    "StreamingConfig",
    "StreamingMAIFProcessor",
    "UltraHighThroughputReader",
]

