"""
MAIF Storage Module

Contains storage backends:
- Columnar storage for analytics
- Hot buffer for frequently accessed data
"""

try:
    from .columnar_storage import ColumnarStorage, ColumnarBlock
except ImportError:
    ColumnarStorage = None
    ColumnarBlock = None

try:
    from .hot_buffer import HotBuffer, BufferConfig
except ImportError:
    HotBuffer = None
    BufferConfig = None

__all__ = [
    "ColumnarStorage",
    "ColumnarBlock",
    "HotBuffer",
    "BufferConfig",
]

