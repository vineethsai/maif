"""
MAIF Media Module

Contains media processing features:
- Video optimization and processing
"""

try:
    from .video_optimized import VideoOptimizer, VideoBlock, VideoMetadata
except ImportError:
    VideoOptimizer = None
    VideoBlock = None
    VideoMetadata = None

__all__ = [
    "VideoOptimizer",
    "VideoBlock",
    "VideoMetadata",
]

