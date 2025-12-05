"""
MAIF Performance Module

Contains performance optimization features:
- GPU acceleration
- Self-optimizing algorithms
- Adaptive processing rules
"""

try:
    from .performance_features import PerformanceOptimizer, GPUAccelerator
except ImportError:
    PerformanceOptimizer = None
    GPUAccelerator = None

try:
    from .self_optimizing import SelfOptimizingMAIF, OptimizationStrategy
except ImportError:
    SelfOptimizingMAIF = None
    OptimizationStrategy = None

try:
    from .adaptation_rules import AdaptationEngine, AdaptationRule
except ImportError:
    AdaptationEngine = None
    AdaptationRule = None

__all__ = [
    "PerformanceOptimizer",
    "GPUAccelerator",
    "SelfOptimizingMAIF",
    "OptimizationStrategy",
    "AdaptationEngine",
    "AdaptationRule",
]

