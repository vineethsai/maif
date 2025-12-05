"""
MAIF Integration Module

Contains integration features:
- Framework adapters (LangChain, LlamaIndex, etc.)
- Enhanced processing
"""

try:
    from .integration import MAIFIntegration
except ImportError:
    MAIFIntegration = None

try:
    from .integration_enhanced import (
        EnhancedMAIFProcessor,
        ConversionResult,
        EnhancedMAIF,
    )
except ImportError:
    EnhancedMAIFProcessor = None
    ConversionResult = None
    EnhancedMAIF = None

try:
    from .framework_adapters import (
        LangChainAdapter,
        LlamaIndexAdapter,
        AutoGenAdapter,
    )
except ImportError:
    LangChainAdapter = None
    LlamaIndexAdapter = None
    AutoGenAdapter = None

__all__ = [
    "MAIFIntegration",
    "EnhancedMAIFProcessor",
    "ConversionResult",
    "EnhancedMAIF",
    "LangChainAdapter",
    "LlamaIndexAdapter",
    "AutoGenAdapter",
]

