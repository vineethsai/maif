"""
MAIF Integration Module

.. deprecated::
    This module is deprecated. Use ``maif.integrations`` instead for
    framework integrations (LangChain, CrewAI, LangGraph, Strands).
    
    Example migration:
        # Old (deprecated)
        from maif.integration.framework_adapters import MAIFLangChainVectorStore
        
        # New
        from maif.integrations.langchain import MAIFVectorStore

Contains integration features:
- Framework adapters (LangChain, LlamaIndex, etc.) - DEPRECATED
- Enhanced processing
"""

import warnings

# Emit deprecation warning when module is imported
warnings.warn(
    "The 'maif.integration' module is deprecated. "
    "Use 'maif.integrations' instead for framework integrations. "
    "See docs/guide/integrations/ for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

try:
    from .integration import MAIFConverter
except ImportError:
    MAIFConverter = None

MAIFIntegration = None  # Legacy alias, not implemented

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

# Legacy framework adapters - deprecated
try:
    from .framework_adapters import (
        MAIFLangChainVectorStore,
        MAIFLlamaIndexVectorStore,
        MAIFMemGPTBackend,
        MAIFSemanticKernelConnector,
    )
    
    # Emit specific warnings for deprecated classes
    def _deprecated_adapter_warning(name: str):
        warnings.warn(
            f"{name} is deprecated. Use maif.integrations instead.",
            DeprecationWarning,
            stacklevel=3
        )
    
except ImportError:
    MAIFLangChainVectorStore = None
    MAIFLlamaIndexVectorStore = None
    MAIFMemGPTBackend = None
    MAIFSemanticKernelConnector = None

# Legacy aliases (these never existed but were in __all__)
LangChainAdapter = None
LlamaIndexAdapter = None
AutoGenAdapter = None

__all__ = [
    "MAIFIntegration",
    "MAIFConverter",
    "EnhancedMAIFProcessor",
    "ConversionResult",
    "EnhancedMAIF",
    # Legacy adapters (deprecated)
    "MAIFLangChainVectorStore",
    "MAIFLlamaIndexVectorStore",
    "MAIFMemGPTBackend",
    "MAIFSemanticKernelConnector",
    # Legacy aliases (never implemented)
    "LangChainAdapter",
    "LlamaIndexAdapter",
    "AutoGenAdapter",
]

