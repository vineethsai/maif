"""
MAIF Semantic Module

Contains semantic/ML features:
- Semantic embeddings (TF-IDF based, lightweight)
- Knowledge graph support
- Novel algorithms (ACAM, HSC, CSB)
- Neural embeddings support (optional, via sentence-transformers)

Note: This module uses TF-IDF for embeddings by default.
No TensorFlow or sentence-transformers dependencies required.
Neural embeddings are opt-in and loaded lazily if available.
"""

try:
    from .semantic import (
        SemanticEmbedder,
        SemanticEmbedding,
        KnowledgeTriple,
        CrossModalAttention,
        HierarchicalSemanticCompression,
        CryptographicSemanticBinding,
        DeepSemanticUnderstanding,
        KnowledgeGraphBuilder,
        AttentionWeights,
        # TF-IDF fallback (lightweight, no TensorFlow)
        TFIDFEmbedder,
        get_embedder,
        _check_tfidf_available,
    )
except ImportError:
    SemanticEmbedder = None
    SemanticEmbedding = None
    KnowledgeTriple = None
    CrossModalAttention = None
    HierarchicalSemanticCompression = None
    CryptographicSemanticBinding = None
    DeepSemanticUnderstanding = None
    KnowledgeGraphBuilder = None
    AttentionWeights = None
    TFIDFEmbedder = None
    get_embedder = None
    _check_tfidf_available = lambda: False

try:
    from .semantic_optimized import (
        OptimizedSemanticEmbedder,
        AdaptiveCrossModalAttention,
        HierarchicalSemanticCompression as EnhancedHSC,
        CryptographicSemanticBinding as EnhancedCSB,
    )
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    OptimizedSemanticEmbedder = None
    AdaptiveCrossModalAttention = None
    EnhancedHSC = None
    EnhancedCSB = None
    ENHANCED_ALGORITHMS_AVAILABLE = False

# Neural embeddings support (lazy import mechanism)
NEURAL_AVAILABLE = None
_SentenceTransformer = None


def _check_neural_available():
    """Check if sentence-transformers is available (lazy import)."""
    global NEURAL_AVAILABLE, _SentenceTransformer
    if NEURAL_AVAILABLE is not None:
        return NEURAL_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
        NEURAL_AVAILABLE = True
    except ImportError:
        NEURAL_AVAILABLE = False
    return NEURAL_AVAILABLE


def _get_sentence_transformer():
    """Get the SentenceTransformer class if available."""
    _check_neural_available()
    return _SentenceTransformer

__all__ = [
    # Core semantic
    "SemanticEmbedder",
    "SemanticEmbedding",
    "KnowledgeTriple",
    "KnowledgeGraphBuilder",
    "AttentionWeights",
    # TF-IDF embeddings (lightweight, no TensorFlow)
    "TFIDFEmbedder",
    "get_embedder",
    "_check_tfidf_available",
    # Novel algorithms
    "CrossModalAttention",
    "HierarchicalSemanticCompression",
    "CryptographicSemanticBinding",
    "DeepSemanticUnderstanding",
    # Enhanced/Optimized algorithms
    "OptimizedSemanticEmbedder",
    "AdaptiveCrossModalAttention",
    "EnhancedHSC",
    "EnhancedCSB",
    "ENHANCED_ALGORITHMS_AVAILABLE",
    # Neural embeddings support (opt-in)
    "NEURAL_AVAILABLE",
    "_check_neural_available",
    "_get_sentence_transformer",
]
