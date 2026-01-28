"""
MAIF Semantic Module

Contains semantic/ML features:
- Semantic embeddings (TF-IDF based, lightweight) ✓ WORKING
- Knowledge graph support ✓ WORKING
- Novel algorithms: ACAM, HSC, CSB ⚠ RESEARCH IMPLEMENTATIONS

IMPORTANT: Read HONEST_STATUS.md and LIMITATIONS.md before using advanced features.

STATUS SUMMARY:
✓ TF-IDF embeddings: Production-ready
✓ FAISS indexing: Production-ready
⚠ HSC (Hierarchical Semantic Compression): ~1.5x compression, NOT the claimed 2.5-4x
⚠ ACAM (Adaptive Cross-Modal Attention): Research implementation, very slow training
⚠ CSB (Cryptographic Semantic Binding): Not validated for high-security use
❌ Neural embeddings: NOT IMPLEMENTED (planned for v2.1)

Note: This module uses TF-IDF for embeddings by default.
No TensorFlow or sentence-transformers dependencies required.

See LIMITATIONS.md for honest descriptions of what works and what doesn't.
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
]
