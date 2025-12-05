"""
MAIF Semantic Module

Contains semantic/ML features:
- Semantic embeddings
- Knowledge graph support
- Novel algorithms (ACAM, HSC, CSB)
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
        SENTENCE_TRANSFORMERS_AVAILABLE,
        SentenceTransformer,
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
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

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
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "SentenceTransformer",
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

