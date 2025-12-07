"""
MAIF (Multimodal Artifact File Format) Library

A comprehensive library for creating, managing, and analyzing MAIF files.
MAIF is an AI-native file format designed for multimodal content with
embedded security (Ed25519 signatures), semantics, and provenance tracking.

Version 3.0 - Secure Format
- Self-contained binary files (no external manifest)
- Ed25519 cryptographic signatures
- Immutable blocks with tamper detection
- Embedded provenance chain

Quick Start:
    from maif import MAIFEncoder, MAIFDecoder

    # Create a MAIF file
    encoder = MAIFEncoder("output.maif", agent_id="my-agent")
    encoder.add_text_block("Hello, world!")
    encoder.finalize()

    # Read and verify a MAIF file
    decoder = MAIFDecoder("output.maif")
    is_valid, errors = decoder.verify_integrity()
    if is_valid:
        blocks = decoder.get_blocks()

Package Structure:
    maif/
    ├── core/          # Core encoding/decoding
    ├── security/      # Signing and verification
    ├── privacy/       # Encryption and anonymization
    ├── semantic/      # Embeddings and ML features
    ├── streaming/     # Streaming I/O
    ├── agents/        # Multi-agent framework
    ├── compliance/    # Logging and forensics
    ├── compression/   # Compression algorithms
    ├── storage/       # Storage backends
    ├── transactions/  # ACID transactions
    ├── integration/   # Framework integrations
    ├── performance/   # Performance optimization
    ├── media/         # Media processing
    └── utils/         # Utilities
"""

# =============================================================================
# Core API - Main encoding/decoding functionality
# =============================================================================

from .core import (
    # Primary classes
    MAIFEncoder,
    MAIFDecoder,
    MAIFParser,
    BlockType,
    # Data classes
    MAIFBlock,
    MAIFVersion,
    MAIFHeader,
    # Secure format structures
    SecureBlock,
    SecureBlockHeader,
    SecureFileHeader,
    ProvenanceEntry,
    FileFooter,
    BlockFlags,
    FileFlags,
    # Convenience functions
    create_maif,
    verify_maif,
    quick_create,
    quick_verify,
    quick_read,
    # Constants
    MAGIC_HEADER,
    MAGIC_FOOTER,
    FORMAT_VERSION_MAJOR,
    FORMAT_VERSION_MINOR,
)

# Legacy aliases for backwards compatibility
SecureMAIFWriter = MAIFEncoder
SecureMAIFReader = MAIFDecoder
SecureBlockType = BlockType

# =============================================================================
# Security - Signing and verification
# =============================================================================

try:
    from .security import MAIFSigner, MAIFVerifier, SecurityManager
except ImportError:
    MAIFSigner = None
    MAIFVerifier = None
    SecurityManager = None

# =============================================================================
# Privacy - Privacy controls and encryption
# =============================================================================

try:
    from .privacy import (
        PrivacyEngine,
        PrivacyPolicy,
        PrivacyLevel,
        EncryptionMode,
        AccessRule,
        DifferentialPrivacy,
        SecureMultipartyComputation,
        ZeroKnowledgeProof,
    )
except ImportError:
    PrivacyEngine = None
    PrivacyPolicy = None
    PrivacyLevel = None
    EncryptionMode = None
    AccessRule = None
    DifferentialPrivacy = None
    SecureMultipartyComputation = None
    ZeroKnowledgeProof = None

# =============================================================================
# Semantics - Embeddings and knowledge graphs
# =============================================================================

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

# Enhanced algorithms
try:
    from .semantic import (
        AdaptiveCrossModalAttention,
        EnhancedHSC as EnhancedHierarchicalSemanticCompression,
        EnhancedCSB as EnhancedCryptographicSemanticBinding,
        AttentionWeights,
        ENHANCED_ALGORITHMS_AVAILABLE,
    )
except ImportError:
    AdaptiveCrossModalAttention = None
    EnhancedHierarchicalSemanticCompression = None
    EnhancedCryptographicSemanticBinding = None
    AttentionWeights = None
    ENHANCED_ALGORITHMS_AVAILABLE = False

# =============================================================================
# Forensics and Validation
# =============================================================================

try:
    from .compliance import ForensicAnalyzer, ForensicEvidence
except ImportError:
    ForensicAnalyzer = None
    ForensicEvidence = None

try:
    from .utils import MAIFValidator, MAIFRepairTool
except ImportError:
    MAIFValidator = None
    MAIFRepairTool = None

# =============================================================================
# Compression
# =============================================================================

try:
    from .compression import CompressionManager, CompressionMetadata
except ImportError:
    CompressionManager = None
    CompressionMetadata = None

# =============================================================================
# Streaming
# =============================================================================

try:
    from .streaming import MAIFStreamReader, MAIFStreamWriter
except ImportError:
    MAIFStreamReader = None
    MAIFStreamWriter = None

# =============================================================================
# Metadata
# =============================================================================

try:
    from .utils import MAIFMetadataManager
except ImportError:
    MAIFMetadataManager = None

# =============================================================================
# Agent Framework
# =============================================================================

try:
    from .agents import (
        MAIFAgent,
        PerceptionSystem,
        ReasoningSystem,
        ExecutionSystem,
    )
except ImportError:
    MAIFAgent = None
    PerceptionSystem = None
    ReasoningSystem = None
    ExecutionSystem = None

# =============================================================================
# Production Features
# =============================================================================

try:
    from .utils import HealthChecker, HealthStatus
except ImportError:
    HealthChecker = None
    HealthStatus = None

try:
    from .utils import (
        RateLimiter,
        RateLimitConfig,
        CostBasedRateLimiter,
        rate_limit,
    )
except ImportError:
    RateLimiter = None
    RateLimitConfig = None
    CostBasedRateLimiter = None
    rate_limit = None

try:
    from .utils import (
        MetricsAggregator,
        MAIFMetrics,
        initialize_metrics,
        get_metrics,
    )
except ImportError:
    MetricsAggregator = None
    MAIFMetrics = None
    initialize_metrics = None
    get_metrics = None

try:
    from .utils import (
        CostTracker,
        Budget,
        BudgetExceededException,
        initialize_cost_tracking,
        get_cost_tracker,
        with_cost_tracking,
    )
except ImportError:
    CostTracker = None
    Budget = None
    BudgetExceededException = None
    initialize_cost_tracking = None
    get_cost_tracker = None
    with_cost_tracking = None

try:
    from .utils import (
        BatchProcessor,
        StreamBatchProcessor,
        DistributedBatchProcessor,
        batch_process,
    )
except ImportError:
    BatchProcessor = None
    StreamBatchProcessor = None
    DistributedBatchProcessor = None
    batch_process = None

# =============================================================================
# Integration (Legacy - use maif.integrations instead for framework support)
# =============================================================================

try:
    from .integration import EnhancedMAIFProcessor, ConversionResult, EnhancedMAIF
except ImportError:
    EnhancedMAIFProcessor = None
    ConversionResult = None
    EnhancedMAIF = None

# =============================================================================
# Framework Integrations (NEW - LangGraph, LangChain, CrewAI, Strands)
# =============================================================================
# Use: from maif.integrations.langgraph import MAIFCheckpointer
# See docs/guide/integrations/ for documentation

INTEGRATIONS_AVAILABLE = True
try:
    from .integrations import (
        EventType,
        MAIFProvenanceTracker,
        BaseMAIFCallback,
    )
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    EventType = None
    MAIFProvenanceTracker = None
    BaseMAIFCallback = None

# =============================================================================
# AWS Integrations (optional)
# =============================================================================

AWS_IMPORTS_AVAILABLE = False
try:
    from .aws_lambda_integration import AWSLambdaIntegration
    from .aws_stepfunctions_integration import AWSStepFunctionsIntegration
    from .aws_xray_integration import MAIFXRayIntegration, xray_trace, xray_subsegment
    from .aws_deployment import (
        DeploymentManager,
        CloudFormationGenerator,
        LambdaPackager,
        DockerfileGenerator,
    )

    AWS_IMPORTS_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# Convenience API
# =============================================================================

try:
    from .utils import SimpleMAIFAgent, create_agent

    CONVENIENCE_API_AVAILABLE = True
except ImportError:
    SimpleMAIFAgent = None
    create_agent = None
    CONVENIENCE_API_AVAILABLE = False

try:
    from .utils import VectorDBMigrator, migrate_to_maif

    MIGRATION_TOOLS_AVAILABLE = True
except ImportError:
    VectorDBMigrator = None
    migrate_to_maif = None
    MIGRATION_TOOLS_AVAILABLE = False

try:
    from .utils import MAIFDebugger, debug_maif

    DEBUG_TOOLS_AVAILABLE = True
except ImportError:
    MAIFDebugger = None
    debug_maif = None
    DEBUG_TOOLS_AVAILABLE = False

# =============================================================================
# Module Info
# =============================================================================

__version__ = "3.0.0"
__author__ = "MAIF Development Team"
__license__ = "MIT"

__all__ = [
    # Core API
    "MAIFEncoder",
    "MAIFDecoder",
    "MAIFParser",
    "BlockType",
    "MAIFBlock",
    "MAIFVersion",
    "MAIFHeader",
    # Secure format
    "SecureBlock",
    "SecureBlockHeader",
    "SecureFileHeader",
    "ProvenanceEntry",
    "FileFooter",
    "BlockFlags",
    "FileFlags",
    # Legacy aliases
    "SecureMAIFWriter",
    "SecureMAIFReader",
    "SecureBlockType",
    # Convenience functions
    "create_maif",
    "verify_maif",
    "quick_create",
    "quick_verify",
    "quick_read",
    # Security
    "MAIFSigner",
    "MAIFVerifier",
    "SecurityManager",
    # Privacy
    "PrivacyEngine",
    "PrivacyPolicy",
    "PrivacyLevel",
    "EncryptionMode",
    "AccessRule",
    # Semantics
    "SemanticEmbedder",
    "SemanticEmbedding",
    "KnowledgeTriple",
    "KnowledgeGraphBuilder",
    # Forensics & Validation
    "ForensicAnalyzer",
    "MAIFValidator",
    # Streaming
    "MAIFStreamReader",
    "MAIFStreamWriter",
    # Agent
    "MAIFAgent",
    # Integration
    "EnhancedMAIF",
    "EnhancedMAIFProcessor",
    # Production
    "HealthChecker",
    "RateLimiter",
    "MetricsAggregator",
    "CostTracker",
    "BatchProcessor",
    # Constants
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
    # Feature flags
    "ENHANCED_ALGORITHMS_AVAILABLE",
    "AWS_IMPORTS_AVAILABLE",
    "CONVENIENCE_API_AVAILABLE",
    "MIGRATION_TOOLS_AVAILABLE",
    "DEBUG_TOOLS_AVAILABLE",
    "INTEGRATIONS_AVAILABLE",
    # Framework Integrations (base classes)
    "EventType",
    "MAIFProvenanceTracker",
    "BaseMAIFCallback",
]
