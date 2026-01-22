"""
MAIF (Multimodal Artifact File Format) Library
==============================================

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

Feature Status
--------------
Each feature is classified by maturity level:

STABLE (Production-ready, API stable):
    - MAIFEncoder, MAIFDecoder, MAIFParser - Core encoding/decoding
    - BlockType, MAIFBlock, MAIFVersion, MAIFHeader - Data structures
    - SecureBlock, SecureBlockHeader, SecureFileHeader - Secure format
    - ProvenanceEntry, FileFooter, BlockFlags, FileFlags - Format structures
    - create_maif, verify_maif, quick_create, quick_verify, quick_read - Convenience API
    - MAGIC_HEADER, MAGIC_FOOTER, FORMAT_VERSION_* - Constants
    - MAIFSigner, MAIFVerifier - Ed25519 signing (requires cryptography)
    - PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode - Privacy controls
    - CompressionManager - Multi-algorithm compression
    - MAIFStreamReader, MAIFStreamWriter - Streaming I/O
    - MAIFValidator - File validation

BETA (Working, API may change):
    - MAIFRootCA, AgentIdentity, MAIFCertificate - PKI/Certificate Authority
    - SecurityManager - Security management
    - ForensicAnalyzer - Forensic analysis
    - HealthChecker - Production health checks
    - RateLimiter - Rate limiting
    - MetricsAggregator - Metrics collection
    - BatchProcessor - Batch processing
    - MAIFMetadataManager - Metadata management
    - Framework Integrations (LangChain, CrewAI, LangGraph, Strands)

EXPERIMENTAL (Working but unstable, significant changes expected):
    - SemanticEmbedder - TF-IDF based embeddings
    - KnowledgeGraphBuilder - Knowledge graph construction
    - CrossModalAttention, HierarchicalSemanticCompression - Novel algorithms
    - CryptographicSemanticBinding - Semantic security binding
    - MAIFAgent - Agent framework
    - CostTracker - Cost tracking
    - DifferentialPrivacy - Privacy-preserving analysis
    - SecureMultipartyComputation, ZeroKnowledgeProof - Advanced crypto

PLANNED (Not yet implemented or placeholders):
    - AWS Lambda/StepFunctions/X-Ray integrations - Files not present
    - DeploymentManager, CloudFormationGenerator - Not implemented
    - Enhanced algorithm variants - Partial implementation
    - VectorDBMigrator - Database migration tools

Package Structure:
    maif/
    |-- core/          # Core encoding/decoding [STABLE]
    |-- security/      # Signing and verification [STABLE/BETA]
    |-- privacy/       # Encryption and anonymization [STABLE]
    |-- semantic/      # Embeddings and ML features [EXPERIMENTAL]
    |-- streaming/     # Streaming I/O [STABLE]
    |-- agents/        # Multi-agent framework [EXPERIMENTAL]
    |-- compliance/    # Logging and forensics [BETA]
    |-- compression/   # Compression algorithms [STABLE]
    |-- storage/       # Storage backends [BETA]
    |-- transactions/  # ACID transactions [BETA]
    |-- integrations/  # Framework integrations [BETA]
    |-- performance/   # Performance optimization [EXPERIMENTAL]
    |-- media/         # Media processing [EXPERIMENTAL]
    |-- utils/         # Utilities [STABLE/BETA]
"""

import warnings
import logging

_logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flags - Check these before using optional features
# =============================================================================

# Core features - always available
CORE_AVAILABLE = True

# Security features - requires cryptography package
SECURITY_AVAILABLE = False

# Certificate Authority - requires security module
CA_AVAILABLE = False

# Privacy features - requires cryptography package
PRIVACY_AVAILABLE = False

# Semantic features - TF-IDF based (lightweight, no ML dependencies)
SEMANTIC_AVAILABLE = False

# Enhanced semantic algorithms
ENHANCED_ALGORITHMS_AVAILABLE = False

# Streaming features
STREAMING_AVAILABLE = False

# Agent framework
AGENT_FRAMEWORK_AVAILABLE = False

# Compression features
COMPRESSION_AVAILABLE = False

# Forensics and compliance
FORENSICS_AVAILABLE = False

# Production utilities
HEALTH_CHECK_AVAILABLE = False
RATE_LIMITER_AVAILABLE = False
METRICS_AVAILABLE = False
COST_TRACKER_AVAILABLE = False
BATCH_PROCESSOR_AVAILABLE = False

# Framework integrations
INTEGRATIONS_AVAILABLE = False

# Convenience API
CONVENIENCE_API_AVAILABLE = False
MIGRATION_TOOLS_AVAILABLE = False
DEBUG_TOOLS_AVAILABLE = False

# AWS integrations - NOT IMPLEMENTED (files do not exist)
AWS_IMPORTS_AVAILABLE = False  # Planned feature, not yet available


# =============================================================================
# Core API - STABLE - Main encoding/decoding functionality
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
# Security - STABLE/BETA - Signing and verification
# =============================================================================

try:
    from .security import MAIFSigner, MAIFVerifier, SecurityManager
    SECURITY_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Security module not available: {e}")
    MAIFSigner = None
    MAIFVerifier = None
    SecurityManager = None

# Certificate Authority for provenance verification [BETA]
try:
    from .security import MAIFRootCA, AgentIdentity, MAIFCertificate, CertificateVerifier
    CA_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Certificate Authority not available: {e}")
    MAIFRootCA = None
    AgentIdentity = None
    MAIFCertificate = None
    CertificateVerifier = None


# =============================================================================
# Privacy - STABLE - Privacy controls and encryption
# =============================================================================

try:
    from .privacy import (
        PrivacyEngine,
        PrivacyPolicy,
        PrivacyLevel,
        EncryptionMode,
        AccessRule,
    )
    PRIVACY_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Privacy module not available: {e}")
    PrivacyEngine = None
    PrivacyPolicy = None
    PrivacyLevel = None
    EncryptionMode = None
    AccessRule = None

# Advanced privacy features [EXPERIMENTAL]
try:
    from .privacy import (
        DifferentialPrivacy,
        SecureMultipartyComputation,
        ZeroKnowledgeProof,
    )
except ImportError:
    DifferentialPrivacy = None
    SecureMultipartyComputation = None
    ZeroKnowledgeProof = None


# =============================================================================
# Semantics - EXPERIMENTAL - Embeddings and knowledge graphs
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
        # TF-IDF based embeddings (lightweight, no TensorFlow)
        TFIDFEmbedder,
        get_embedder,
    )
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Semantic module not available: {e}")
    SemanticEmbedder = None
    SemanticEmbedding = None
    KnowledgeTriple = None
    CrossModalAttention = None
    HierarchicalSemanticCompression = None
    CryptographicSemanticBinding = None
    DeepSemanticUnderstanding = None
    KnowledgeGraphBuilder = None
    TFIDFEmbedder = None
    get_embedder = None

# Enhanced algorithms [EXPERIMENTAL]
try:
    from .semantic import (
        AdaptiveCrossModalAttention,
        EnhancedHSC as EnhancedHierarchicalSemanticCompression,
        EnhancedCSB as EnhancedCryptographicSemanticBinding,
        AttentionWeights,
        ENHANCED_ALGORITHMS_AVAILABLE as _enhanced_available,
    )
    ENHANCED_ALGORITHMS_AVAILABLE = _enhanced_available
except ImportError:
    AdaptiveCrossModalAttention = None
    EnhancedHierarchicalSemanticCompression = None
    EnhancedCryptographicSemanticBinding = None
    AttentionWeights = None


# =============================================================================
# Forensics and Validation - BETA
# =============================================================================

try:
    from .compliance import ForensicAnalyzer, ForensicEvidence
    FORENSICS_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Forensics module not available: {e}")
    ForensicAnalyzer = None
    ForensicEvidence = None

try:
    from .utils import MAIFValidator, MAIFRepairTool
except ImportError:
    MAIFValidator = None
    MAIFRepairTool = None


# =============================================================================
# Compression - STABLE
# =============================================================================

try:
    from .compression import CompressionManager, CompressionMetadata
    COMPRESSION_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Compression module not available: {e}")
    CompressionManager = None
    CompressionMetadata = None


# =============================================================================
# Streaming - STABLE
# =============================================================================

try:
    from .streaming import MAIFStreamReader, MAIFStreamWriter
    STREAMING_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Streaming module not available: {e}")
    MAIFStreamReader = None
    MAIFStreamWriter = None


# =============================================================================
# Metadata - BETA
# =============================================================================

try:
    from .utils import MAIFMetadataManager
except ImportError:
    MAIFMetadataManager = None


# =============================================================================
# Agent Framework - EXPERIMENTAL
# =============================================================================

try:
    from .agents import (
        MAIFAgent,
        PerceptionSystem,
        ReasoningSystem,
        ExecutionSystem,
    )
    AGENT_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Agent framework not available: {e}")
    MAIFAgent = None
    PerceptionSystem = None
    ReasoningSystem = None
    ExecutionSystem = None


# =============================================================================
# Production Utilities - BETA
# =============================================================================

# Health checks
try:
    from .utils import HealthChecker, HealthStatus
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    HealthChecker = None
    HealthStatus = None

# Rate limiting
try:
    from .utils import (
        RateLimiter,
        RateLimitConfig,
        CostBasedRateLimiter,
        rate_limit,
    )
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RateLimiter = None
    RateLimitConfig = None
    CostBasedRateLimiter = None
    rate_limit = None

# Metrics
try:
    from .utils import (
        MetricsAggregator,
        MAIFMetrics,
        initialize_metrics,
        get_metrics,
    )
    METRICS_AVAILABLE = True
except ImportError:
    MetricsAggregator = None
    MAIFMetrics = None
    initialize_metrics = None
    get_metrics = None

# Cost tracking [EXPERIMENTAL]
try:
    from .utils import (
        CostTracker,
        Budget,
        BudgetExceededException,
        initialize_cost_tracking,
        get_cost_tracker,
        with_cost_tracking,
    )
    COST_TRACKER_AVAILABLE = True
except ImportError:
    CostTracker = None
    Budget = None
    BudgetExceededException = None
    initialize_cost_tracking = None
    get_cost_tracker = None
    with_cost_tracking = None

# Batch processing
try:
    from .utils import (
        BatchProcessor,
        StreamBatchProcessor,
        DistributedBatchProcessor,
        batch_process,
    )
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BatchProcessor = None
    StreamBatchProcessor = None
    DistributedBatchProcessor = None
    batch_process = None


# =============================================================================
# Integration (Legacy - DEPRECATED)
# =============================================================================

try:
    from .integration import EnhancedMAIFProcessor, ConversionResult, EnhancedMAIF
except ImportError:
    EnhancedMAIFProcessor = None
    ConversionResult = None
    EnhancedMAIF = None


# =============================================================================
# Framework Integrations - BETA (LangGraph, LangChain, CrewAI, Strands)
# =============================================================================
# Use: from maif.integrations.langgraph import MAIFCheckpointer
# See docs/guide/integrations/ for documentation

try:
    from .integrations import (
        EventType,
        MAIFProvenanceTracker,
        BaseMAIFCallback,
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    _logger.debug(f"Framework integrations not available: {e}")
    EventType = None
    MAIFProvenanceTracker = None
    BaseMAIFCallback = None


# =============================================================================
# AWS Integrations - PLANNED (NOT YET IMPLEMENTED)
# =============================================================================
# NOTE: These files do not exist in the codebase. This is a planned feature.
# AWS_IMPORTS_AVAILABLE remains False. Do not use these imports.

# Placeholder for future implementation:
# - AWSLambdaIntegration
# - AWSStepFunctionsIntegration
# - MAIFXRayIntegration
# - DeploymentManager
# - CloudFormationGenerator
# - LambdaPackager
# - DockerfileGenerator


# =============================================================================
# Convenience API - BETA
# =============================================================================

try:
    from .utils import SimpleMAIFAgent, create_agent
    CONVENIENCE_API_AVAILABLE = True
except ImportError:
    SimpleMAIFAgent = None
    create_agent = None

try:
    from .utils import VectorDBMigrator, migrate_to_maif
    MIGRATION_TOOLS_AVAILABLE = True
except ImportError:
    VectorDBMigrator = None
    migrate_to_maif = None

try:
    from .utils import MAIFDebugger, debug_maif
    DEBUG_TOOLS_AVAILABLE = True
except ImportError:
    MAIFDebugger = None
    debug_maif = None


# =============================================================================
# Module Info
# =============================================================================

__version__ = "1.0.2"
__author__ = "MAIF Development Team"
__license__ = "MIT"


# =============================================================================
# Utility functions for checking feature availability
# =============================================================================

def check_feature(feature_name: str) -> bool:
    """
    Check if a feature is available.

    Args:
        feature_name: Name of the feature to check (e.g., 'security', 'semantic')

    Returns:
        bool: True if the feature is available

    Example:
        if maif.check_feature('security'):
            signer = maif.MAIFSigner(...)
    """
    feature_flags = {
        'core': CORE_AVAILABLE,
        'security': SECURITY_AVAILABLE,
        'ca': CA_AVAILABLE,
        'certificate_authority': CA_AVAILABLE,
        'privacy': PRIVACY_AVAILABLE,
        'semantic': SEMANTIC_AVAILABLE,
        'enhanced_algorithms': ENHANCED_ALGORITHMS_AVAILABLE,
        'streaming': STREAMING_AVAILABLE,
        'agent': AGENT_FRAMEWORK_AVAILABLE,
        'agents': AGENT_FRAMEWORK_AVAILABLE,
        'compression': COMPRESSION_AVAILABLE,
        'forensics': FORENSICS_AVAILABLE,
        'health_check': HEALTH_CHECK_AVAILABLE,
        'rate_limiter': RATE_LIMITER_AVAILABLE,
        'metrics': METRICS_AVAILABLE,
        'cost_tracker': COST_TRACKER_AVAILABLE,
        'batch_processor': BATCH_PROCESSOR_AVAILABLE,
        'integrations': INTEGRATIONS_AVAILABLE,
        'convenience_api': CONVENIENCE_API_AVAILABLE,
        'migration_tools': MIGRATION_TOOLS_AVAILABLE,
        'debug_tools': DEBUG_TOOLS_AVAILABLE,
        'aws': AWS_IMPORTS_AVAILABLE,
    }
    return feature_flags.get(feature_name.lower(), False)


def get_available_features() -> dict:
    """
    Get a dictionary of all features and their availability status.

    Returns:
        dict: Feature names mapped to (available: bool, status: str)

    Example:
        features = maif.get_available_features()
        for name, (available, status) in features.items():
            print(f"{name}: {'OK' if available else 'Missing'} ({status})")
    """
    return {
        'core': (CORE_AVAILABLE, 'STABLE'),
        'security': (SECURITY_AVAILABLE, 'STABLE'),
        'certificate_authority': (CA_AVAILABLE, 'BETA'),
        'privacy': (PRIVACY_AVAILABLE, 'STABLE'),
        'semantic': (SEMANTIC_AVAILABLE, 'EXPERIMENTAL'),
        'enhanced_algorithms': (ENHANCED_ALGORITHMS_AVAILABLE, 'EXPERIMENTAL'),
        'streaming': (STREAMING_AVAILABLE, 'STABLE'),
        'agent_framework': (AGENT_FRAMEWORK_AVAILABLE, 'EXPERIMENTAL'),
        'compression': (COMPRESSION_AVAILABLE, 'STABLE'),
        'forensics': (FORENSICS_AVAILABLE, 'BETA'),
        'health_check': (HEALTH_CHECK_AVAILABLE, 'BETA'),
        'rate_limiter': (RATE_LIMITER_AVAILABLE, 'BETA'),
        'metrics': (METRICS_AVAILABLE, 'BETA'),
        'cost_tracker': (COST_TRACKER_AVAILABLE, 'EXPERIMENTAL'),
        'batch_processor': (BATCH_PROCESSOR_AVAILABLE, 'BETA'),
        'integrations': (INTEGRATIONS_AVAILABLE, 'BETA'),
        'convenience_api': (CONVENIENCE_API_AVAILABLE, 'BETA'),
        'migration_tools': (MIGRATION_TOOLS_AVAILABLE, 'BETA'),
        'debug_tools': (DEBUG_TOOLS_AVAILABLE, 'BETA'),
        'aws': (AWS_IMPORTS_AVAILABLE, 'PLANNED'),
    }


# =============================================================================
# __all__ - Only export what actually works
# =============================================================================

# Core exports (always available) - STABLE
__all__ = [
    # Core API - STABLE
    "MAIFEncoder",
    "MAIFDecoder",
    "MAIFParser",
    "BlockType",
    "MAIFBlock",
    "MAIFVersion",
    "MAIFHeader",
    # Secure format - STABLE
    "SecureBlock",
    "SecureBlockHeader",
    "SecureFileHeader",
    "ProvenanceEntry",
    "FileFooter",
    "BlockFlags",
    "FileFlags",
    # Legacy aliases - STABLE
    "SecureMAIFWriter",
    "SecureMAIFReader",
    "SecureBlockType",
    # Convenience functions - STABLE
    "create_maif",
    "verify_maif",
    "quick_create",
    "quick_verify",
    "quick_read",
    # Constants - STABLE
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
    # Feature flags - Always available
    "CORE_AVAILABLE",
    "SECURITY_AVAILABLE",
    "CA_AVAILABLE",
    "PRIVACY_AVAILABLE",
    "SEMANTIC_AVAILABLE",
    "ENHANCED_ALGORITHMS_AVAILABLE",
    "STREAMING_AVAILABLE",
    "AGENT_FRAMEWORK_AVAILABLE",
    "COMPRESSION_AVAILABLE",
    "FORENSICS_AVAILABLE",
    "HEALTH_CHECK_AVAILABLE",
    "RATE_LIMITER_AVAILABLE",
    "METRICS_AVAILABLE",
    "COST_TRACKER_AVAILABLE",
    "BATCH_PROCESSOR_AVAILABLE",
    "INTEGRATIONS_AVAILABLE",
    "CONVENIENCE_API_AVAILABLE",
    "MIGRATION_TOOLS_AVAILABLE",
    "DEBUG_TOOLS_AVAILABLE",
    "AWS_IMPORTS_AVAILABLE",
    # Utility functions
    "check_feature",
    "get_available_features",
]

# Conditionally add exports based on availability
# Security - STABLE (when available)
if SECURITY_AVAILABLE:
    __all__.extend([
        "MAIFSigner",
        "MAIFVerifier",
        "SecurityManager",
    ])

# Certificate Authority - BETA (when available)
if CA_AVAILABLE:
    __all__.extend([
        "MAIFRootCA",
        "AgentIdentity",
        "MAIFCertificate",
        "CertificateVerifier",
    ])

# Privacy - STABLE (when available)
if PRIVACY_AVAILABLE:
    __all__.extend([
        "PrivacyEngine",
        "PrivacyPolicy",
        "PrivacyLevel",
        "EncryptionMode",
        "AccessRule",
    ])

# Semantic - EXPERIMENTAL (when available)
if SEMANTIC_AVAILABLE:
    __all__.extend([
        "SemanticEmbedder",
        "SemanticEmbedding",
        "KnowledgeTriple",
        "KnowledgeGraphBuilder",
        "TFIDFEmbedder",
        "get_embedder",
    ])

# Streaming - STABLE (when available)
if STREAMING_AVAILABLE:
    __all__.extend([
        "MAIFStreamReader",
        "MAIFStreamWriter",
    ])

# Compression - STABLE (when available)
if COMPRESSION_AVAILABLE:
    __all__.extend([
        "CompressionManager",
        "CompressionMetadata",
    ])

# Forensics - BETA (when available)
if FORENSICS_AVAILABLE:
    __all__.extend([
        "ForensicAnalyzer",
        "ForensicEvidence",
    ])

# Validator (always try to include)
if MAIFValidator is not None:
    __all__.append("MAIFValidator")

# Agent - EXPERIMENTAL (when available)
if AGENT_FRAMEWORK_AVAILABLE:
    __all__.append("MAIFAgent")

# Production utilities - BETA (when available)
if HEALTH_CHECK_AVAILABLE:
    __all__.extend(["HealthChecker", "HealthStatus"])

if RATE_LIMITER_AVAILABLE:
    __all__.append("RateLimiter")

if METRICS_AVAILABLE:
    __all__.append("MetricsAggregator")

if COST_TRACKER_AVAILABLE:
    __all__.append("CostTracker")

if BATCH_PROCESSOR_AVAILABLE:
    __all__.append("BatchProcessor")

# Framework integrations - BETA (when available)
if INTEGRATIONS_AVAILABLE:
    __all__.extend([
        "EventType",
        "MAIFProvenanceTracker",
        "BaseMAIFCallback",
    ])
