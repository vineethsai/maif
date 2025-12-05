"""
MAIF Privacy Module

Contains privacy features:
- Encryption (AES-GCM, ChaCha20)
- PII detection and anonymization
- Access control policies
- Differential privacy
"""

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
except (ImportError, AttributeError) as e:
    PrivacyEngine = None
    PrivacyPolicy = None
    PrivacyLevel = None
    EncryptionMode = None
    AccessRule = None
    DifferentialPrivacy = None
    SecureMultipartyComputation = None
    ZeroKnowledgeProof = None

__all__ = [
    "PrivacyEngine",
    "PrivacyPolicy",
    "PrivacyLevel",
    "EncryptionMode",
    "AccessRule",
    "DifferentialPrivacy",
    "SecureMultipartyComputation",
    "ZeroKnowledgeProof",
]
