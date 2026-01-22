"""
MAIF Privacy Module

Contains privacy features:
- Encryption (AES-GCM, ChaCha20)
- PII detection and anonymization
- Access control policies
- Differential privacy (Laplace mechanism)
- Schnorr Zero-Knowledge Proofs (real ZKP)
- Shamir Secret Sharing (real threshold MPC)
"""

try:
    from .privacy import (
        PrivacyEngine,
        PrivacyPolicy,
        PrivacyLevel,
        EncryptionMode,
        AccessRule,
        DifferentialPrivacy,
        # Real implementations
        SchnorrZKP,
        ShamirSecretSharing,
        SchnorrProof,
        # Backward compatibility aliases
        SecureMultipartyComputation,  # -> ShamirSecretSharing
        ZeroKnowledgeProof,  # -> SchnorrZKP
    )
except (ImportError, AttributeError) as e:
    PrivacyEngine = None
    PrivacyPolicy = None
    PrivacyLevel = None
    EncryptionMode = None
    AccessRule = None
    DifferentialPrivacy = None
    SchnorrZKP = None
    ShamirSecretSharing = None
    SchnorrProof = None
    SecureMultipartyComputation = None
    ZeroKnowledgeProof = None

__all__ = [
    "PrivacyEngine",
    "PrivacyPolicy",
    "PrivacyLevel",
    "EncryptionMode",
    "AccessRule",
    "DifferentialPrivacy",
    # Real implementations
    "SchnorrZKP",
    "ShamirSecretSharing",
    "SchnorrProof",
    # Backward compatibility aliases
    "SecureMultipartyComputation",
    "ZeroKnowledgeProof",
]
