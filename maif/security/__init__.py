"""
MAIF Security Module

Contains security features:
- Cryptographic signing and verification (Ed25519)
- Certificate Authority for provenance verification
- Access control and permissions
- Zero-knowledge proofs
- PKI and certificate management

Import specific modules directly:
    from maif.security.security import MAIFSigner, MAIFVerifier
    from maif.security.ca import MAIFRootCA, AgentIdentity
    from maif.security.signature_verification import SignatureVerifier
"""

# Only import core security classes that don't have circular dependencies
try:
    from .security import (
        MAIFSigner,
        MAIFVerifier,
        SecurityManager,
        AccessController,
        ProvenanceEntry,
        generate_key_pair,
        sign_data,
        verify_signature,
        hash_data,
    )
except (ImportError, AttributeError) as e:
    MAIFSigner = None
    MAIFVerifier = None
    SecurityManager = None
    AccessController = None
    ProvenanceEntry = None
    generate_key_pair = None
    sign_data = None
    verify_signature = None
    hash_data = None

# Import CA classes for provenance verification
try:
    from .ca import (
        MAIFRootCA,
        AgentIdentity,
        MAIFCertificate,
        CertificateVerifier,
        RevocationList,
    )
except (ImportError, AttributeError) as e:
    MAIFRootCA = None
    AgentIdentity = None
    MAIFCertificate = None
    CertificateVerifier = None
    RevocationList = None

# Import did:key functions for self-certifying identities
try:
    from .did_key import (
        public_key_to_did_key,
        did_key_to_public_key,
        is_valid_did_key,
        verify_did_key_ownership,
    )
except (ImportError, AttributeError) as e:
    public_key_to_did_key = None
    did_key_to_public_key = None
    is_valid_did_key = None
    verify_did_key_ownership = None

__all__ = [
    # Core signing
    "MAIFSigner",
    "MAIFVerifier",
    "SecurityManager",
    "AccessController",
    "ProvenanceEntry",
    "generate_key_pair",
    "sign_data",
    "verify_signature",
    "hash_data",
    # Certificate Authority
    "MAIFRootCA",
    "AgentIdentity",
    "MAIFCertificate",
    "CertificateVerifier",
    "RevocationList",
    # did:key (self-certifying identities)
    "public_key_to_did_key",
    "did_key_to_public_key",
    "is_valid_did_key",
    "verify_did_key_ownership",
]
