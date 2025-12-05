"""
MAIF Security Module

Contains security features:
- Cryptographic signing and verification (Ed25519)
- Access control and permissions
- Zero-knowledge proofs
- PKI and certificate management

Import specific modules directly:
    from maif.security.security import MAIFSigner, MAIFVerifier
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

__all__ = [
    "MAIFSigner",
    "MAIFVerifier",
    "SecurityManager",
    "AccessController",
    "ProvenanceEntry",
    "generate_key_pair",
    "sign_data",
    "verify_signature",
    "hash_data",
]
