"""
DID Key Method Implementation for MAIF
======================================

Implements the did:key method (W3C DID specification) for self-certifying
decentralized identifiers.

With did:key, the identifier IS derived from the public key, making it
impossible to claim an identity without possessing the private key.

Format:
    did:key:<multibase-encoded-public-key>

    For Ed25519: did:key:z6Mk...
    - 'z' = base58-btc multibase prefix
    - Multicodec prefix 0xed01 = Ed25519 public key
    - Followed by 32-byte raw public key

Example:
    Public key (hex): 8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c
    did:key: did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK

Usage:
    from maif.security.did_key import public_key_to_did_key, did_key_to_public_key

    # Create did:key from public key
    did = public_key_to_did_key(public_key_bytes)
    # -> "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"

    # Extract public key from did:key
    public_key = did_key_to_public_key(did)
    # -> bytes (32 bytes Ed25519 public key)

Reference:
    https://w3c-ccg.github.io/did-method-key/
"""

from typing import Tuple, Optional
import re

# Multicodec prefixes
# https://github.com/multiformats/multicodec/blob/master/table.csv
MULTICODEC_ED25519_PUB = bytes([0xed, 0x01])  # ed25519-pub
MULTICODEC_SECP256K1_PUB = bytes([0xe7, 0x01])  # secp256k1-pub

# Base58-btc alphabet (Bitcoin alphabet)
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    """Encode bytes to base58-btc string."""
    # Count leading zeros
    leading_zeros = 0
    for byte in data:
        if byte == 0:
            leading_zeros += 1
        else:
            break

    # Convert to integer
    num = int.from_bytes(data, 'big')

    # Encode
    result = []
    while num > 0:
        num, remainder = divmod(num, 58)
        result.append(BASE58_ALPHABET[remainder])

    # Add leading '1's for leading zero bytes
    result.extend(['1'] * leading_zeros)

    return ''.join(reversed(result))


def _base58_decode(s: str) -> bytes:
    """Decode base58-btc string to bytes."""
    # Count leading '1's (representing zero bytes)
    leading_ones = 0
    for char in s:
        if char == '1':
            leading_ones += 1
        else:
            break

    # Convert to integer
    num = 0
    for char in s:
        num = num * 58 + BASE58_ALPHABET.index(char)

    # Convert to bytes
    if num == 0:
        result = b''
    else:
        result = []
        while num > 0:
            result.append(num & 0xff)
            num >>= 8
        result = bytes(reversed(result))

    # Add leading zero bytes
    return b'\x00' * leading_ones + result


def public_key_to_did_key(public_key_bytes: bytes, key_type: str = "ed25519") -> str:
    """
    Convert a public key to did:key format.

    Args:
        public_key_bytes: Raw public key bytes (32 bytes for Ed25519)
        key_type: Key type, currently only "ed25519" supported

    Returns:
        DID string in format "did:key:z6Mk..."

    Raises:
        ValueError: If key type not supported or invalid key length
    """
    if key_type.lower() != "ed25519":
        raise ValueError(f"Unsupported key type: {key_type}. Only 'ed25519' is supported.")

    if len(public_key_bytes) != 32:
        raise ValueError(f"Ed25519 public key must be 32 bytes, got {len(public_key_bytes)}")

    # Prepend multicodec prefix
    prefixed = MULTICODEC_ED25519_PUB + public_key_bytes

    # Encode with base58-btc and add 'z' multibase prefix
    encoded = 'z' + _base58_encode(prefixed)

    return f"did:key:{encoded}"


def did_key_to_public_key(did: str) -> bytes:
    """
    Extract public key bytes from a did:key identifier.

    Args:
        did: DID string in format "did:key:z6Mk..."

    Returns:
        Raw public key bytes (32 bytes for Ed25519)

    Raises:
        ValueError: If DID format is invalid or unsupported key type
    """
    # Validate format
    if not did.startswith("did:key:"):
        raise ValueError(f"Invalid did:key format: must start with 'did:key:', got '{did[:20]}...'")

    # Extract multibase-encoded part
    multibase_value = did[8:]  # Remove "did:key:"

    if not multibase_value:
        raise ValueError("Empty did:key identifier")

    # Check multibase prefix
    multibase_prefix = multibase_value[0]
    if multibase_prefix != 'z':
        raise ValueError(f"Unsupported multibase encoding: '{multibase_prefix}'. Only 'z' (base58-btc) supported.")

    # Decode base58-btc (remove 'z' prefix first)
    try:
        decoded = _base58_decode(multibase_value[1:])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid base58 encoding: {e}")

    # Check multicodec prefix and extract key
    if decoded.startswith(MULTICODEC_ED25519_PUB):
        public_key = decoded[2:]  # Remove 2-byte prefix
        if len(public_key) != 32:
            raise ValueError(f"Invalid Ed25519 key length: expected 32 bytes, got {len(public_key)}")
        return public_key
    else:
        prefix_hex = decoded[:2].hex() if len(decoded) >= 2 else decoded.hex()
        raise ValueError(f"Unsupported multicodec prefix: 0x{prefix_hex}. Only Ed25519 (0xed01) supported.")


def is_valid_did_key(did: str) -> bool:
    """
    Check if a string is a valid did:key identifier.

    Args:
        did: String to validate

    Returns:
        True if valid did:key format
    """
    try:
        did_key_to_public_key(did)
        return True
    except ValueError:
        return False


def get_did_key_type(did: str) -> Optional[str]:
    """
    Get the key type from a did:key identifier.

    Args:
        did: DID string

    Returns:
        Key type string (e.g., "ed25519") or None if invalid
    """
    try:
        if not did.startswith("did:key:z"):
            return None

        decoded = _base58_decode(did[9:])  # Remove "did:key:z"

        if decoded.startswith(MULTICODEC_ED25519_PUB):
            return "ed25519"
        elif decoded.startswith(MULTICODEC_SECP256K1_PUB):
            return "secp256k1"
        else:
            return None
    except (ValueError, IndexError):
        return None


def verify_did_key_ownership(did: str, public_key_bytes: bytes) -> bool:
    """
    Verify that a public key corresponds to a did:key.

    This is the core of did:key verification - the DID must be
    derived from the public key.

    Args:
        did: The did:key identifier
        public_key_bytes: The claimed public key

    Returns:
        True if the public key matches the DID
    """
    try:
        extracted_key = did_key_to_public_key(did)
        return extracted_key == public_key_bytes
    except ValueError:
        return False


def create_did_key_identifier(public_key_bytes: bytes) -> dict:
    """
    Create a complete did:key identifier with verification method.

    Returns a dict suitable for inclusion in a DID Document or
    MAIF security section.

    Args:
        public_key_bytes: Ed25519 public key (32 bytes)

    Returns:
        Dict with DID and verification method info
    """
    did = public_key_to_did_key(public_key_bytes)

    return {
        "id": did,
        "type": "Ed25519VerificationKey2020",
        "controller": did,
        "publicKeyMultibase": "z" + _base58_encode(MULTICODEC_ED25519_PUB + public_key_bytes),
    }


__all__ = [
    "public_key_to_did_key",
    "did_key_to_public_key",
    "is_valid_did_key",
    "get_did_key_type",
    "verify_did_key_ownership",
    "create_did_key_identifier",
]
