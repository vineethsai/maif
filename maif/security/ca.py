"""
MAIF Certificate Authority
==========================

Built-in Certificate Authority for MAIF that provides:
- Root CA creation and management
- Agent certificate issuance (binding agent-did to public keys)
- Certificate revocation (CRL)
- Trust anchor distribution
- Certificate chain validation

This solves the provenance problem by cryptographically binding
agent identities to their signing keys through X.509 certificates.

Usage:
    # Create CA (one-time setup)
    ca = MAIFRootCA()
    ca.create_root(validity_years=10)

    # Issue agent certificate
    agent = AgentIdentity("did:maif:my-agent")
    agent.generate_keypair()
    cert = ca.issue_agent_certificate(agent.agent_did, agent.public_key_bytes)

    # Verify certificate chain
    valid, details = ca.verify_certificate(cert.certificate_pem)
"""

import os
import json
import time
import hashlib
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


@dataclass
class MAIFCertificate:
    """Represents a MAIF certificate (agent or CA)."""

    certificate_pem: str
    agent_did: str
    serial_number: str
    fingerprint_sha256: str
    subject: str
    issuer: str
    not_valid_before: datetime
    not_valid_after: datetime
    is_ca: bool = False
    private_key_pem: Optional[str] = None  # Only present for own certs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes private key)."""
        return {
            "certificate_pem": self.certificate_pem,
            "agent_did": self.agent_did,
            "serial_number": self.serial_number,
            "fingerprint_sha256": self.fingerprint_sha256,
            "subject": self.subject,
            "issuer": self.issuer,
            "not_valid_before": self.not_valid_before.isoformat(),
            "not_valid_after": self.not_valid_after.isoformat(),
            "is_ca": self.is_ca,
        }

    @classmethod
    def from_pem(cls, cert_pem: str, private_key_pem: Optional[str] = None) -> "MAIFCertificate":
        """Create MAIFCertificate from PEM string."""
        cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())

        # Extract agent_did from Subject Alternative Name or CN
        agent_did = None
        try:
            san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            for name in san.value:
                if isinstance(name, x509.UniformResourceIdentifier):
                    if name.value.startswith("did:maif:"):
                        agent_did = name.value
                        break
        except x509.ExtensionNotFound:
            pass

        if not agent_did:
            # Fall back to CN
            for attr in cert.subject:
                if attr.oid == NameOID.COMMON_NAME:
                    cn = attr.value
                    if cn.startswith("did:maif:"):
                        agent_did = cn
                    else:
                        agent_did = f"did:maif:{cn}"
                    break

        # Check if CA
        is_ca = False
        try:
            bc = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
            is_ca = bc.value.ca
        except x509.ExtensionNotFound:
            pass

        return cls(
            certificate_pem=cert_pem,
            agent_did=agent_did or "unknown",
            serial_number=str(cert.serial_number),
            fingerprint_sha256=cert.fingerprint(hashes.SHA256()).hex(),
            subject=cert.subject.rfc4514_string(),
            issuer=cert.issuer.rfc4514_string(),
            not_valid_before=cert.not_valid_before_utc,
            not_valid_after=cert.not_valid_after_utc,
            is_ca=is_ca,
            private_key_pem=private_key_pem,
        )


@dataclass
class RevocationEntry:
    """A single revocation entry."""
    serial_number: str
    revocation_date: datetime
    reason: str


class RevocationList:
    """Certificate Revocation List management."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.revocations: Dict[str, RevocationEntry] = {}
        self.storage_path = storage_path
        self._lock = threading.RLock()

        if storage_path and storage_path.exists():
            self._load()

    def add_revocation(
        self,
        serial_number: str,
        reason: str = "unspecified"
    ) -> None:
        """Add certificate to revocation list."""
        with self._lock:
            self.revocations[serial_number] = RevocationEntry(
                serial_number=serial_number,
                revocation_date=datetime.utcnow(),
                reason=reason,
            )
            self._save()
        logger.info(f"Revoked certificate: {serial_number}, reason: {reason}")

    def is_revoked(self, serial_number: str) -> Tuple[bool, Optional[RevocationEntry]]:
        """Check if certificate is revoked."""
        with self._lock:
            entry = self.revocations.get(serial_number)
            return (entry is not None, entry)

    def _save(self):
        """Save revocation list to file."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            sn: {
                "serial_number": e.serial_number,
                "revocation_date": e.revocation_date.isoformat(),
                "reason": e.reason,
            }
            for sn, e in self.revocations.items()
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load revocation list from file."""
        try:
            data = json.loads(self.storage_path.read_text())
            for sn, entry in data.items():
                self.revocations[sn] = RevocationEntry(
                    serial_number=entry["serial_number"],
                    revocation_date=datetime.fromisoformat(entry["revocation_date"]),
                    reason=entry["reason"],
                )
        except Exception as e:
            logger.warning(f"Failed to load revocation list: {e}")


class MAIFRootCA:
    """
    MAIF Root Certificate Authority.

    Creates and manages a root CA that can issue certificates
    binding agent-did identities to Ed25519 public keys.
    """

    def __init__(self, ca_dir: str = None):
        """
        Initialize or load existing root CA.

        Args:
            ca_dir: Directory for CA storage. Defaults to ~/.maif/ca
        """
        if ca_dir is None:
            ca_dir = os.path.expanduser("~/.maif/ca")

        self.ca_dir = Path(ca_dir)
        self.root_dir = self.ca_dir / "root"
        self.issued_dir = self.ca_dir / "issued"
        self.crl_dir = self.ca_dir / "crl"

        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._certificate: Optional[MAIFCertificate] = None
        self._serial_counter = 1
        self._lock = threading.RLock()

        # Initialize revocation list
        self.revocation_list = RevocationList(self.crl_dir / "revoked.json")

        # Try to load existing CA
        if (self.root_dir / "certificate.pem").exists():
            self._load_root()

    @property
    def is_initialized(self) -> bool:
        """Check if root CA is initialized."""
        return self._certificate is not None

    def create_root(
        self,
        validity_years: int = 10,
        organization: str = "MAIF",
        common_name: str = "MAIF Root CA",
    ) -> MAIFCertificate:
        """
        Create new root CA certificate.

        Args:
            validity_years: Certificate validity period
            organization: Organization name for certificate
            common_name: Common name for certificate

        Returns:
            The root CA certificate
        """
        if self._certificate is not None:
            raise ValueError("Root CA already exists. Use load_root() or delete existing CA.")

        # Generate Ed25519 key pair
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = self._private_key.public_key()

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])

        now = datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_years * 365))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=1),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.UniformResourceIdentifier("did:maif:root-ca"),
                ]),
                critical=False,
            )
            .sign(self._private_key, None)  # Ed25519 doesn't use hash algorithm
        )

        # Save to disk
        self._save_root(cert)

        # Create MAIFCertificate wrapper
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        private_key_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        self._certificate = MAIFCertificate.from_pem(cert_pem, private_key_pem)

        logger.info(f"Created root CA: {self._certificate.fingerprint_sha256}")
        return self._certificate

    def _save_root(self, cert: x509.Certificate):
        """Save root CA to disk."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.issued_dir.mkdir(parents=True, exist_ok=True)
        self.crl_dir.mkdir(parents=True, exist_ok=True)

        # Save certificate
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        (self.root_dir / "certificate.pem").write_bytes(cert_pem)

        # Save private key
        key_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_path = self.root_dir / "private_key.pem"
        key_path.write_bytes(key_pem)
        os.chmod(key_path, 0o600)  # Restrict access

        # Save serial counter
        (self.root_dir / "serial").write_text(str(self._serial_counter))

    def _load_root(self):
        """Load existing root CA from disk."""
        try:
            cert_pem = (self.root_dir / "certificate.pem").read_text()
            private_key_pem = (self.root_dir / "private_key.pem").read_text()

            self._private_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=None,
                backend=default_backend(),
            )

            self._certificate = MAIFCertificate.from_pem(cert_pem, private_key_pem)

            # Load serial counter
            serial_file = self.root_dir / "serial"
            if serial_file.exists():
                self._serial_counter = int(serial_file.read_text())

            logger.info(f"Loaded root CA: {self._certificate.fingerprint_sha256}")
        except Exception as e:
            logger.error(f"Failed to load root CA: {e}")
            raise

    def load_root(self) -> MAIFCertificate:
        """Load existing root CA."""
        if self._certificate is None:
            self._load_root()
        return self._certificate

    def export_trust_anchor(self) -> str:
        """
        Export root certificate PEM for distribution.

        This is what agents use to verify certificate chains.
        """
        if self._certificate is None:
            raise ValueError("Root CA not initialized")
        return self._certificate.certificate_pem

    def issue_agent_certificate(
        self,
        agent_did: str,
        public_key: bytes,
        validity_days: int = 365,
        metadata: Optional[Dict] = None,
    ) -> MAIFCertificate:
        """
        Issue certificate binding agent_did to public key.

        Args:
            agent_did: Agent's DID (e.g., "did:maif:my-agent")
            public_key: Agent's Ed25519 public key (32 bytes)
            validity_days: Certificate validity period
            metadata: Optional metadata to include

        Returns:
            Issued certificate
        """
        if self._certificate is None or self._private_key is None:
            raise ValueError("Root CA not initialized")

        if not agent_did.startswith("did:maif:"):
            agent_did = f"did:maif:{agent_did}"

        # Load agent's public key
        agent_public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)

        # Get next serial number
        with self._lock:
            serial = self._serial_counter
            self._serial_counter += 1
            (self.root_dir / "serial").write_text(str(self._serial_counter))

        # Extract short ID for CN (X.509 CN has 64 char limit)
        # Full DID is stored in SubjectAlternativeName extension
        if len(agent_did) <= 64:
            cn_value = agent_did
        else:
            # Truncate did:key - keep prefix and last 48 chars of the key
            # e.g., "did:key:z6Mk...last48chars"
            cn_value = agent_did[:15] + "..." + agent_did[-46:]

        # Build certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MAIF Agent"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn_value),
        ])

        # Load CA certificate for issuer name
        ca_cert = x509.load_pem_x509_certificate(
            self._certificate.certificate_pem.encode(),
            default_backend()
        )

        now = datetime.utcnow()
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(agent_public_key)
            .serial_number(serial)
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.UniformResourceIdentifier(agent_did),
                ]),
                critical=False,
            )
        )

        # Sign with CA private key
        cert = builder.sign(self._private_key, None)
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()

        # Save issued certificate
        cert_file = self.issued_dir / f"{serial}.pem"
        cert_file.write_text(cert_pem)

        # Save metadata
        if metadata:
            meta_file = self.issued_dir / f"{serial}.json"
            meta_file.write_text(json.dumps({
                "agent_did": agent_did,
                "issued_at": now.isoformat(),
                "valid_until": (now + timedelta(days=validity_days)).isoformat(),
                "metadata": metadata,
            }, indent=2))

        result = MAIFCertificate.from_pem(cert_pem)
        logger.info(f"Issued certificate for {agent_did}: serial={serial}")
        return result

    def revoke_certificate(
        self,
        serial_number: str,
        reason: str = "unspecified"
    ) -> bool:
        """
        Revoke a certificate.

        Args:
            serial_number: Certificate serial number
            reason: Revocation reason

        Returns:
            True if revoked successfully
        """
        self.revocation_list.add_revocation(serial_number, reason)
        return True

    def is_revoked(self, serial_number: str) -> Tuple[bool, Optional[RevocationEntry]]:
        """Check if certificate is revoked."""
        return self.revocation_list.is_revoked(serial_number)

    def verify_certificate(
        self,
        cert_pem: str,
        check_revocation: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a certificate was issued by this CA.

        Args:
            cert_pem: Certificate to verify (PEM format)
            check_revocation: Whether to check revocation status

        Returns:
            (is_valid, details_dict)
        """
        result = {
            "valid": False,
            "agent_did": None,
            "serial_number": None,
            "errors": [],
            "warnings": [],
        }

        if self._certificate is None:
            result["errors"].append("Root CA not initialized")
            return False, result

        try:
            # Load certificate
            cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())
            maif_cert = MAIFCertificate.from_pem(cert_pem)

            result["agent_did"] = maif_cert.agent_did
            result["serial_number"] = maif_cert.serial_number

            # Check validity period (use timezone-aware comparison)
            from datetime import timezone
            now = datetime.now(timezone.utc)
            if now < cert.not_valid_before_utc:
                result["errors"].append("Certificate not yet valid")
                return False, result
            if now > cert.not_valid_after_utc:
                result["errors"].append("Certificate expired")
                return False, result

            # Verify issuer matches our CA
            ca_cert = x509.load_pem_x509_certificate(
                self._certificate.certificate_pem.encode(),
                default_backend()
            )
            if cert.issuer != ca_cert.subject:
                result["errors"].append("Certificate not issued by this CA")
                return False, result

            # Verify signature
            try:
                ca_public_key = ca_cert.public_key()
                ca_public_key.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                )
            except InvalidSignature:
                result["errors"].append("Invalid signature")
                return False, result

            # Check revocation
            if check_revocation:
                is_revoked, revocation_entry = self.is_revoked(maif_cert.serial_number)
                if is_revoked:
                    result["errors"].append(
                        f"Certificate revoked: {revocation_entry.reason}"
                    )
                    return False, result

            result["valid"] = True
            return True, result

        except Exception as e:
            result["errors"].append(f"Verification error: {e}")
            return False, result


class AgentIdentity:
    """
    Manages an agent's cryptographic identity.

    Handles keypair generation, certificate requests, and signing.

    By default, uses did:key format where the DID is derived from the public key,
    making identity unforgeable without the private key.

    Usage:
        # With did:key (recommended - self-certifying)
        agent = AgentIdentity("my-agent")  # or use_did_key=True
        agent.generate_keypair()
        # agent.agent_did is now "did:key:z6Mk..." derived from public key

        # With did:maif (legacy - requires CA for trust)
        agent = AgentIdentity("my-agent", use_did_key=False)
        agent.generate_keypair()
        # agent.agent_did is "did:maif:my-agent"
    """

    def __init__(
        self,
        agent_id: str,
        identity_dir: Optional[str] = None,
        use_did_key: bool = True,
    ):
        """
        Initialize agent identity.

        Args:
            agent_id: Agent identifier (human-readable name)
            identity_dir: Directory for identity storage
            use_did_key: If True (default), derive DID from public key (did:key:).
                         If False, use legacy did:maif: format.
        """
        # Store the human-readable agent ID
        self.agent_id = agent_id.replace("did:maif:", "").replace("did:key:", "")
        self._use_did_key = use_did_key

        # DID will be set after keypair generation if using did:key
        if use_did_key:
            self._agent_did = None  # Will be derived from public key
        else:
            self._agent_did = f"did:maif:{self.agent_id}"

        if identity_dir is None:
            identity_dir = os.path.expanduser(f"~/.maif/identity/{self.agent_id}")
        self.identity_dir = Path(identity_dir)

        self._private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self._public_key: Optional[ed25519.Ed25519PublicKey] = None
        self._certificate: Optional[MAIFCertificate] = None
        self._ca_certificate: Optional[str] = None

        # Try to load existing identity
        if (self.identity_dir / "private_key.pem").exists():
            self._load()

    @property
    def agent_did(self) -> str:
        """
        Get the agent's DID.

        For did:key mode, this is derived from the public key.
        For legacy mode, this is did:maif:{agent_id}.
        """
        if self._use_did_key:
            if self._public_key is None:
                raise ValueError("Keypair not generated - cannot derive did:key")
            try:
                from .did_key import public_key_to_did_key
            except ImportError:
                from maif.security.did_key import public_key_to_did_key
            return public_key_to_did_key(self.public_key_bytes)
        else:
            return self._agent_did

    @property
    def public_key_bytes(self) -> bytes:
        """Get raw public key bytes (32 bytes)."""
        if self._public_key is None:
            raise ValueError("Keypair not generated")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_hex(self) -> str:
        """Get public key as hex string."""
        return self.public_key_bytes.hex()

    @property
    def certificate(self) -> Optional[MAIFCertificate]:
        """Get agent's certificate if issued."""
        return self._certificate

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Ed25519 keypair.

        For did:key mode, the agent_did is derived from the public key
        after this method is called.

        Returns:
            (private_key_bytes, public_key_bytes)
        """
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()

        self._save_keypair()

        private_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Now agent_did property will return the correct did:key
        logger.info(f"Generated keypair for {self.agent_id} -> {self.agent_did}")
        return private_bytes, self.public_key_bytes

    def _save_keypair(self):
        """Save keypair to disk."""
        self.identity_dir.mkdir(parents=True, exist_ok=True)

        key_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        key_path = self.identity_dir / "private_key.pem"
        key_path.write_bytes(key_pem)
        os.chmod(key_path, 0o600)

    def _load(self):
        """Load existing identity from disk."""
        try:
            key_pem = (self.identity_dir / "private_key.pem").read_bytes()
            self._private_key = serialization.load_pem_private_key(
                key_pem,
                password=None,
                backend=default_backend(),
            )
            self._public_key = self._private_key.public_key()

            # Load certificate if exists
            cert_path = self.identity_dir / "certificate.pem"
            if cert_path.exists():
                cert_pem = cert_path.read_text()
                self._certificate = MAIFCertificate.from_pem(cert_pem)

            # Load CA certificate if exists
            ca_cert_path = self.identity_dir / "ca_certificate.pem"
            if ca_cert_path.exists():
                self._ca_certificate = ca_cert_path.read_text()

            logger.info(f"Loaded identity for {self.agent_did}")
        except Exception as e:
            logger.error(f"Failed to load identity: {e}")
            raise

    def request_certificate(self, ca: MAIFRootCA) -> MAIFCertificate:
        """
        Request certificate from CA.

        Args:
            ca: The Certificate Authority to request from

        Returns:
            Issued certificate
        """
        if self._public_key is None:
            self.generate_keypair()

        # Request certificate from CA
        cert = ca.issue_agent_certificate(
            agent_did=self.agent_did,
            public_key=self.public_key_bytes,
        )

        self._certificate = cert
        self._ca_certificate = ca.export_trust_anchor()

        # Save certificate
        (self.identity_dir / "certificate.pem").write_text(cert.certificate_pem)
        (self.identity_dir / "ca_certificate.pem").write_text(self._ca_certificate)

        logger.info(f"Obtained certificate for {self.agent_did}")
        return cert

    def save_certificate(self, cert: MAIFCertificate, ca_cert_pem: Optional[str] = None):
        """Save an externally obtained certificate."""
        self._certificate = cert
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        (self.identity_dir / "certificate.pem").write_text(cert.certificate_pem)

        if ca_cert_pem:
            self._ca_certificate = ca_cert_pem
            (self.identity_dir / "ca_certificate.pem").write_text(ca_cert_pem)

    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data with private key.

        Args:
            data: Data to sign

        Returns:
            64-byte Ed25519 signature
        """
        if self._private_key is None:
            raise ValueError("No private key available")
        return self._private_key.sign(data)

    def sign_data_hex(self, data: bytes) -> str:
        """Sign data and return hex-encoded signature."""
        return self.sign_data(data).hex()

    def get_certificate_chain(self) -> List[str]:
        """
        Get certificate chain (agent cert + CA cert).

        Returns:
            List of PEM certificates, from end entity to root
        """
        chain = []
        if self._certificate:
            chain.append(self._certificate.certificate_pem)
        if self._ca_certificate:
            chain.append(self._ca_certificate)
        return chain

    def get_security_section(self, include_certificate: bool = True) -> Dict[str, Any]:
        """
        Get security section for MAIF manifest.

        For did:key mode, the DID itself proves key ownership - no CA needed.
        For did:maif mode, certificate chain is required.

        Args:
            include_certificate: Include certificate chain if available

        Returns:
            Security section dictionary
        """
        section = {
            "version": "2.0",
            "signature_algorithm": "Ed25519",
            "agent_did": self.agent_did,
            "public_key": self.public_key_hex,
        }

        # Add did:key specific info
        if self._use_did_key:
            section["did_method"] = "key"
            # For did:key, the public key can be extracted from the DID itself
            # No certificate required - the DID is self-certifying

        # Add certificate chain if available and requested
        if include_certificate and self._certificate is not None:
            section["signer_certificate"] = self._certificate.certificate_pem
            section["certificate_chain"] = self.get_certificate_chain()
            section["certificate_fingerprint"] = self._certificate.fingerprint_sha256

        return section


class CertificateVerifier:
    """
    Verifies MAIF certificates against trust anchors.

    Used by MAIFVerifier to validate certificate-based signatures.
    """

    def __init__(self, trust_anchors: List[str] = None):
        """
        Initialize verifier with trust anchors.

        Args:
            trust_anchors: List of trusted CA certificate PEMs
        """
        self.trust_anchors: Dict[str, x509.Certificate] = {}
        self._lock = threading.RLock()

        if trust_anchors:
            for anchor in trust_anchors:
                self.add_trust_anchor(anchor)

    def add_trust_anchor(self, cert_pem: str) -> str:
        """
        Add a trust anchor.

        Returns:
            Fingerprint of added certificate
        """
        cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())
        fingerprint = cert.fingerprint(hashes.SHA256()).hex()

        with self._lock:
            self.trust_anchors[fingerprint] = cert

        logger.info(f"Added trust anchor: {fingerprint}")
        return fingerprint

    def verify_certificate_chain(
        self,
        end_cert_pem: str,
        chain: List[str] = None,
        check_validity: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify certificate chain to a trust anchor.

        Args:
            end_cert_pem: End entity certificate
            chain: Intermediate certificates
            check_validity: Check validity periods

        Returns:
            (is_valid, details)
        """
        result = {
            "valid": False,
            "agent_did": None,
            "issuer": None,
            "errors": [],
        }

        try:
            # Load end certificate
            end_cert = x509.load_pem_x509_certificate(
                end_cert_pem.encode(),
                default_backend()
            )
            maif_cert = MAIFCertificate.from_pem(end_cert_pem)
            result["agent_did"] = maif_cert.agent_did

            # Check validity (use timezone-aware comparison)
            if check_validity:
                from datetime import timezone
                now = datetime.now(timezone.utc)
                if now < end_cert.not_valid_before_utc:
                    result["errors"].append("Certificate not yet valid")
                    return False, result
                if now > end_cert.not_valid_after_utc:
                    result["errors"].append("Certificate expired")
                    return False, result

            # Find issuer in chain or trust anchors
            issuer_cert = None

            # Check chain
            if chain:
                for cert_pem in chain:
                    cert = x509.load_pem_x509_certificate(
                        cert_pem.encode(),
                        default_backend()
                    )
                    if cert.subject == end_cert.issuer:
                        issuer_cert = cert
                        break

            # Check trust anchors
            if issuer_cert is None:
                for fingerprint, cert in self.trust_anchors.items():
                    if cert.subject == end_cert.issuer:
                        issuer_cert = cert
                        result["issuer"] = fingerprint
                        break

            if issuer_cert is None:
                result["errors"].append("Issuer not found in chain or trust anchors")
                return False, result

            # Verify signature
            try:
                issuer_public_key = issuer_cert.public_key()
                issuer_public_key.verify(
                    end_cert.signature,
                    end_cert.tbs_certificate_bytes,
                )
            except InvalidSignature:
                result["errors"].append("Invalid certificate signature")
                return False, result

            result["valid"] = True
            return True, result

        except Exception as e:
            result["errors"].append(f"Verification error: {e}")
            return False, result

    def extract_public_key(self, cert_pem: str) -> bytes:
        """Extract public key from certificate."""
        cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())
        public_key = cert.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


__all__ = [
    "MAIFCertificate",
    "MAIFRootCA",
    "AgentIdentity",
    "RevocationList",
    "RevocationEntry",
    "CertificateVerifier",
]
