"""
Tests for MAIF Certificate Authority.

Tests cover:
- Root CA creation and loading
- Agent certificate issuance
- Certificate revocation
- Certificate chain verification
- Integration with MAIFSigner and MAIFVerifier
- Backward compatibility with legacy signing
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta

from maif.security.ca import (
    MAIFRootCA,
    AgentIdentity,
    MAIFCertificate,
    CertificateVerifier,
    RevocationList,
)
from maif.security import MAIFSigner, MAIFVerifier


class TestMAIFRootCA:
    """Tests for MAIFRootCA class."""

    @pytest.fixture
    def temp_ca_dir(self):
        """Create temporary directory for CA storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_root_ca(self, temp_ca_dir):
        """Test root CA creation."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        cert = ca.create_root(validity_years=10)

        assert cert is not None
        assert cert.is_ca is True
        assert cert.agent_did == "did:maif:root-ca"
        assert cert.certificate_pem.startswith("-----BEGIN CERTIFICATE-----")
        assert ca.is_initialized is True

    def test_load_existing_root_ca(self, temp_ca_dir):
        """Test loading existing root CA."""
        # Create CA
        ca1 = MAIFRootCA(ca_dir=temp_ca_dir)
        cert1 = ca1.create_root()

        # Load CA
        ca2 = MAIFRootCA(ca_dir=temp_ca_dir)
        cert2 = ca2.load_root()

        assert cert1.fingerprint_sha256 == cert2.fingerprint_sha256
        assert cert1.serial_number == cert2.serial_number

    def test_export_trust_anchor(self, temp_ca_dir):
        """Test exporting root cert for distribution."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        ca.create_root()

        trust_anchor = ca.export_trust_anchor()

        assert trust_anchor.startswith("-----BEGIN CERTIFICATE-----")
        assert "-----END CERTIFICATE-----" in trust_anchor

    def test_issue_agent_certificate(self, temp_ca_dir):
        """Test issuing certificate for an agent."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        ca.create_root()

        # Create agent identity and get public key
        agent = AgentIdentity("test-agent", identity_dir=f"{temp_ca_dir}/identity", use_did_key=False)
        agent.generate_keypair()

        # Issue certificate
        cert = ca.issue_agent_certificate(
            agent_did=agent.agent_did,
            public_key=agent.public_key_bytes,
            validity_days=365,
        )

        assert cert is not None
        assert cert.agent_did == "did:maif:test-agent"
        assert cert.is_ca is False
        assert cert.certificate_pem.startswith("-----BEGIN CERTIFICATE-----")

    def test_certificate_contains_agent_did(self, temp_ca_dir):
        """Test that agent_did is embedded in certificate."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        ca.create_root()

        agent = AgentIdentity("my-special-agent", identity_dir=f"{temp_ca_dir}/identity", use_did_key=False)
        agent.generate_keypair()

        cert = ca.issue_agent_certificate(
            agent_did="did:maif:my-special-agent",
            public_key=agent.public_key_bytes,
        )

        # Verify DID is in certificate
        assert "my-special-agent" in cert.subject
        assert cert.agent_did == "did:maif:my-special-agent"

    def test_revoke_certificate(self, temp_ca_dir):
        """Test certificate revocation."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        ca.create_root()

        agent = AgentIdentity("revoke-test", identity_dir=f"{temp_ca_dir}/identity", use_did_key=False)
        agent.generate_keypair()

        cert = ca.issue_agent_certificate(
            agent_did=agent.agent_did,
            public_key=agent.public_key_bytes,
        )

        # Verify not revoked
        is_revoked, _ = ca.is_revoked(cert.serial_number)
        assert is_revoked is False

        # Revoke
        ca.revoke_certificate(cert.serial_number, reason="compromised")

        # Verify revoked
        is_revoked, entry = ca.is_revoked(cert.serial_number)
        assert is_revoked is True
        assert entry.reason == "compromised"

    def test_verify_revoked_certificate_fails(self, temp_ca_dir):
        """Test that revoked certificates fail verification."""
        ca = MAIFRootCA(ca_dir=temp_ca_dir)
        ca.create_root()

        agent = AgentIdentity("verify-revoked", identity_dir=f"{temp_ca_dir}/identity", use_did_key=False)
        agent.generate_keypair()

        cert = ca.issue_agent_certificate(
            agent_did=agent.agent_did,
            public_key=agent.public_key_bytes,
        )

        # Verify works before revocation
        valid, details = ca.verify_certificate(cert.certificate_pem)
        assert valid is True

        # Revoke
        ca.revoke_certificate(cert.serial_number, reason="test")

        # Verify fails after revocation
        valid, details = ca.verify_certificate(cert.certificate_pem)
        assert valid is False
        assert any("revoked" in e.lower() for e in details["errors"])


class TestAgentIdentity:
    """Tests for AgentIdentity class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_generate_ed25519_keypair(self, temp_dir):
        """Test Ed25519 keypair generation."""
        agent = AgentIdentity("keypair-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        private_bytes, public_bytes = agent.generate_keypair()

        assert len(private_bytes) == 32  # Ed25519 private key
        assert len(public_bytes) == 32  # Ed25519 public key
        assert agent.public_key_bytes == public_bytes

    def test_request_certificate(self, temp_dir):
        """Test certificate request flow."""
        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        agent = AgentIdentity("request-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent.generate_keypair()
        cert = agent.request_certificate(ca)

        assert cert is not None
        assert agent.certificate is not None
        assert agent.certificate.agent_did == "did:maif:request-test"

    def test_sign_data_with_certificate(self, temp_dir):
        """Test signing with certificate-bound key."""
        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        agent = AgentIdentity("sign-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent.generate_keypair()
        agent.request_certificate(ca)

        # Sign some data
        data = b"Hello, certified world!"
        signature = agent.sign_data(data)

        assert len(signature) == 64  # Ed25519 signature
        assert signature != data

    def test_get_certificate_chain(self, temp_dir):
        """Test getting certificate chain."""
        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        agent = AgentIdentity("chain-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent.generate_keypair()
        agent.request_certificate(ca)

        chain = agent.get_certificate_chain()

        assert len(chain) == 2  # Agent cert + CA cert
        assert all(c.startswith("-----BEGIN CERTIFICATE-----") for c in chain)

    def test_load_existing_identity(self, temp_dir):
        """Test loading existing identity from disk."""
        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        # Create and save identity
        agent1 = AgentIdentity("persist-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent1.generate_keypair()
        agent1.request_certificate(ca)
        original_pubkey = agent1.public_key_hex

        # Load identity
        agent2 = AgentIdentity("persist-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)

        assert agent2.public_key_hex == original_pubkey
        assert agent2.certificate is not None


class TestCertificateIntegration:
    """Tests for integration with MAIFSigner and MAIFVerifier."""

    @pytest.fixture
    def ca_and_agent(self):
        """Create CA and agent with certificate."""
        temp_dir = tempfile.mkdtemp()

        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        agent = AgentIdentity("integration-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent.generate_keypair()
        agent.request_certificate(ca)

        yield ca, agent, temp_dir

        shutil.rmtree(temp_dir)

    def test_sign_maif_with_certificate(self, ca_and_agent):
        """Test MAIF signing with certificate."""
        ca, agent, _ = ca_and_agent

        signer = MAIFSigner(agent_identity=agent)
        manifest = {"blocks": [{"type": "TEXT", "data": "test"}]}
        signed = signer.sign_manifest(manifest)

        assert "security" in signed
        assert signed["security"]["version"] == "2.0"
        assert signed["security"]["agent_did"] == "did:maif:integration-test"
        assert "signer_certificate" in signed["security"]
        assert "certificate_chain" in signed["security"]
        assert "signature" in signed["security"]

    def test_verify_maif_with_trust_anchor(self, ca_and_agent):
        """Test verification with trust anchor."""
        ca, agent, _ = ca_and_agent

        # Sign
        signer = MAIFSigner(agent_identity=agent)
        manifest = {"blocks": [{"type": "TEXT", "data": "test"}]}
        signed = signer.sign_manifest(manifest)

        # Verify
        verifier = MAIFVerifier(trust_anchors=[ca.export_trust_anchor()])
        valid, details = verifier.verify_manifest_with_chain(signed)

        assert valid is True
        assert details["chain_valid"] is True
        assert details["signature_valid"] is True
        assert details["agent_did"] == "did:maif:integration-test"

    def test_verify_fails_without_trust_anchor(self, ca_and_agent):
        """Test that verification fails without trust anchor."""
        ca, agent, _ = ca_and_agent

        # Sign
        signer = MAIFSigner(agent_identity=agent)
        manifest = {"blocks": []}
        signed = signer.sign_manifest(manifest)

        # Verify without trust anchor
        verifier = MAIFVerifier()  # No trust anchors
        valid, details = verifier.verify_manifest_with_chain(signed)

        assert valid is False
        assert any("trust" in e.lower() for e in details["errors"])

    def test_agent_did_binding(self, ca_and_agent):
        """Test that signature is bound to agent_did."""
        ca, agent, _ = ca_and_agent

        signer = MAIFSigner(agent_identity=agent)
        manifest = {"data": "test"}
        signed = signer.sign_manifest(manifest)

        # Tamper with agent_did
        signed["security"]["agent_did"] = "did:maif:attacker"

        # Verify should fail
        verifier = MAIFVerifier(trust_anchors=[ca.export_trust_anchor()])
        valid, details = verifier.verify_manifest_with_chain(signed)

        assert valid is False
        assert any("mismatch" in e.lower() for e in details["errors"])

    def test_different_agent_cannot_use_cert(self, ca_and_agent):
        """Test that one agent cannot use another's certificate."""
        ca, agent1, temp_dir = ca_and_agent

        # Create second agent
        agent2 = AgentIdentity("attacker", identity_dir=f"{temp_dir}/attacker", use_did_key=False)
        agent2.generate_keypair()
        agent2.request_certificate(ca)

        # Agent2 tries to claim agent1's identity by using their cert
        signer = MAIFSigner(agent_identity=agent2)
        manifest = {"data": "malicious"}
        signed = signer.sign_manifest(manifest)

        # Verify - should show agent2's DID, not agent1's
        verifier = MAIFVerifier(trust_anchors=[ca.export_trust_anchor()])
        valid, details = verifier.verify_manifest_with_chain(signed)

        assert valid is True  # Signature is valid
        assert details["agent_did"] == "did:maif:attacker"  # But identity is agent2


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy signing."""

    def test_legacy_maif_still_verifies(self):
        """Test that existing MAIF files (without certs) still verify."""
        # Legacy signing (no certificate)
        signer = MAIFSigner(agent_id="legacy-agent")
        manifest = {"data": "legacy content"}
        signed = signer.sign_manifest(manifest)

        # Should have legacy fields
        assert "signature" in signed
        assert "public_key" in signed
        assert "signature_metadata" in signed
        assert "security" not in signed

        # Verify using legacy verification
        verifier = MAIFVerifier(public_key=signed["public_key"])
        valid, message = verifier.verify_manifest(signed)
        assert valid is True

    def test_new_verifier_handles_legacy(self):
        """Test new verifier with legacy manifests."""
        # Legacy signing
        signer = MAIFSigner(agent_id="legacy-agent")
        manifest = {"data": "legacy"}
        signed = signer.sign_manifest(manifest)

        # New verifier without trust anchors falls back to legacy
        verifier = MAIFVerifier(public_key=signed["public_key"])
        valid, details = verifier.verify_manifest_with_chain(signed)

        assert valid is True
        assert "Legacy verification" in str(details.get("warnings", []))


class TestRevocationList:
    """Tests for RevocationList class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_add_and_check_revocation(self, temp_dir):
        """Test adding and checking revocation."""
        crl = RevocationList(temp_dir / "revoked.json")

        crl.add_revocation("12345", reason="key compromise")

        is_revoked, entry = crl.is_revoked("12345")
        assert is_revoked is True
        assert entry.reason == "key compromise"

    def test_persistence(self, temp_dir):
        """Test that revocations persist to disk."""
        crl1 = RevocationList(temp_dir / "revoked.json")
        crl1.add_revocation("67890", reason="test")

        # Load new instance
        crl2 = RevocationList(temp_dir / "revoked.json")
        is_revoked, entry = crl2.is_revoked("67890")

        assert is_revoked is True
        assert entry.reason == "test"


class TestCertificateVerifier:
    """Tests for CertificateVerifier class."""

    @pytest.fixture
    def ca_and_cert(self):
        """Create CA and certificate."""
        temp_dir = tempfile.mkdtemp()

        ca = MAIFRootCA(ca_dir=f"{temp_dir}/ca")
        ca.create_root()

        agent = AgentIdentity("verifier-test", identity_dir=f"{temp_dir}/identity", use_did_key=False)
        agent.generate_keypair()
        cert = ca.issue_agent_certificate(agent.agent_did, agent.public_key_bytes)

        yield ca, cert, agent, temp_dir

        shutil.rmtree(temp_dir)

    def test_verify_valid_chain(self, ca_and_cert):
        """Test verifying a valid certificate chain."""
        ca, cert, agent, _ = ca_and_cert

        verifier = CertificateVerifier([ca.export_trust_anchor()])
        valid, details = verifier.verify_certificate_chain(
            cert.certificate_pem,
            [ca.export_trust_anchor()],
        )

        assert valid is True
        assert details["agent_did"] == "did:maif:verifier-test"

    def test_extract_public_key(self, ca_and_cert):
        """Test extracting public key from certificate."""
        ca, cert, agent, _ = ca_and_cert

        verifier = CertificateVerifier()
        public_key = verifier.extract_public_key(cert.certificate_pem)

        assert len(public_key) == 32
        assert public_key == agent.public_key_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
