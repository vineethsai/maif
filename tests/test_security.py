"""
Comprehensive tests for MAIF security functionality.
"""

import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from maif.security import MAIFSigner, MAIFVerifier, ProvenanceEntry, AccessController


class TestProvenanceEntry:
    """Test ProvenanceEntry data structure."""
    
    def test_provenance_entry_creation(self):
        """Test basic ProvenanceEntry creation."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="abc123def456",
            previous_hash="def456ghi789"
        )
        
        assert entry.timestamp == 1234567890.0
        assert entry.agent_id == "test_agent"
        assert entry.action == "create_block"
        assert entry.block_hash == "abc123def456"
        assert entry.previous_hash == "def456ghi789"
    
    def test_provenance_entry_to_dict(self):
        """Test ProvenanceEntry serialization."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="abc123def456",
            previous_hash="def456ghi789"
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["timestamp"] == 1234567890.0
        assert entry_dict["agent_id"] == "test_agent"
        assert entry_dict["action"] == "create_block"
        assert entry_dict["block_hash"] == "abc123def456"
        assert entry_dict["previous_hash"] == "def456ghi789"


class TestMAIFSigner:
    """Test MAIFSigner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signer = MAIFSigner(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_signer_initialization(self):
        """Test MAIFSigner initialization."""
        assert self.signer.agent_id == "test_agent"
        assert self.signer.private_key is not None
        # Should have genesis block
        assert len(self.signer.provenance_chain) == 1
        assert self.signer.provenance_chain[0].action == "genesis"
    
    def test_signer_with_existing_key(self):
        """Test MAIFSigner with existing private key."""
        # Generate a test key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Save key to file
        key_path = os.path.join(self.temp_dir, "test_key.pem")
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Create signer with existing key
        signer = MAIFSigner(private_key_path=key_path, agent_id="test_agent")
        
        assert signer.private_key is not None
        assert signer.agent_id == "test_agent"
    
    def test_get_public_key_pem(self):
        """Test public key extraction."""
        public_key_pem = self.signer.get_public_key_pem()
        
        assert public_key_pem is not None
        assert b"BEGIN PUBLIC KEY" in public_key_pem
        assert b"END PUBLIC KEY" in public_key_pem
    
    def test_sign_data(self):
        """Test data signing."""
        test_data = b"Hello, MAIF security!"
        signature = self.signer.sign_data(test_data)
        
        assert signature is not None
        assert isinstance(signature, str)
        assert len(signature) > 0
    
    def test_add_provenance_entry(self):
        """Test adding provenance entries."""
        entry = self.signer.add_provenance_entry(
            action="create_block",
            block_hash="abc123def456"
        )
        
        # Genesis + 1 entry
        assert len(self.signer.provenance_chain) == 2
        assert entry.agent_id == "test_agent"
        assert entry.action == "create_block"
        assert entry.block_hash == "abc123def456"
        assert entry.timestamp > 0
    
    def test_provenance_chain_linking(self):
        """Test that provenance entries are properly linked."""
        # Add first entry
        entry1 = self.signer.add_provenance_entry(
            action="create_block",
            block_hash="abc123"
        )
        
        # Add second entry
        entry2 = self.signer.add_provenance_entry(
            action="update_block",
            block_hash="def456"
        )
        
        # Genesis + 2 entries
        assert len(self.signer.provenance_chain) == 3
        assert entry2.previous_hash == entry1.block_hash
    
    def test_sign_maif_manifest(self):
        """Test MAIF manifest signing."""
        manifest = {
            "header": {
                "version": "2.0",
                "agent_id": "test_agent",
                "timestamp": time.time()
            },
            "blocks": [
                {
                    "block_id": "block_001",
                    "block_type": "text",
                    "hash": "abc123def456"
                }
            ]
        }
        
        signed_manifest = self.signer.sign_maif_manifest(manifest)
        
        assert "signature" in signed_manifest
        assert "public_key" in signed_manifest
        assert signed_manifest["signature"] is not None
        assert signed_manifest["public_key"] is not None
        
        # Original manifest should be preserved
        assert signed_manifest["header"] == manifest["header"]
        assert signed_manifest["blocks"] == manifest["blocks"]


class TestMAIFVerifier:
    """Test MAIFVerifier functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signer = MAIFSigner(agent_id="test_agent")
        self.verifier = MAIFVerifier()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_verify_signature(self):
        """Test signature verification."""
        test_data = b"Hello, MAIF security!"
        signature = self.signer.sign_data(test_data)
        public_key_pem = self.signer.get_public_key_pem().decode('utf-8')
        
        # Valid signature should verify
        result = self.verifier.verify_signature(test_data, signature, public_key_pem)
        assert result is True
        
        # Invalid signature should not verify
        invalid_signature = "invalid_signature"
        result = self.verifier.verify_signature(test_data, invalid_signature, public_key_pem)
        assert result is False
        
        # Modified data should not verify
        modified_data = b"Modified data"
        result = self.verifier.verify_signature(modified_data, signature, public_key_pem)
        assert result is False
    
    def test_verify_maif_signature(self):
        """Test MAIF manifest signature verification."""
        manifest = {
            "header": {
                "version": "2.0",
                "agent_id": "test_agent",
                "timestamp": time.time()
            },
            "blocks": [
                {
                    "block_id": "block_001",
                    "block_type": "text",
                    "hash": "abc123def456"
                }
            ]
        }
        
        signed_manifest = self.signer.sign_maif_manifest(manifest)
        
        # Valid signature should verify
        result = self.verifier.verify_maif_signature(signed_manifest)
        assert result is True
        
        # Tampered manifest should not verify
        tampered_manifest = signed_manifest.copy()
        tampered_manifest["blocks"][0]["hash"] = "tampered_hash"
        
        result = self.verifier.verify_maif_signature(tampered_manifest)
        assert result is False
    
    def test_verify_provenance_chain(self):
        """Test provenance chain verification."""
        # Create a provenance chain
        self.signer.add_provenance_entry("create_block", "hash1")
        self.signer.add_provenance_entry("update_block", "hash2")
        self.signer.add_provenance_entry("delete_block", "hash3")
        
        provenance_data = {
            "chain": [entry.to_dict() for entry in self.signer.provenance_chain],
            "agent_id": "test_agent"
        }
        
        is_valid, errors = self.verifier.verify_provenance_chain(provenance_data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_verify_broken_provenance_chain(self):
        """Test verification of broken provenance chain."""
        # Create a broken chain
        # We need a genesis block first
        genesis = ProvenanceEntry(
            timestamp=1234567889.0,
            agent_id="test_agent",
            action="genesis",
            block_hash="genesis_hash",
            previous_hash=None
        )
        
        entry1 = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="hash1",
            previous_hash="genesis_hash"
        )
        
        entry2 = ProvenanceEntry(
            timestamp=1234567891.0,
            agent_id="test_agent",
            action="update_block",
            block_hash="hash2",
            previous_hash="wrong_hash"  # Should be "hash1"
        )
        
        provenance_data = {
            "chain": [genesis.to_dict(), entry1.to_dict(), entry2.to_dict()],
            "agent_id": "test_agent"
        }
        
        is_valid, errors = self.verifier.verify_provenance_chain(provenance_data)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("link broken" in e for e in errors)
    
    def test_verify_maif_manifest_comprehensive(self):
        """Test comprehensive MAIF manifest verification."""
        # Create a complete manifest with provenance
        self.signer.add_provenance_entry("create_block", "block_hash_1")
        
        manifest = {
            "header": {
                "version": "2.0",
                "agent_id": "test_agent",
                "timestamp": time.time()
            },
            "blocks": [
                {
                    "block_id": "block_001",
                    "block_type": "text",
                    "hash": "block_hash_1"
                }
            ],
            "provenance": {
                "chain": [entry.to_dict() for entry in self.signer.provenance_chain],
                "agent_id": "test_agent"
            }
        }
        
        signed_manifest = self.signer.sign_maif_manifest(manifest)
        
        is_valid, errors = self.verifier.verify_maif_manifest(signed_manifest)
        
        assert is_valid is True
        assert len(errors) == 0


class TestAccessController:
    """Test AccessController functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.access_controller = AccessController()
    
    def test_set_block_permissions(self):
        """Test setting block permissions."""
        self.access_controller.set_block_permissions(
            block_hash="block_123",
            agent_id="agent_456",
            permissions=["read", "write"]
        )
        
        assert "block_123" in self.access_controller.permissions
        assert "agent_456" in self.access_controller.permissions["block_123"]
        assert "read" in self.access_controller.permissions["block_123"]["agent_456"]
        assert "write" in self.access_controller.permissions["block_123"]["agent_456"]
    
    def test_check_permission_granted(self):
        """Test permission checking when access is granted."""
        self.access_controller.set_block_permissions(
            block_hash="block_123",
            agent_id="agent_456",
            permissions=["read"]
        )
        
        result = self.access_controller.check_permission(
            block_hash="block_123",
            agent_id="agent_456",
            action="read"
        )
        
        assert result is True
    
    def test_check_permission_denied(self):
        """Test permission checking when access is denied."""
        self.access_controller.set_block_permissions(
            block_hash="block_123",
            agent_id="agent_456",
            permissions=["read"]
        )
        
        # Check for write permission (not granted)
        result = self.access_controller.check_permission(
            block_hash="block_123",
            agent_id="agent_456",
            action="write"
        )
        
        assert result is False
        
        # Check for different agent
        result = self.access_controller.check_permission(
            block_hash="block_123",
            agent_id="different_agent",
            action="read"
        )
        
        assert result is False
        
        # Check for non-existent block
        result = self.access_controller.check_permission(
            block_hash="non_existent_block",
            agent_id="agent_456",
            action="read"
        )
        
        assert result is False
    
    def test_get_permissions_manifest(self):
        """Test permissions manifest generation."""
        self.access_controller.set_block_permissions(
            block_hash="block_123",
            agent_id="agent_456",
            permissions=["read", "write"]
        )
        
        self.access_controller.set_block_permissions(
            block_hash="block_789",
            agent_id="agent_456",
            permissions=["read"]
        )
        
        manifest = self.access_controller.get_permissions_manifest()
        
        assert "block_123" in manifest
        assert "block_789" in manifest
        assert manifest["block_123"]["agent_456"] == ["read", "write"]
        assert manifest["block_789"]["agent_456"] == ["read"]


class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signer = MAIFSigner(agent_id="test_agent")
        self.verifier = MAIFVerifier()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow from signing to verification."""
        # 1. Create and sign data
        test_data = b"Sensitive MAIF data"
        signature = self.signer.sign_data(test_data)
        public_key_pem = self.signer.get_public_key_pem().decode('utf-8')
        
        # 2. Add provenance entries
        self.signer.add_provenance_entry("create_block", "data_hash_123")
        self.signer.add_provenance_entry("encrypt_block", "encrypted_hash_456")
        
        # 3. Create and sign manifest
        manifest = {
            "header": {
                "version": "2.0",
                "agent_id": "test_agent",
                "timestamp": time.time()
            },
            "blocks": [
                {
                    "block_id": "block_001",
                    "block_type": "text",
                    "hash": "data_hash_123",
                    "signature": signature
                }
            ],
            "provenance": {
                "chain": [entry.to_dict() for entry in self.signer.provenance_chain],
                "agent_id": "test_agent"
            }
        }
        
        signed_manifest = self.signer.sign_maif_manifest(manifest)
        
        # 4. Verify everything
        # Verify data signature
        data_valid = self.verifier.verify_signature(test_data, signature, public_key_pem)
        assert data_valid is True
        
        # Verify manifest signature
        manifest_valid = self.verifier.verify_maif_signature(signed_manifest)
        assert manifest_valid is True
        
        # Verify provenance chain
        provenance_valid, errors = self.verifier.verify_provenance_chain(
            signed_manifest["provenance"]
        )
        assert provenance_valid is True
        assert len(errors) == 0
    
    def test_tamper_detection(self):
        """Test tamper detection capabilities."""
        # Create original data and signature
        original_data = b"Original secure data"
        signature = self.signer.sign_data(original_data)
        public_key_pem = self.signer.get_public_key_pem().decode('utf-8')
        
        # Verify original data
        assert self.verifier.verify_signature(original_data, signature, public_key_pem) is True
        
        # Test various tampering scenarios
        tampered_scenarios = [
            b"Tampered secure data",  # Modified content
            b"Original secure data extra",  # Added content
            b"Original secure dat",  # Removed content
            b"",  # Empty content
        ]
        
        for tampered_data in tampered_scenarios:
            result = self.verifier.verify_signature(tampered_data, signature, public_key_pem)
            assert result is False, f"Tamper detection failed for: {tampered_data}"
    
    def test_multi_agent_security(self):
        """Test security with multiple agents."""
        # Create multiple signers
        agent1 = MAIFSigner(agent_id="agent_1")
        agent2 = MAIFSigner(agent_id="agent_2")
        
        # Each agent signs different data
        data1 = b"Data from agent 1"
        data2 = b"Data from agent 2"
        
        sig1 = agent1.sign_data(data1)
        sig2 = agent2.sign_data(data2)
        
        pub_key1 = agent1.get_public_key_pem().decode('utf-8')
        pub_key2 = agent2.get_public_key_pem().decode('utf-8')
        
        # Verify correct combinations
        assert self.verifier.verify_signature(data1, sig1, pub_key1) is True
        assert self.verifier.verify_signature(data2, sig2, pub_key2) is True
        
        # Verify incorrect combinations fail
        assert self.verifier.verify_signature(data1, sig2, pub_key2) is False
        assert self.verifier.verify_signature(data2, sig1, pub_key1) is False
        assert self.verifier.verify_signature(data1, sig1, pub_key2) is False
        assert self.verifier.verify_signature(data2, sig2, pub_key1) is False


class TestSecurityErrorHandling:
    """Test security error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.verifier = MAIFVerifier()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_private_key_file(self):
        """Test handling of invalid private key files."""
        # Create invalid key file
        invalid_key_path = os.path.join(self.temp_dir, "invalid_key.pem")
        with open(invalid_key_path, "w") as f:
            f.write("invalid key content")
        
        with pytest.raises(Exception):
            MAIFSigner(private_key_path=invalid_key_path, agent_id="test_agent")
    
    def test_nonexistent_key_file(self):
        """Test handling of non-existent key files."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pem")
        
        with pytest.raises(FileNotFoundError):
            MAIFSigner(private_key_path=nonexistent_path, agent_id="test_agent")
    
    def test_invalid_signature_format(self):
        """Test handling of invalid signature formats."""
        test_data = b"test data"
        public_key_pem = "invalid_public_key"
        
        # Various invalid signature formats
        invalid_signatures = [
            "",
            "not_base64",
            "invalid signature format",
            None
        ]
        
        for invalid_sig in invalid_signatures:
            if invalid_sig is not None:
                result = self.verifier.verify_signature(test_data, invalid_sig, public_key_pem)
                assert result is False
    
    def test_malformed_manifest(self):
        """Test handling of malformed manifests."""
        malformed_manifests = [
            {},  # Empty manifest
            {"header": {}},  # Missing blocks
            {"blocks": []},  # Missing header
            {"header": {}, "blocks": [], "signature": "invalid"},  # Invalid signature
        ]
        
        for manifest in malformed_manifests:
            result = self.verifier.verify_maif_signature(manifest)
            assert result is False
    
    def test_empty_provenance_chain(self):
        """Test handling of empty provenance chains."""
        provenance_data = {
            "chain": [],
            "agent_id": "test_agent"
        }
        
        is_valid, errors = self.verifier.verify_provenance_chain(provenance_data)
        
        # Empty chain should be valid (no entries to verify)
        assert is_valid is True
        assert len(errors) == 0


class TestSecurityPerformance:
    """Test security performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.signer = MAIFSigner(agent_id="test_agent")
        self.verifier = MAIFVerifier()
    
    def test_signing_performance(self):
        """Test signing performance with various data sizes."""
        import time
        
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        
        for size in data_sizes:
            test_data = b"x" * size
            
            start_time = time.time()
            signature = self.signer.sign_data(test_data)
            end_time = time.time()
            
            duration = end_time - start_time
            
            assert signature is not None
            assert duration < 1.0  # Should complete in under 1 second
    
    def test_verification_performance(self):
        """Test verification performance."""
        import time
        
        test_data = b"Performance test data"
        signature = self.signer.sign_data(test_data)
        public_key_pem = self.signer.get_public_key_pem().decode('utf-8')
        
        # Test multiple verifications
        start_time = time.time()
        
        for _ in range(100):
            result = self.verifier.verify_signature(test_data, signature, public_key_pem)
            assert result is True
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 100 verifications should complete quickly
        assert duration < 5.0  # Should complete in under 5 seconds
    
    def test_large_provenance_chain_performance(self):
        """Test performance with large provenance chains."""
        import time
        
        # Create large provenance chain
        start_time = time.time()
        
        for i in range(1000):
            self.signer.add_provenance_entry(f"action_{i}", f"hash_{i}")
        
        end_time = time.time()
        creation_duration = end_time - start_time
        
        # Verify the chain
        provenance_data = {
            "chain": [entry.to_dict() for entry in self.signer.provenance_chain],
            "agent_id": "test_agent"
        }
        
        start_time = time.time()
        is_valid, errors = self.verifier.verify_provenance_chain(provenance_data)
        end_time = time.time()
        
        verification_duration = end_time - start_time
        
        assert is_valid is True
        assert len(errors) == 0
        assert creation_duration < 5.0  # Creation should be fast
        assert verification_duration < 10.0  # Verification should be reasonable


if __name__ == "__main__":
    pytest.main([__file__])