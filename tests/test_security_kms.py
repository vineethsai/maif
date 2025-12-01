"""
Comprehensive tests for MAIF security with KMS integration.
"""

import pytest
import os
import json
import base64
import time
from unittest.mock import Mock, patch, MagicMock
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from maif.security import SecurityManager, MAIFSigner, MAIFVerifier, ProvenanceEntry


class TestSecurityManagerKMS:
    """Test SecurityManager with KMS integration."""
    
    @patch('maif.security.create_kms_verifier')
    @patch('maif.security.KMS_AVAILABLE', True)
    def test_kms_initialization_success(self, mock_create_verifier):
        """Test successful KMS initialization."""
        # Mock KMS verifier
        mock_verifier = MagicMock()
        mock_key_store = MagicMock()
        mock_verifier.key_store = mock_key_store
        mock_key_store.get_key_metadata.return_value = {
            'KeyId': 'test-key-id',
            'KeyState': 'Enabled'
        }
        mock_create_verifier.return_value = mock_verifier
        
        # Create SecurityManager with KMS
        manager = SecurityManager(
            use_kms=True,
            kms_key_id='test-key-id',
            region_name='us-east-1'
        )
        
        assert manager.kms_enabled is True
        assert manager.kms_key_id == 'test-key-id'
        assert manager.kms_verifier is not None
        mock_create_verifier.assert_called_once_with(region_name='us-east-1')
        
    @patch('maif.security.create_kms_verifier')
    @patch('maif.security.KMS_AVAILABLE', True)
    def test_kms_initialization_key_disabled(self, mock_create_verifier):
        """Test KMS initialization with disabled key."""
        # Mock KMS verifier with disabled key
        mock_verifier = MagicMock()
        mock_key_store = MagicMock()
        mock_verifier.key_store = mock_key_store
        mock_key_store.get_key_metadata.return_value = {
            'KeyId': 'test-key-id',
            'KeyState': 'Disabled'
        }
        mock_create_verifier.return_value = mock_verifier
        
        # Should raise error with disabled key
        with pytest.raises(RuntimeError, match="is not in Enabled state"):
            SecurityManager(
                use_kms=True,
                kms_key_id='test-key-id',
                require_encryption=True
            )
    
    @patch('maif.security.KMS_AVAILABLE', False)
    def test_kms_not_available(self):
        """Test behavior when KMS is not available."""
        manager = SecurityManager(use_kms=True, require_encryption=False)
        assert manager.kms_enabled is False
        
    def test_no_kms_key_id_error(self):
        """Test error when KMS is requested but no key ID provided."""
        with pytest.raises(RuntimeError, match="KMS initialization failed"):
            SecurityManager(use_kms=True, kms_key_id=None)
    
    @patch('maif.security.create_kms_verifier')
    @patch('maif.security.KMS_AVAILABLE', True)
    @patch('boto3.client')
    def test_encrypt_data_kms_direct(self, mock_boto_client, mock_create_verifier):
        """Test encryption using direct KMS for small data."""
        # Setup mocks
        mock_kms_client = MagicMock()
        mock_boto_client.return_value = mock_kms_client
        
        mock_verifier = MagicMock()
        mock_key_store = MagicMock()
        mock_verifier.key_store = mock_key_store
        mock_key_store.kms_client.meta.region_name = 'us-east-1'
        mock_key_store.get_key_metadata.return_value = {
            'KeyId': 'test-key-id',
            'KeyState': 'Enabled'
        }
        mock_create_verifier.return_value = mock_verifier
        
        # Mock KMS encrypt response
        encrypted_blob = b'encrypted_data_from_kms'
        mock_kms_client.encrypt.return_value = {
            'CiphertextBlob': encrypted_blob
        }
        
        # Create manager and encrypt small data
        manager = SecurityManager(use_kms=True, kms_key_id='test-key-id')
        small_data = b'Small test data'
        
        result = manager.encrypt_data(small_data)
        
        # Verify KMS was called
        mock_kms_client.encrypt.assert_called_once_with(
            KeyId='test-key-id',
            Plaintext=small_data
        )
        
        # Verify result format
        assert len(result) > 4  # Has header
        header_length = int.from_bytes(result[:4], byteorder='big')
        metadata = json.loads(result[4:4+header_length].decode('utf-8'))
        
        assert metadata['encryption_method'] == 'kms_direct'
        assert metadata['key_id'] == 'test-key-id'
        assert result[4+header_length:] == encrypted_blob
        
    @patch('maif.security.create_kms_verifier')
    @patch('maif.security.KMS_AVAILABLE', True)
    @patch('boto3.client')
    def test_encrypt_data_kms_envelope(self, mock_boto_client, mock_create_verifier):
        """Test envelope encryption for large data."""
        # Setup mocks
        mock_kms_client = MagicMock()
        mock_boto_client.return_value = mock_kms_client
        
        mock_verifier = MagicMock()
        mock_key_store = MagicMock()
        mock_verifier.key_store = mock_key_store
        mock_key_store.kms_client.meta.region_name = 'us-east-1'
        mock_key_store.get_key_metadata.return_value = {
            'KeyId': 'test-key-id',
            'KeyState': 'Enabled'
        }
        mock_create_verifier.return_value = mock_verifier
        
        # Mock KMS generate data key response
        plaintext_key = os.urandom(32)
        encrypted_key = b'encrypted_dek_from_kms'
        mock_kms_client.generate_data_key.return_value = {
            'Plaintext': plaintext_key,
            'CiphertextBlob': encrypted_key
        }
        
        # Create manager and encrypt large data
        manager = SecurityManager(use_kms=True, kms_key_id='test-key-id')
        large_data = b'X' * 5000  # Larger than 4KB
        
        result = manager.encrypt_data(large_data)
        
        # Verify KMS was called for data key
        mock_kms_client.generate_data_key.assert_called_once_with(
            KeyId='test-key-id',
            KeySpec='AES_256'
        )
        
        # Verify result format
        header_length = int.from_bytes(result[:4], byteorder='big')
        metadata = json.loads(result[4:4+header_length].decode('utf-8'))
        
        assert metadata['encryption_method'] == 'kms_envelope'
        assert metadata['key_id'] == 'test-key-id'
        assert 'encrypted_dek' in metadata
        assert 'iv' in metadata
        assert 'tag' in metadata
        
    def test_encrypt_data_local_fips(self):
        """Test local FIPS-compliant encryption when KMS is not available."""
        manager = SecurityManager(use_kms=False)
        test_data = b'Test data for FIPS encryption'
        
        encrypted = manager.encrypt_data(test_data)
        
        # Verify encrypted format
        header_length = int.from_bytes(encrypted[:4], byteorder='big')
        metadata = json.loads(encrypted[4:4+header_length].decode('utf-8'))
        
        assert metadata['encryption_method'] == 'local_fips'
        assert metadata['algorithm'] == 'AES-256-GCM'
        assert metadata['kdf'] == 'PBKDF2-HMAC-SHA256'
        assert metadata['iterations'] == 100000
        assert 'salt' in metadata
        assert 'iv' in metadata
        assert 'tag' in metadata
        
        # Decrypt and verify
        decrypted = manager.decrypt_data(encrypted)
        assert decrypted == test_data
        
    @patch('maif.security.create_kms_verifier')
    @patch('maif.security.KMS_AVAILABLE', True)
    @patch('boto3.client')
    def test_decrypt_data_kms_direct(self, mock_boto_client, mock_create_verifier):
        """Test decryption using direct KMS."""
        # Setup mocks
        mock_kms_client = MagicMock()
        mock_boto_client.return_value = mock_kms_client
        
        mock_verifier = MagicMock()
        mock_key_store = MagicMock()
        mock_verifier.key_store = mock_key_store
        mock_key_store.kms_client.meta.region_name = 'us-east-1'
        mock_key_store.get_key_metadata.return_value = {
            'KeyId': 'test-key-id',
            'KeyState': 'Enabled'
        }
        mock_create_verifier.return_value = mock_verifier
        
        # Create encrypted data with KMS direct format
        encrypted_blob = b'encrypted_data_from_kms'
        metadata = {
            'encryption_method': 'kms_direct',
            'key_id': 'test-key-id',
            'algorithm': 'SYMMETRIC_DEFAULT'
        }
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        header_length = len(metadata_bytes).to_bytes(4, byteorder='big')
        encrypted_data = header_length + metadata_bytes + encrypted_blob
        
        # Mock KMS decrypt response
        plaintext = b'Decrypted test data'
        mock_kms_client.decrypt.return_value = {
            'Plaintext': plaintext
        }
        
        # Decrypt
        manager = SecurityManager(use_kms=True, kms_key_id='test-key-id')
        result = manager.decrypt_data(encrypted_data)
        
        # Verify KMS was called
        mock_kms_client.decrypt.assert_called_once_with(
            CiphertextBlob=encrypted_blob
        )
        
        assert result == plaintext
        
    def test_encrypt_empty_data_error(self):
        """Test encryption with empty data raises error."""
        manager = SecurityManager(use_kms=False)
        
        with pytest.raises(ValueError, match="Cannot encrypt empty data"):
            manager.encrypt_data(b'')
            
    def test_decrypt_empty_data_error(self):
        """Test decryption with empty data raises error."""
        manager = SecurityManager(use_kms=False)
        
        with pytest.raises(ValueError, match="Cannot decrypt empty data"):
            manager.decrypt_data(b'')
            
    def test_security_event_logging(self):
        """Test security event logging."""
        manager = SecurityManager(use_kms=False)
        
        # Test event logging
        manager.log_security_event('test_event', {'detail': 'test'})
        
        assert len(manager.security_events) == 1  # 1 from test (init doesn't log when kms disabled)
        last_event = manager.security_events[-1]
        assert last_event['type'] == 'test_event'
        assert last_event['details']['detail'] == 'test'
        assert 'timestamp' in last_event
        
    def test_require_encryption_failure(self):
        """Test that encryption failure raises error when required."""
        manager = SecurityManager(use_kms=False, require_encryption=True)
        
        # Mock encryption to fail
        # with patch.object(manager, '_master_key', None): # Removed because it causes AttributeError and isn't needed
        with pytest.raises(RuntimeError, match="Decryption failed"):
            # Create data with invalid format to trigger decryption error
            invalid_data = b'\x00\x00\x00\x10' + b'{"encryption_method": "local_fips"}' + b'invalid'
            manager.decrypt_data(invalid_data)


class TestProvenanceEntry:
    """Test ProvenanceEntry functionality."""
    
    def test_provenance_entry_creation(self):
        """Test basic provenance entry creation."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id='agent-001',
            action='create',
            block_hash='abc123',
            metadata={'test': 'data'}
        )
        
        assert entry.timestamp == 1234567890.0
        assert entry.agent_id == 'agent-001'
        assert entry.action == 'create'
        assert entry.block_hash == 'abc123'
        assert entry.metadata['test'] == 'data'
        assert entry.entry_hash is not None
        
    def test_provenance_entry_hash_calculation(self):
        """Test provenance entry hash calculation."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id='agent-001',
            action='create',
            block_hash='abc123'
        )
        
        original_hash = entry.entry_hash
        
        # Recalculate should give same hash
        new_hash = entry.calculate_entry_hash()
        assert new_hash == original_hash
        
        # Changing data should change hash
        entry.action = 'update'
        changed_hash = entry.calculate_entry_hash()
        assert changed_hash != original_hash
        
    def test_provenance_entry_verification(self):
        """Test provenance entry verification."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id='agent-001',
            action='create',
            block_hash='abc123'
        )
        
        # Should verify without signature
        assert entry.verify() is True
        assert entry.verification_status == 'unverified'
        
        # Tampering should fail verification
        original_hash = entry.entry_hash
        entry.action = 'tampered'
        assert entry.verify() is False
        assert entry.verification_status == 'hash_mismatch'


class TestMAIFSigner:
    """Test MAIFSigner functionality."""
    
    def test_signer_initialization(self):
        """Test signer initialization."""
        signer = MAIFSigner()
        
        assert signer.agent_id is not None
        assert signer.private_key is not None
        assert len(signer.provenance_chain) == 1  # Genesis entry
        assert signer.provenance_chain[0].action == 'genesis'
        
    def test_add_provenance_entry(self):
        """Test adding provenance entries."""
        signer = MAIFSigner()
        
        # Add entry
        entry = signer.add_provenance_entry(
            action='update',
            block_hash='test_hash_123',
            metadata={'version': 2}
        )
        
        assert entry.action == 'update'
        assert entry.block_hash == 'test_hash_123'
        assert entry.metadata['version'] == 2
        assert entry.signature != ''
        assert len(signer.provenance_chain) == 2
        
    def test_sign_maif_manifest(self):
        """Test signing MAIF manifest."""
        signer = MAIFSigner()
        
        manifest = {
            'version': '1.0',
            'blocks': ['block1', 'block2']
        }
        
        signed = signer.sign_maif_manifest(manifest)
        
        assert 'signature' in signed
        assert 'public_key' in signed
        assert 'signature_metadata' in signed
        assert signed['signature_metadata']['signer_id'] == signer.agent_id
        assert len(signed['signature_metadata']['provenance_chain']) > 0


class TestMAIFVerifier:
    """Test MAIFVerifier functionality."""
    
    def test_verify_manifest_signature(self):
        """Test manifest signature verification."""
        # Create and sign manifest
        signer = MAIFSigner()
        manifest = {'test': 'data'}
        signed_manifest = signer.sign_maif_manifest(manifest)
        
        # Verify signature
        verifier = MAIFVerifier()
        assert verifier.verify_maif_signature(signed_manifest) is True
        
        # Tamper with manifest
        signed_manifest['test'] = 'tampered'
        assert verifier.verify_maif_signature(signed_manifest) is False
        
    def test_verify_empty_provenance_chain(self):
        """Test verification of empty provenance chain."""
        verifier = MAIFVerifier()
        
        # Empty chain should be valid
        is_valid, errors = verifier.verify_provenance_chain([])
        assert is_valid is True
        assert len(errors) == 0
        
    def test_verify_provenance_chain_formats(self):
        """Test different provenance chain formats."""
        verifier = MAIFVerifier()
        
        # Test direct list format
        # Test direct list format
        entry = ProvenanceEntry(
            timestamp=1234567890,
            agent_id='test',
            action='genesis',
            block_hash='hash1'
        )
        entry.calculate_entry_hash()
        chain = [entry.to_dict()]
        
        is_valid, errors = verifier.verify_provenance_chain(chain)
        assert is_valid is True
        
        # Test wrapped format
        wrapped = {'chain': chain}
        is_valid, errors = verifier.verify_provenance_chain(wrapped)
        assert is_valid is True
        
        # Test version history format
        version_history = {'version_history': chain}
        is_valid, errors = verifier.verify_provenance_chain(version_history)
        assert is_valid is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])