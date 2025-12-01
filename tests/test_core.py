"""
Comprehensive tests for MAIF core functionality.
"""

import pytest
import tempfile
import os
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from maif.core import MAIFEncoder, MAIFDecoder, MAIFBlock, MAIFVersion, MAIFHeader, MAIFParser
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode
from maif.security import MAIFSigner, MAIFVerifier


class TestMAIFBlock:
    """Test MAIFBlock data structure."""
    
    def test_maif_block_creation(self):
        """Test basic MAIFBlock creation."""
        block = MAIFBlock(
            block_id="test_block_001",
            block_type="TEXT",
            data=b"Hello, MAIF!",
            metadata={"source": "test", "timestamp": 1234567890}
        )
        
        assert block.block_id == "test_block_001"
        assert block.block_type == "TEXT"
        assert block.data == b"Hello, MAIF!"
        assert block.metadata["source"] == "test"
        assert block.hash is not None
        assert len(block.hash) == 64  # SHA-256 hex length
    
    def test_maif_block_to_dict(self):
        """Test MAIFBlock serialization to dictionary."""
        block = MAIFBlock(
            block_id="test_block_001",
            block_type="TEXT",
            data=b"Hello, MAIF!",
            metadata={"source": "test"}
        )
        
        block_dict = block.to_dict()
        
        assert block_dict["block_id"] == "test_block_001"
        assert block_dict["block_type"] == "TEXT"
        assert "hash" in block_dict
        assert "metadata" in block_dict
        assert block_dict["metadata"]["source"] == "test"


class TestMAIFVersion:
    """Test MAIFVersion data structure."""
    
    def test_maif_version_creation(self):
        """Test MAIFVersion creation."""
        version = MAIFVersion(
            version=1,
            timestamp=1234567890.0,
            agent_id="test_agent",
            operation="create",
            block_hash="abc123"
        )
        
        assert version.version == 1
        assert version.timestamp == 1234567890.0
        assert version.agent_id == "test_agent"
        assert version.operation == "create"
        assert version.block_hash == "abc123"
    
    def test_maif_version_to_dict(self):
        """Test MAIFVersion serialization."""
        version = MAIFVersion(
            version=1,
            timestamp=1234567890.0,
            agent_id="test_agent",
            operation="create",
            block_hash="abc123"
        )
        
        version_dict = version.to_dict()
        
        assert version_dict["version"] == 1
        assert version_dict["timestamp"] == 1234567890.0
        assert version_dict["agent_id"] == "test_agent"
        assert version_dict["operation"] == "create"
        assert version_dict["block_hash"] == "abc123"


class TestMAIFEncoder:
    """Test MAIFEncoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = MAIFEncoder(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_encoder_initialization(self):
        """Test MAIFEncoder initialization."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        assert encoder.agent_id == "test_agent"
        assert encoder.blocks == []
        assert encoder.version_history == {}
        assert encoder.access_rules == []
        assert encoder.privacy_engine is not None
    
    def test_add_text_block(self):
        """Test adding text blocks."""
        self.encoder.add_text_block(
            text="Hello, MAIF world!",
            metadata={"source": "test", "language": "en"}
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        assert block.block_type == "TEXT"
        assert "Hello, MAIF world!" in block.data.decode('utf-8')
        assert block.metadata["source"] == "test"
        assert block.metadata["language"] == "en"
    
    def test_add_binary_block(self):
        """Test adding binary blocks."""
        binary_data = b"\x89PNG\r\n\x1a\n"  # PNG header
        
        self.encoder.add_binary_block(
            data=binary_data,
            block_type="image",
            metadata={"format": "png", "size": len(binary_data)}
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        assert block.block_type == "IDAT"
        assert block.data == binary_data
        assert block.metadata["format"] == "png"
    
    def test_add_embeddings_block(self):
        """Test adding embeddings blocks."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        self.encoder.add_embeddings_block(
            embeddings=embeddings,
            metadata={"model": "test-model", "dimensions": 3}
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        assert block.block_type == "EMBD"
        assert block.metadata["model"] == "test-model"
        assert block.metadata["dimensions"] == 3
    
    @patch('maif.semantic.DeepSemanticUnderstanding')
    def test_add_cross_modal_block(self, mock_semantic):
        """Test adding cross-modal blocks."""
        mock_semantic_instance = Mock()
        mock_semantic_instance.process_multimodal_input.return_value = {
            "unified_embedding": [0.1, 0.2, 0.3],
            "attention_weights": {"text": 0.6, "image": 0.4},
            "semantic_features": {"entities": ["test"], "sentiment": "positive"}
        }
        mock_semantic.return_value = mock_semantic_instance
        
        multimodal_data = {
            "text": "A beautiful sunset",
            "image": b"fake_image_data"
        }
        
        self.encoder.add_cross_modal_block(
            multimodal_data=multimodal_data,
            metadata={"scene": "sunset"}
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        assert block.block_type == "XMOD"
        assert block.metadata["scene"] == "sunset"
    
    def test_delete_block(self):
        """Test block deletion."""
        self.encoder.add_text_block("Test text")
        block_id = self.encoder.blocks[0].block_id
        
        result = self.encoder.delete_block(block_id, reason="test deletion")
        
        assert result is True
        assert len(self.encoder.blocks) == 1  # Block still exists but marked as deleted
        assert self.encoder.blocks[0].metadata.get("deleted") is True
        assert self.encoder.blocks[0].metadata.get("deletion_reason") == "test deletion"
    
    def test_access_rules(self):
        """Test access rule management."""
        self.encoder.add_access_rule(
            subject="user123",
            resource="block_*",
            permissions=["read", "write"],
            conditions={"time_limit": "2024-12-31"}
        )
        
        assert len(self.encoder.access_rules) == 1
        rule = self.encoder.access_rules[0]
        assert rule.subject == "user123"
        assert rule.resource == "block_*"
        assert "read" in rule.permissions
        assert "write" in rule.permissions
    
    def test_check_access(self):
        """Test access checking."""
        self.encoder.add_text_block("Test text")
        block_id = self.encoder.blocks[0].block_id
        
        self.encoder.add_access_rule(
            subject="user123",
            resource=f"{block_id}",
            permissions=["read"]
        )
        
        # Should have access
        assert self.encoder.check_access("user123", block_id, "read") is True
        
        # Should not have write access
        assert self.encoder.check_access("user123", block_id, "write") is False
        
        # Different user should not have access
        assert self.encoder.check_access("user456", block_id, "read") is False
    
    def test_build_maif(self):
        """Test MAIF file building."""
        self.encoder.add_text_block("Hello, MAIF!")
        self.encoder.add_binary_block(b"binary_data", "data")
        
        output_path = os.path.join(self.temp_dir, "test.maif")
        manifest_path = os.path.join(self.temp_dir, "test_manifest.json")
        
        self.encoder.build_maif(output_path, manifest_path)
        
        # Check files were created
        assert os.path.exists(output_path)
        assert os.path.exists(manifest_path)
        
        # Check manifest content
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert "header" in manifest
        assert "blocks" in manifest
        assert len(manifest["blocks"]) == 2
        assert manifest["header"]["agent_id"] == "test_agent"


class TestMAIFDecoder:
    """Test MAIFDecoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Hello, MAIF!", metadata={"source": "test"})
        encoder.add_binary_block(b"binary_data", "data", metadata={"type": "test_data"})
        
        self.maif_path = os.path.join(self.temp_dir, "test.maif")
        self.manifest_path = os.path.join(self.temp_dir, "test_manifest.json")
        
        encoder.build_maif(self.maif_path, self.manifest_path)
        
        self.decoder = MAIFDecoder(self.maif_path, self.manifest_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_decoder_initialization(self):
        """Test MAIFDecoder initialization."""
        assert self.decoder.maif_path == self.maif_path
        assert self.decoder.manifest_path == self.manifest_path
        assert self.decoder.manifest is not None
        assert len(self.decoder.blocks) == 2
    
    def test_verify_integrity(self):
        """Test integrity verification."""
        # First verify that a clean file passes integrity check
        result = self.decoder.verify_integrity()
        assert result is True
        
        # Now test that tampering is detected
        # Tamper with the MAIF file (seek past header + some data)
        with open(self.maif_path, 'r+b') as f:
            f.seek(80)  # Seek well past the 32-byte header into data
            original_byte = f.read(1)
            f.seek(80)
            f.write(b'X')  # Change one byte
        
        # Create new decoder for tampered file
        tampered_decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        tampered_result = tampered_decoder.verify_integrity()
        assert tampered_result is False  # Should detect tampering
        
        # Restore the original byte for other tests
        with open(self.maif_path, 'r+b') as f:
            f.seek(50)
            f.write(original_byte)
    
    def test_get_text_blocks(self):
        """Test retrieving text blocks."""
        text_blocks = self.decoder.get_text_blocks()
        
        assert len(text_blocks) == 1
        assert "Hello, MAIF!" in text_blocks[0]
    
    def test_get_block_data(self):
        """Test retrieving specific block data."""
        # Get text block data
        text_data = self.decoder.get_block_data("TEXT")
        assert text_data is not None
        assert "Hello, MAIF!" in text_data.decode('utf-8')
        
        # Get binary block data
        binary_data = self.decoder.get_block_data("BDAT")
        assert binary_data is not None
        assert binary_data == b"binary_data"
    
    def test_get_accessible_blocks(self):
        """Test getting accessible blocks."""
        accessible_blocks = self.decoder.get_accessible_blocks()
        
        # Should return all blocks when no privacy restrictions
        assert len(accessible_blocks) == 2
    
    def test_privacy_summary(self):
        """Test privacy summary generation."""
        summary = self.decoder.get_privacy_summary()
        
        assert "total_blocks" in summary
        assert "encrypted_blocks" in summary
        assert "access_controlled_blocks" in summary
        assert summary["total_blocks"] == 2


class TestMAIFParser:
    """Test MAIFParser high-level interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Hello, MAIF!", metadata={"source": "test"})
        
        self.maif_path = os.path.join(self.temp_dir, "test.maif")
        self.manifest_path = os.path.join(self.temp_dir, "test_manifest.json")
        
        encoder.build_maif(self.maif_path, self.manifest_path)
        
        self.parser = MAIFParser(self.maif_path, self.manifest_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_metadata(self):
        """Test metadata extraction."""
        metadata = self.parser.get_metadata()
        
        assert "header" in metadata
        assert "blocks" in metadata
        assert metadata["header"]["agent_id"] == "test_agent"
    
    def test_extract_content(self):
        """Test content extraction."""
        content = self.parser.extract_content()
        
        assert "text_blocks" in content
        assert len(content["text_blocks"]) == 1
        assert "Hello, MAIF!" in content["text_blocks"][0]


class TestPrivacyIntegration:
    """Test privacy features integration with core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.privacy_engine = PrivacyEngine()
        self.encoder = MAIFEncoder(
            agent_id="test_agent",
            privacy_engine=self.privacy_engine,
            enable_privacy=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_encrypted_text_block(self):
        """Test adding encrypted text blocks."""
        self.encoder.add_text_block(
            text="Sensitive information",
            metadata={"classification": "confidential"},
            privacy_level=PrivacyLevel.HIGH,
            encryption_mode=EncryptionMode.AES_GCM
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        
        # Data should be encrypted
        assert block.data != b"Sensitive information"
        # Encryption metadata is now stored under _system
        assert block.metadata.get("_system", {}).get("encrypted") is True
        assert block.metadata.get("_system", {}).get("encryption_mode") == "aes_gcm"
    
    def test_anonymized_text_block(self):
        """Test adding anonymized text blocks."""
        self.encoder.add_text_block(
            text="John Smith works at ACME Corp",
            metadata={"source": "hr_document"},
            anonymize=True
        )
        
        assert len(self.encoder.blocks) == 1
        block = self.encoder.blocks[0]
        
        # Text should be anonymized
        text_content = block.data.decode('utf-8')
        assert "John Smith" not in text_content
        assert "ACME Corp" not in text_content
        # Anonymized flag is now stored under _system
        assert block.metadata.get("_system", {}).get("anonymized") is True
    
    def test_privacy_report(self):
        """Test privacy report generation."""
        # Add various blocks with different privacy levels
        self.encoder.add_text_block("Public information")
        self.encoder.add_text_block(
            "Confidential data",
            privacy_level=PrivacyLevel.HIGH,
            encryption_mode=EncryptionMode.AES_GCM
        )
        
        report = self.encoder.get_privacy_report()
        
        assert "total_blocks" in report
        assert "encrypted_blocks" in report
        assert "anonymized_blocks" in report
        print(f"DEBUG: blocks count = {len(self.encoder.blocks)}")
        print(f"DEBUG: report = {report}")
        assert report["total_blocks"] == 2


class TestVersioning:
    """Test versioning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = MAIFEncoder(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_version_history_tracking(self):
        """Test that version history is properly tracked."""
        # Add initial block
        self.encoder.add_text_block("Initial text")
        block_id = self.encoder.blocks[0].block_id
        
        # Modify block (simulate update)
        self.encoder._add_block(
            block_type="text",
            data=b"Updated text",
            metadata={"updated": True},
            update_block_id=block_id  # Same ID for update
        )
        
        # Check version history
        assert block_id in self.encoder.version_history
        versions = self.encoder.version_history[block_id]
        assert len(versions) >= 1
    
    def test_get_block_at_version(self):
        """Test retrieving specific block versions."""
        # Add and update a block
        self.encoder.add_text_block("Version 1")
        block_id = self.encoder.blocks[0].block_id
        
        # Get block at version 1
        block_v1 = self.encoder.get_block_at_version(block_id, 1)
        assert block_v1 is not None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = MAIFEncoder(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_block_type(self):
        """Test handling of invalid block types."""
        with pytest.raises(ValueError):
            self.encoder._add_block(
                block_type="",  # Empty block type
                data=b"test data"
            )
    
    def test_empty_data(self):
        """Test handling of empty data."""
        # Should not raise an error
        self.encoder.add_text_block("")
        assert len(self.encoder.blocks) == 1
    
    def test_large_data_block(self):
        """Test handling of large data blocks."""
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        self.encoder.add_binary_block(large_data, "large_data")
        assert len(self.encoder.blocks) == 1
        assert len(self.encoder.blocks[0].data) == len(large_data)
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        with pytest.raises(FileNotFoundError):
            MAIFDecoder("nonexistent.maif", "nonexistent.json")
    
    def test_corrupted_manifest(self):
        """Test handling of corrupted manifest files."""
        # Create a corrupted manifest
        manifest_path = os.path.join(self.temp_dir, "corrupted.json")
        with open(manifest_path, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            with open(manifest_path, 'r') as f:
                json.load(f)


class TestPerformance:
    """Test performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = MAIFEncoder(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_many_small_blocks(self):
        """Test performance with many small blocks."""
        import time
        
        start_time = time.time()
        
        # Add 1000 small text blocks
        for i in range(1000):
            self.encoder.add_text_block(f"Block {i}", metadata={"index": i})
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert len(self.encoder.blocks) == 1000
        assert duration < 10.0  # Should complete in under 10 seconds
    
    def test_large_embeddings_block(self):
        """Test performance with large embeddings."""
        import time
        
        # Create large embeddings (1000 vectors of 512 dimensions)
        large_embeddings = [[0.1] * 512 for _ in range(1000)]
        
        start_time = time.time()
        self.encoder.add_embeddings_block(large_embeddings)
        end_time = time.time()
        
        duration = end_time - start_time
        
        assert len(self.encoder.blocks) == 1
        assert duration < 5.0  # Should complete in under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__])