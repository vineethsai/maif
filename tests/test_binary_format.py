"""
Comprehensive tests for MAIF binary format functionality.
"""

import pytest
import tempfile
import os
import struct
import hashlib
from unittest.mock import Mock, patch, MagicMock

from maif.core import MAIFEncoder, MAIFDecoder, MAIFBlock, MAIFHeader


class TestBinaryFormat:
    """Test MAIF binary format structure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_binary_header_format(self):
        """Test binary header format."""
        encoder = MAIFEncoder(agent_id="binary_test")
        encoder.add_text_block("Test content", metadata={"test": True})
        
        maif_path = os.path.join(self.temp_dir, "binary_test.maif")
        manifest_path = os.path.join(self.temp_dir, "binary_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Read binary header
        with open(maif_path, 'rb') as f:
            header_data = f.read(16)  # Basic header size
            
        # Should have valid binary data
        assert len(header_data) == 16
        assert header_data != b'\x00' * 16  # Not all zeros
    
    def test_binary_block_structure(self):
        """Test binary block structure."""
        encoder = MAIFEncoder(agent_id="binary_test")
        encoder.add_text_block("Block structure test", metadata={"block_test": True})
        encoder.add_binary_block(b"Binary data test", "data", metadata={"binary_test": True})
        
        maif_path = os.path.join(self.temp_dir, "block_test.maif")
        manifest_path = os.path.join(self.temp_dir, "block_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify blocks can be read back
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) >= 2
        assert decoder.blocks[0].block_type in ["TEXT", "text_data", "text"]
        assert decoder.blocks[1].block_type in ["BDAT", "binary_data", "data"]
    
    def test_binary_data_integrity(self):
        """Test binary data integrity."""
        test_data = b"Binary integrity test data with special chars: \x00\x01\x02\xFF"
        
        encoder = MAIFEncoder(agent_id="integrity_test")
        encoder.add_binary_block(test_data, "test_data", metadata={"integrity_test": True})
        
        maif_path = os.path.join(self.temp_dir, "integrity_test.maif")
        manifest_path = os.path.join(self.temp_dir, "integrity_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Read back and verify
        decoder = MAIFDecoder(maif_path, manifest_path)
        retrieved_data = decoder.get_block_data(decoder.blocks[0].block_id)
        
        assert retrieved_data == test_data
    
    def test_binary_large_data(self):
        """Test handling of large binary data."""
        # Create large binary data (1MB)
        large_data = b"A" * (1024 * 1024)
        
        encoder = MAIFEncoder(agent_id="large_test")
        encoder.add_binary_block(large_data, "large_data", metadata={"size": "1MB"})
        
        maif_path = os.path.join(self.temp_dir, "large_test.maif")
        manifest_path = os.path.join(self.temp_dir, "large_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify file was created and has reasonable size
        assert os.path.exists(maif_path)
        file_size = os.path.getsize(maif_path)
        assert file_size > 1024 * 1024  # Should be at least 1MB
    
    def test_binary_mixed_content(self):
        """Test mixed binary and text content."""
        encoder = MAIFEncoder(agent_id="mixed_test")
        
        # Add various types of content
        encoder.add_text_block("Text content", metadata={"type": "text"})
        encoder.add_binary_block(b"Binary content", "data", metadata={"type": "binary"})
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]], metadata={"type": "embeddings"})
        
        maif_path = os.path.join(self.temp_dir, "mixed_test.maif")
        manifest_path = os.path.join(self.temp_dir, "mixed_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify all content types are preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) >= 3
        
        # Should have different block types
        block_types = [block.block_type for block in decoder.blocks]
        assert len(set(block_types)) >= 2  # At least 2 different types
    
    def test_binary_compression_compatibility(self):
        """Test binary format compatibility with compression."""
        # Create compressible data
        compressible_data = b"AAAA" * 1000  # Highly compressible
        
        encoder = MAIFEncoder(agent_id="compression_test")
        encoder.add_binary_block(compressible_data, "compressible", metadata={"compression_test": True})
        
        maif_path = os.path.join(self.temp_dir, "compression_test.maif")
        manifest_path = os.path.join(self.temp_dir, "compression_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Should be able to read back
        decoder = MAIFDecoder(maif_path, manifest_path)
        retrieved_data = decoder.get_block_data(decoder.blocks[0].block_id)
        
        assert retrieved_data == compressible_data


class TestBinaryEncoding:
    """Test binary encoding and decoding."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_binary_encoding_text(self):
        """Test binary encoding of text data."""
        text_content = "Hello, binary encoding world! 🌍"
        
        encoder = MAIFEncoder(agent_id="encoding_test")
        encoder.add_text_block(text_content, metadata={"encoding_test": True})
        
        maif_path = os.path.join(self.temp_dir, "encoding_test.maif")
        manifest_path = os.path.join(self.temp_dir, "encoding_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify encoding/decoding
        decoder = MAIFDecoder(maif_path, manifest_path)
        text_blocks = decoder.get_text_blocks()
        
        assert len(text_blocks) > 0
        assert text_content in text_blocks
    
    def test_binary_encoding_unicode(self):
        """Test binary encoding of Unicode text."""
        unicode_content = "Unicode test: 你好世界 🚀 العالم مرحبا"
        
        encoder = MAIFEncoder(agent_id="unicode_test")
        encoder.add_text_block(unicode_content, metadata={"unicode_test": True})
        
        maif_path = os.path.join(self.temp_dir, "unicode_test.maif")
        manifest_path = os.path.join(self.temp_dir, "unicode_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify Unicode is preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        text_blocks = decoder.get_text_blocks()
        
        assert len(text_blocks) > 0
        assert unicode_content in text_blocks
    
    def test_binary_encoding_embeddings(self):
        """Test binary encoding of embeddings."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        encoder = MAIFEncoder(agent_id="embeddings_test")
        encoder.add_embeddings_block(embeddings, metadata={"embeddings_test": True})
        
        maif_path = os.path.join(self.temp_dir, "embeddings_test.maif")
        manifest_path = os.path.join(self.temp_dir, "embeddings_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify embeddings are preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        retrieved_embeddings = decoder.get_embeddings()
        
        assert len(retrieved_embeddings) > 0
        # Check if embeddings are approximately equal
        import numpy as np
        # Compare each embedding individually since retrieved_embeddings[0] is a single embedding
        # but embeddings is a list of embeddings
        assert len(retrieved_embeddings) == len(embeddings)
        for i, (retrieved, original) in enumerate(zip(retrieved_embeddings, embeddings)):
            assert np.allclose(retrieved, original, rtol=1e-5, atol=1e-6)
    
    def test_binary_encoding_metadata(self):
        """Test binary encoding of metadata."""
        complex_metadata = {
            "string": "test",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "unicode": "测试 🎯"
        }
        
        encoder = MAIFEncoder(agent_id="metadata_test")
        encoder.add_text_block("Metadata test", metadata=complex_metadata)
        
        maif_path = os.path.join(self.temp_dir, "metadata_test.maif")
        manifest_path = os.path.join(self.temp_dir, "metadata_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify metadata is preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) > 0
        retrieved_metadata = decoder.blocks[0].metadata
        
        assert retrieved_metadata is not None
        assert retrieved_metadata["string"] == "test"
        assert retrieved_metadata["number"] == 42
        assert retrieved_metadata["unicode"] == "测试 🎯"


class TestBinaryPerformance:
    """Test binary format performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_binary_write_performance(self):
        """Test binary write performance."""
        import time
        
        encoder = MAIFEncoder(agent_id="perf_test")
        
        # Add multiple blocks
        start_time = time.time()
        for i in range(100):
            encoder.add_text_block(f"Performance test block {i}", metadata={"index": i})
        
        maif_path = os.path.join(self.temp_dir, "perf_test.maif")
        manifest_path = os.path.join(self.temp_dir, "perf_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds)
        elapsed = end_time - start_time
        assert elapsed < 10.0
    
    def test_binary_read_performance(self):
        """Test binary read performance."""
        import time
        
        # Create test file
        encoder = MAIFEncoder(agent_id="read_perf_test")
        for i in range(50):
            encoder.add_text_block(f"Read performance test {i}", metadata={"index": i})
        
        maif_path = os.path.join(self.temp_dir, "read_perf_test.maif")
        manifest_path = os.path.join(self.temp_dir, "read_perf_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Test read performance
        start_time = time.time()
        decoder = MAIFDecoder(maif_path, manifest_path)
        text_blocks = decoder.get_text_blocks()
        end_time = time.time()
        
        # Should read quickly
        elapsed = end_time - start_time
        assert elapsed < 5.0
        assert len(text_blocks) == 50
    
    def test_binary_size_efficiency(self):
        """Test binary format size efficiency."""
        # Create test content
        test_content = "Size efficiency test content"
        
        encoder = MAIFEncoder(agent_id="size_test")
        encoder.add_text_block(test_content, metadata={"size_test": True})
        
        maif_path = os.path.join(self.temp_dir, "size_test.maif")
        manifest_path = os.path.join(self.temp_dir, "size_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Check file sizes
        maif_size = os.path.getsize(maif_path)
        manifest_size = os.path.getsize(manifest_path)
        content_size = len(test_content.encode('utf-8'))
        
        # MAIF file should not be excessively larger than content
        # Allow for reasonable overhead (10x should be more than enough)
        assert maif_size < content_size * 10
        # Manifest can be larger due to metadata, allow more generous overhead
        assert manifest_size < content_size * 100  # Increased tolerance for manifest


class TestBinaryCompatibility:
    """Test binary format compatibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_binary_version_compatibility(self):
        """Test binary format version compatibility."""
        encoder = MAIFEncoder(agent_id="version_test")
        encoder.add_text_block("Version compatibility test", metadata={"version_test": True})
        
        maif_path = os.path.join(self.temp_dir, "version_test.maif")
        manifest_path = os.path.join(self.temp_dir, "version_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Should be readable
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) > 0
        assert decoder.manifest.get("maif_version") is not None
    
    def test_binary_cross_platform_compatibility(self):
        """Test binary format cross-platform compatibility."""
        # Test with different endianness considerations
        encoder = MAIFEncoder(agent_id="platform_test")
        
        # Add data that might be affected by endianness
        encoder.add_binary_block(struct.pack('>I', 0x12345678), "big_endian", metadata={"endian": "big"})
        encoder.add_binary_block(struct.pack('<I', 0x12345678), "little_endian", metadata={"endian": "little"})
        
        maif_path = os.path.join(self.temp_dir, "platform_test.maif")
        manifest_path = os.path.join(self.temp_dir, "platform_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Should be readable regardless of platform
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) >= 2
        
        # Verify data integrity
        for block in decoder.blocks:
            data = decoder.get_block_data(block.block_id)
            assert data is not None
            assert len(data) == 4  # 4 bytes for uint32