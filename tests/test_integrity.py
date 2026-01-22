"""
Comprehensive integrity tests for MAIF secure format.

Tests cover:
1. Merkle tree calculation and verification
2. Hash chain linking and verification
3. Tamper detection (modify a block, verify it's caught)
4. Provenance chain integrity
5. End-to-end: create file, tamper, detect
"""

import pytest
import tempfile
import os
import hashlib
import json
import struct
import shutil
from typing import List

from maif.core.secure_format import (
    SecureMAIFWriter,
    SecureMAIFReader,
    SecureBlockType,
    SecureBlock,
    SecureBlockHeader,
    SecureFileHeader,
    BlockFlags,
    FileFlags,
    ProvenanceEntry,
    MAGIC_HEADER,
    MAGIC_FOOTER,
)


class TestMerkleTreeCalculation:
    """Tests for Merkle tree calculation and verification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merkle_root_single_block(self):
        """Test Merkle root with a single block."""
        file_path = os.path.join(self.temp_dir, "single_block.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Hello, World!")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # With single block, merkle root should equal that block's content hash
        assert len(reader.blocks) == 1
        single_hash = reader.blocks[0].header.content_hash
        calculated_root = reader._calculate_merkle_root()
        assert calculated_root == single_hash

    def test_merkle_root_two_blocks(self):
        """Test Merkle root with two blocks."""
        file_path = os.path.join(self.temp_dir, "two_blocks.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block one")
        writer.add_text_block("Block two")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        assert len(reader.blocks) == 2
        h1 = reader.blocks[0].header.content_hash
        h2 = reader.blocks[1].header.content_hash

        # Merkle root for two leaves = hash(h1 + h2)
        expected_root = hashlib.sha256(h1 + h2).digest()
        calculated_root = reader._calculate_merkle_root()
        assert calculated_root == expected_root

    def test_merkle_root_three_blocks(self):
        """Test Merkle root with three blocks (odd number)."""
        file_path = os.path.join(self.temp_dir, "three_blocks.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block one")
        writer.add_text_block("Block two")
        writer.add_text_block("Block three")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        assert len(reader.blocks) == 3
        h1 = reader.blocks[0].header.content_hash
        h2 = reader.blocks[1].header.content_hash
        h3 = reader.blocks[2].header.content_hash

        # For odd number, duplicate last: [h1, h2, h3, h3]
        # Level 1: hash(h1+h2), hash(h3+h3)
        # Level 2: hash(level1[0] + level1[1])
        level1_0 = hashlib.sha256(h1 + h2).digest()
        level1_1 = hashlib.sha256(h3 + h3).digest()
        expected_root = hashlib.sha256(level1_0 + level1_1).digest()

        calculated_root = reader._calculate_merkle_root()
        assert calculated_root == expected_root

    def test_merkle_root_four_blocks(self):
        """Test Merkle root with four blocks (power of 2)."""
        file_path = os.path.join(self.temp_dir, "four_blocks.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        for i in range(4):
            writer.add_text_block(f"Block {i}")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        assert len(reader.blocks) == 4
        hashes = [b.header.content_hash for b in reader.blocks]

        # Level 1: hash(h0+h1), hash(h2+h3)
        # Level 2: hash(level1[0] + level1[1])
        level1_0 = hashlib.sha256(hashes[0] + hashes[1]).digest()
        level1_1 = hashlib.sha256(hashes[2] + hashes[3]).digest()
        expected_root = hashlib.sha256(level1_0 + level1_1).digest()

        calculated_root = reader._calculate_merkle_root()
        assert calculated_root == expected_root

    def test_merkle_root_stored_matches_calculated(self):
        """Test that stored Merkle root matches recalculated root."""
        file_path = os.path.join(self.temp_dir, "merkle_verify.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        for i in range(5):
            writer.add_text_block(f"Content block {i}")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        stored_root = reader.file_header.merkle_root
        calculated_root = reader._calculate_merkle_root()
        assert stored_root == calculated_root

    def test_merkle_root_empty_file(self):
        """Test Merkle root calculation for empty file (no blocks)."""
        file_path = os.path.join(self.temp_dir, "empty.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # Empty file should have zero merkle root
        expected_root = b"\x00" * 32
        assert reader._calculate_merkle_root() == expected_root


class TestHashChainLinking:
    """Tests for hash chain linking and verification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_first_block_previous_hash_is_zero(self):
        """Test that first block has zero previous hash."""
        file_path = os.path.join(self.temp_dir, "chain_first.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("First block")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        first_block = reader.blocks[0]
        expected_prev = b"\x00" * 32
        assert first_block.header.previous_hash == expected_prev

    def test_chain_linking_multiple_blocks(self):
        """Test that blocks are properly chain-linked."""
        file_path = os.path.join(self.temp_dir, "chain_multiple.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        for i in range(5):
            writer.add_text_block(f"Block {i}")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # First block should have zero previous hash
        assert reader.blocks[0].header.previous_hash == b"\x00" * 32

        # Each subsequent block should reference previous block's content hash
        for i in range(1, len(reader.blocks)):
            prev_content_hash = reader.blocks[i - 1].header.content_hash
            curr_prev_hash = reader.blocks[i].header.previous_hash
            assert curr_prev_hash == prev_content_hash, f"Chain break at block {i}"

    def test_chain_verification_in_integrity_check(self):
        """Test that chain verification is included in integrity check."""
        file_path = os.path.join(self.temp_dir, "chain_verify.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block 1")
        writer.add_text_block("Block 2")
        writer.add_text_block("Block 3")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is True
        assert len(errors) == 0

    def test_content_hash_includes_metadata(self):
        """Test that content hash includes metadata when present."""
        file_path = os.path.join(self.temp_dir, "metadata_hash.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Text with metadata", {"key": "value", "count": 42})
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        block = reader.blocks[0]
        # Recalculate content hash
        data_hash = hashlib.sha256(block.data).digest()
        meta_bytes = json.dumps(block.metadata, sort_keys=True).encode()
        expected_hash = hashlib.sha256(data_hash + meta_bytes).digest()

        assert block.header.content_hash == expected_hash


class TestTamperDetection:
    """Tests for tamper detection capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_modified_block_data(self):
        """Test detection of modified block data."""
        file_path = os.path.join(self.temp_dir, "tamper_data.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Original content")
        writer.finalize()

        # Read original file
        reader = SecureMAIFReader(file_path)
        reader.load()
        block = reader.blocks[0]
        original_offset = SecureFileHeader.HEADER_SIZE + SecureBlockHeader.HEADER_SIZE

        # Tamper with the file: modify the text content
        with open(file_path, "r+b") as f:
            f.seek(original_offset)
            f.write(b"Tampered content")

        # Verify tampering is detected
        tampered_reader = SecureMAIFReader(file_path)
        is_valid, errors = tampered_reader.verify_integrity()

        assert is_valid is False
        assert len(errors) > 0
        assert tampered_reader.is_tampered() is True

    def test_detect_modified_block_header(self):
        """Test detection of modified block header."""
        file_path = os.path.join(self.temp_dir, "tamper_header.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Test content")
        writer.finalize()

        # Tamper with block header - modify block_type field
        header_offset = SecureFileHeader.HEADER_SIZE
        with open(file_path, "r+b") as f:
            f.seek(header_offset + 4)  # Skip size field, modify block_type
            original_type = struct.unpack(">I", f.read(4))[0]
            f.seek(header_offset + 4)
            f.write(struct.pack(">I", original_type ^ 0xFF))  # Flip some bits

        # Verify tampering is detected
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        assert reader.is_tampered() is True

    def test_detect_modified_merkle_root(self):
        """Test detection when Merkle root is tampered."""
        file_path = os.path.join(self.temp_dir, "tamper_merkle.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block 1")
        writer.add_text_block("Block 2")
        writer.finalize()

        # Tamper with Merkle root in file header
        # Merkle root starts at offset: 4+2+2+4+8+8+16+16+64+4+8+8+8 = 152
        merkle_offset = 4 + 2 + 2 + 4 + 8 + 8 + 16 + 16 + 64 + 4 + 8 + 8 + 8
        with open(file_path, "r+b") as f:
            f.seek(merkle_offset)
            f.write(b"\xFF" * 32)  # Corrupt Merkle root

        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        # Should contain error about Merkle root mismatch
        merkle_error = any("merkle" in e.lower() for e in errors)
        assert merkle_error is True

    def test_detect_modified_signature(self):
        """Test detection when block signature is tampered."""
        file_path = os.path.join(self.temp_dir, "tamper_sig.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Signed content")
        writer.finalize()

        # Tamper with signature field in block header
        # Signature starts at: size(4) + type(4) + flags(4) + version(4) +
        # timestamp(8) + block_id(16) + prev_hash(32) + content_hash(32) = 104
        block_offset = SecureFileHeader.HEADER_SIZE
        sig_offset = block_offset + 4 + 4 + 4 + 4 + 8 + 16 + 32 + 32
        with open(file_path, "r+b") as f:
            f.seek(sig_offset)
            f.write(b"\x00" * 64)  # Corrupt signature

        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        assert reader.is_tampered() is True

    def test_get_tampered_block_indices(self):
        """Test that specific tampered blocks are identified."""
        file_path = os.path.join(self.temp_dir, "tamper_indices.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block 0")
        writer.add_text_block("Block 1")
        writer.add_text_block("Block 2")
        writer.finalize()

        # Get the offset of block 1's data
        reader_pre = SecureMAIFReader(file_path)
        reader_pre.load()
        block0_size = reader_pre.blocks[0].header.size
        block1_data_offset = (
            SecureFileHeader.HEADER_SIZE
            + block0_size
            + SecureBlockHeader.HEADER_SIZE
        )

        # Tamper with block 1 only
        with open(file_path, "r+b") as f:
            f.seek(block1_data_offset)
            f.write(b"TAMPERED!")

        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        tampered = reader.get_tampered_blocks()
        assert 1 in tampered  # Block 1 should be identified as tampered

    def test_tampered_flag_set_on_blocks(self):
        """Test that TAMPERED flag is set on detected blocks."""
        file_path = os.path.join(self.temp_dir, "tamper_flag.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Original")
        writer.finalize()

        # Tamper with block data
        data_offset = SecureFileHeader.HEADER_SIZE + SecureBlockHeader.HEADER_SIZE
        with open(file_path, "r+b") as f:
            f.seek(data_offset)
            f.write(b"Modified")

        reader = SecureMAIFReader(file_path)
        reader.verify_integrity()

        block = reader.blocks[0]
        assert block.header.flags & BlockFlags.TAMPERED


class TestProvenanceChainIntegrity:
    """Tests for provenance chain integrity."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_genesis_entry_exists(self):
        """Test that genesis entry is created."""
        file_path = os.path.join(self.temp_dir, "prov_genesis.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        assert len(reader.provenance) > 0
        genesis = reader.provenance[0]
        assert genesis.action == "genesis"
        assert genesis.chain_position == 0

    def test_provenance_entry_for_each_block(self):
        """Test that each block addition creates a provenance entry."""
        file_path = os.path.join(self.temp_dir, "prov_blocks.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Text 1")
        writer.add_text_block("Text 2")
        writer.add_embeddings_block([[0.1, 0.2, 0.3]])
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # genesis + 3 blocks + finalize = 5 entries
        assert len(reader.provenance) == 5

        actions = [e.action for e in reader.provenance]
        assert "genesis" in actions
        assert "add_text_block" in actions
        assert "add_embeddings_block" in actions
        assert "finalize" in actions

    def test_provenance_chain_linking(self):
        """Test that provenance entries are properly linked."""
        file_path = os.path.join(self.temp_dir, "prov_chain.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Block 1")
        writer.add_text_block("Block 2")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # Check chain linking
        for i in range(1, len(reader.provenance)):
            prev_entry = reader.provenance[i - 1]
            curr_entry = reader.provenance[i]
            assert curr_entry.previous_entry_hash == prev_entry.entry_hash, \
                f"Provenance chain break at position {i}"

    def test_provenance_entry_hash_calculation(self):
        """Test that provenance entry hashes are valid."""
        file_path = os.path.join(self.temp_dir, "prov_hash.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Test")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        for entry in reader.provenance:
            # Recalculate hash
            original_hash = entry.entry_hash
            entry.entry_hash = ""  # Clear for recalculation
            recalculated = entry.calculate_hash()
            assert recalculated == original_hash

    def test_provenance_signatures_exist(self):
        """Test that provenance entries are signed."""
        file_path = os.path.join(self.temp_dir, "prov_signed.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        for entry in reader.provenance:
            assert entry.signature is not None
            assert len(entry.signature) > 0

    def test_provenance_chain_position_increments(self):
        """Test that chain positions increment correctly."""
        file_path = os.path.join(self.temp_dir, "prov_position.maif")
        writer = SecureMAIFWriter(file_path, agent_id="test-agent")
        for i in range(3):
            writer.add_text_block(f"Block {i}")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        for i, entry in enumerate(reader.provenance):
            assert entry.chain_position == i

    def test_provenance_agent_info_recorded(self):
        """Test that agent information is recorded in provenance."""
        file_path = os.path.join(self.temp_dir, "prov_agent.maif")
        agent_id = "my-test-agent"
        writer = SecureMAIFWriter(file_path, agent_id=agent_id)
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        for entry in reader.provenance:
            assert entry.agent_id == agent_id
            assert agent_id in entry.agent_did


class TestEndToEndIntegrity:
    """End-to-end integrity tests: create, tamper, detect."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_verify_valid_file(self):
        """Test creating and verifying a valid file."""
        file_path = os.path.join(self.temp_dir, "e2e_valid.maif")

        # Create
        writer = SecureMAIFWriter(file_path, agent_id="creator-agent")
        writer.add_text_block("Document title", {"type": "title"})
        writer.add_text_block("Document body content goes here.")
        writer.add_embeddings_block([[0.1, 0.2, 0.3, 0.4, 0.5]])
        writer.add_binary_block(b"\x00\x01\x02\x03", metadata={"format": "raw"})
        writer.finalize()

        # Verify
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is True
        assert len(errors) == 0
        assert reader.is_tampered() is False
        assert len(reader.get_tampered_blocks()) == 0

    def test_create_tamper_first_block_detect(self):
        """Test tampering with first block is detected."""
        file_path = os.path.join(self.temp_dir, "e2e_tamper_first.maif")

        # Create
        writer = SecureMAIFWriter(file_path, agent_id="creator")
        writer.add_text_block("First block - important header")
        writer.add_text_block("Second block")
        writer.finalize()

        # Verify original is valid
        reader1 = SecureMAIFReader(file_path)
        is_valid1, _ = reader1.verify_integrity()
        assert is_valid1 is True

        # Tamper with first block's data
        data_offset = SecureFileHeader.HEADER_SIZE + SecureBlockHeader.HEADER_SIZE
        with open(file_path, "r+b") as f:
            f.seek(data_offset)
            f.write(b"HACKED! Modified header!!!!")

        # Detect
        reader2 = SecureMAIFReader(file_path)
        is_valid2, errors = reader2.verify_integrity()

        assert is_valid2 is False
        assert reader2.is_tampered() is True
        assert 0 in reader2.get_tampered_blocks()

    def test_create_tamper_middle_block_detect(self):
        """Test tampering with middle block is detected."""
        file_path = os.path.join(self.temp_dir, "e2e_tamper_middle.maif")

        # Create
        writer = SecureMAIFWriter(file_path, agent_id="creator")
        for i in range(5):
            writer.add_text_block(f"Block number {i} with some content padding")
        writer.finalize()

        # Find middle block offset
        reader_pre = SecureMAIFReader(file_path)
        reader_pre.load()

        # Calculate offset to block 2
        offset = SecureFileHeader.HEADER_SIZE
        for i in range(2):
            offset += reader_pre.blocks[i].header.size
        offset += SecureBlockHeader.HEADER_SIZE  # Skip block 2's header

        # Tamper
        with open(file_path, "r+b") as f:
            f.seek(offset)
            f.write(b"TAMPERED MIDDLE BLOCK!")

        # Detect
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        assert reader.is_tampered() is True
        assert 2 in reader.get_tampered_blocks()

    def test_create_tamper_last_block_detect(self):
        """Test tampering with last block is detected."""
        file_path = os.path.join(self.temp_dir, "e2e_tamper_last.maif")

        # Create
        writer = SecureMAIFWriter(file_path, agent_id="creator")
        writer.add_text_block("Block 0")
        writer.add_text_block("Block 1")
        writer.add_text_block("Last block - final content here")
        writer.finalize()

        # Find last block offset
        reader_pre = SecureMAIFReader(file_path)
        reader_pre.load()

        offset = SecureFileHeader.HEADER_SIZE
        for i in range(len(reader_pre.blocks) - 1):
            offset += reader_pre.blocks[i].header.size
        offset += SecureBlockHeader.HEADER_SIZE

        # Tamper
        with open(file_path, "r+b") as f:
            f.seek(offset)
            f.write(b"TAMPERED LAST!!!")

        # Detect
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        assert reader.is_tampered() is True
        last_idx = len(reader.blocks) - 1
        assert last_idx in reader.get_tampered_blocks()

    def test_full_workflow_with_embeddings(self):
        """Test full workflow with embeddings data."""
        file_path = os.path.join(self.temp_dir, "e2e_embeddings.maif")

        # Create file with embeddings
        writer = SecureMAIFWriter(file_path, agent_id="embedding-agent")
        writer.add_text_block("Source text for embeddings")
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
        writer.add_embeddings_block(embeddings, {"model": "test-model"})
        writer.finalize()

        # Verify
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is True
        assert len(reader.blocks) == 2

        # Check embeddings metadata
        emb_block = reader.blocks[1]
        assert emb_block.metadata["count"] == 3
        assert emb_block.metadata["dimensions"] == 4

    def test_binary_data_integrity(self):
        """Test integrity with binary data."""
        file_path = os.path.join(self.temp_dir, "e2e_binary.maif")

        # Create with binary data
        binary_data = bytes(range(256)) * 4  # 1KB of binary data
        writer = SecureMAIFWriter(file_path, agent_id="binary-agent")
        writer.add_binary_block(binary_data, SecureBlockType.BINARY, {"size": len(binary_data)})
        writer.finalize()

        # Verify
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is True
        assert reader.blocks[0].data == binary_data

        # Tamper with binary data (flip a byte)
        data_offset = SecureFileHeader.HEADER_SIZE + SecureBlockHeader.HEADER_SIZE + 100
        with open(file_path, "r+b") as f:
            f.seek(data_offset)
            original = f.read(1)
            f.seek(data_offset)
            f.write(bytes([original[0] ^ 0xFF]))

        # Detect tampering
        reader2 = SecureMAIFReader(file_path)
        is_valid2, _ = reader2.verify_integrity()
        assert is_valid2 is False

    def test_multiple_tampering_detected(self):
        """Test that multiple tampered blocks are all detected."""
        file_path = os.path.join(self.temp_dir, "e2e_multi_tamper.maif")

        # Create
        writer = SecureMAIFWriter(file_path, agent_id="creator")
        for i in range(4):
            writer.add_text_block(f"Block {i}: " + "x" * 50)
        writer.finalize()

        # Get block offsets
        reader_pre = SecureMAIFReader(file_path)
        reader_pre.load()

        offsets = []
        offset = SecureFileHeader.HEADER_SIZE
        for block in reader_pre.blocks:
            offsets.append(offset + SecureBlockHeader.HEADER_SIZE)
            offset += block.header.size

        # Tamper with blocks 0 and 2
        with open(file_path, "r+b") as f:
            f.seek(offsets[0])
            f.write(b"TAMPERED_0")
            f.seek(offsets[2])
            f.write(b"TAMPERED_2")

        # Detect
        reader = SecureMAIFReader(file_path)
        is_valid, errors = reader.verify_integrity()

        assert is_valid is False
        tampered = reader.get_tampered_blocks()
        assert 0 in tampered
        assert 2 in tampered

    def test_verify_clean_file_multiple_times(self):
        """Test that clean file passes verification repeatedly."""
        file_path = os.path.join(self.temp_dir, "e2e_repeat.maif")

        writer = SecureMAIFWriter(file_path, agent_id="creator")
        writer.add_text_block("Consistent content")
        writer.finalize()

        # Verify multiple times
        for i in range(5):
            reader = SecureMAIFReader(file_path)
            is_valid, errors = reader.verify_integrity()
            assert is_valid is True, f"Failed on verification {i}"

    def test_file_info_after_verification(self):
        """Test file info retrieval after integrity verification."""
        file_path = os.path.join(self.temp_dir, "e2e_info.maif")

        writer = SecureMAIFWriter(file_path, agent_id="info-agent")
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.verify_integrity()

        info = reader.get_file_info()
        assert info["agent_did"].startswith("did:maif:info-agent")
        assert info["is_signed"] is True
        assert info["is_finalized"] is True
        assert info["key_algorithm"] == "Ed25519"
        assert info["block_count"] == 1

    def test_export_manifest_includes_integrity(self):
        """Test that exported manifest includes integrity info."""
        file_path = os.path.join(self.temp_dir, "e2e_manifest.maif")

        writer = SecureMAIFWriter(file_path, agent_id="manifest-agent")
        writer.add_text_block("Block 1")
        writer.add_text_block("Block 2")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.verify_integrity()

        manifest = reader.export_manifest()

        assert "integrity" in manifest
        assert manifest["integrity"]["verified"] is True
        assert manifest["integrity"]["tampering_detected"] is False
        assert len(manifest["integrity"]["tampered_blocks"]) == 0

        assert "blocks" in manifest
        assert len(manifest["blocks"]) == 2
        for block_info in manifest["blocks"]:
            assert block_info["is_signed"] is True
            assert block_info["is_tampered"] is False


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_text_block(self):
        """Test handling of empty text block."""
        file_path = os.path.join(self.temp_dir, "edge_empty.maif")

        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_text_block("")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert reader.blocks[0].data == b""

    def test_large_block(self):
        """Test handling of large block."""
        file_path = os.path.join(self.temp_dir, "edge_large.maif")

        # Create 1MB of data
        large_data = b"x" * (1024 * 1024)
        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_binary_block(large_data)
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert len(reader.blocks[0].data) == 1024 * 1024

    def test_unicode_content(self):
        """Test handling of unicode content."""
        file_path = os.path.join(self.temp_dir, "edge_unicode.maif")

        unicode_text = "Hello, \u4e16\u754c! \U0001f600 \u0420\u043e\u0441\u0441\u0438\u044f"
        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_text_block(unicode_text)
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert reader.get_text_content(0) == unicode_text

    def test_special_characters_in_metadata(self):
        """Test handling of special characters in metadata."""
        file_path = os.path.join(self.temp_dir, "edge_special.maif")

        metadata = {
            "quotes": 'He said "Hello"',
            "backslash": "path\\to\\file",
            "newline": "line1\nline2",
            "unicode": "\u4e2d\u6587",
        }
        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_text_block("Content", metadata)
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert reader.blocks[0].metadata["quotes"] == metadata["quotes"]
        assert reader.blocks[0].metadata["unicode"] == metadata["unicode"]

    def test_many_small_blocks(self):
        """Test handling of many small blocks."""
        file_path = os.path.join(self.temp_dir, "edge_many.maif")

        writer = SecureMAIFWriter(file_path, agent_id="test")
        for i in range(100):
            writer.add_text_block(f"Block {i}")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert len(reader.blocks) == 100

    def test_read_without_verify(self):
        """Test that reading works without explicit verification."""
        file_path = os.path.join(self.temp_dir, "edge_no_verify.maif")

        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        reader.load()

        # Should be able to read without calling verify_integrity
        assert len(reader.blocks) == 1
        assert reader.get_text_content(0) == "Content"

    def test_verify_triggers_load(self):
        """Test that verification triggers loading if needed."""
        file_path = os.path.join(self.temp_dir, "edge_auto_load.maif")

        writer = SecureMAIFWriter(file_path, agent_id="test")
        writer.add_text_block("Content")
        writer.finalize()

        reader = SecureMAIFReader(file_path)
        # Don't call load(), just verify
        is_valid, _ = reader.verify_integrity()

        assert is_valid is True
        assert len(reader.blocks) == 1  # Verify should have loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
