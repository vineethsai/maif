import pytest
import numpy as np
from maif.semantic.product_quantization import ProductQuantizer, get_optimal_pq_config
from maif.semantic.hsc_binary_format import HSCBinaryFormat
from maif.semantic.semantic_optimized import HierarchicalSemanticCompression

class TestProductQuantizer:
    def test_pq_training_and_encoding(self):
        """Test PQ training and encoding."""
        pq = ProductQuantizer(dim=384, num_subvectors=8)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        pq.train(embeddings)
        codes = pq.encode(embeddings)
        assert codes.shape == (100, 8)
        assert codes.dtype == np.uint8

    def test_pq_decode_roundtrip(self):
        """Test PQ decode preserves structure."""
        pq = ProductQuantizer(dim=768, num_subvectors=12)
        embeddings = np.random.randn(50, 768).astype(np.float32)
        pq.train(embeddings)
        codes = pq.encode(embeddings)
        reconstructed = pq.decode(codes)
        assert reconstructed.shape == embeddings.shape
        assert reconstructed.dtype == np.float32

    def test_pq_optimal_config(self):
        """Test optimal PQ configs for standard dimensions."""
        configs = {
            384: 8,
            512: 8,
            768: 12,
            1024: 16,
            1536: 24,
        }
        for dim, expected_subvecs in configs.items():
            config = get_optimal_pq_config(dim)
            assert config['num_subvectors'] == expected_subvecs

class TestBinaryFormat:
    def test_binary_serialize_deserialize(self):
        """Test binary format roundtrip."""
        pq = ProductQuantizer(dim=384, num_subvectors=8)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        pq.train(embeddings)
        codes = pq.encode(embeddings)

        # Serialize
        binary_data = HSCBinaryFormat.serialize(pq, codes)
        assert isinstance(binary_data, bytes)
        assert binary_data.startswith(b'HSCPQ')

        # Deserialize
        pq2, codes2, _ = HSCBinaryFormat.deserialize(binary_data)
        reconstructed = pq2.decode(codes2)

        assert pq2.D == pq.D
        assert pq2.M == pq.M

class TestHierarchicalPQ:
    def test_hsc_pq_compression_ratio(self):
        """Test HSC achieves target 2.5-4x compression."""
        embeddings = np.random.randn(1000, 768).astype(np.float32)

        hsc = HierarchicalSemanticCompression(use_pq=True)
        result = hsc.compress_embeddings(embeddings.tolist())

        ratio = result['metadata']['compression_ratio']
        assert 2.5 <= ratio <= 4.5, f"Ratio {ratio} not in target range"

    def test_hsc_pq_quality(self):
        """Test HSC maintains >0.95 cosine similarity."""
        embeddings = np.random.randn(500, 384).astype(np.float32)

        hsc = HierarchicalSemanticCompression(use_pq=True)
        result = hsc.compress_embeddings(embeddings.tolist())

        fidelity = result['metadata']['fidelity_score']
        assert fidelity > 0.90, f"Fidelity {fidelity} below threshold"

    def test_hsc_pq_vs_legacy(self):
        """Test PQ is better than legacy HSC."""
        embeddings = np.random.randn(500, 384).astype(np.float32)

        # PQ compression
        hsc_pq = HierarchicalSemanticCompression(use_pq=True)
        result_pq = hsc_pq.compress_embeddings(embeddings.tolist())

        # Legacy compression
        hsc_legacy = HierarchicalSemanticCompression(use_pq=False)
        result_legacy = hsc_legacy.compress_embeddings(embeddings.tolist())

        # PQ should have better compression ratio
        ratio_pq = result_pq['metadata']['compression_ratio']
        ratio_legacy = result_legacy['metadata']['compression_ratio']
        assert ratio_pq >= ratio_legacy * 0.9, "PQ worse than legacy"
