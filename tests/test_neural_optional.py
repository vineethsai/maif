import pytest
from unittest import mock
from maif.semantic import NEURAL_AVAILABLE, get_embedder

class TestNeuralAvailability:
    def test_neural_flag_exists(self):
        """Test NEURAL_AVAILABLE flag is defined."""
        assert NEURAL_AVAILABLE is not None
        assert isinstance(NEURAL_AVAILABLE, bool)

    @pytest.mark.skipif(not NEURAL_AVAILABLE, reason="sentence-transformers not installed")
    def test_neural_embeddings_when_available(self):
        """Test neural embeddings work when installed."""
        embedder = get_embedder(model_name="all-MiniLM-L6-v2", prefer_neural=True)
        assert embedder.__class__.__name__ == "NeuralEmbedder"

        # Test embedding
        embedding = embedder.embed_text("Test text")
        assert len(embedding.vector) == 384  # MiniLM dimension
        assert embedding.metadata["embedder_type"] == "neural"

    def test_graceful_fallback_to_tfidf(self):
        """Test fallback to TF-IDF when neural unavailable."""
        with mock.patch('maif.semantic.NEURAL_AVAILABLE', False):
            embedder = get_embedder(prefer_neural=True)
            # Should be TF-IDF, not NeuralEmbedder
            assert embedder.__class__.__name__ == "TFIDFEmbedder"

class TestNeuralOptional:
    def test_neural_deps_truly_optional(self):
        """Test MAIF works without neural deps."""
        # If neural deps not installed, should not raise ImportError
        # unless explicitly importing NeuralEmbedder
        try:
            from maif.semantic import get_embedder
            embedder = get_embedder()  # Default, no neural
            assert embedder is not None
        except ImportError as e:
            if "sentence_transformers" in str(e):
                pytest.fail("Neural deps should be optional")

    def test_factory_selection_logic(self):
        """Test factory correctly selects embedder."""
        # No neural requested, no neural available
        embedder1 = get_embedder(prefer_neural=False)
        assert embedder1.__class__.__name__ == "TFIDFEmbedder"

        # Neural requested but fallback if unavailable
        embedder2 = get_embedder(prefer_neural=True)
        if NEURAL_AVAILABLE:
            assert embedder2.__class__.__name__ == "NeuralEmbedder"
        else:
            assert embedder2.__class__.__name__ == "TFIDFEmbedder"

    def test_backward_compatibility(self):
        """Test old code still works."""
        from maif.semantic import TFIDFEmbedder, SemanticEmbedder

        # Old way still works
        embedder = SemanticEmbedder()
        assert embedder is not None

        # New way works
        embedder = get_embedder()
        assert embedder is not None

class TestNeuralDeviceSelection:
    @pytest.mark.skipif(not NEURAL_AVAILABLE, reason="sentence-transformers not installed")
    def test_device_auto_detection(self):
        """Test device auto-detection works."""
        embedder = get_embedder(prefer_neural=True)
        # Should auto-detect cuda/mps/cpu
        assert hasattr(embedder, 'device')
        assert embedder.device in ['cuda', 'mps', 'cpu']

    @pytest.mark.skipif(not NEURAL_AVAILABLE, reason="sentence-transformers not installed")
    def test_device_override(self):
        """Test device can be overridden."""
        embedder = get_embedder(prefer_neural=True, device='cpu')
        assert embedder.device == 'cpu'
