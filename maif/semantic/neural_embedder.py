"""
Neural embeddings using sentence-transformers (optional dependency).

This module provides neural embedding support via sentence-transformers.
It is an optional feature and requires the sentence-transformers package.
"""

import time
import hashlib
from typing import Dict, List, Optional

from . import _check_neural_available, _get_sentence_transformer
from .semantic import SemanticEmbedding


class NeuralEmbedder:
    """
    Neural text embedder using sentence-transformers.

    This embedder uses pre-trained neural models from the sentence-transformers
    library to create dense vector embeddings of text. It supports GPU acceleration
    and automatic device detection.

    Attributes:
        model_name: The name/identifier of the sentence-transformer model.
        device: The device to run inference on ('cuda', 'mps', or 'cpu').
        model: The loaded SentenceTransformer model instance.

    Example:
        >>> embedder = NeuralEmbedder(model_name="all-MiniLM-L6-v2")
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(len(embedding.vector))
        384
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the neural embedder.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to "all-MiniLM-L6-v2".
            device: Device to run inference on. If None, auto-detects
                   (cuda > mps > cpu). Can be explicitly set to 'cuda',
                   'mps', or 'cpu'.

        Raises:
            ImportError: If sentence-transformers is not installed.

        Example:
            >>> embedder = NeuralEmbedder()  # Auto-detects device
            >>> embedder = NeuralEmbedder(device="cpu")  # Force CPU
        """
        # Check if sentence-transformers is available
        if not _check_neural_available():
            raise ImportError(
                "sentence-transformers is required for NeuralEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()

        self.device = device

        # Load the sentence-transformers model
        SentenceTransformer = _get_sentence_transformer()
        self.model = SentenceTransformer(model_name, device=device)

    def _auto_detect_device(self) -> str:
        """
        Auto-detect the best available device for inference.

        Tries devices in order of preference:
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple Silicon)
        3. CPU (fallback)

        Returns:
            str: The name of the detected device ('cuda', 'mps', or 'cpu').
        """
        try:
            import torch

            # Try CUDA first (most performant)
            if torch.cuda.is_available():
                return "cuda"

            # Try MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            # torch not available, fall back to cpu
            pass

        # Default to CPU
        return "cpu"

    def embed_text(
        self, text: str, metadata: Optional[Dict] = None
    ) -> SemanticEmbedding:
        """
        Generate a neural embedding for a single text.

        Args:
            text: The input text to embed.
            metadata: Optional metadata dictionary to attach to the embedding.

        Returns:
            SemanticEmbedding object containing the neural vector and metadata.

        Example:
            >>> embedder = NeuralEmbedder()
            >>> embedding = embedder.embed_text("The quick brown fox")
            >>> print(embedding.vector[:5])  # First 5 dimensions
            [-0.123, 0.456, ...]
        """
        # Encode text using the neural model
        vector = self.model.encode(text, convert_to_numpy=True)

        # Convert numpy array to list
        vector_list = vector.tolist()

        # Compute source hash
        source_hash = hashlib.sha256(text.encode()).hexdigest()

        # Prepare metadata
        final_metadata = metadata.copy() if metadata else {}
        final_metadata["embedder_type"] = "neural"

        # Create and return SemanticEmbedding
        embedding = SemanticEmbedding(
            vector=vector_list,
            source_hash=source_hash,
            model_name=self.model_name,
            timestamp=time.time(),
            metadata=final_metadata,
        )

        return embedding

    def embed_texts(
        self, texts: List[str], **kwargs
    ) -> List[SemanticEmbedding]:
        """
        Generate neural embeddings for multiple texts efficiently.

        Uses batch encoding for improved performance compared to
        calling embed_text multiple times.

        Args:
            texts: List of input texts to embed.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            List of SemanticEmbedding objects containing neural vectors.

        Example:
            >>> embedder = NeuralEmbedder()
            >>> texts = ["Hello world", "How are you", "Goodbye"]
            >>> embeddings = embedder.embed_texts(texts)
            >>> print(len(embeddings))
            3
        """
        if not texts:
            return []

        # Batch encode texts using the neural model
        vectors = self.model.encode(texts, convert_to_numpy=True)

        # Create SemanticEmbedding objects for each text
        embeddings = []
        current_time = time.time()

        for text, vector in zip(texts, vectors):
            # Convert numpy array to list
            vector_list = vector.tolist()

            # Compute source hash
            source_hash = hashlib.sha256(text.encode()).hexdigest()

            # Prepare metadata
            final_metadata = {"embedder_type": "neural"}

            # Create SemanticEmbedding
            embedding = SemanticEmbedding(
                vector=vector_list,
                source_hash=source_hash,
                model_name=self.model_name,
                timestamp=current_time,
                metadata=final_metadata,
            )

            embeddings.append(embedding)

        return embeddings
