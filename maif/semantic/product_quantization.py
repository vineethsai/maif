"""
Product Quantization (PQ) for Embedding Compression

This module implements Product Quantization, the core technique for achieving
2.5-4x compression on high-dimensional embeddings while maintaining semantic quality.

Key Concepts:
- Split D-dimensional embedding into M subvectors of D/M dimensions
- Train independent K-means codebook for each subvector
- Encode each subvector with its codebook index (8-bit = 256 codes)
- Decode by looking up codebook entries and concatenating

References:
- Product Quantization for Similarity Search (Jegou et al., 2011)
- Similarity Search the Right Way (Douze et al., 2016)
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class ProductQuantizer:
    """
    Core Product Quantization implementation.

    Compresses D-dimensional vectors to M bytes by splitting into subvectors
    and quantizing each independently.

    Parameters:
        dim: Original embedding dimension (e.g., 384, 768)
        num_subvectors: Number of subvectors M (default 8)
        codebook_size: Number of codes per codebook K (default 256 = 8 bits)

    Example:
        pq = ProductQuantizer(dim=768, num_subvectors=12)
        pq.train(embeddings)  # Train codebooks
        codes = pq.encode(embeddings)  # Encode to indices
        reconstructed = pq.decode(codes)  # Reconstruct vectors
    """

    def __init__(
        self,
        dim: int,
        num_subvectors: int = 8,
        codebook_size: int = 256,
    ):
        """Initialize Product Quantizer."""
        if KMeans is None:
            raise ImportError(
                "Product Quantization requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        if dim % num_subvectors != 0:
            raise ValueError(
                f"Dimension {dim} must be divisible by num_subvectors {num_subvectors}. "
                f"Consider using num_subvectors={dim // 32} for {dim}-dim vectors."
            )

        self.D = dim  # Total dimension
        self.M = num_subvectors  # Number of subvectors
        self.K = codebook_size  # Number of centroids per subvector
        self.subvec_dim = dim // num_subvectors

        self.codebooks = []  # List of M codebooks, each shape (K, subvec_dim)
        self._is_trained = False

    def train(
        self,
        vectors: np.ndarray,
        max_iter: int = 100,
        n_init: int = 10,
        verbose: int = 0,
    ) -> None:
        """
        Train PQ codebooks using K-means on subvectors.

        Parameters:
            vectors: Training data, shape (N, D)
            max_iter: Maximum K-means iterations
            n_init: Number of K-means initializations
            verbose: Verbosity level for K-means
        """
        if vectors.shape[1] != self.D:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"configured dimension {self.D}"
            )

        vectors = vectors.astype(np.float32)
        self.codebooks = []

        # Train independent codebook for each subvector
        for m in range(self.M):
            start_idx = m * self.subvec_dim
            end_idx = (m + 1) * self.subvec_dim

            # Extract subvectors for this position
            subvectors = vectors[:, start_idx:end_idx]

            # Train K-means on subvector space
            kmeans = KMeans(
                n_clusters=self.K,
                max_iter=max_iter,
                n_init=n_init,
                verbose=verbose,
                random_state=42,
            )
            kmeans.fit(subvectors)

            # Store codebook (cluster centers)
            codebook = kmeans.cluster_centers_.astype(np.float32)
            self.codebooks.append(codebook)

        self._is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to PQ codes (indices).

        Parameters:
            vectors: Input vectors, shape (N, D)

        Returns:
            Codes array of shape (N, M) with dtype uint8
        """
        if not self._is_trained:
            raise RuntimeError("Must call train() before encode()")

        if vectors.shape[1] != self.D:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"configured dimension {self.D}"
            )

        vectors = vectors.astype(np.float32)
        n_vectors = vectors.shape[0]
        codes = np.zeros((n_vectors, self.M), dtype=np.uint8)

        # Encode each subvector
        for m in range(self.M):
            start_idx = m * self.subvec_dim
            end_idx = (m + 1) * self.subvec_dim

            # Extract subvectors
            subvectors = vectors[:, start_idx:end_idx]
            codebook = self.codebooks[m]

            # Find nearest codebook entry for each subvector
            # Using efficient distance computation
            distances = self._compute_distances(subvectors, codebook)
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to (approximate) vectors.

        Parameters:
            codes: PQ codes, shape (N, M) with dtype uint8

        Returns:
            Reconstructed vectors, shape (N, D) with dtype float32
        """
        if not self._is_trained:
            raise RuntimeError("Must call train() before decode()")

        if codes.shape[1] != self.M:
            raise ValueError(
                f"Code dimension {codes.shape[1]} doesn't match "
                f"configured num_subvectors {self.M}"
            )

        codes = codes.astype(np.uint8)
        n_vectors = codes.shape[0]
        vectors = np.zeros((n_vectors, self.D), dtype=np.float32)

        # Reconstruct each subvector
        for m in range(self.M):
            start_idx = m * self.subvec_dim
            end_idx = (m + 1) * self.subvec_dim

            codebook = self.codebooks[m]

            # Lookup centroid for each code
            vectors[:, start_idx:end_idx] = codebook[codes[:, m]]

        return vectors

    def compute_compression_ratio(self, original_size_bytes: int) -> float:
        """
        Compute theoretical compression ratio.

        Parameters:
            original_size_bytes: Original uncompressed size in bytes

        Returns:
            Compression ratio (original_size / compressed_size)
        """
        # Compressed size = codebooks + codes
        codebook_size = self.M * self.K * self.subvec_dim * 4  # float32
        codes_size_per_vector = self.M  # uint8 index per subvector

        # For N vectors:
        # compressed_size = codebook_size + N * codes_size_per_vector
        # But this depends on N, so we estimate for typical batch size

        n_vectors_estimate = original_size_bytes // (self.D * 4)  # Assume float32
        codes_size = n_vectors_estimate * codes_size_per_vector
        total_compressed = codebook_size + codes_size

        return original_size_bytes / total_compressed if total_compressed > 0 else 1.0

    @staticmethod
    def _compute_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Efficiently compute squared Euclidean distances.

        Parameters:
            vectors: Shape (N, D)
            centroids: Shape (K, D)

        Returns:
            Distances, shape (N, K)
        """
        # ||v - c||^2 = ||v||^2 + ||c||^2 - 2 <v, c>
        v_norms = np.sum(vectors ** 2, axis=1, keepdims=True)  # (N, 1)
        c_norms = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        dot_product = np.dot(vectors, centroids.T)  # (N, K)

        distances = v_norms + c_norms - 2 * dot_product
        return np.maximum(distances, 0.0)  # Clamp to avoid numerical issues

    def get_config(self) -> dict:
        """Get configuration dict for serialization."""
        return {
            "dim": self.D,
            "num_subvectors": self.M,
            "codebook_size": self.K,
            "subvec_dim": self.subvec_dim,
            "is_trained": self._is_trained,
        }

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"ProductQuantizer(dim={self.D}, num_subvectors={self.M}, "
            f"codebook_size={self.K}, {status})"
        )


def get_optimal_pq_config(embedding_dim: int) -> dict:
    """
    Get optimal PQ configuration for given embedding dimension.

    Uses empirically-determined configurations that balance compression
    ratio and reconstruction quality.

    Parameters:
        embedding_dim: Embedding dimension (e.g., 384, 768)

    Returns:
        Dict with 'num_subvectors' and 'codebook_size'
    """
    configs = {
        384: {"num_subvectors": 8, "codebook_size": 256},    # 384/8 = 48 dims per subvec
        512: {"num_subvectors": 8, "codebook_size": 256},    # 512/8 = 64 dims per subvec
        768: {"num_subvectors": 12, "codebook_size": 256},   # 768/12 = 64 dims per subvec
        1024: {"num_subvectors": 16, "codebook_size": 256},  # 1024/16 = 64 dims per subvec
        1536: {"num_subvectors": 24, "codebook_size": 256},  # 1536/24 = 64 dims per subvec
    }

    # If exact match, use it
    if embedding_dim in configs:
        return configs[embedding_dim]

    # Otherwise, infer from similar dimensions
    # Prefer subvector dimension of 32-64
    for num_subvectors in [8, 12, 16, 24, 32]:
        if embedding_dim % num_subvectors == 0:
            subvec_dim = embedding_dim // num_subvectors
            if 32 <= subvec_dim <= 64:
                return {
                    "num_subvectors": num_subvectors,
                    "codebook_size": 256
                }

    # Fallback: divide into ~48-dim subvectors
    num_subvectors = max(1, embedding_dim // 48)
    return {
        "num_subvectors": num_subvectors,
        "codebook_size": 256
    }
