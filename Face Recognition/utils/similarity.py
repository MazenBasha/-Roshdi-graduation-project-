"""
Similarity computation for face recognition.

Implements cosine similarity for face embedding matching and recognition.
"""

from typing import Union, Tuple
import numpy as np
import tensorflow as tf


class SimilarityMetric:
    """Compute similarity metrics between embeddings."""

    @staticmethod
    def cosine_similarity(
        embedding1: Union[np.ndarray, tf.Tensor],
        embedding2: Union[np.ndarray, tf.Tensor],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Assumes embeddings are already L2-normalized. If not, they will be normalized.

        Args:
            embedding1: First embedding vector (D,) or (1, D).
            embedding2: Second embedding vector (D,) or (1, D).

        Returns:
            Cosine similarity score in [-1, 1]. Higher = more similar.
        """
        # Convert to numpy if tensor
        if isinstance(embedding1, tf.Tensor):
            embedding1 = embedding1.numpy()
        if isinstance(embedding2, tf.Tensor):
            embedding2 = embedding2.numpy()

        # Flatten to 1D if needed
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # Normalize if not already normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2

        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(embedding1, embedding2)

        # Clip to [-1, 1] for numerical stability
        similarity = np.clip(similarity, -1.0, 1.0)

        return float(similarity)

    @staticmethod
    def batch_cosine_similarity(
        embeddings1: Union[np.ndarray, tf.Tensor],
        embeddings2: Union[np.ndarray, tf.Tensor],
    ) -> np.ndarray:
        """
        Compute cosine similarity between batches of embeddings.

        Args:
            embeddings1: Batch of embeddings (N, D) or (M, D).
            embeddings2: Batch of embeddings (N, D) or (K, D).

        Returns:
            Similarity matrix of shape (N, K).
        """
        # Convert to numpy
        if isinstance(embeddings1, tf.Tensor):
            embeddings1 = embeddings1.numpy()
        if isinstance(embeddings2, tf.Tensor):
            embeddings2 = embeddings2.numpy()

        # Normalize
        embeddings1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        similarity = np.dot(embeddings1, embeddings2.T)

        return np.clip(similarity, -1.0, 1.0)

    @staticmethod
    def euclidean_distance(
        embedding1: Union[np.ndarray, tf.Tensor],
        embedding2: Union[np.ndarray, tf.Tensor],
    ) -> float:
        """
        Compute Euclidean distance between two embeddings.

        Note: For normalized embeddings, cosine similarity is preferred.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Euclidean distance.
        """
        # Convert to numpy
        if isinstance(embedding1, tf.Tensor):
            embedding1 = embedding1.numpy()
        if isinstance(embedding2, tf.Tensor):
            embedding2 = embedding2.numpy()

        # Flatten
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # Compute distance
        distance = np.linalg.norm(embedding1 - embedding2)

        return float(distance)


class FaceMatch:
    """Face matching and recognition using embeddings."""

    def __init__(self, threshold: float = 0.45):
        """
        Initialize face matcher.

        Args:
            threshold: Similarity threshold for positive match.
                      Default: 0.45 (suitable for cosine similarity).
        """
        self.threshold = threshold

    def match(
        self,
        query_embedding: Union[np.ndarray, tf.Tensor],
        template_embedding: Union[np.ndarray, tf.Tensor],
    ) -> Tuple[bool, float]:
        """
        Check if query embedding matches template embedding.

        Args:
            query_embedding: Query face embedding.
            template_embedding: Template face embedding.

        Returns:
            Tuple of (is_match, similarity_score).
        """
        similarity = SimilarityMetric.cosine_similarity(
            query_embedding, template_embedding
        )

        is_match = similarity >= self.threshold

        return is_match, similarity

    def identify(
        self,
        query_embedding: Union[np.ndarray, tf.Tensor],
        templates: dict,
    ) -> Tuple[str, float]:
        """
        Identify query embedding against a set of templates.

        Args:
            query_embedding: Query face embedding.
            templates: Dictionary mapping person_name -> embedding.

        Returns:
            Tuple of (identified_person, max_similarity).
                     Returns ("Unknown", similarity) if no match above threshold.
        """
        if not templates:
            return "Unknown", 0.0

        max_similarity = -2.0  # Start below the minimum possible similarity
        best_match = "Unknown"

        for person_name, template_emb in templates.items():
            similarity = SimilarityMetric.cosine_similarity(
                query_embedding, template_emb
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = person_name

        # Check if best match exceeds threshold
        if max_similarity < self.threshold:
            return "Unknown", max_similarity

        return best_match, max_similarity

    def identify_with_scores(
        self,
        query_embedding: Union[np.ndarray, tf.Tensor],
        templates: dict,
    ) -> dict:
        """
        Get similarity scores for all templates.

        Args:
            query_embedding: Query face embedding.
            templates: Dictionary mapping person_name -> embedding.

        Returns:
            Dictionary mapping person_name -> similarity_score.
        """
        scores = {}

        for person_name, template_emb in templates.items():
            similarity = SimilarityMetric.cosine_similarity(
                query_embedding, template_emb
            )
            scores[person_name] = float(similarity)

        return scores
