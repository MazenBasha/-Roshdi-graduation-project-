"""
Metrics utilities for face recognition evaluation.

Computes metrics like accuracy, precision, recall for classification
and provides face-specific metrics like TAR/FAR.
"""

from typing import Tuple
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FaceRecognitionMetrics:
    """Compute metrics for face recognition models."""

    @staticmethod
    def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            y_true: True class indices.
            y_pred: Predicted class indices.

        Returns:
            Accuracy as float [0, 1].
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
        """
        Compute precision.

        Args:
            y_true: True class indices.
            y_pred: Predicted class indices.
            average: Averaging method ("weighted", "macro", "micro").

        Returns:
            Precision score.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
        """
        Compute recall.

        Args:
            y_true: True class indices.
            y_pred: Predicted class indices.
            average: Averaging method.

        Returns:
            Recall score.
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
        """
        Compute F1 score.

        Args:
            y_true: True class indices.
            y_pred: Predicted class indices.
            average: Averaging method.

        Returns:
            F1 score.
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """
        Compute all classification metrics.

        Args:
            y_true: True class indices.
            y_pred: Predicted class indices.

        Returns:
            Dictionary of metrics.
        """
        return {
            "accuracy": FaceRecognitionMetrics.classification_accuracy(y_true, y_pred),
            "precision": FaceRecognitionMetrics.precision(y_true, y_pred),
            "recall": FaceRecognitionMetrics.recall(y_true, y_pred),
            "f1_score": FaceRecognitionMetrics.f1_score(y_true, y_pred),
        }


class EmbeddingMetrics:
    """Metrics for face embeddings."""

    @staticmethod
    def cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embeddings1: First embedding or batch of embeddings (N, D).
            embeddings2: Second embedding or batch of embeddings (N, D).

        Returns:
            Cosine similarity score(s) in [-1, 1].
        """
        # Normalize embeddings
        emb1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-8)
        emb2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.sum(emb1 * emb2, axis=-1)

        return similarity

    @staticmethod
    def intra_class_distance(embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute average intra-class distances (should be small).

        Args:
            embeddings: Embedding vectors (N, D).
            labels: Class labels (N,).

        Returns:
            Dictionary of per-class average distances.
        """
        unique_labels = np.unique(labels)
        intra_distances = {}

        for label in unique_labels:
            class_embeddings = embeddings[labels == label]

            if len(class_embeddings) < 2:
                continue

            # Compute pairwise distances
            distances = []
            for i in range(len(class_embeddings)):
                for j in range(i + 1, len(class_embeddings)):
                    dist = 1.0 - EmbeddingMetrics.cosine_similarity(
                        class_embeddings[i:i+1],
                        class_embeddings[j:j+1],
                    )[0]
                    distances.append(dist)

            if distances:
                intra_distances[int(label)] = np.mean(distances)

        return intra_distances

    @staticmethod
    def inter_class_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute average inter-class distance (should be large).

        Args:
            embeddings: Embedding vectors (N, D).
            labels: Class labels (N,).

        Returns:
            Average inter-class distance.
        """
        unique_labels = np.unique(labels)
        distances = []

        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                emb1 = embeddings[labels == label1].mean(axis=0)
                emb2 = embeddings[labels == label2].mean(axis=0)

                dist = 1.0 - EmbeddingMetrics.cosine_similarity(
                    emb1[np.newaxis, :],
                    emb2[np.newaxis, :],
                )[0]
                distances.append(dist)

        return np.mean(distances) if distances else 0.0


class TrainingMetrics:
    """Metrics and callbacks for training monitoring."""

    @staticmethod
    def create_metrics_dict() -> dict:
        """Create metrics dictionary for training."""
        return {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    @staticmethod
    def compute_top_k_accuracy(
        y_true: np.ndarray, y_pred_logits: np.ndarray, k: int = 5
    ) -> float:
        """
        Compute top-k accuracy.

        Args:
            y_true: True labels.
            y_pred_logits: Predicted logits (batch_size, num_classes).
            k: Top-k to consider.

        Returns:
            Top-k accuracy.
        """
        top_k_pred = np.argsort(y_pred_logits, axis=1)[:, -k:]
        correct = sum(np.any(top_k_pred == y_true[:, np.newaxis], axis=1))
        return correct / len(y_true)
