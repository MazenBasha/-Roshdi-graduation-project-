"""
Face verification evaluation utilities.

Implements:
- Face pair similarity computation (cosine, euclidean)
- ROC curve generation
- AUC-ROC calculation
- EER (Equal Error Rate) calculation
- Verification accuracy at different thresholds
- Standard LFW evaluation
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class FaceVerificationEvaluator:
    """Evaluate face recognition model on verification task."""

    def __init__(self, model: tf.keras.Model, similarity_metric: str = "cosine"):
        """
        Initialize evaluator.

        Args:
            model: The face recognition model (should return embeddings).
            similarity_metric: 'cosine' or 'euclidean'.
        """
        self.model = model
        self.similarity_metric = similarity_metric.lower()

        if self.similarity_metric not in ["cosine", "euclidean", "l2"]:
            raise ValueError(f"Unknown metric: {similarity_metric}")

    def compute_embeddings(
        self, images: np.ndarray, batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute embeddings for images.

        Args:
            images: Array of shape (N, H, W, 3) in range [-1, 1].
            batch_size: Batch size for inference.

        Returns:
            Embeddings of shape (N, embedding_size).
        """
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        embeddings_list = []
        for batch in dataset:
            # Get embeddings (specify return_embedding=True for inference mode)
            batch_embeddings = self.model(batch, training=False, return_embedding=True)
            embeddings_list.append(batch_embeddings.numpy())

        return np.concatenate(embeddings_list, axis=0)

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: Embedding vector.
            embedding2: Embedding vector.

        Returns:
            Similarity score (higher = more similar).
        """
        # Normalize embeddings
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-7)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-7)

        if self.similarity_metric == "cosine":
            # Cosine similarity: [-1, 1] -> rescale to [0, 1]
            similarity = np.dot(emb1, emb2)
            return (similarity + 1.0) / 2.0

        elif self.similarity_metric in ["euclidean", "l2"]:
            # L2 distance: smaller = more similar
            distance = np.linalg.norm(emb1 - emb2)
            # Convert to similarity (max distance is ~2 for normalized vectors)
            similarity = 1.0 - distance / 2.0
            return max(0.0, similarity)

    def evaluate_pairs(
        self,
        pair_paths: List[Tuple[str, str]],
        image_loader: callable,
        thresholds: Optional[List[float]] = None,
    ) -> Dict:
        """
        Evaluate on face pairs.

        Args:
            pair_paths: List of (path1, path2) tuples.
            image_loader: Function to load images from paths.
            thresholds: Thresholds to evaluate (if None, compute ROC).

        Returns:
            Dictionary with metrics.
        """
        print("\n[Evaluation] Computing similarities for pairs...")

        similarities = []
        labels = []

        for i, (path1, path2, label) in enumerate(pair_paths):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(pair_paths)} pairs")

            try:
                img1 = image_loader(path1)
                img2 = image_loader(path2)

                # Get embeddings
                emb1 = self.model(
                    tf.expand_dims(img1, 0), training=False, return_embedding=True
                )[0].numpy()
                emb2 = self.model(
                    tf.expand_dims(img2, 0), training=False, return_embedding=True
                )[0].numpy()

                similarity = self.compute_similarity(emb1, emb2)
                similarities.append(similarity)
                labels.append(label)
            except Exception as e:
                print(f"  Error processing pair {i}: {e}")
                continue

        similarities = np.array(similarities)
        labels = np.array(labels)

        # Compute metrics
        results = self._compute_metrics(similarities, labels, thresholds)

        return results

    def _compute_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> Dict:
        """Compute verification metrics."""
        results = {}

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)

        results["roc_auc"] = float(roc_auc)
        results["fpr"] = fpr.tolist()
        results["tpr"] = tpr.tolist()
        results["roc_thresholds"] = roc_thresholds.tolist()

        # EER
        eer = self._compute_eer(fpr, tpr)
        results["eer"] = float(eer)

        # Best threshold (at EER)
        best_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        best_threshold = roc_thresholds[best_idx]
        results["best_threshold_eer"] = float(best_threshold)

        # Standard thresholds
        if thresholds is None:
            thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            accuracy = np.mean(predictions == labels)
            results[f"accuracy@{threshold}"] = float(accuracy)

        # Additional metrics at best threshold
        predictions = (similarities >= best_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        results["best_accuracy"] = float(accuracy)
        results["best_precision"] = float(precision)
        results["best_recall"] = float(recall)
        results["best_f1"] = float(f1)

        # TPR@FPR thresholds
        fpr_thresholds = [0.001, 0.01, 0.1]
        for fpr_threshold in fpr_thresholds:
            valid_idx = fpr <= fpr_threshold
            if np.any(valid_idx):
                best_tpr = np.max(tpr[valid_idx])
                results[f"tpr@fpr_{fpr_threshold}"] = float(best_tpr)

        return results

    def _compute_eer(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Compute Equal Error Rate.

        Args:
            fpr: False Positive Rates.
            tpr: True Positive Rates.

        Returns:
            EER value.
        """
        # EER is where FPR == FNR (1 - TPR)
        fnr = 1 - tpr

        # Find point where FPR closest to FNR
        differences = np.abs(fpr - fnr)
        min_idx = np.argmin(differences)

        eer = (fpr[min_idx] + fnr[min_idx]) / 2.0

        return eer


def load_lfw_pairs(
    pairs_file: Path, lfw_root: Path
) -> Tuple[List[Tuple[str, str, int]], Dict]:
    """
    Load LFW verification pairs.

    Args:
        pairs_file: Path to pairs.txt file.
        lfw_root: Root directory of LFW dataset.

    Returns:
        Tuple of (pair_list, metadata).
    """
    pairs = []
    metadata = {"num_pairs": 0, "num_positive": 0, "num_negative": 0}

    with open(pairs_file, "r") as f:
        lines = f.readlines()

    # Skip header
    parts = lines[0].strip().split()
    num_pairs = int(parts[0])

    for line in lines[1:]:
        parts = line.strip().split()

        if len(parts) == 3:
            # Same person pair
            name = parts[0]
            idx1 = int(parts[1])
            idx2 = int(parts[2])

            path1 = lfw_root / name / f"{name}_{idx1:04d}.jpg"
            path2 = lfw_root / name / f"{name}_{idx2:04d}.jpg"

            if path1.exists() and path2.exists():
                pairs.append((str(path1), str(path2), 1))  # Same person
                metadata["num_positive"] += 1

        elif len(parts) == 4:
            # Different person pair
            name1 = parts[0]
            idx1 = int(parts[1])
            name2 = parts[2]
            idx2 = int(parts[3])

            path1 = lfw_root / name1 / f"{name1}_{idx1:04d}.jpg"
            path2 = lfw_root / name2 / f"{name2}_{idx2:04d}.jpg"

            if path1.exists() and path2.exists():
                pairs.append((str(path1), str(path2), 0))  # Different people
                metadata["num_negative"] += 1

    metadata["num_pairs"] = len(pairs)

    return pairs, metadata


def load_image_for_verification(image_path: str, img_size: int = 112) -> np.ndarray:
    """Load and preprocess image for verification."""
    image = tf.io.read_file(image_path)

    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except:
        try:
            image = tf.image.decode_png(image, channels=3)
        except:
            raise ValueError(f"Failed to decode image: {image_path}")

    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1.0

    return image.numpy()
