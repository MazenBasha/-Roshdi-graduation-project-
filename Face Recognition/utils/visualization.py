"""
Visualization utilities for face recognition.

Provides functions for visualizing training history, embeddings, and results.
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingVisualizer:
    """Visualize training metrics."""

    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot training history.

        Args:
            history: Dictionary with keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
            save_path: Optional path to save figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved training history plot to {save_path}")

        plt.show()

    @staticmethod
    def plot_embedding_space(
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot embedding space using dimensionality reduction.

        Args:
            embeddings: Embedding vectors (N, D).
            labels: Class labels (N,).
            method: Dimensionality reduction method ('tsne' or 'pca').
            save_path: Optional path to save figure.
        """
        try:
            if method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, verbose=1, random_state=42)
            elif method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
            else:
                raise ValueError(f"Unknown method: {method}")

            reduced = reducer.fit_transform(embeddings)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=labels,
                cmap='tab20',
                alpha=0.6,
            )
            plt.colorbar(scatter, label='Class')
            plt.xlabel(f'{method.upper()} 1')
            plt.ylabel(f'{method.upper()} 2')
            plt.title(f'Embedding Space ({method.upper()})')
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Saved embedding plot to {save_path}")

            plt.show()

        except ImportError:
            print(f"Warning: sklearn not available for {method} visualization")

    @staticmethod
    def plot_similarity_matrix(
        embeddings: np.ndarray,
        labels: np.ndarray,
        top_k: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot similarity matrix between class prototypes.

        Args:
            embeddings: Embedding vectors (N, D).
            labels: Class labels (N,).
            top_k: Number of top classes to show.
            save_path: Optional path to save figure.
        """
        # Compute class prototypes
        unique_labels = np.unique(labels)[:top_k]
        prototypes = []

        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            prototype = class_embeddings.mean(axis=0)
            prototypes.append(prototype)

        prototypes = np.array(prototypes)

        # Normalize
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        similarity = np.dot(prototypes, prototypes.T)

        # Plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(similarity, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.title(f'Class Similarity Matrix (Top {top_k})')

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved similarity matrix to {save_path}")

        plt.show()


class RecognitionVisualizer:
    """Visualize recognition results."""

    @staticmethod
    def show_match_results(
        query_image: np.ndarray,
        template_images: List[np.ndarray],
        similarities: List[float],
        person_names: List[str],
        threshold: float = 0.45,
    ) -> None:
        """
        Show recognition results with images and similarity scores.

        Args:
            query_image: Query face image.
            template_images: List of template face images.
            similarities: Similarity scores.
            person_names: Names of persons.
            threshold: Matching threshold.
        """
        n_templates = len(template_images)
        fig, axes = plt.subplots(1, n_templates + 1, figsize=(15, 3))

        # Show query image
        if query_image.max() > 1:
            query_image = query_image / 255.0
        axes[0].imshow(query_image)
        axes[0].set_title("Query")
        axes[0].axis("off")

        # Show templates with scores
        for i, (template_img, similarity, name) in enumerate(
            zip(template_images, similarities, person_names)
        ):
            if template_img.max() > 1:
                template_img = template_img / 255.0

            color = "green" if similarity >= threshold else "red"
            axes[i + 1].imshow(template_img)
            axes[i + 1].set_title(
                f"{name}\n{similarity:.3f}",
                color=color,
                fontweight="bold",
            )
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_recognition_scores(
        scores: Dict[str, float],
        threshold: float = 0.45,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot recognition scores as bar chart.

        Args:
            scores: Dictionary mapping person_name -> similarity_score.
            threshold: Matching threshold (shown as horizontal line).
            save_path: Optional path to save figure.
        """
        names = list(scores.keys())
        values = list(scores.values())

        # Sort by score
        sorted_pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_pairs)

        colors = ["green" if v >= threshold else "red" for v in values]

        plt.figure(figsize=(10, 6))
        plt.bar(names, values, color=colors, alpha=0.7)
        plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel('Person')
        plt.ylabel('Similarity Score')
        plt.title('Face Recognition Scores')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved scores plot to {save_path}")

        plt.show()
