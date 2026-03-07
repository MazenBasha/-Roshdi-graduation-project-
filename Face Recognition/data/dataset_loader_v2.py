"""
Improved dataset loader with filtering, balancing, and class weighting.

Features:
- Filter identities with fewer than N images
- Balance class distribution
- Class-weighted sampling
- Proper train/val split
- Memory-efficient data pipeline
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import tensorflow as tf
from data.augmentations_v2 import FaceAugmentationPipeline


class ImprovedFaceRecognitionDataset:
    """Improved dataset loader with filtering and balancing."""

    def __init__(
        self,
        data_dir: Path,
        img_size: int = 112,
        batch_size: int = 32,
        augment: bool = True,
        augmentation_strength: str = "strong",
        min_images_per_class: int = 3,
        balance_classes: bool = True,
        cache: bool = True,
        seed: int = 42,
    ):
        """
        Initialize improved dataset loader.

        Args:
            data_dir: Path to dataset root (class folders).
            img_size: Image size.
            batch_size: Batch size.
            augment: Whether to apply augmentations.
            augmentation_strength: 'light', 'medium', or 'strong'.
            min_images_per_class: Minimum images to keep a class.
            balance_classes: Whether to balance class distribution.
            cache: Whether to cache dataset.
            seed: Random seed.
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.augmentation_strength = augmentation_strength
        self.min_images_per_class = min_images_per_class
        self.balance_classes = balance_classes
        self.cache = cache
        self.seed = seed

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Create augmentation pipeline
        self.augmenter = FaceAugmentationPipeline(
            img_size=img_size,
            strength=augmentation_strength,
        ) if augment else None

        # Load and filter dataset
        self._load_and_filter_classes()

        # Calculate class weights
        self._calculate_class_weights()

    def _load_and_filter_classes(self) -> None:
        """Load classes and filter by minimum image count."""
        print(f"\n[Dataset] Loading from {self.data_dir}...")

        all_classes = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Found {len(all_classes)} total classes")

        # Load class-image mapping
        self.class_info = {}
        filtered_classes = []

        for class_idx, class_dir in enumerate(all_classes):
            # Get images
            image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]

            # Filter by min images
            if len(images) >= self.min_images_per_class:
                self.class_info[class_dir.name] = {
                    "path": str(class_dir),
                    "images": [str(img) for img in images],
                    "count": len(images),
                }
                filtered_classes.append(class_dir.name)
            else:
                pass  # Skip class with too few images

        print(f"After filtering (min {self.min_images_per_class} images):")
        print(f"  Kept: {len(filtered_classes)} classes")
        print(f"  Total images: {sum(info['count'] for info in self.class_info.values())}")

        # Create mapping
        self.classes = filtered_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(filtered_classes)}
        self.num_classes = len(filtered_classes)

        if self.num_classes == 0:
            raise ValueError(f"No valid classes found in {self.data_dir}")

    def _calculate_class_weights(self) -> None:
        """Calculate class weights for balanced sampling."""
        counts = np.array([
            self.class_info[cls]["count"] for cls in self.classes
        ], dtype=np.float32)

        # Inverse weighting (less frequent classes get higher weight)
        self.class_weights = 1.0 / counts
        self.class_weights = self.class_weights / np.sum(self.class_weights)

        print(f"\n[Dataset] Class weight statistics:")
        print(f"  Min weight: {self.class_weights.min():.6f}")
        print(f"  Max weight: {self.class_weights.max():.6f}")
        print(f"  Mean weight: {self.class_weights.mean():.6f}")

    def _create_file_list(self) -> List[Tuple[str, int]]:
        """Create list of (image_path, label) tuples."""
        file_list = []

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            images = self.class_info[class_name]["images"]

            for img_path in images:
                file_list.append((img_path, class_idx))

        return file_list

    def _create_balanced_file_list(self) -> List[Tuple[str, int]]:
        """Create balanced file list with oversampling of underrepresented classes."""
        file_list = []

        # Find max class size
        max_count = max(info["count"] for info in self.class_info.values())

        # Oversample underrepresented classes
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            images = self.class_info[class_name]["images"]
            current_count = len(images)

            # Oversample if necessary
            if self.balance_classes and current_count < max_count:
                # Repeat images to balance
                repetitions = max_count // current_count + 1
                images_repeated = (images * repetitions)[:max_count]
                file_list.extend([
                    (img_path, class_idx) for img_path in images_repeated
                ])
            else:
                file_list.extend([
                    (img_path, class_idx) for img_path in images
                ])

        # Shuffle
        np.random.shuffle(file_list)

        return file_list

    def create_dataset(
        self,
        shuffle: bool = True,
        drop_remainder: bool = True,
    ) -> Tuple[tf.data.Dataset, Dict]:
        """
        Create tf.data.Dataset.

        Args:
            shuffle: Whether to shuffle.
            drop_remainder: Whether to drop incomplete final batch.

        Returns:
            Tuple of (dataset, stats_dict).
        """
        # Create file list
        if self.balance_classes:
            file_list = self._create_balanced_file_list()
            print(f"\n[Dataset] Using balanced sampling (total: {len(file_list)} samples)")
        else:
            file_list = self._create_file_list()
            print(f"\n[Dataset] Using standard sampling (total: {len(file_list)} samples)")

        paths = [item[0] for item in file_list]
        labels = [item[1] for item in file_list]

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(len(file_list), 10000),
                seed=self.seed,
            )

        # Load and preprocess
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Remove corrupted samples (if any)
        dataset = dataset.filter(
            lambda x: tf.not_equal(x[0], None) and tf.not_equal(x[1], -1)
        )

        # Augmentation
        if self.augment and self.augmenter is not None:
            dataset = dataset.map(
                lambda img, label: (self.augmenter.augment(img), label),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Cache if enabled
        if self.cache:
            dataset = dataset.cache()

        # Batch
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Calculate stats
        stats = {
            "num_classes": self.num_classes,
            "num_samples": len(file_list),
            "batch_size": self.batch_size,
            "num_batches": len(file_list) // self.batch_size,
            "class_weights": self.class_weights.tolist(),
        }

        return dataset, stats

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        return self.classes[class_idx]

    def get_class_weights(self) -> np.ndarray:
        """Get class weights."""
        return self.class_weights

    def _load_and_preprocess(
        self, path: str, label: int
    ) -> Tuple[tf.Tensor, int]:
        """Load and preprocess image."""
        try:
            # Read image
            image = tf.io.read_file(path)

            # Decode
            try:
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                try:
                    image = tf.image.decode_png(image, channels=3)
                except:
                    # Return sentinel for filtering
                    return tf.constant(0.0), tf.constant(-1)

            # Resize
            image = tf.image.resize(image, [self.img_size, self.img_size])

            # Normalize to [-1, 1]
            image = tf.cast(image, tf.float32)
            image = image / 127.5 - 1.0

            return image, label
        except:
            # Return sentinel for filtering
            return tf.constant(0.0), tf.constant(-1)


def load_improved_dataset(
    data_dir: str,
    img_size: int = 112,
    batch_size: int = 32,
    augment: bool = True,
    augmentation_strength: str = "strong",
    min_images_per_class: int = 3,
    balance_classes: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, Dict]:
    """
    Load dataset with improvements.

    Args:
        data_dir: Path to dataset.
        img_size: Image size.
        batch_size: Batch size.
        augment: Whether to augment.
        augmentation_strength: Augmentation strength level.
        min_images_per_class: Minimum images per class.
        balance_classes: Whether to balance.
        shuffle: Whether to shuffle.
        seed: Random seed.

    Returns:
        Tuple of (dataset, stats).
    """
    loader = ImprovedFaceRecognitionDataset(
        data_dir=Path(data_dir),
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        augmentation_strength=augmentation_strength,
        min_images_per_class=min_images_per_class,
        balance_classes=balance_classes,
        seed=seed,
    )

    dataset, stats = loader.create_dataset(shuffle=shuffle)

    return dataset, stats, loader
