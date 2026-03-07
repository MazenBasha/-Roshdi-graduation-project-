"""
Dataset loader for face recognition training.

Loads face recognition datasets in folder-per-class format and creates
tf.data pipelines with augmentation support.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from data.preprocessing import normalize_image
from data.augmentations import get_augmentation_pipeline, get_light_augmentation_pipeline


class FaceRecognitionDataset:
    """Load face recognition dataset and create tf.data pipelines."""

    def __init__(
        self,
        data_dir: Path,
        img_size: int = 112,
        batch_size: int = 32,
        augment: bool = True,
        cache: bool = True,
    ):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to dataset root (containing class folders).
            img_size: Size of images to load.
            batch_size: Batch size for training.
            augment: Whether to apply augmentations.
            cache: Whether to cache dataset in memory.
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.cache = cache

        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        # Get list of classes (folders)
        self.classes = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.classes)

        if self.num_classes == 0:
            raise ValueError(f"No classes found in {self.data_dir}")

        # Map class folder names to indices
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(self.classes)}

        print(f"Found {self.num_classes} classes in {self.data_dir}")

    def _load_and_preprocess_image(self, path: str, label: int) -> Tuple[tf.Tensor, int]:
        """
        Load and preprocess an image.

        Args:
            path: Path to image file.
            label: Class label.

        Returns:
            Tuple of (image, label).
        """
        # Read image file
        image = tf.io.read_file(path)

        # Decode image (handles JPG, PNG, etc.)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            try:
                image = tf.image.decode_png(image, channels=3)
            except:
                # Skip corrupted images
                return None, None

        # Resize
        image = tf.image.resize(image, [self.img_size, self.img_size])

        # Normalize to [-1, 1]
        image = normalize_image(image)

        return image, label

    def _augment_image(self, image: tf.Tensor, label: int) -> Tuple[tf.Tensor, int]:
        """
        Apply augmentations to image.

        Args:
            image: Input image (in range [-1, 1]).
            label: Class label.

        Returns:
            Augmented (image, label).
        """
        # Ensure image is in the right dtype
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        
        # Random brightness (works in [-1, 1] range)
        image = tf.image.random_brightness(image, 0.15)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.85, 1.15)
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Clip to valid range
        image = tf.clip_by_value(image, -1.0, 1.0)
        
        return image, label

    def _create_file_list(self) -> List[Tuple[str, int]]:
        """
        Create list of (image_path, label) tuples.

        Returns:
            List of (path, label) tuples.
        """
        file_list = []

        for class_idx, class_dir in enumerate(self.classes):
            # Get all image files in class folder
            image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]

            if len(image_files) == 0:
                print(f"Warning: No images found in {class_dir}")
                continue

            for img_path in image_files:
                file_list.append((str(img_path), class_idx))

        return file_list

    def create_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create tf.data.Dataset for training or validation.

        Args:
            shuffle: Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset with (image, label) pairs.
        """
        # Create file list
        file_list = self._create_file_list()

        if len(file_list) == 0:
            raise ValueError("No image files found in dataset")

        # Convert to tensor dataset
        paths = [item[0] for item in file_list]
        labels = [item[1] for item in file_list]

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(file_list))

        # Load and preprocess images
        def load_and_preprocess(path, label):
            """Load and preprocess single image."""
            image = tf.io.read_file(path)
            
            # Decode image
            try:
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                try:
                    image = tf.image.decode_png(image, channels=3)
                except:
                    # Return dummy image on error
                    image = tf.zeros((self.img_size, self.img_size, 3), dtype=tf.uint8)
            
            # Resize
            image = tf.image.resize(image, [self.img_size, self.img_size])
            
            # Normalize to [-1, 1]
            image = tf.cast(image, tf.float32)
            image = image / 127.5 - 1.0
            
            return image, label

        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Set shape information explicitly
        dataset = dataset.map(
            lambda img, label: (
                tf.ensure_shape(img, [self.img_size, self.img_size, 3]),
                label,
            )
        )

        # Apply augmentations if enabled
        if self.augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Cache if requested
        if self.cache:
            dataset = dataset.cache()

        # Batch
        dataset = dataset.batch(self.batch_size)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        return self.classes[class_idx].name

    def get_class_index(self, class_name: str) -> int:
        """Get class index from name."""
        return self.class_to_idx[class_name]


def load_dataset(
    data_dir: str,
    img_size: int = 112,
    batch_size: int = 32,
    augment: bool = True,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, int]:
    """
    Convenience function to load dataset.

    Args:
        data_dir: Path to dataset directory.
        img_size: Image size.
        batch_size: Batch size.
        augment: Whether to augment.
        shuffle: Whether to shuffle.

    Returns:
        Tuple of (dataset, num_classes).
    """
    dataset_loader = FaceRecognitionDataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
    )

    dataset = dataset_loader.create_dataset(shuffle=shuffle)

    return dataset, dataset_loader.num_classes, dataset_loader
