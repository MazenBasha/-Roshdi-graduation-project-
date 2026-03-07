"""
Image preprocessing utilities for face recognition.

Handles image loading, resizing, and normalization.
"""

from typing import Tuple
import tensorflow as tf
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file.

    Returns:
        Numpy array of the image in RGB format.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be read.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def resize_image(image: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Resize image to specified size.

    Args:
        image: Input image as numpy array.
        size: Target size (size x size).

    Returns:
        Resized image as numpy array.
    """
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)

    resized = tf.image.resize(image, [size, size])
    return resized.numpy().astype(np.uint8)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize image to [-1, 1] range.

    Args:
        image: Input image tensor (values in [0, 255] or [0, 1]).

    Returns:
        Normalized image in [-1, 1] range.
    """
    # Ensure image is float32
    image = tf.cast(image, tf.float32)

    # If values are in [0, 255], divide by 127.5 and subtract 1
    if tf.reduce_max(image) > 1.0:
        image = image / 127.5 - 1.0
    else:
        # If already in [0, 1], scale to [-1, 1]
        image = image * 2.0 - 1.0

    return image


def preprocess_image(
    image_path: str,
    size: int = 112,
    normalize: bool = True,
) -> tf.Tensor:
    """
    Complete preprocessing pipeline for a single image.

    Args:
        image_path: Path to image file.
        size: Target size.
        normalize: Whether to normalize to [-1, 1].

    Returns:
        Preprocessed image tensor of shape (size, size, 3).
    """
    # Load image
    image = load_image(image_path)

    # Convert to tensor
    image = tf.convert_to_tensor(image, dtype=tf.uint8)

    # Resize
    image = tf.image.resize(image, [size, size])

    # Normalize
    if normalize:
        image = normalize_image(image)

    return image


def batch_preprocess_images(
    image_paths: list,
    size: int = 112,
    normalize: bool = True,
) -> Tuple[tf.Tensor, list]:
    """
    Preprocess a batch of images.

    Args:
        image_paths: List of image file paths.
        size: Target size.
        normalize: Whether to normalize.

    Returns:
        Tuple of:
            - Batch tensor of shape (batch_size, size, size, 3)
            - List of paths that failed to process
    """
    images = []
    failed_paths = []

    for path in image_paths:
        try:
            image = preprocess_image(path, size=size, normalize=normalize)
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to process {path}: {str(e)}")
            failed_paths.append(path)

    if not images:
        raise ValueError("No images could be processed")

    batch = tf.stack(images)
    return batch, failed_paths
