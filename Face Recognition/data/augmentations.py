"""
Data augmentation utilities for face recognition training.

Provides augmentation functions suitable for face recognition:
- Random brightness
- Random contrast
- Random horizontal flip
- Random rotation
- Random zoom/crop
"""

from typing import Callable
import tensorflow as tf


def random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """
    Apply random brightness adjustment.

    Args:
        image: Input image tensor in range [-1, 1] or [0, 1].
        max_delta: Maximum brightness delta (0-1 scale).

    Returns:
        Image with adjusted brightness.
    """
    return tf.image.random_brightness(image, max_delta)


def random_contrast(
    image: tf.Tensor, lower: float = 0.8, upper: float = 1.2
) -> tf.Tensor:
    """
    Apply random contrast adjustment.

    Args:
        image: Input image tensor.
        lower: Lower contrast factor.
        upper: Upper contrast factor.

    Returns:
        Image with adjusted contrast.
    """
    return tf.image.random_contrast(image, lower, upper)


def random_flip_left_right(image: tf.Tensor, probability: float = 0.5) -> tf.Tensor:
    """
    Apply random horizontal flip.

    Args:
        image: Input image tensor.
        probability: Probability of flipping.

    Returns:
        Flipped image with given probability.
    """
    if tf.random.uniform(()) < probability:
        return tf.image.flip_left_right(image)
    return image


def random_rotation(image: tf.Tensor, rotation_range: int = 10) -> tf.Tensor:
    """
    Apply random rotation using TensorFlow ops.

    Note: Limited rotation support due to TensorFlow limitations.
    For now, we use a simple random crop instead.

    Args:
        image: Input image tensor of shape (height, width, channels).
        rotation_range: Maximum rotation in degrees (unused, for compatibility).

    Returns:
        Augmented image.
    """
    # Skip rotation due to TensorFlow limitations with rotate op in graph mode
    # Instead, apply a slight random crop and resize for similar augmentation effect
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    
    crop_size = tf.cast(tf.cast(height, tf.float32) * 0.9, tf.int32)
    offset = tf.cast((tf.cast(height, tf.float32) - tf.cast(crop_size, tf.float32)) / 2.0, tf.int32)
    
    cropped = tf.image.crop_to_bounding_box(image, offset, offset, crop_size, crop_size)
    resized = tf.image.resize(cropped, [height, width])
    
    return resized


def random_zoom(image: tf.Tensor, zoom_range: float = 0.1) -> tf.Tensor:
    """
    Apply random zoom/crop.

    Args:
        image: Input image tensor of shape (height, width, channels).
        zoom_range: Maximum zoom factor (0-1).

    Returns:
        Zoomed image.
    """
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    # Random zoom factor
    zoom_factor = tf.random.uniform(
        [],
        minval=1.0 - zoom_range,
        maxval=1.0 + zoom_range,
        dtype=tf.float32,
    )

    # New crop size
    crop_height = tf.cast(
        tf.cast(height, tf.float32) / zoom_factor, tf.int32
    )
    crop_width = tf.cast(
        tf.cast(width, tf.float32) / zoom_factor, tf.int32
    )

    # Ensure crop size doesn't exceed image size
    crop_height = tf.minimum(crop_height, height)
    crop_width = tf.minimum(crop_width, width)

    # Ensure crop size is positive
    crop_height = tf.maximum(crop_height, 1)
    crop_width = tf.maximum(crop_width, 1)

    # Random offset
    offset_height = tf.random.uniform(
        [], 0, tf.maximum(height - crop_height + 1, 1), dtype=tf.int32
    )
    offset_width = tf.random.uniform(
        [], 0, tf.maximum(width - crop_width + 1, 1), dtype=tf.int32
    )

    # Crop
    cropped = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width
    )

    # Resize back to original size
    resized = tf.image.resize(cropped, [height, width])

    return resized


def get_augmentation_pipeline(
    brightness_delta: float = 0.2,
    contrast_factor: tuple = (0.8, 1.2),
    rotation_range: int = 10,
    zoom_range: float = 0.1,
    flip_probability: float = 0.5,
) -> Callable:
    """
    Create an augmentation pipeline function.

    Args:
        brightness_delta: Brightness delta for random_brightness.
        contrast_factor: Tuple of (lower, upper) for contrast.
        rotation_range: Rotation range in degrees.
        zoom_range: Zoom range.
        flip_probability: Probability of horizontal flip.

    Returns:
        Function that applies augmentations to an image.
    """

    def augment(image: tf.Tensor) -> tf.Tensor:
        """Apply augmentation pipeline to image."""
        # Random brightness
        image = random_brightness(image, brightness_delta)

        # Random contrast
        image = random_contrast(image, contrast_factor[0], contrast_factor[1])

        # Random horizontal flip
        image = random_flip_left_right(image, flip_probability)

        # Random rotation
        image = random_rotation(image, rotation_range)

        # Random zoom
        image = random_zoom(image, zoom_range)

        # Slight Gaussian noise
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
        image = image + noise

        # Clip to valid range
        image = tf.clip_by_value(image, -1.0, 1.0)

        return image

    return augment


def get_light_augmentation_pipeline() -> Callable:
    """
    Create a light augmentation pipeline (suitable for validation).

    Returns:
        Function that applies light augmentations.
    """

    def augment(image: tf.Tensor) -> tf.Tensor:
        """Apply light augmentations."""
        # Only horizontal flip on validation
        image = random_flip_left_right(image, 0.5)
        return image

    return augment
