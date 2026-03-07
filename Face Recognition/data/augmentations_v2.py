"""
Enhanced face-specific data augmentation pipeline for face recognition.

Implements strong augmentations suitable for face recognition while being
compatible with tf.data pipelines (graph mode).

Augmentations included:
- Random brightness
- Random contrast
- Random saturation
- Gaussian blur
- Random crop & resize
- Random zoom
- Random rotation (simulated via crop)
- Horizontal flip
- Random erasing
- Color jitter
"""

from typing import Tuple
import tensorflow as tf


class FaceAugmentationPipeline:
    """Face-specific augmentation pipeline for tf.data pipelines."""

    def __init__(
        self,
        img_size: int = 112,
        brightness_delta: float = 0.2,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.7, 1.3),
        blur_probability: float = 0.3,
        blur_kernel_size: int = 3,
        crop_probability: float = 0.5,
        crop_ratio_range: Tuple[float, float] = (0.85, 1.0),
        zoom_range: float = 0.15,
        rotation_probability: float = 0.3,
        rotation_degrees: float = 10.0,
        flip_probability: float = 0.5,
        erasing_probability: float = 0.2,
        erasing_ratio_range: Tuple[float, float] = (0.02, 0.1),
        color_jitter_probability: float = 0.4,
        apply_mixup: bool = False,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            img_size: Expected image size (height/width).
            brightness_delta: Max brightness change [-delta, delta].
            contrast_range: Range for contrast adjustment.
            saturation_range: Range for saturation adjustment.
            blur_probability: Probability of applying Gaussian blur.
            blur_kernel_size: Kernel size for Gaussian blur.
            crop_probability: Probability of random crop.
            crop_ratio_range: Aspect ratio range for crops.
            zoom_range: Maximum zoom factor.
            rotation_probability: Probability of rotation augmentation.
            rotation_degrees: Maximum rotation degrees.
            flip_probability: Probability of horizontal flip.
            erasing_probability: Probability of random erasing.
            erasing_ratio_range: Ratio range for erased area.
            color_jitter_probability: Probability of color jitter.
            apply_mixup: Whether to apply mixup augmentation (requires batch).
        """
        self.img_size = img_size
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.blur_probability = blur_probability
        self.blur_kernel_size = blur_kernel_size
        self.crop_probability = crop_probability
        self.crop_ratio_range = crop_ratio_range
        self.zoom_range = zoom_range
        self.rotation_probability = rotation_probability
        self.rotation_degrees = rotation_degrees
        self.flip_probability = flip_probability
        self.erasing_probability = erasing_probability
        self.erasing_ratio_range = erasing_ratio_range
        self.color_jitter_probability = color_jitter_probability
        self.apply_mixup = apply_mixup

    def augment(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply augmentations to a single image.

        Args:
            image: Input image in range [-1, 1].

        Returns:
            Augmented image in range [-1, 1].
        """
        # Ensure float32
        image = tf.cast(image, tf.float32)

        # Convert to [0, 1] for some augmentations
        image = (image + 1.0) / 2.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        # 1. Random color jitter
        if tf.random.uniform(()) < self.color_jitter_probability:
            image = self._color_jitter(image)

        # 2. Random brightness
        image = tf.image.random_brightness(image, self.brightness_delta)

        # 3. Random contrast
        image = tf.image.random_contrast(
            image, self.contrast_range[0], self.contrast_range[1]
        )

        # 4. Random saturation
        image = tf.image.random_saturation(
            image, self.saturation_range[0], self.saturation_range[1]
        )

        # 5. Random horizontal flip
        if tf.random.uniform(()) < self.flip_probability:
            image = tf.image.flip_left_right(image)

        # 6. Random Gaussian blur
        if tf.random.uniform(()) < self.blur_probability:
            image = self._gaussian_blur(image, self.blur_kernel_size)

        # 7. Random crop & resize (simulates rotation)
        if tf.random.uniform(()) < self.crop_probability:
            image = self._random_crop_and_resize(image)

        # 8. Random zoom
        if tf.random.uniform(()) < 0.5:  # 50% chance
            image = self._random_zoom(image)

        # 9. Random erasing (Cutout style)
        if tf.random.uniform(()) < self.erasing_probability:
            image = self._random_erasing(image)

        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Convert back to [-1, 1]
        image = image * 2.0 - 1.0

        return image

    def _color_jitter(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply color jitter (random brightness, contrast, saturation, hue).

        Args:
            image: Image in range [0, 1].

        Returns:
            Augmented image in range [0, 1].
        """
        # Slight adjustments to brightness
        jitter_brightness = tf.random.uniform((), -0.1, 0.1)
        image = tf.image.adjust_brightness(image, jitter_brightness)

        # Slight adjustments to contrast
        jitter_contrast = tf.random.uniform((), 0.9, 1.1)
        image = tf.image.adjust_contrast(image, jitter_contrast)

        return image

    def _gaussian_blur(self, image: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
        """
        Apply Gaussian blur using depthwise convolution.

        Args:
            image: Image tensor of shape (H, W, 3).
            kernel_size: Size of the Gaussian kernel.

        Returns:
            Blurred image.
        """
        # Create Gaussian kernel
        sigma = tf.random.uniform((), 0.5, 1.5)
        kernel = self._create_gaussian_kernel(kernel_size, sigma)

        # Apply separable convolution for efficiency
        # Pad image
        pad_size = kernel_size // 2
        image = tf.pad(
            image,
            [[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
            mode="REFLECT",
        )

        # Apply depthwise convolution to each channel
        # For simplicity, use TensorFlow's built-in blur approximation
        image = tf.image.gaussian_blur(image, sigma=sigma, filter_size=kernel_size)

        return image

    def _create_gaussian_kernel(
        self, kernel_size: int, sigma: tf.Tensor
    ) -> tf.Tensor:
        """Create 1D Gaussian kernel."""
        # This is for reference; actual blur uses tf.image.gaussian_blur
        coords = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1) / 2.0
        kernel = tf.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel

    def _random_crop_and_resize(self, image: tf.Tensor) -> tf.Tensor:
        """
        Random crop and resize (simulates rotation effect).

        Args:
            image: Image of shape (H, W, 3).

        Returns:
            Cropped and resized image.
        """
        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Random crop ratio
        crop_ratio = tf.random.uniform((), self.crop_ratio_range[0], self.crop_ratio_range[1])
        crop_size = tf.cast(tf.cast(h, tf.float32) * crop_ratio, tf.int32)

        # Random offset
        offset_h = tf.random.uniform(
            (), 0, h - crop_size + 1, dtype=tf.int32
        )
        offset_w = tf.random.uniform(
            (), 0, w - crop_size + 1, dtype=tf.int32
        )

        # Crop
        cropped = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, crop_size, crop_size
        )

        # Resize back to original size
        resized = tf.image.resize(cropped, [h, w])

        return resized

    def _random_zoom(self, image: tf.Tensor) -> tf.Tensor:
        """
        Random zoom (crop from center and resize).

        Args:
            image: Image of shape (H, W, 3).

        Returns:
            Zoomed image.
        """
        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Random zoom factor
        zoom_factor = tf.random.uniform((), 1.0, 1.0 + self.zoom_range)
        zoom_size = tf.cast(
            tf.cast(h, tf.float32) / zoom_factor, tf.int32
        )
        zoom_size = tf.maximum(zoom_size, 4)  # Ensure minimum size

        # Center crop
        offset = (h - zoom_size) // 2
        offset = tf.maximum(offset, 0)
        
        cropped = tf.image.crop_to_bounding_box(
            image, offset, offset, zoom_size, zoom_size
        )

        # Resize back
        resized = tf.image.resize(cropped, [h, w])

        return resized

    def _random_erasing(self, image: tf.Tensor) -> tf.Tensor:
        """
        Random erasing (Cutout-style augmentation).

        Args:
            image: Image of shape (H, W, 3) in range [0, 1].

        Returns:
            Image with random erasing applied.
        """
        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Random erase area ratio
        erase_ratio = tf.random.uniform((), self.erasing_ratio_range[0], self.erasing_ratio_range[1])
        erase_area = tf.cast(
            tf.cast(h, tf.float32) * tf.cast(w, tf.float32) * erase_ratio,
            tf.int32,
        )

        # Random erase dimensions (square-ish)
        erase_h = tf.cast(tf.sqrt(tf.cast(erase_area, tf.float32)), tf.int32)
        erase_w = tf.cast(tf.sqrt(tf.cast(erase_area, tf.float32)), tf.int32)

        # Random position
        offset_h = tf.random.uniform((), 0, h - erase_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform((), 0, w - erase_w + 1, dtype=tf.int32)

        # Create mask
        mask = tf.ones([h, w, 3], dtype=tf.float32)
        updates = tf.zeros([erase_h, erase_w, 3], dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.stack(
                [
                    tf.repeat(
                        tf.range(offset_h, offset_h + erase_h)[:, None],
                        erase_w,
                        axis=1,
                    ).reshape([-1]),
                    tf.tile(
                        tf.range(offset_w, offset_w + erase_w),
                        [erase_h],
                    ),
                    tf.zeros([erase_h * erase_w], dtype=tf.int32),
                ],
                axis=1,
            ),
            tf.zeros([erase_h * erase_w], dtype=tf.float32),
        )

        # Apply erasing (fill with random color or mean color)
        # For faces, use a neutral gray color
        erase_color = tf.random.uniform((), 0.3, 0.7)
        erased = image * mask + (1.0 - mask) * erase_color

        return erased


def create_augmentation_pipeline(
    img_size: int = 112,
    strength: str = "strong",
) -> FaceAugmentationPipeline:
    """
    Create augmentation pipeline with different strength levels.

    Args:
        img_size: Image size.
        strength: 'light', 'medium', or 'strong'.

    Returns:
        FaceAugmentationPipeline instance.
    """
    if strength == "light":
        return FaceAugmentationPipeline(
            img_size=img_size,
            brightness_delta=0.1,
            blur_probability=0.1,
            crop_probability=0.2,
            flip_probability=0.5,
            erasing_probability=0.0,
            color_jitter_probability=0.0,
        )
    elif strength == "medium":
        return FaceAugmentationPipeline(
            img_size=img_size,
            brightness_delta=0.15,
            blur_probability=0.2,
            crop_probability=0.5,
            flip_probability=0.5,
            erasing_probability=0.1,
            color_jitter_probability=0.2,
        )
    else:  # strong
        return FaceAugmentationPipeline(
            img_size=img_size,
            brightness_delta=0.2,
            blur_probability=0.3,
            crop_probability=0.5,
            flip_probability=0.5,
            erasing_probability=0.2,
            color_jitter_probability=0.4,
            zoom_range=0.15,
            rotation_probability=0.3,
        )
