"""
Simplified MobileFaceNet backbone using standard Keras layers.

A lightweight face recognition backbone using depthwise separable convolutions
and standard Keras building blocks.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class L2Normalize(layers.Layer):
    """L2 normalization layer with proper shape handling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        """Apply L2 normalization."""
        return tf.nn.l2_normalize(x, axis=-1)

    def compute_output_shape(self, input_shape):
        """Return output shape (same as input)."""
        return input_shape


def create_mobilefacenet(embedding_size: int = 128):
    """
    Create a simplified MobileFaceNet backbone model.

    Uses standard Keras layers without custom layer classes to avoid shape inference issues.

    Args:
        embedding_size: Dimension of the output embedding.

    Returns:
        Keras Model with input shape (None, 112, 112, 3) and output shape (None, 128).
    """
    inputs = keras.Input(shape=(112, 112, 3))

    # Initial convolution: 112x112 -> 56x56
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)

    # Depthwise separable conv: 56x56
    x = layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)

    # Residual block 1: 56x56 -> 28x28
    shortcut = layers.Conv2D(
        filters=64,
        kernel_size=1,
        strides=2,
        padding="same",
        use_bias=False,
    )(x)
    
    residual = layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    residual = layers.BatchNormalization(momentum=0.9)(residual)
    residual = layers.ReLU()(residual)
    x = layers.Add()([shortcut, residual])

    # Residual block 2: 28x28
    residual = layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    residual = layers.BatchNormalization(momentum=0.9)(residual)
    residual = layers.ReLU()(residual)
    x = layers.Add()([x, residual])

    # Residual block 3: 28x28 -> 14x14
    shortcut = layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=2,
        padding="same",
        use_bias=False,
    )(x)
    
    residual = layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    residual = layers.BatchNormalization(momentum=0.9)(residual)
    residual = layers.ReLU()(residual)
    x = layers.Add()([shortcut, residual])

    # Residual blocks: 14x14
    for _ in range(2):
        residual = layers.SeparableConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_regularizer=keras.regularizers.l2(1e-4),
            pointwise_regularizer=keras.regularizers.l2(1e-4),
        )(x)
        residual = layers.BatchNormalization(momentum=0.9)(residual)
        residual = layers.ReLU()(residual)
        x = layers.Add()([x, residual])

    # Residual block: 14x14 -> 7x7
    shortcut = layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=2,
        padding="same",
        use_bias=False,
    )(x)
    
    residual = layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    residual = layers.BatchNormalization(momentum=0.9)(residual)
    residual = layers.ReLU()(residual)
    x = layers.Add()([shortcut, residual])

    # Residual blocks: 7x7
    for _ in range(2):
        residual = layers.SeparableConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_regularizer=keras.regularizers.l2(1e-4),
            pointwise_regularizer=keras.regularizers.l2(1e-4),
        )(x)
        residual = layers.BatchNormalization(momentum=0.9)(residual)
        residual = layers.ReLU()(residual)
        x = layers.Add()([x, residual])

    # Final layers
    x = layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_regularizer=keras.regularizers.l2(1e-4),
        pointwise_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Embedding layer
    x = layers.Dense(
        units=embedding_size,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="embedding",
    )(x)
    # BatchNorm after Dense(128)
    x = layers.BatchNormalization(momentum=0.9, name="embedding_bn")(x)
    # L2 normalization using custom layer
    embeddings = L2Normalize(name="l2_normalize")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=embeddings, name="mobilefacenet")

    return model
