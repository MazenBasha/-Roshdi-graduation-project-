"""
MobileFaceNet-inspired backbone architecture.

This module implements a lightweight face recognition backbone inspired by MobileFaceNet,
using depthwise separable convolutions and residual bottleneck blocks for efficient
mobile deployment.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DepthwiseSeparableConv2D(layers.Layer):
    """
    Depthwise separable convolution: depthwise conv + pointwise conv.

    This is more efficient than standard convolution and is key to MobileFaceNet's
    lightweight design.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = False,
        **kwargs
    ):
        """
        Initialize depthwise separable convolution.

        Args:
            filters: Number of output filters for pointwise convolution.
            kernel_size: Size of the depthwise convolution kernel.
            strides: Stride of the depthwise convolution.
            padding: Padding mode.
            use_bias: Whether to use bias in convolutions.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        # Depthwise convolution
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            depthwise_regularizer=keras.regularizers.l2(1e-4),
        )

        # Batch normalization after depthwise
        self.bn1 = layers.BatchNormalization(momentum=0.9)

        # Activation
        self.relu = layers.ReLU()

        # Pointwise convolution (1x1)
        self.pointwise = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )

        # Batch normalization after pointwise
        self.bn2 = layers.BatchNormalization(momentum=0.9)

    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.depthwise(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        return x


class BottleneckResidualBlock(layers.Layer):
    """
    Residual bottleneck block with depthwise separable convolutions.

    Used in MobileFaceNet for efficient residual connections.
    """

    def __init__(
        self, filters: int, strides: int = 1, expansion_factor: int = 2, **kwargs
    ):
        """
        Initialize bottleneck residual block.

        Args:
            filters: Output filters.
            strides: Stride for depthwise convolution.
            expansion_factor: Expansion factor for intermediate channels.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor

        # 1x1 expansion convolution
        expanded_filters = filters * expansion_factor
        self.expand_conv = layers.Conv2D(
            filters=expanded_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.expand_bn = layers.BatchNormalization(momentum=0.9)

        # Depthwise convolution
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
            depthwise_regularizer=keras.regularizers.l2(1e-4),
        )
        self.depthwise_bn = layers.BatchNormalization(momentum=0.9)

        # 1x1 projection convolution
        self.project_conv = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.project_bn = layers.BatchNormalization(momentum=0.9)

        self.relu = layers.ReLU()

        # Whether to use residual connection
        self.use_residual = strides == 1

    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = self.relu(x)

        x = self.depthwise(x)
        x = self.depthwise_bn(x, training=training)
        x = self.relu(x)

        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        # Add residual connection if strides == 1
        if self.use_residual:
            x = x + inputs

        return x


class MobileFaceNet(keras.Model):
    """
    MobileFaceNet-inspired lightweight face recognition backbone.

    Architecture:
    - Efficient depthwise separable convolutions
    - Residual bottleneck blocks
    - Global average pooling
    - L2 normalization on embeddings

    Input: 112x112 RGB image
    Output: 128-dim L2-normalized embedding
    """

    def __init__(self, embedding_size: int = 128, **kwargs):
        """
        Initialize MobileFaceNet.

        Args:
            embedding_size: Dimension of output embedding vector.
        """
        super().__init__(**kwargs)
        self.embedding_size = embedding_size

        # Initial convolution: 112x112 -> 56x56
        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()

        # Depthwise separable conv: 56x56
        self.dw_sep1 = DepthwiseSeparableConv2D(filters=64, strides=1)

        # Bottleneck blocks: 56x56 -> 28x28
        self.bottleneck1 = BottleneckResidualBlock(filters=64, strides=2)
        self.bottleneck2 = BottleneckResidualBlock(filters=64, strides=1)

        # Bottleneck blocks: 28x28
        self.bottleneck3 = BottleneckResidualBlock(filters=128, strides=2)
        self.bottleneck4 = BottleneckResidualBlock(filters=128, strides=1)

        # Bottleneck blocks: 14x14 -> 7x7
        self.bottleneck5 = BottleneckResidualBlock(filters=128, strides=2)
        self.bottleneck6 = BottleneckResidualBlock(filters=128, strides=1)

        # Depthwise separable conv: 7x7
        self.dw_sep2 = DepthwiseSeparableConv2D(filters=128, strides=1)

        # Pointwise convolution for dimension reduction: 7x7
        self.conv2 = layers.Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.bn2 = layers.BatchNormalization(momentum=0.9)

        # Global average pooling: 7x7 -> 1x1
        self.global_pool = layers.GlobalAveragePooling2D()

        # Embedding layer with L2 normalization
        self.embedding = layers.Dense(
            units=embedding_size,
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="embedding",
        )

    def call(self, inputs, training=False, normalize=True):
        """
        Forward pass.

        Args:
            inputs: Input images of shape (batch_size, 112, 112, 3).
            training: Training flag for batch normalization.
            normalize: Whether to apply L2 normalization.

        Returns:
            Embeddings of shape (batch_size, embedding_size).
        """
        # Initial convolution and activation
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        # Depthwise separable convolution
        x = self.dw_sep1(x, training=training)

        # Bottleneck residual blocks
        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        x = self.bottleneck3(x, training=training)
        x = self.bottleneck4(x, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)

        # Depthwise separable convolution
        x = self.dw_sep2(x, training=training)

        # Pointwise convolution
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Global average pooling
        x = self.global_pool(x)

        # Embedding
        x = self.embedding(x)

        # L2 normalization (always applied in call)
        if normalize:
            x = tf.nn.l2_normalize(x, axis=1)

        return x

    def get_config(self):
        """Return config for serialization."""
        return {"embedding_size": self.embedding_size}


def create_mobilefacenet(
    embedding_size: int = 128,
) -> Tuple[keras.Model, keras.Model]:
    """
    Create MobileFaceNet backbone model.

    Args:
        embedding_size: Dimension of the output embedding.

    Returns:
        Tuple of:
            - backbone: Full model for training
            - embedding_model: Model for inference (no training components)
    """
    backbone = MobileFaceNet(embedding_size=embedding_size)
    return backbone
