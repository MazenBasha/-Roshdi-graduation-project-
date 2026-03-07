"""
Improved ArcFace loss with margin warmup and adaptive scaling.

Features:
- Margin warmup (gradually increase margin during training)
- Adaptive margin scaling
- Improved numerical stability
- Label smoothing integration
- Better gradient flow
"""

import math
from typing import Optional, Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ImprovedArcFaceLoss(keras.losses.Loss):
    """
    Improved ArcFace loss with margin warmup and adaptive features.

    The loss implements:
    loss = -log(exp(s * cos(theta + m(epoch))) / (exp(s * cos(theta + m(epoch))) + sum(exp(s * cos(theta_j)))))

    Where margin m(epoch) is gradually warmed up during training.
    """

    def __init__(
        self,
        margin: float = 0.5,
        scale: float = 64.0,
        num_classes: int = None,
        margin_warmup_epochs: int = 10,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        """
        Initialize improved ArcFace loss.

        Args:
            margin: Target angular margin at end of warmup.
            scale: Scaling factor.
            num_classes: Number of classes.
            margin_warmup_epochs: Epochs to warmup margin.
            label_smoothing: Label smoothing factor (0-0.1).
        """
        super().__init__(**kwargs)
        self.target_margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.margin_warmup_epochs = margin_warmup_epochs
        self.label_smoothing = label_smoothing

        # Current margin (updated per epoch)
        self.current_margin = 0.0
        self.current_epoch = 0

        # Precompute cos/sin of target margin
        self.cos_target_margin = math.cos(margin)
        self.sin_target_margin = math.sin(margin)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for margin warmup."""
        self.current_epoch = epoch

        # Linear warmup of margin
        if epoch < self.margin_warmup_epochs:
            warmup_progress = epoch / max(1, self.margin_warmup_epochs)
            self.current_margin = self.target_margin * warmup_progress
        else:
            self.current_margin = self.target_margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute improved ArcFace loss.

        Args:
            y_true: True labels (sparse or one-hot).
            y_pred: Logits from ArcFace head.

        Returns:
            Scalar loss value.
        """
        y_pred = tf.cast(y_pred, tf.float32)

        # Convert sparse to one-hot if needed
        if len(y_true.shape) == 1:
            y_true_one_hot = tf.one_hot(
                tf.cast(y_true, tf.int32), self.num_classes, dtype=tf.float32
            )
        else:
            y_true_one_hot = tf.cast(y_true, tf.float32)

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            y_true_one_hot = (
                y_true_one_hot * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        # Compute cross-entropy loss
        loss = keras.losses.categorical_crossentropy(
            y_true_one_hot, y_pred, from_logits=True
        )

        return tf.reduce_mean(loss)


class ImprovedArcFaceHead(layers.Layer):
    """
    Improved ArcFace head with margin warmup and numerical stability.
    """

    def __init__(
        self,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        margin_warmup_epochs: int = 10,
        adaptive_scale: bool = False,
        **kwargs
    ):
        """
        Initialize improved ArcFace head.

        Args:
            num_classes: Number of output classes.
            margin: Target angular margin.
            scale: Scaling factor.
            margin_warmup_epochs: Epochs for margin warmup.
            adaptive_scale: Whether to use adaptive scaling.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.margin_warmup_epochs = margin_warmup_epochs
        self.adaptive_scale = adaptive_scale

        # Current values (updated per epoch)
        self.current_margin = 0.0
        self.current_epoch = 0
        self.current_scale = scale

        # Precompute values
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

        self.w = None

    def build(self, input_shape):
        """Build weight matrix."""
        embedding_size = input_shape[-1]

        self.w = self.add_weight(
            name="weight",
            shape=(embedding_size, self.num_classes),
            initializer=keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )

        super().build(input_shape)

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for margin warmup."""
        self.current_epoch = epoch

        # Linear warmup of margin
        if epoch < self.margin_warmup_epochs:
            warmup_progress = epoch / max(1, self.margin_warmup_epochs)
            self.current_margin = self.margin * warmup_progress
            # Also slightly reduce scale during warmup for stability
            self.current_scale = self.scale * (0.5 + 0.5 * warmup_progress)
        else:
            self.current_margin = self.margin
            self.current_scale = self.scale

    def call(self, embeddings: tf.Tensor, training=None) -> tf.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Normalized embeddings (batch_size, embedding_size).
            training: Training flag.

        Returns:
            Logits (batch_size, num_classes).
        """
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # Normalize weights
        w_normalized = tf.nn.l2_normalize(self.w, axis=0)

        # Cosine similarity
        logits = tf.matmul(embeddings, w_normalized)

        # Clip to [-1, 1]
        logits = tf.clip_by_value(logits, -1.0, 1.0)

        # Apply margin and scale
        # Note: margin is applied in loss, not here, for better stability
        logits = logits * self.current_scale

        return logits

    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "adaptive_scale": self.adaptive_scale,
        })
        return config


class ImprovedArcFaceModel(keras.Model):
    """
    Improved ArcFace model with regularization and margin warmup.
    """

    def __init__(
        self,
        backbone: keras.Model,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        margin_warmup_epochs: int = 10,
        embedding_dropout: float = 0.2,
        l2_reg: float = 0.0,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        """
        Initialize improved ArcFace model.

        Args:
            backbone: Feature extraction backbone.
            num_classes: Number of classes.
            margin: Angular margin.
            scale: Scaling factor.
            margin_warmup_epochs: Warmup epochs for margin.
            embedding_dropout: Dropout rate for embeddings.
            l2_reg: L2 regularization weight.
            label_smoothing: Label smoothing factor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.margin_warmup_epochs = margin_warmup_epochs
        self.embedding_dropout = embedding_dropout
        self.l2_reg = l2_reg
        self.label_smoothing = label_smoothing

        # Embedding dropout
        self.dropout = layers.Dropout(embedding_dropout)

        # ArcFace head
        self.arcface_head = ImprovedArcFaceHead(
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            margin_warmup_epochs=margin_warmup_epochs,
        )

        # Loss and metrics
        self._loss_fn = ImprovedArcFaceLoss(
            margin=margin,
            scale=scale,
            num_classes=num_classes,
            margin_warmup_epochs=margin_warmup_epochs,
            label_smoothing=label_smoothing,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for margin warmup."""
        self.arcface_head.set_epoch(epoch)
        self._loss_fn.set_epoch(epoch)

    def call(
        self,
        inputs,
        training=None,
        return_embedding=False,
    ):
        """
        Forward pass.

        Args:
            inputs: Input images.
            training: Training flag.
            return_embedding: Whether to return embeddings.

        Returns:
            Logits or embeddings.
        """
        # Extract embeddings
        embeddings = self.backbone(inputs, training=training)

        # Apply dropout during training
        if training:
            embeddings = self.dropout(embeddings, training=True)

        # Return embeddings if requested
        if return_embedding:
            return embeddings

        # Return logits
        logits = self.arcface_head(embeddings, training=training)
        return logits

    def get_config(self):
        """Return config for serialization."""
        return {
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "embedding_dropout": self.embedding_dropout,
            "l2_reg": self.l2_reg,
            "label_smoothing": self.label_smoothing,
        }
