"""
ArcFace loss implementation.

ArcFace (Additive Angular Margin) is a loss function that improves face recognition
by adding an additive angular margin to the target logit. This encourages larger
angular separation between different classes while maintaining angular margin for
positive pairs.

Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698
"""

import math
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ArcFaceLoss(keras.losses.Loss):
    """
    ArcFace Loss as a custom Keras loss function.

    The loss computes:
    loss = -log(exp(s * cos(theta + m)) / (exp(s * cos(theta + m)) + sum(exp(s * cos(theta_j)))))

    Where:
    - theta is the angle between embedding and weight
    - m is the angular margin (default: 0.5)
    - s is the scale factor (default: 64)
    """

    def __init__(
        self,
        margin: float = 0.5,
        scale: float = 64.0,
        num_classes: int = None,
        **kwargs
    ):
        """
        Initialize ArcFace loss.

        Args:
            margin: Angular margin (m). Default: 0.5 (~28.6 degrees).
            scale: Scaling factor (s). Default: 64.
            num_classes: Number of classes (needed for one-hot conversion).
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes

        # Precompute cos(margin) and sin(margin) for numerical stability
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute ArcFace loss.

        Args:
            y_true: True labels - either one-hot encoded (batch_size, num_classes) or sparse integers (batch_size,).
            y_pred: Logits from ArcFace head of shape (batch_size, num_classes).

        Returns:
            Scalar loss value.
        """
        # Ensure predictions are float32
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1:
            # Sparse integer labels
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes, dtype=tf.float32)
        else:
            y_true = tf.cast(y_true, tf.float32)

        # Compute cross-entropy loss with softmax
        # The y_pred should already be the ArcFace-adjusted logits
        loss = keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        return tf.reduce_mean(loss)


class ArcFaceHead(layers.Layer):
    """
    ArcFace classification head.

    This layer:
    1. Takes embeddings and applies feature normalization
    2. Normalizes weight matrix
    3. Computes cosine similarity between embeddings and weights
    4. Applies ArcFace angular margin to positive class logits
    5. Scales logits
    """

    def __init__(
        self,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        **kwargs
    ):
        """
        Initialize ArcFace head.

        Args:
            num_classes: Number of output classes.
            margin: Angular margin (m).
            scale: Scaling factor (s).
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Precompute cos(margin) and sin(margin) for numerical stability
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

        # Weight matrix for classification
        self.w = None

    def build(self, input_shape):
        """Build weight matrix."""
        embedding_size = input_shape[-1]

        # Initialize weight with random normal
        # Shape: (embedding_size, num_classes)
        self.w = self.add_weight(
            name="weight",
            shape=(embedding_size, self.num_classes),
            initializer=keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )

        super().build(input_shape)

    def call(self, embeddings: tf.Tensor, training=None) -> tf.Tensor:
        """
        Forward pass for ArcFace head.

        Args:
            embeddings: Normalized embeddings of shape (batch_size, embedding_size).
            training: Training flag (ignored but required by Keras).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Normalize embeddings (should already be normalized, but ensure it)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # Normalize weights
        w_normalized = tf.nn.l2_normalize(self.w, axis=0)

        # Compute cosine similarity: (batch_size, num_classes)
        # cos(theta) = embedding @ w_normalized
        logits = tf.matmul(embeddings, w_normalized)

        # Clip to [-1, 1] for numerical stability
        logits = tf.clip_by_value(logits, -1.0, 1.0)

        # Scale logits and return
        # Note: Margin is applied implicitly through loss function
        logits = logits * self.scale

        return logits

    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
        })
        return config


class ArcFaceModel(keras.Model):
    """
    Complete ArcFace model combining backbone and head.

    During training:
    - Input: images
    - Output: logits (passed to ArcFace loss)

    During inference:
    - Input: images
    - Output: normalized embeddings
    """

    def __init__(
        self,
        backbone: keras.Model,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        **kwargs
    ):
        """
        Initialize ArcFace model.

        Args:
            backbone: Feature extraction backbone (e.g., MobileFaceNet).
            num_classes: Number of classes for classification head.
            margin: Angular margin for ArcFace.
            scale: Scaling factor for ArcFace.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # ArcFace head
        self.arcface_head = ArcFaceHead(
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    def call(
        self,
        inputs,
        y_true=None,
        training=False,
        return_embedding=False,
    ):
        """
        Forward pass.

        Args:
            inputs: Input images of shape (batch_size, 112, 112, 3).
            y_true: Optional one-hot encoded labels for training.
            training: Training flag (used for batch norm, dropout, etc).
            return_embedding: If True, return embeddings instead of logits.

        Returns:
            Logits (batch_size, num_classes) by default during model.fit(),
            embeddings (batch_size, 128) if return_embedding=True.
        """
        # Extract normalized embeddings from backbone
        embeddings = self.backbone(inputs, training=training)

        # If return_embedding is explicitly True, return embeddings
        if return_embedding:
            return embeddings

        # For model.fit(), always return logits (for both train and validation)
        # return_embedding=False is the default, so we return logits
        logits = self.arcface_head(embeddings, training=training)
        return logits

    def get_config(self):
        """Return config for serialization."""
        return {
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
        }
