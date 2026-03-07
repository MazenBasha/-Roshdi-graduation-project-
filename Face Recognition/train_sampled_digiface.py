"""
Training script for face recognition model with ArcFace loss.

This script trains a MobileFaceNet backbone with ArcFace loss on a sampled
subset of DigiFace-1M.

Usage:
    python train_sampled_digiface.py --data_dir data_subset/train --val_dir data_subset/val
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config.config import Config
from models.mobilefacenet_simple import create_mobilefacenet
from data.dataset_loader import load_dataset


def train_face_recognition(
    train_dir: str,
    val_dir: str,
    output_dir: str = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
):
    """
    Train face recognition model with ArcFace loss.

    Args:
        train_dir: Path to training dataset directory.
        val_dir: Path to validation dataset directory.
        output_dir: Directory to save checkpoints and logs.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        early_stopping_patience: Patience for early stopping.
    """

    print("=" * 80)
    print("Face Recognition Training with ArcFace Loss")
    print("=" * 80)

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\n[1/6] Loading training dataset...")
    train_dataset, num_classes, train_loader = load_dataset(
        data_dir=train_dir,
        img_size=Config.INPUT_SIZE,
        batch_size=batch_size,
        augment=True,
        shuffle=True,
    )
    print(f"Training set: {num_classes} classes")

    print("\n[2/6] Loading validation dataset...")
    val_dataset, _, _ = load_dataset(
        data_dir=val_dir,
        img_size=Config.INPUT_SIZE,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    # Create model
    print("\n[3/6] Creating model architecture...")
    backbone = create_mobilefacenet(embedding_size=Config.EMBEDDING_SIZE)

    from models.arcface import ArcFaceModel, ArcFaceLoss

    # Create ArcFace model
    model = ArcFaceModel(
        backbone=backbone,
        num_classes=num_classes,
        margin=Config.ARCFACE_MARGIN,
        scale=Config.ARCFACE_SCALE,
    )

    # Compile model
    print("\n[4/6] Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=ArcFaceLoss(margin=Config.ARCFACE_MARGIN, scale=Config.ARCFACE_SCALE, num_classes=num_classes),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Print model summary
    print("\nBackbone Architecture:")
    backbone.summary()

    # Callbacks
    callbacks = [
        # Checkpoint best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate scheduler
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_REDUCE_PATIENCE,
            min_lr=Config.MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    # Train model
    print("\n[5/6] Training model...")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Classes: {num_classes}")
    print("-" * 80)

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=Config.VERBOSE,
    )

    print("\n[6/6] Saving results...")

    # Save backbone
    backbone_path = output_dir / "backbone.h5"
    backbone.save(str(backbone_path))
    print(f"Saved backbone to {backbone_path}")

    # Save full model
    model_path = output_dir / "training_model.h5"
    model.save(str(model_path))
    print(f"Saved training model to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    history_dict = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
    }
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_classes": num_classes,
                "embedding_size": Config.EMBEDDING_SIZE,
                "input_size": Config.INPUT_SIZE,
                "arcface_margin": Config.ARCFACE_MARGIN,
                "arcface_scale": Config.ARCFACE_SCALE,
                "training_date": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"Saved configuration to {config_path}")

    # Print final metrics
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")

    return backbone, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train face recognition model with ArcFace loss"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Config.TRAIN_DIR),
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=str(Config.VAL_DIR),
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.CHECKPOINT_DIR),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=Config.LEARNING_RATE,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=Config.EARLY_STOPPING_PATIENCE,
        help="Early stopping patience",
    )

    args = parser.parse_args()

    # Check if GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nGPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("\nNo GPU found. Training will use CPU (slower).")

    # Train model
    train_face_recognition(
        train_dir=args.data_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
    )


if __name__ == "__main__":
    main()
