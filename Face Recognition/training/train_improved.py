"""
Improved training script for face recognition with all enhancements.

Features implemented:
1. Improved dataset loading with filtering and balancing
2. Strong face-specific augmentation
3. L2 regularization and dropout
4. Cosine annealing learning rate schedule with warmup
5. ArcFace with margin warmup
6. Gradient clipping
7. Experiment tracking and checkpointing
8. Automatic report generation
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Import custom modules
from config.config import Config
from models.mobilefacenet_simple import create_mobilefacenet
from models.arcface_improved import ImprovedArcFaceModel, ImprovedArcFaceLoss
from data.dataset_loader_v2 import load_improved_dataset
from training.lr_schedules import create_learning_rate_schedule, CustomCosineDecaySchedule
from training.experiment_tracker import ExperimentTracker


class ImprovedTrainingPipeline:
    """Improved training pipeline with all enhancements."""

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        output_dir: str = "experiments",
        exp_name: str = None,
        # Dataset parameters
        batch_size: int = 32,
        img_size: int = 112,
        min_images_per_class: int = 3,
        balance_classes: bool = True,
        augmentation_strength: str = "strong",
        # Model parameters
        embedding_size: int = 128,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64,
        embedding_dropout: float = 0.2,
        l2_reg: float = 0.0001,
        label_smoothing: float = 0.1,
        margin_warmup_epochs: int = 10,
        # Training parameters
        num_epochs: int = 50,
        learning_rate: float = 0.0003,
        warmup_epochs: int = 5,
        lr_strategy: str = "cosine_warmup",
        early_stopping_patience: int = 15,
        # Optimization
        gradient_clip_norm: float = 1.0,
        mixed_precision: bool = False,
        seed: int = 42,
    ):
        """Initialize training pipeline."""
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.lr_strategy = lr_strategy
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        # Create experiment tracker
        self.tracker = ExperimentTracker(Path(output_dir))
        self.exp_dir = self.tracker.create_experiment(exp_name)

        # Save configuration
        self.config = self._build_config(
            train_dir, val_dir, batch_size, img_size, min_images_per_class,
            balance_classes, augmentation_strength, embedding_size,
            arcface_margin, arcface_scale, embedding_dropout, l2_reg,
            label_smoothing, margin_warmup_epochs, num_epochs, learning_rate,
            warmup_epochs, lr_strategy, early_stopping_patience, gradient_clip_norm,
            mixed_precision, seed,
        )
        self.tracker.save_config(self.config)

        # Enable mixed precision if requested
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[Training] Mixed precision enabled")

        # Store parameters
        self.embedding_size = embedding_size
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        self.embedding_dropout = embedding_dropout
        self.l2_reg = l2_reg
        self.label_smoothing = label_smoothing
        self.margin_warmup_epochs = margin_warmup_epochs
        self.augmentation_strength = augmentation_strength
        self.min_images_per_class = min_images_per_class
        self.balance_classes = balance_classes

    def _build_config(self, *args, **kwargs) -> Dict:
        """Build configuration dictionary."""
        return {
            "train_dir": args[0],
            "val_dir": args[1],
            "batch_size": args[2],
            "img_size": args[3],
            "min_images_per_class": args[4],
            "balance_classes": args[5],
            "augmentation_strength": args[6],
            "embedding_size": args[7],
            "arcface_margin": args[8],
            "arcface_scale": args[9],
            "embedding_dropout": args[10],
            "l2_reg": args[11],
            "label_smoothing": args[12],
            "margin_warmup_epochs": args[13],
            "num_epochs": args[14],
            "learning_rate": args[15],
            "warmup_epochs": args[16],
            "lr_strategy": args[17],
            "early_stopping_patience": args[18],
            "gradient_clip_norm": args[19],
            "mixed_precision": args[20],
            "seed": args[21],
        }

    def train(self):
        """Run training pipeline."""
        print("=" * 80)
        print("IMPROVED FACE RECOGNITION TRAINING")
        print("=" * 80)

        # 1. Load datasets
        print("\n[1/6] Loading and preparing datasets...")
        train_dataset, train_stats, train_loader = load_improved_dataset(
            data_dir=self.train_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            augment=True,
            augmentation_strength=self.augmentation_strength,
            min_images_per_class=self.min_images_per_class,
            balance_classes=self.balance_classes,
            seed=self.seed,
        )

        val_dataset, val_stats, _ = load_improved_dataset(
            data_dir=self.val_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            augment=False,
            min_images_per_class=self.min_images_per_class,
            balance_classes=False,
            seed=self.seed,
        )

        num_classes = train_stats["num_classes"]
        print(f"Training:   {train_stats['num_classes']} classes, {train_stats['num_samples']} samples")
        print(f"Validation: {val_stats['num_classes']} classes, {val_stats['num_samples']} samples")

        # 2. Create model
        print("\n[2/6] Building model architecture...")
        backbone = create_mobilefacenet(embedding_size=self.embedding_size)

        model = ImprovedArcFaceModel(
            backbone=backbone,
            num_classes=num_classes,
            margin=self.arcface_margin,
            scale=self.arcface_scale,
            margin_warmup_epochs=self.margin_warmup_epochs,
            embedding_dropout=self.embedding_dropout,
            l2_reg=self.l2_reg,
            label_smoothing=self.label_smoothing,
        )

        # Add L2 regularization to backbone if specified
        if self.l2_reg > 0:
            for layer in backbone.layers:
                if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
                    layer.add_loss(
                        self.l2_reg * tf.nn.l2_loss(layer.kernel)
                    )
            print(f"L2 regularization added: {self.l2_reg}")

        # 3. Compile model
        print("\n[3/6] Compiling model...")

        # Create loss function
        loss_fn = ImprovedArcFaceLoss(
            margin=self.arcface_margin,
            scale=self.arcface_scale,
            num_classes=num_classes,
            margin_warmup_epochs=self.margin_warmup_epochs,
            label_smoothing=self.label_smoothing,
        )

        # Create learning rate schedule
        total_steps = train_stats["num_batches"] * self.num_epochs
        warmup_steps = train_stats["num_batches"] * self.warmup_epochs

        lr_schedule = create_learning_rate_schedule(
            strategy=self.lr_strategy,
            initial_learning_rate=self.learning_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        # Create optimizer with gradient clipping
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=self.gradient_clip_norm,
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print(f"LR Schedule: {self.lr_strategy}")
        print(f"Gradient Clipping: {self.gradient_clip_norm}")

        # 4. Setup callbacks
        print("\n[4/6] Setting up callbacks...")
        callbacks = self._create_callbacks(num_classes, loss_fn, model, train_stats)

        # 5. Train model
        print("\n[5/6] Training model...")
        print(f"Epochs: {self.num_epochs}, Batches/epoch: {train_stats['num_batches']}")
        print("-" * 80)

        history = model.fit(
            train_dataset,
            epochs=self.num_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
        )

        # 6. Save results
        print("\n[6/6] Saving results...")
        self._save_results(model, backbone, history, num_classes)

        return model, history

    def _create_callbacks(self, num_classes, loss_fn, model, train_stats):
        """Create training callbacks."""
        callbacks = []

        # Model checkpoint
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(self.exp_dir / "best_model.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ))

        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ))

        # Epoch callback for margin warmup
        class EpochCallback(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                model.set_epoch(epoch)
                loss_fn.set_epoch(epoch)

                # Log current margin
                if logs:
                    logs["current_margin"] = float(loss_fn.current_margin)

        callbacks.append(EpochCallback())

        # TensorBoard
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=str(self.exp_dir / "logs"),
            update_freq="epoch",
        ))

        # Learning rate logging
        class LRLoggingCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    logs["learning_rate"] = float(
                        self.model.optimizer.learning_rate.numpy()
                    )

        callbacks.append(LRLoggingCallback())

        return callbacks

    def _save_results(self, model, backbone, history, num_classes):
        """Save training results."""
        # Save models
        backbone.save(str(self.exp_dir / "backbone.h5"))
        model.save(str(self.exp_dir / "training_model.h5"))
        print(f"Models saved to {self.exp_dir}")

        # Save training history
        history_dict = {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        }
        self.tracker.save_training_history(history_dict)

        # Save final metrics
        final_metrics = {
            "num_classes": num_classes,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "best_val_loss": float(min(history.history["val_loss"])),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "epochs_trained": len(history.history["loss"]),
        }
        self.tracker.save_metrics(final_metrics, "final")

        # Generate training curves
        self._plot_training_curves(history)

        # Generate and save report
        report = self.tracker.generate_report()
        report_path = self.exp_dir / "REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\nTraining completed!")
        print(f"Experiment saved to: {self.exp_dir}")
        print(f"Report: {report_path}")

    def _plot_training_curves(self, history):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.history["loss"]) + 1)

        # Loss curves
        ax1.plot(epochs, history.history["loss"], "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, np.array(history.history["accuracy"]) * 100, "b-", label="Training Accuracy", linewidth=2)
        ax2.plot(epochs, np.array(history.history["val_accuracy"]) * 100, "r-", label="Validation Accuracy", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(self.exp_dir / "training_curves.png"), dpi=150)
        print(f"Training curves saved: {self.exp_dir / 'training_curves.png'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Improved face recognition training")

    # Dataset arguments
    parser.add_argument("--train_dir", type=str, required=True, help="Training dataset directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation dataset directory")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")

    # Dataset processing
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=112, help="Image size")
    parser.add_argument("--min_images_per_class", type=int, default=3, help="Min images per class")
    parser.add_argument("--balance_classes", action="store_true", default=True, help="Balance class distribution")
    parser.add_argument("--augmentation_strength", type=str, default="strong", help="Augmentation strength")

    # Model arguments
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--arcface_margin", type=float, default=0.5, help="ArcFace margin")
    parser.add_argument("--arcface_scale", type=int, default=64, help="ArcFace scale")
    parser.add_argument("--embedding_dropout", type=float, default=0.2, help="Embedding dropout")
    parser.add_argument("--l2_reg", type=float, default=0.0001, help="L2 regularization")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--margin_warmup_epochs", type=int, default=10, help="Margin warmup epochs")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--lr_strategy", type=str, default="cosine_warmup", help="LR strategy")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = ImprovedTrainingPipeline(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        img_size=args.img_size,
        min_images_per_class=args.min_images_per_class,
        balance_classes=args.balance_classes,
        augmentation_strength=args.augmentation_strength,
        embedding_size=args.embedding_size,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        embedding_dropout=args.embedding_dropout,
        l2_reg=args.l2_reg,
        label_smoothing=args.label_smoothing,
        margin_warmup_epochs=args.margin_warmup_epochs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_strategy=args.lr_strategy,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_norm=args.gradient_clip_norm,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )

    model, history = pipeline.train()


if __name__ == "__main__":
    main()
