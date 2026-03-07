#!/usr/bin/env python3
"""
Quick start script for improved face recognition training.

This script sets up and runs training with recommended configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_training(
    config: str = "balanced",
    train_dir: str = "lfw_funneled",
    val_dir: str = "lfw_funneled",
    num_epochs: int = None,
    batch_size: int = None,
    exp_name: str = None,
):
    """Run training with specified configuration."""
    
    # Configuration presets
    configs = {
        "balanced": {
            "batch_size": 32,
            "learning_rate": 0.0003,
            "augmentation_strength": "strong",
            "embedding_dropout": 0.2,
            "l2_reg": 0.0001,
            "label_smoothing": 0.1,
            "arcface_margin": 0.5,
            "arcface_scale": 64,
            "margin_warmup_epochs": 10,
            "warmup_epochs": 5,
            "num_epochs": 50,
            "early_stopping_patience": 15,
            "gradient_clip_norm": 1.0,
        },
        "conservative": {
            "batch_size": 32,
            "learning_rate": 0.0003,
            "augmentation_strength": "strong",
            "embedding_dropout": 0.3,
            "l2_reg": 0.0005,
            "label_smoothing": 0.15,
            "arcface_margin": 0.45,
            "arcface_scale": 64,
            "margin_warmup_epochs": 10,
            "warmup_epochs": 5,
            "num_epochs": 50,
            "early_stopping_patience": 20,
            "gradient_clip_norm": 0.5,
        },
        "aggressive": {
            "batch_size": 64,
            "learning_rate": 0.0005,
            "augmentation_strength": "strong",
            "embedding_dropout": 0.1,
            "l2_reg": 0.00001,
            "label_smoothing": 0.05,
            "arcface_margin": 0.5,
            "arcface_scale": 64,
            "margin_warmup_epochs": 5,
            "warmup_epochs": 3,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "gradient_clip_norm": 2.0,
        },
        "light": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "augmentation_strength": "medium",
            "embedding_dropout": 0.1,
            "l2_reg": 0.00001,
            "label_smoothing": 0.05,
            "arcface_margin": 0.5,
            "arcface_scale": 64,
            "margin_warmup_epochs": 5,
            "warmup_epochs": 3,
            "num_epochs": 30,
            "early_stopping_patience": 10,
            "gradient_clip_norm": 1.0,
        },
    }

    if config not in configs:
        print(f"Unknown configuration: {config}")
        print(f"Available: {', '.join(configs.keys())}")
        sys.exit(1)

    cfg = configs[config].copy()

    # Override with command-line arguments
    if num_epochs is not None:
        cfg["num_epochs"] = num_epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    # Build command
    cmd = [
        sys.executable,
        "training/train_improved.py",
        f"--train_dir={train_dir}",
        f"--val_dir={val_dir}",
        f"--output_dir=experiments",
    ]

    if exp_name:
        cmd.append(f"--exp_name={exp_name}")
    else:
        cmd.append(f"--exp_name={config}")

    # Add configuration arguments
    for key, value in cfg.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")

    # Print configuration
    print("=" * 80)
    print(f"CONFIGURATION: {config.upper()}")
    print("=" * 80)
    for key, value in cfg.items():
        print(f"  {key:<30} {value}")
    print("=" * 80)
    print()

    # Run training
    print("Starting training...")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick start improved face recognition training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="balanced",
        choices=["balanced", "conservative", "aggressive", "light"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="lfw_funneled",
        help="Training dataset directory",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="lfw_funneled",
        help="Validation dataset directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name",
    )

    args = parser.parse_args()

    # Check if training script exists
    if not Path("training/train_improved.py").exists():
        print("Error: training/train_improved.py not found")
        print("Please run this script from the project root directory")
        sys.exit(1)

    # Run training
    returncode = run_training(
        config=args.config,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        exp_name=args.exp_name,
    )

    sys.exit(returncode)


if __name__ == "__main__":
    main()
