"""
Script to sample a subset from DigiFace-1M dataset.

DigiFace-1M is too large for training on limited hardware. This script samples
a manageable subset while preserving class folders structure.

Usage:
    python sample_digiface_subset.py --digiface_dir /path/to/DigiFace-1M \
                                     --output_dir data_subset \
                                     --num_identities 500 \
                                     --images_per_identity 20
"""

import argparse
import random
from pathlib import Path
import shutil


def sample_digiface_subset(
    digiface_dir: str,
    output_dir: str,
    num_identities: int = 500,
    images_per_identity: int = 20,
    random_seed: int = 42,
    train_split: float = 0.8,
) -> None:
    """
    Sample a subset from DigiFace-1M dataset.

    Args:
        digiface_dir: Path to DigiFace-1M dataset root.
        output_dir: Path to output directory for subset.
        num_identities: Number of identities to sample.
        images_per_identity: Number of images per identity.
        random_seed: Random seed for reproducibility.
        train_split: Fraction for training (rest goes to validation).
    """

    print("=" * 80)
    print("DigiFace-1M Subset Sampling")
    print("=" * 80)

    # Set random seed
    random.seed(random_seed)

    # Validate input directory
    digiface_path = Path(digiface_dir)
    if not digiface_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {digiface_dir}")

    # Get list of identity folders
    print(f"\n[1/3] Scanning dataset...")
    print(f"Source directory: {digiface_path}")

    identity_folders = sorted([
        d for d in digiface_path.iterdir()
        if d.is_dir()
    ])

    print(f"Found {len(identity_folders)} identities in dataset")

    if len(identity_folders) < num_identities:
        print(f"Warning: Dataset has only {len(identity_folders)} identities")
        print(f"         Requested {num_identities} identities")
        num_identities = len(identity_folders)

    # Sample identities
    print(f"\n[2/3] Sampling {num_identities} identities...")
    sampled_identities = random.sample(identity_folders, k=num_identities)

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")

    # Copy sampled identities
    print(f"\n[3/3] Copying images...")
    stats = {
        "total_identities": 0,
        "train_identities": 0,
        "val_identities": 0,
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
    }

    for identity_idx, identity_folder in enumerate(sampled_identities, 1):
        identity_name = identity_folder.name

        # Get all images in this identity
        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = [
            f for f in identity_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if len(image_files) == 0:
            continue

        # Sample images
        sampled_images = random.sample(
            image_files,
            k=min(images_per_identity, len(image_files))
        )

        # Split into train/val
        num_train = max(1, int(len(sampled_images) * train_split))
        train_images = sampled_images[:num_train]
        val_images = sampled_images[num_train:]

        # Copy train images
        if train_images:
            train_identity_dir = train_dir / identity_name
            train_identity_dir.mkdir(exist_ok=True)

            for img_path in train_images:
                try:
                    shutil.copy2(img_path, train_identity_dir / img_path.name)
                except Exception as e:
                    print(f"Warning: Failed to copy {img_path}: {str(e)}")

            stats["train_identities"] += 1
            stats["train_images"] += len(train_images)

        # Copy val images
        if val_images:
            val_identity_dir = val_dir / identity_name
            val_identity_dir.mkdir(exist_ok=True)

            for img_path in val_images:
                try:
                    shutil.copy2(img_path, val_identity_dir / img_path.name)
                except Exception as e:
                    print(f"Warning: Failed to copy {img_path}: {str(e)}")

            stats["val_identities"] += 1
            stats["val_images"] += len(val_images)

        stats["total_identities"] += 1
        stats["total_images"] += len(sampled_images)

        # Progress
        if identity_idx % 50 == 0:
            print(f"  Processed {identity_idx}/{num_identities} identities...")

    # Print summary
    print("\n" + "=" * 80)
    print("Sampling Complete!")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Total identities sampled: {stats['total_identities']}")
    print(f"  Total images sampled: {stats['total_images']}")
    print(f"\nTraining Set:")
    print(f"  Identities: {stats['train_identities']}")
    print(f"  Images: {stats['train_images']}")
    print(f"\nValidation Set:")
    print(f"  Identities: {stats['val_identities']}")
    print(f"  Images: {stats['val_images']}")
    print(f"\nOutput directory: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sample subset from DigiFace-1M dataset"
    )
    parser.add_argument(
        "--digiface_dir",
        type=str,
        required=True,
        help="Path to DigiFace-1M dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_subset",
        help="Output directory for sampled subset",
    )
    parser.add_argument(
        "--num_identities",
        type=int,
        default=500,
        help="Number of identities to sample",
    )
    parser.add_argument(
        "--images_per_identity",
        type=int,
        default=20,
        help="Number of images per identity",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of images for training (rest for validation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    sample_digiface_subset(
        digiface_dir=args.digiface_dir,
        output_dir=args.output_dir,
        num_identities=args.num_identities,
        images_per_identity=args.images_per_identity,
        random_seed=args.seed,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
