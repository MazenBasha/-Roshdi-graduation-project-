"""
Enrollment script for adding new persons to the face database.

This script takes a folder of face images for one person and enrolls them
in the database by computing mean embedding.

Usage:
    python enroll_person.py --model_path checkpoints/backbone.h5 --person_folder images/alice --name "Alice"
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.config import Config
from data.preprocessing import load_image, resize_image, normalize_image
from utils.database import TemplateDatabase


def enroll_person(
    model_path: str,
    person_folder: str,
    person_name: str,
    db_path: str = str(Config.TEMPLATES_JSON_PATH),
) -> None:
    """
    Enroll a new person in the face database.

    Args:
        model_path: Path to trained backbone model.
        person_folder: Path to folder containing person's face images.
        person_name: Name of the person.
        db_path: Path to templates database.
    """

    print("=" * 80)
    print(f"Enrolling Person: {person_name}")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    model = keras.models.load_model(str(model_path))
    print(f"Model loaded from {model_path}")

    # Find all images in folder
    print("\n[2/4] Finding images...")
    person_path = Path(person_folder)

    if not person_path.exists():
        raise FileNotFoundError(f"Folder not found: {person_folder}")

    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [
        f for f in person_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise ValueError(f"No images found in {person_folder}")

    print(f"Found {len(image_files)} images in {person_path}")

    # Extract embeddings
    print(f"\n[3/4] Extracting embeddings...")
    embeddings = []

    for img_path in sorted(image_files):
        try:
            # Load and preprocess image
            image = load_image(str(img_path))
            image = resize_image(image, size=Config.INPUT_SIZE)
            image = normalize_image(tf.convert_to_tensor(image, dtype=tf.float32))

            # Extract embedding
            image_batch = tf.expand_dims(image, axis=0)
            embedding = model(image_batch, training=False)[0].numpy()

            embeddings.append(embedding)
            print(f"  ✓ {img_path.name}")

        except Exception as e:
            print(f"  ✗ {img_path.name}: {str(e)}")

    if not embeddings:
        raise ValueError("Could not extract embeddings from any images")

    embeddings = np.array(embeddings)

    # Compute mean embedding
    print(f"\n[4/4] Computing mean embedding...")
    mean_embedding = embeddings.mean(axis=0)

    # Normalize
    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

    print(f"Mean embedding computed from {len(embeddings)} images")
    print(f"Embedding shape: {mean_embedding.shape}")

    # Save to database
    db = TemplateDatabase(Path(db_path))
    db.add_template(
        person_name=person_name,
        embedding=mean_embedding,
        num_samples=len(embeddings),
        metadata={
            "image_files": [f.name for f in image_files],
        },
    )

    print("\n" + "=" * 80)
    print(f"✓ Successfully enrolled {person_name}")
    print("=" * 80)
    print(f"Database location: {db_path}")
    print(f"Enrolled persons: {db.list_all_persons()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enroll a person in the face database")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(Config.CHECKPOINT_DIR / "backbone.h5"),
        help="Path to trained backbone model",
    )
    parser.add_argument(
        "--person_folder",
        type=str,
        required=True,
        help="Path to folder containing person's face images",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the person to enroll",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=str(Config.TEMPLATES_JSON_PATH),
        help="Path to templates database",
    )

    args = parser.parse_args()

    enroll_person(
        model_path=args.model_path,
        person_folder=args.person_folder,
        person_name=args.name,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
