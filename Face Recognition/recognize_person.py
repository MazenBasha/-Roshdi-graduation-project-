"""
Face recognition script for identifying persons from an image.

This script loads a query face image and identifies the person using
the enrolled templates in the database.

Usage:
    python recognize_person.py --model_path checkpoints/backbone.h5 --image test_image.jpg
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.config import Config
from data.preprocessing import load_image, resize_image, normalize_image
from utils.database import TemplateDatabase
from utils.similarity import FaceMatch


def recognize_person(
    model_path: str,
    image_path: str,
    db_path: str = str(Config.TEMPLATES_JSON_PATH),
    threshold: float = Config.RECOGNITION_THRESHOLD,
    verbose: bool = True,
) -> tuple:
    """
    Recognize a person from an image.

    Args:
        model_path: Path to trained backbone model.
        image_path: Path to query face image.
        db_path: Path to templates database.
        threshold: Similarity threshold for recognition.
        verbose: Whether to print detailed results.

    Returns:
        Tuple of (person_name, similarity_score, all_scores).
    """

    # Load model
    if verbose:
        print("=" * 80)
        print("Face Recognition")
        print("=" * 80)
        print("\n[1/4] Loading model...")

    model = keras.models.load_model(str(model_path))

    # Load and preprocess image
    if verbose:
        print("\n[2/4] Loading query image...")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    image = resize_image(image, size=Config.INPUT_SIZE)
    image_normalized = normalize_image(tf.convert_to_tensor(image, dtype=tf.float32))

    if verbose:
        print(f"Image loaded: {image_path}")
        print(f"Image shape: {image.shape}")

    # Extract embedding
    if verbose:
        print("\n[3/4] Extracting embedding...")

    image_batch = tf.expand_dims(image_normalized, axis=0)
    query_embedding = model(image_batch, training=False)[0].numpy()

    if verbose:
        print(f"Embedding shape: {query_embedding.shape}")

    # Load database and templates
    if verbose:
        print("\n[4/4] Matching against templates...")

    db = TemplateDatabase(Path(db_path))
    templates = db.get_all_templates()

    if not templates:
        if verbose:
            print("\nNo templates in database!")
        return "Unknown", 0.0, {}

    # Find best match
    matcher = FaceMatch(threshold=threshold)
    person_name, similarity = matcher.identify(query_embedding, templates)
    all_scores = matcher.identify_with_scores(query_embedding, templates)

    if verbose:
        print("\n" + "=" * 80)
        print("Recognition Results")
        print("=" * 80)

        if person_name != "Unknown":
            print(f"\n✓ IDENTIFIED: {person_name}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Threshold: {threshold:.4f}")
        else:
            print(f"\n✗ UNKNOWN PERSON")
            print(f"  Max similarity: {max(all_scores.values()) if all_scores else 0:.4f}")
            print(f"  Threshold: {threshold:.4f}")

        # Sort and display all scores
        print(f"\nAll Scores:")
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_scores:
            marker = "✓" if score >= threshold else "✗"
            print(f"  {marker} {name:<20} {score:.4f}")

        print("\n" + "=" * 80)

    return person_name, similarity, all_scores


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Recognize a person from an image")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(Config.CHECKPOINT_DIR / "backbone.h5"),
        help="Path to trained backbone model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to query face image",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=str(Config.TEMPLATES_JSON_PATH),
        help="Path to templates database",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.RECOGNITION_THRESHOLD,
        help="Similarity threshold for recognition",
    )

    args = parser.parse_args()

    person_name, similarity, all_scores = recognize_person(
        model_path=args.model_path,
        image_path=args.image,
        db_path=args.db_path,
        threshold=args.threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
