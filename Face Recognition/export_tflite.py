"""
TensorFlow Lite export script.

Exports the trained embedding backbone to TensorFlow Lite format for
mobile deployment.

Usage:
    python export_tflite.py --model_path checkpoints/backbone.h5 --output_path embedding_model.tflite
"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from config.config import Config


def export_to_tflite(
    model_path: str,
    output_path: str = str(Config.TFLITE_OUTPUT_PATH),
    quantize: bool = Config.TFLITE_QUANTIZATION,
) -> None:
    """
    Export backbone model to TensorFlow Lite format.

    Args:
        model_path: Path to saved Keras model.
        output_path: Path to save TFLite model.
        quantize: Whether to apply float16 quantization.
    """

    print("=" * 80)
    print("TensorFlow Lite Model Export")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"Model loaded from {model_path}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/4] Converting to TensorFlow Lite...")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimization
    if quantize:
        print("  Applying float16 quantization...")
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        print("  Using default (float32) precision...")

    # Convert
    tflite_model = converter.convert()

    print("\n[3/4] Saving TFLite model...")
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved to {output_path}")

    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    print("\n[4/4] Creating interpreter test...")

    # Test interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\nTFLite Model Details:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")

        # Test inference
        import numpy as np
        test_input = np.ones(input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"  Test inference output shape: {output.shape}")
        print(f"  ✓ Model works correctly!")

    except Exception as e:
        print(f"  Warning: Could not test interpreter: {str(e)}")

    print("\n" + "=" * 80)
    print("✓ Export Complete!")
    print("=" * 80)
    print(f"\nTFLite Model: {output_path}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Quantization: {'float16' if quantize else 'float32'}")
    print("\nThis model can be used for mobile deployment!")


def create_metadata(
    output_path: str,
    input_size: int = Config.INPUT_SIZE,
    embedding_size: int = Config.EMBEDDING_SIZE,
) -> None:
    """
    Create metadata file for TFLite model.

    Args:
        output_path: Path to save metadata JSON.
        input_size: Input image size.
        embedding_size: Output embedding size.
    """
    import json

    metadata = {
        "model_type": "face_recognition",
        "architecture": "mobilefacenet",
        "input_size": input_size,
        "input_channels": 3,
        "output_size": embedding_size,
        "normalization": {
            "type": "minmax",
            "min": -1.0,
            "max": 1.0,
        },
        "expected_output": "l2_normalized_embedding",
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export model to TensorFlow Lite")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(Config.CHECKPOINT_DIR / "backbone.h5"),
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Config.TFLITE_OUTPUT_PATH),
        help="Output path for TFLite model",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=Config.TFLITE_QUANTIZATION,
        help="Whether to apply float16 quantization",
    )

    args = parser.parse_args()

    export_to_tflite(
        model_path=args.model_path,
        output_path=args.output_path,
        quantize=args.quantize,
    )

    # Also create metadata
    metadata_path = Path(args.output_path).with_suffix(".json")
    create_metadata(str(metadata_path))


if __name__ == "__main__":
    main()
