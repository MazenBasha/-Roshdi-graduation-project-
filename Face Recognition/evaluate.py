"""
Evaluation script for face recognition model.

Evaluates the trained model on validation set and computes metrics.

Usage:
    python evaluate.py --model_path checkpoints/backbone.h5 --val_dir data_subset/val
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py

from config.config import Config
from data.dataset_loader import load_dataset
from utils.metrics import FaceRecognitionMetrics, EmbeddingMetrics
from models.arcface import ArcFaceLoss
from models.mobilefacenet_simple import create_mobilefacenet


def _load_weights_from_h5(model, h5_path):
    """Load weights from h5 file with proper nested model handling."""
    try:
        with h5py.File(h5_path, 'r') as f:
            model_weights_group = f['model_weights']
            weights_loaded = 0
            
            # First, load the mobilefacenet (sub-model) weights
            if 'mobilefacenet' in model_weights_group:
                mobilefacenet_group = model_weights_group['mobilefacenet']
                mobilefacenet = model.get_layer('mobilefacenet')
                
                # Get weight names attribute which lists all weights
                weight_names = mobilefacenet_group.attrs.get('weight_names', [])
                print(f"  Found {len(weight_names)} weight entries in mobilefacenet")
                
                # Process each weight in the flat list
                weight_dict_by_layer = {}
                for full_wname in weight_names:
                    full_wname_str = full_wname.decode() if isinstance(full_wname, bytes) else full_wname
                    
                    # Parse the weight name to get layer name
                    # Format: "layername/kernelname" or "layername/kernel"
                    parts = full_wname_str.split('/')
                    if len(parts) >= 2:
                        layer_name = parts[0]
                        if layer_name not in weight_dict_by_layer:
                            weight_dict_by_layer[layer_name] = []
                        weight_dict_by_layer[layer_name].append(full_wname_str)
                
                # Load weights for each layer
                loaded_layers = []
                failed_layers = []
                for layer_name, weight_names_for_layer in sorted(weight_dict_by_layer.items()):
                    try:
                        sub_layer = mobilefacenet.get_layer(layer_name)
                        if sub_layer is None:
                            failed_layers.append((layer_name, "layer not found"))
                            continue
                        
                        weights = []
                        for wname_str in weight_names_for_layer:
                            try:
                                w = mobilefacenet_group[wname_str][()]
                                weights.append(w)
                            except:
                                pass
                        
                        if weights:
                            sub_layer.set_weights(weights)
                            weights_loaded += len(weights)
                            loaded_layers.append((layer_name, len(weights)))
                        else:
                            failed_layers.append((layer_name, "no weights found"))
                    except Exception as e:
                        failed_layers.append((layer_name, str(e)[:50]))
                
                print(f"  Loaded weights for {len(loaded_layers)} layers in mobilefacenet")
                if failed_layers:
                    print(f"  Failed to load {len(failed_layers)} layers: {failed_layers[:3]}")
            
            # Load classification layer weights
            if 'classification' in model_weights_group:
                classification = model.get_layer('classification')
                if classification is not None:
                    weights = []
                    class_group = model_weights_group['classification']
                    weight_names = class_group.attrs.get('weight_names', [])
                    
                    for wname in weight_names:
                        wname_str = wname.decode() if isinstance(wname, bytes) else wname
                        try:
                            weights.append(class_group[wname_str][()])
                        except:
                            pass
                    
                    if weights:
                        classification.set_weights(weights)
                        weights_loaded += len(weights)
                        print(f"  Loaded {len(weights)} weights for classification layer")
            
            print(f"  Total {weights_loaded} weight arrays loaded")
            return weights_loaded > 0
    except Exception as e:
        print(f"Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model(
    model_path: str,
    val_dir: str,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate face recognition model.

    Args:
        model_path: Path to trained backbone model.
        val_dir: Path to validation dataset directory.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary of evaluation metrics.
    """

    print("=" * 80)
    print("Face Recognition Model Evaluation")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading trained model...")
    try:
        # Try direct load first
        model = keras.models.load_model(
            str(model_path),
            custom_objects={"ArcFaceLoss": ArcFaceLoss},
            safe_mode=False
        )
    except Exception as e:
        # Fallback: rebuild model architecture and load weights properly
        print(f"Direct loading failed, rebuilding model...")
        
        # Create model architecture
        mobilefacenet = create_mobilefacenet(
            embedding_size=Config.EMBEDDING_SIZE,
        )
        
        # Wrap it with the classification head
        inputs = keras.Input(shape=(Config.INPUT_SIZE, Config.INPUT_SIZE, 3))
        embeddings = mobilefacenet(inputs, training=False)
        logits = keras.layers.Dense(
            1600,  # num classes
            activation="linear",
            use_bias=True,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="classification"
        )(embeddings)
        model = keras.Model(inputs=inputs, outputs=logits)
        
        # Build the model
        dummy_input = np.random.randn(1, Config.INPUT_SIZE, Config.INPUT_SIZE, 3).astype(np.float32)
        _ = model(dummy_input, training=False)
        
        # Load weights
        success = _load_weights_from_h5(model, str(model_path))
        if not success:
            print("Warning: Could not load weights, model will use random initialization")
    
    print(f"Model loaded from {model_path}")

    # Load validation dataset
    print("\n[2/4] Loading validation dataset...")
    val_dataset, num_classes, dataset_loader = load_dataset(
        data_dir=val_dir,
        img_size=Config.INPUT_SIZE,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )
    print(f"Validation set: {num_classes} classes")

    # Evaluate classification accuracy
    print("\n[3/4] Evaluating classification accuracy...")

    all_embeddings = []
    all_labels = []

    for images, labels in val_dataset:
        # Extract embeddings
        embeddings = model(images, training=False)
        all_embeddings.append(embeddings.numpy())
        all_labels.append(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)

    print(f"Extracted {len(all_embeddings)} embeddings")

    # Classification accuracy (using cosine similarity)
    # For each embedding, find closest class prototype
    unique_labels = np.unique(all_labels)
    class_prototypes = []

    for label in unique_labels:
        class_embeddings = all_embeddings[all_labels == label]
        prototype = class_embeddings.mean(axis=0)
        class_prototypes.append(prototype)

    class_prototypes = np.array(class_prototypes)

    # Normalize prototypes
    class_prototypes = class_prototypes / (
        np.linalg.norm(class_prototypes, axis=1, keepdims=True) + 1e-8
    )

    # Compute similarities and predictions
    test_embeddings_normalized = all_embeddings / (
        np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
    )

    similarities = np.dot(test_embeddings_normalized, class_prototypes.T)
    predictions = np.argmax(similarities, axis=1)

    # Compute metrics
    accuracy = FaceRecognitionMetrics.classification_accuracy(all_labels, predictions)
    precision = FaceRecognitionMetrics.precision(all_labels, predictions)
    recall = FaceRecognitionMetrics.recall(all_labels, predictions)
    f1 = FaceRecognitionMetrics.f1_score(all_labels, predictions)

    # Compute embedding quality metrics
    print("\n[4/4] Computing embedding quality metrics...")

    intra_distances = EmbeddingMetrics.intra_class_distance(all_embeddings, all_labels)
    inter_distance = EmbeddingMetrics.inter_class_distance(all_embeddings, all_labels)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\nEmbedding Quality Metrics:")
    print(f"  Intra-class distance (within class):")
    for label, dist in sorted(intra_distances.items())[:5]:
        class_name = dataset_loader.get_class_name(label)
        print(f"    {class_name}: {dist:.4f}")
    if len(intra_distances) > 5:
        print(f"    ... and {len(intra_distances) - 5} more classes")

    print(f"  Inter-class distance (between class):")
    print(f"    Average: {inter_distance:.4f}")

    # Prepare summary
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_test_samples": len(all_embeddings),
        "num_classes": int(num_classes),
        "intra_class_distance_mean": float(np.mean(list(intra_distances.values()))),
        "inter_class_distance": float(inter_distance),
    }

    print("\n" + "=" * 80)

    return metrics_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(Config.CHECKPOINT_DIR / "backbone.h5"),
        help="Path to trained backbone model",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=str(Config.VAL_DIR),
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    # Evaluate
    metrics = evaluate_model(
        model_path=args.model_path,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
    )

    # Save metrics
    metrics_path = Path(args.model_path).parent / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
