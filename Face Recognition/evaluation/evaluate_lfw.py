"""
Evaluation script for face recognition on LFW verification pairs.

Usage:
    python evaluate_lfw.py --model_path path/to/model.h5 --lfw_dir lfw_funneled --output_dir results
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from evaluation.face_verification import (
    FaceVerificationEvaluator,
    load_lfw_pairs,
    load_image_for_verification,
)


def evaluate_on_lfw(
    model_path: str,
    lfw_dir: str,
    pairs_file: str = "pairs.txt",
    output_dir: str = "evaluation_results",
    custom_objects=None,
):
    """
    Evaluate face recognition model on LFW verification pairs.

    Args:
        model_path: Path to trained model.
        lfw_dir: Path to LFW dataset directory.
        pairs_file: Name of pairs file in lfw_dir.
        output_dir: Output directory for results.
        custom_objects: Custom objects for model loading.
    """
    print("=" * 80)
    print("LFW VERIFICATION EVALUATION")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n[1/4] Loading model from {model_path}...")
    if custom_objects is None:
        custom_objects = {}
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully")

    # Load LFW pairs
    print(f"\n[2/4] Loading LFW pairs from {lfw_dir}...")
    lfw_path = Path(lfw_dir)
    pairs_path = lfw_path / pairs_file
    pairs, metadata = load_lfw_pairs(pairs_path, lfw_path)
    print(f"Loaded {metadata['num_pairs']} pairs")
    print(f"  Positive pairs: {metadata['num_positive']}")
    print(f"  Negative pairs: {metadata['num_negative']}")

    # Create evaluator
    print("\n[3/4] Evaluating...")
    evaluator = FaceVerificationEvaluator(model, similarity_metric="cosine")

    # Evaluate
    results = evaluator.evaluate_pairs(pairs, load_image_for_verification)

    # Save results
    print(f"\n[4/4] Saving results to {output_dir}...")
    results_file = output_path / "verification_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(results["fpr"], results["tpr"], lw=2, label=f"AUC = {results['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LFW Verification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(str(output_path / "roc_curve.png"), dpi=150, bbox_inches="tight")
    print("ROC curve saved")

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"AUC-ROC:              {results['roc_auc']:.6f}")
    print(f"EER:                  {results['eer']:.6f}")
    print(f"Best Threshold (EER): {results['best_threshold_eer']:.6f}")
    print(f"Best Accuracy:        {results['best_accuracy']:.6f}")
    print(f"Best Precision:       {results['best_precision']:.6f}")
    print(f"Best Recall:          {results['best_recall']:.6f}")
    print(f"Best F1-Score:        {results['best_f1']:.6f}")

    print("\nAccuracy at standard thresholds:")
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        key = f"accuracy@{threshold}"
        if key in results:
            print(f"  Threshold {threshold}: {results[key]:.6f}")

    print("\nTPR at fixed FPR:")
    for fpr_threshold in [0.001, 0.01, 0.1]:
        key = f"tpr@fpr_{fpr_threshold}"
        if key in results:
            print(f"  TPR @ FPR={fpr_threshold}: {results[key]:.6f}")

    print("=" * 80)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate on LFW verification pairs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--lfw_dir", type=str, required=True, help="Path to LFW directory")
    parser.add_argument("--pairs_file", type=str, default="pairs.txt", help="Pairs filename")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")

    args = parser.parse_args()

    evaluate_on_lfw(
        model_path=args.model_path,
        lfw_dir=args.lfw_dir,
        pairs_file=args.pairs_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
