"""
Evaluation script for Egyptian Currency Classification Model.

Evaluates the best trained checkpoint on the test set and reports:
- Overall accuracy, precision, recall, F1, mAP
- Per-class metrics
- Confusion matrix
- Sample prediction visualization

Usage:
    python evaluate.py
    python evaluate.py --checkpoint outputs/best_model.pth
    python evaluate.py --split valid
"""

import argparse
import os

import torch
import torch.nn as nn

import config
from dataset import CurrencyDataset, get_eval_transforms, validate_dataset
from model import build_model
from torch.utils.data import DataLoader
from utils import (
    set_seed,
    get_device,
    MetricTracker,
    print_metrics,
    print_confusion_matrix,
    save_predictions_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Egyptian Currency Classifier")
    parser.add_argument(
        "--checkpoint", type=str, default=config.BEST_MODEL_PATH,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "valid", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Optional custom dataset directory containing class subfolders",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--save-viz", action="store_true", help="Save prediction visualization")
    return parser.parse_args()


def resolve_eval_data_dir(args) -> tuple[str, str]:
    """Resolve an evaluation data directory with sensible fallback behavior."""
    if args.data_dir:
        return args.data_dir, "custom"

    split_dirs = {
        "train": config.TRAIN_DIR,
        "valid": config.VALID_DIR,
        "test": config.TEST_DIR,
    }

    requested_dir = split_dirs[args.split]
    if os.path.isdir(requested_dir):
        return requested_dir, args.split

    if args.split == "test":
        for fallback_split in ("valid", "train"):
            fallback_dir = split_dirs[fallback_split]
            if os.path.isdir(fallback_dir):
                print(
                    f"Warning: test split not found at {requested_dir}. "
                    f"Falling back to '{fallback_split}' split at {fallback_dir}."
                )
                return fallback_dir, fallback_split

    raise FileNotFoundError(
        f"Dataset directory not found: {requested_dir}\n"
        "Create the expected split folders under data/ (train, valid, test), or run with "
        "--split valid/--split train, or provide --data-dir <path>."
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Full evaluation on a dataset split."""
    model.eval()
    tracker = MetricTracker(config.NUM_CLASSES, config.CLASS_NAMES)

    all_images = []
    all_preds = []
    all_targets = []
    all_confidences = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(dim=1)

        tracker.update(preds, targets, loss.item(), images.size(0))

        # Collect samples for visualization (detach + cpu immediately)
        if len(all_images) < 16:
            remaining = 16 - len(all_images)
            all_images.extend(images[:remaining].detach().cpu())
            all_preds.extend(preds[:remaining].detach().cpu().tolist())
            all_targets.extend(targets[:remaining].detach().cpu().tolist())
            all_confidences.extend(confidences[:remaining].detach().cpu().tolist())

        # Free batch tensors immediately
        del outputs, loss, probs, confidences, preds, images, targets

    metrics = tracker.compute()

    # Attach visualization data
    metrics["_viz_data"] = {
        "images": torch.stack(all_images[:16]) if all_images else None,
        "preds": all_preds[:16],
        "targets": all_targets[:16],
        "confidences": all_confidences[:16],
    }

    return metrics


def main():
    args = parse_args()
    set_seed()
    device = get_device()

    # Determine split directory (with fallback if test is missing)
    data_dir, resolved_split = resolve_eval_data_dir(args)

    # Validate dataset
    label = args.split if resolved_split == args.split else f"{args.split} -> {resolved_split}"
    print(f"\nEvaluating on '{label}' split")
    stats = validate_dataset(data_dir, config.CLASS_NAMES)
    total = sum(stats.values())
    print(f"Total images: {total}")

    # Create dataset and loader
    dataset = CurrencyDataset(data_dir, config.CLASS_NAMES, get_eval_transforms())
    use_pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train the model first with: python train.py"
        )

    model = build_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    ckpt_epoch = checkpoint.get("epoch", "?")
    print(f"Checkpoint from epoch: {ckpt_epoch}")

    # Loss (unweighted for fair evaluation)
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate(model, loader, criterion, device)

    # Print results
    print_metrics(metrics, prefix=f"{resolved_split.upper()} Set")
    print_confusion_matrix(metrics["confusion_matrix"], config.CLASS_NAMES)

    # Save visualization
    viz_data = metrics.pop("_viz_data")
    if args.save_viz and viz_data["images"] is not None:
        viz_path = os.path.join(config.OUTPUT_DIR, f"{resolved_split}_predictions.png")
        save_predictions_grid(
            viz_data["images"],
            viz_data["targets"],
            viz_data["preds"],
            viz_data["confidences"],
            config.CLASS_NAMES,
            viz_path,
        )

    print(f"\nEvaluation complete.")
    return metrics


if __name__ == "__main__":
    main()
