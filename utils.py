"""
Utility functions for training, evaluation, metrics, and visualization.
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import config


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ─── Metrics ────────────────────────────────────────────────────────────────

class MetricTracker:
    """Tracks and computes classification metrics."""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.total_loss = 0.0
        self.num_samples = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float, batch_size: int):
        """Update metrics with a batch of predictions."""
        # Defensively detach and move to CPU to avoid retaining
        # computation graphs or device memory across batches.
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        for p, t in zip(preds_np, targets_np):
            self.confusion_matrix[t, p] += 1
        self.total_loss += loss * batch_size
        self.num_samples += batch_size

    def compute(self) -> Dict:
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix
        avg_loss = self.total_loss / max(self.num_samples, 1)

        # Per-class metrics
        per_class = {}
        precisions = []
        recalls = []
        f1s = []
        aps = []

        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            support = cm[i, :].sum()

            # AP approximation (for classification, AP ≈ precision at recall threshold)
            ap = precision * recall  # Simplified AP for classification

            per_class[self.class_names[i]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "ap": float(ap),
                "support": int(support),
            }
            if support > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                aps.append(ap)

        # Overall accuracy
        correct = np.trace(cm)
        total = cm.sum()
        accuracy = correct / max(total, 1)

        # Macro averages
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1s) if f1s else 0.0
        mAP = np.mean(aps) if aps else 0.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "mAP": float(mAP),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }


# ─── Early Stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ─── Checkpoint Management ──────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    path: str,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# ─── Logging ────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, log_dir: str = config.LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.csv")
        self.history: List[Dict] = []

        # Write header
        with open(self.log_path, "w") as f:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,"
                "val_precision,val_recall,val_f1,val_mAP,lr\n"
            )

    def log(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log one epoch of metrics."""
        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["macro_precision"],
            "val_recall": val_metrics["macro_recall"],
            "val_f1": val_metrics["macro_f1"],
            "val_mAP": val_metrics["mAP"],
            "lr": lr,
        }
        self.history.append(entry)

        with open(self.log_path, "a") as f:
            f.write(
                f"{entry['epoch']},"
                f"{entry['train_loss']:.6f},{entry['train_acc']:.4f},"
                f"{entry['val_loss']:.6f},{entry['val_acc']:.4f},"
                f"{entry['val_precision']:.4f},{entry['val_recall']:.4f},"
                f"{entry['val_f1']:.4f},{entry['val_mAP']:.4f},"
                f"{entry['lr']:.8f}\n"
            )


# ─── Display Helpers ────────────────────────────────────────────────────────

def print_metrics(metrics: Dict, prefix: str = ""):
    """Print metrics in a formatted table."""
    print(f"\n{'=' * 60}")
    print(f"{prefix} Results")
    print(f"{'=' * 60}")
    print(f"  Loss:            {metrics['loss']:.4f}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  mAP:             {metrics['mAP']:.4f}")

    if "per_class" in metrics:
        print(f"\n  {'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP':>8} {'Support':>8}")
        print(f"  {'-'*55}")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(
                f"  {cls_name:<15} "
                f"{cls_metrics['precision']:>8.4f} "
                f"{cls_metrics['recall']:>8.4f} "
                f"{cls_metrics['f1']:>8.4f} "
                f"{cls_metrics['ap']:>8.4f} "
                f"{cls_metrics['support']:>8d}"
            )
    print(f"{'=' * 60}")


def print_confusion_matrix(cm: list, class_names: List[str]):
    """Print confusion matrix."""
    print(f"\nConfusion Matrix:")
    header = "  " + "  ".join(f"{name[:6]:>6}" for name in class_names)
    print(f"{'Pred ->':>12}{header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {class_names[i][:10]:<10}{row_str}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ─── Visualization ──────────────────────────────────────────────────────────

def save_predictions_grid(
    images: torch.Tensor,
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    class_names: List[str],
    save_path: str,
    max_images: int = 16,
):
    """
    Save a grid of sample predictions as an image.
    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization.")
        return

    n = min(len(images), max_images)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = [ax for row in (axes if rows > 1 else [axes]) for ax in (row if isinstance(row, (list, np.ndarray)) else [row])]

    mean = torch.tensor(config.IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMG_STD).view(3, 1, 1)

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n:
            img = images[i].cpu() * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            color = "green" if true_labels[i] == pred_labels[i] else "red"
            ax.set_title(
                f"True: {class_names[true_labels[i]]}\n"
                f"Pred: {class_names[pred_labels[i]]} ({confidences[i]:.1%})",
                color=color,
                fontsize=9,
            )
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions grid saved to: {save_path}")
