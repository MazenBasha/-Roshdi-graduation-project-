"""
Training script for Egyptian Currency Classification Model.

Trains a lightweight MobileNet-style model from scratch with:
- Weighted sampling for class imbalance
- Mixed precision training (AMP)
- Cosine/step learning rate scheduling
- Early stopping
- Checkpoint saving
- Metric logging

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 64 --lr 0.005
    python train.py --resume outputs/checkpoints/checkpoint_epoch_10.pth
"""

import argparse
import gc
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler

import config
from dataset import CurrencyDataset, create_dataloaders, validate_dataset
from model import build_model, model_summary, count_parameters
from utils import (
    set_seed,
    get_device,
    MetricTracker,
    EarlyStopping,
    TrainingLogger,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
    format_time,
    save_predictions_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Egyptian Currency Classifier")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--width-mult", type=float, default=config.MODEL_WIDTH_MULT)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    tracker = MetricTracker(config.NUM_CLASSES, config.CLASS_NAMES)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Zero gradients are already handled at the top, but explicitly
        # ensure the grad graph from this iteration is fully released.
        optimizer.zero_grad(set_to_none=True)

        # Detach predictions from the computation graph and extract loss
        # as a plain float BEFORE any reference keeps the graph alive.
        preds = outputs.detach().argmax(dim=1)
        loss_val = loss.item()
        batch_sz = images.size(0)

        # Update metrics with plain Python values / detached tensors only
        tracker.update(preds, targets, loss_val, batch_sz)

        # Explicitly free large tensors to release memory immediately.
        # Without this, the previous batch's tensors stay alive until
        # the next iteration overwrites local variables — doubling peak usage.
        del outputs, loss, images, targets, preds

        # Progress logging
        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            current_metrics = tracker.compute()
            print(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {current_metrics['loss']:.4f} "
                f"Acc: {current_metrics['accuracy']:.4f}"
            )

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    tracker = MetricTracker(config.NUM_CLASSES, config.CLASS_NAMES)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        tracker.update(preds, targets, loss.item(), images.size(0))

        # Free batch tensors immediately
        del outputs, loss, images, targets, preds

    return tracker.compute()


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)
    device = get_device()

    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Validate dataset
    print("\n--- Dataset Validation ---")
    for split_name, split_dir in [
        ("Train", config.TRAIN_DIR),
        ("Valid", config.VALID_DIR),
    ]:
        stats = validate_dataset(split_dir, config.CLASS_NAMES)
        total = sum(stats.values())
        print(f"{split_name}: {total} images across {len(stats)} classes")
        for cls, count in stats.items():
            print(f"  {cls}: {count}")

    # Create dataloaders
    print("\n--- Creating DataLoaders ---")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model FROM SCRATCH
    print("\n--- Building Model (FROM SCRATCH) ---")
    model = build_model(
        num_classes=config.NUM_CLASSES,
        width_mult=args.width_mult,
    )
    model_summary(model)
    model = model.to(device)

    # Loss function with label smoothing and class weights
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    )
    print(f"\nClass weights: {class_weights.cpu().tolist()}")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=config.MOMENTUM,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Learning rate scheduler
    if config.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=config.LR_MIN
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )

    # Mixed precision
    use_amp = args.no_amp is False and config.USE_AMP
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    print(f"Mixed precision: {'enabled' if use_amp and device.type == 'cuda' else 'disabled'}")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
    )

    # Logger
    logger = TrainingLogger()

    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["metrics"].get("accuracy", 0.0)
        model = model.to(device)
        print(f"Resumed at epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # ─── Training Loop ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Device: {device} | Batch size: {args.batch_size} | Workers: {args.num_workers}")
    print(f"{'=' * 60}")

    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.6f})")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, epoch, args.epochs,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

        # Free any lingering tensors between train/val and next epoch
        gc.collect()

        # Step scheduler
        scheduler.step()

        # Log metrics
        logger.log(epoch, train_metrics, val_metrics, current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(
            f"\n  Train  - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Valid  - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['macro_f1']:.4f}, "
            f"mAP: {val_metrics['mAP']:.4f}"
        )
        print(f"  Time: {format_time(epoch_time)}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, config.BEST_MODEL_PATH,
            )
            print(f"  >>> New best model saved! Val Acc: {best_val_acc:.4f}")

        # Periodic checkpoint
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, ckpt_path,
            )

        # Early stopping
        if early_stopping(val_metrics["accuracy"]):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break

    total_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Training log saved to: {logger.log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
