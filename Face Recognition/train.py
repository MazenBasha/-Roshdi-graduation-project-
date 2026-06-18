"""
Two-stage face recognition training (from scratch).

Why two stages?
---------------
ArcFace's additive angular margin makes the loss surface hostile at random
initialization: cos(theta + 0.5) for the target class is roughly -0.48 while
non-target classes sit near 0. Multiplied by scale=64, the optimizer is told
"every wrong class is more correct than the right one" — and a tiny model
with no learned features can't crawl out of that pit. The published symptom
is exactly what the previous attempt hit: loss plateaus near log(C * exp(s))
and accuracy stays at 0%.

The fix used in InsightFace, CurricularFace, and most production face
training pipelines is a two-stage schedule:

  Stage 1  ->  CosineHead + CrossEntropy  (no margin)
                Lets the backbone learn an embedding geometry where the
                target class direction is consistently the maximum cosine.
                Trains for 5-10 epochs.
                Optionally freezes the backbone for the first 1-2 epochs so
                the head can adapt to whatever random features it sees,
                then unfreezes everything.

  Stage 2  ->  ArcFaceHead (margin=0.5, scale=64)
                With Stage-1 weights, cos(theta_y) is already ~0.6+ for the
                right class, so cos(theta_y + 0.5) ~= 0.1 -> the additive
                margin now actually pushes the cluster tighter instead of
                breaking the model. Trains for 20-50 epochs.
                ArcFace head's class-direction matrix is initialized from
                Stage-1's CosineHead weights (same shape and same geometry).

Usage:
    python train.py --data data/casia-webface --epochs-stage1 8 --epochs-stage2 30
    python train.py --data data/digiface_smoke --epochs-stage1 5 --epochs-stage2 15 \
                    --backbone iresnet18 --batch-size 64

Resume across stages:
    python train.py --data ... --resume checkpoints/run.pt
The checkpoint records which stage was active; we pick up there.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import ArcFaceHead, CosineHead, build_iresnet, count_parameters
from utils import get_device, set_seed
import mlflow_utils as mlu


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)

    # Architecture
    p.add_argument("--backbone", type=str, default="iresnet18",
                   choices=["iresnet18", "iresnet34", "iresnet50", "iresnet100"])
    p.add_argument("--embedding-dim", type=int, default=config.EMBEDDING_DIM)

    # Stage 1 (warmup, CrossEntropy + CosineHead)
    p.add_argument("--epochs-stage1", type=int, default=8)
    p.add_argument("--lr-stage1", type=float, default=0.1)
    p.add_argument("--scale-stage1", type=float, default=30.0,
                   help="Cosine softmax scale during Stage 1. Smaller than "
                        "Stage-2's scale=64 by design: with low scale, the "
                        "softmax is less peaky, so cross-entropy needs cos "
                        "to actually be HIGH (not just larger than others) "
                        "to drive the loss down. This pre-conditions the "
                        "embeddings so Stage 2's angular margin lands on "
                        "cos values it can actually shift.")
    p.add_argument("--freeze-backbone-epochs", type=int, default=1,
                   help="In Stage 1, keep backbone frozen for this many epochs "
                        "so the head adapts to random features. Set to 0 to "
                        "always train end-to-end.")

    # Stage 2 (ArcFace)
    p.add_argument("--epochs-stage2", type=int, default=30)
    p.add_argument("--lr-stage2", type=float, default=0.01,
                   help="Lower than Stage-1 LR by default. Stage 1 already "
                        "shaped the embeddings; we don't want to undo it.")
    p.add_argument("--scale", type=float, default=config.ARC_SCALE)
    p.add_argument("--margin", type=float, default=config.ARC_MARGIN)
    p.add_argument("--margin-warmup-epochs", type=int, default=3,
                   help="Linearly ramp the ArcFace margin from 0 to --margin "
                        "over this many epochs at the start of Stage 2. "
                        "Avoids the loss explosion when the angular penalty "
                        "exceeds what Stage-1's modest cosine separations can "
                        "support. Set to 0 to use the full margin from epoch 0.")
    p.add_argument("--margin-hold-epochs", type=int, default=0,
                   help="Hold margin at 0 for this many Stage-2 epochs BEFORE "
                        "starting the linear warmup. Lets Stage-2's lower LR "
                        "consolidate the embeddings under the new ArcFace head "
                        "before the angular penalty turns on. The full margin "
                        "schedule is: hold (margin=0) -> warmup (linear 0->m) "
                        "-> full (margin=m). Total Stage-2 epochs should be "
                        ">= margin_hold_epochs + margin_warmup_epochs.")

    # Optimizer + schedule (shared)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of each stage spent in linear LR warmup")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Max grad norm; 0 disables clipping")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--min-samples-per-class", type=int, default=2)

    # Misc
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--output", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--log-dir", type=str, default=str(config.LOG_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision; only effective on CUDA")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warmup -> cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def freeze_backbone(backbone: nn.Module, frozen: bool) -> None:
    for p in backbone.parameters():
        p.requires_grad = not frozen
    # The final BN's `weight` was always frozen in the iResNet definition;
    # leave that as-is so we don't accidentally re-enable it.


@dataclass
class EpochStats:
    loss: float
    acc: float
    seconds: float


# ---------------------------------------------------------------------------
# Train / val one-epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch_label: str,
    writer: SummaryWriter,
    global_step: int,
    use_amp: bool,
    grad_clip: float,
    head_takes_labels: bool,
) -> tuple[EpochStats, int]:
    backbone.train()
    head.train()
    is_cuda_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda_amp)

    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()

    pbar = tqdm(loader, desc=epoch_label, leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=is_cuda_amp):
            embeddings = backbone(imgs)
            logits = head(embeddings, labels) if head_takes_labels else head(embeddings)
            loss = nn.functional.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            params = [p for p in backbone.parameters() if p.requires_grad]
            params += list(head.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

        global_step += 1
        if global_step % 50 == 0:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{(correct / max(1, total)) * 100:.2f}%",
                lr=f"{optimizer.param_groups[0]['lr']:.4f}",
            )

    return (
        EpochStats(loss=loss_sum / max(1, total),
                   acc=correct / max(1, total),
                   seconds=time.time() - t0),
        global_step,
    )


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    head: nn.Module,
    loader,
    device: torch.device,
    head_takes_labels: bool,
) -> EpochStats:
    backbone.eval()
    head.eval()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings = backbone(imgs)
        logits = head(embeddings, labels) if head_takes_labels else head(embeddings)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
    return EpochStats(
        loss=loss_sum / max(1, total),
        acc=correct / max(1, total),
        seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"[train] device={device} | arch={args.backbone} | dim={args.embedding_dim}")
    run_name = f"train_{args.backbone}_{Path(args.data).name}_{time.strftime('%Y%m%d_%H%M%S')}"
    mlflow_run_cm = mlu.init_run(
        experiment="train",
        run_name=run_name,
        params={
            **vars(args),
            "loss_stage1": "CrossEntropy+CosineHead",
            "loss_stage2": f"ArcFace(margin={args.margin}, scale={args.scale})",
            "optimizer": "SGD(nesterov=True)",
            "dataset": Path(args.data).name,
        },
        category="Training",
        tags={"step": "train", "backbone": args.backbone,
              "device": str(device)},
    )
    mlflow_run = mlflow_run_cm.__enter__()
    try:
        _main_impl(args, device)
    finally:
        mlflow_run_cm.__exit__(None, None, None)


def _main_impl(args, device) -> None:
    if args.amp and device.type != "cuda":
        print("[train] --amp ignored (only effective on CUDA; current device is "
              f"{device.type}).")

    # --- Data ---
    train_loader, val_loader, num_classes = build_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        min_samples_per_class=args.min_samples_per_class,
    )
    print(f"[train] num_classes={num_classes} | "
          f"train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    # --- Backbone (random init, both stages share these weights) ---
    backbone = build_iresnet(args.backbone, embedding_dim=args.embedding_dim).to(device)
    print(f"[train] backbone params = {count_parameters(backbone) / 1e6:.2f}M")

    # --- Setup logging + checkpoints ---
    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[train] tb log dir = {log_dir}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_acc = 0.0
    start_stage = 1
    start_epoch = 0
    cosine_state = None  # weights from CosineHead used to init ArcFace head
    arcface_state = None  # weights from ArcFaceHead, restored when resuming Stage 2

    # --- Resume ---
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["model_state"])
        start_stage = ckpt.get("stage", 1)
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        cosine_state = ckpt.get("cosine_state")
        arcface_state = ckpt.get("head_state")
        print(f"[train] resumed {args.resume} | stage {start_stage} epoch {start_epoch}")

    # ----------------------------------------------------------------------
    # STAGE 1: CrossEntropy + CosineHead (warmup)
    # ----------------------------------------------------------------------
    if start_stage <= 1:
        print("\n" + "=" * 70)
        print(f"STAGE 1  (CrossEntropy + CosineHead)  -  {args.epochs_stage1} epochs")
        print("=" * 70)
        head1 = CosineHead(args.embedding_dim, num_classes, scale=args.scale_stage1).to(device)

        # First N epochs: backbone frozen so the head can adapt to whatever
        # random features the conv stack produces. Avoids wasting compute on
        # gradients that fight a random head.
        backbone_currently_frozen = args.freeze_backbone_epochs > 0
        freeze_backbone(backbone, backbone_currently_frozen)
        if backbone_currently_frozen:
            print(f"[stage1] backbone FROZEN for first {args.freeze_backbone_epochs} epoch(s)")

        params = [p for p in backbone.parameters() if p.requires_grad]
        params += list(head1.parameters())
        optimizer = SGD(params, lr=args.lr_stage1, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)
        total_steps_s1 = max(1, args.epochs_stage1 * len(train_loader))
        warmup_steps_s1 = max(1, int(args.warmup_frac * total_steps_s1))
        scheduler = build_scheduler(optimizer, total_steps_s1, warmup_steps_s1)

        global_step = 0
        for ep in range(start_epoch, args.epochs_stage1):
            # Unfreeze the backbone partway through Stage 1 if needed.
            if backbone_currently_frozen and ep >= args.freeze_backbone_epochs:
                print(f"[stage1] epoch {ep}: UNFREEZING backbone")
                freeze_backbone(backbone, False)
                backbone_currently_frozen = False
                # Rebuild optimizer so it sees the newly-trainable params.
                optimizer = SGD(
                    [p for p in backbone.parameters() if p.requires_grad] + list(head1.parameters()),
                    lr=optimizer.param_groups[0]["lr"],   # keep current LR
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True,
                )
                # Continue the same schedule from the same step count.
                remaining_steps = max(1, total_steps_s1 - global_step)
                remaining_warmup = max(0, warmup_steps_s1 - global_step)
                scheduler = build_scheduler(optimizer, remaining_steps + global_step, warmup_steps_s1)
                # Fast-forward scheduler to the current global_step.
                for _ in range(global_step):
                    scheduler.step()

            ts, global_step = train_one_epoch(
                backbone, head1, train_loader, optimizer, scheduler, device,
                f"s1 ep{ep}", writer, global_step, args.amp, args.grad_clip,
                head_takes_labels=False,
            )
            vs = evaluate(backbone, head1, val_loader, device, head_takes_labels=False)
            print(f"[s1 ep {ep:02d}] "
                  f"train_loss={ts.loss:.4f} train_acc={ts.acc * 100:.2f}% | "
                  f"val_loss={vs.loss:.4f} val_acc={vs.acc * 100:.2f}% | "
                  f"{ts.seconds:.1f}s")
            history.append({"stage": 1, "epoch": ep,
                            "train": asdict(ts), "val": asdict(vs)})

            writer.add_scalar("stage1/train_loss", ts.loss, ep)
            writer.add_scalar("stage1/train_acc", ts.acc, ep)
            writer.add_scalar("stage1/val_loss", vs.loss, ep)
            writer.add_scalar("stage1/val_acc", vs.acc, ep)
            mlu.log_metrics_flat({
                "stage1.train_loss": ts.loss, "stage1.train_acc": ts.acc,
                "stage1.val_loss": vs.loss, "stage1.val_acc": vs.acc,
                "stage1.epoch_seconds": ts.seconds,
                "best_val_acc": max(best_val_acc, vs.acc),
            }, step=ep)

            (log_dir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save({
                "stage": 1,
                "epoch": ep,
                "model_state": backbone.state_dict(),
                "cosine_state": head1.state_dict(),
                "best_val_acc": max(best_val_acc, vs.acc),
                "args": vars(args),
                "num_classes": num_classes,
            }, args.output)
            if vs.acc > best_val_acc:
                best_val_acc = vs.acc
                torch.save(
                    {"stage": 1, "epoch": ep, "model_state": backbone.state_dict(),
                     "best_val_acc": best_val_acc, "args": vars(args),
                     "num_classes": num_classes},
                    str(Path(args.output).with_suffix(".best.pt")),
                )

        cosine_state = head1.state_dict()
        del head1
        start_epoch = 0  # reset for stage 2
    else:
        print(f"[train] resuming straight into Stage 2 (cosine_state from checkpoint)")

    # ----------------------------------------------------------------------
    # STAGE 2: ArcFace
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STAGE 2  (ArcFace, margin={args.margin}, scale={args.scale})  "
          f"-  {args.epochs_stage2} epochs")
    print("=" * 70)
    head2 = ArcFaceHead(args.embedding_dim, num_classes,
                        scale=args.scale, margin=args.margin).to(device)
    if arcface_state is not None:
        # Resuming mid-Stage-2: restore the trained ArcFace head verbatim so
        # the optimizer continues from the exact geometry the checkpoint saved.
        head2.load_state_dict(arcface_state)
        print("[stage2] ArcFace head restored from resumed Stage-2 checkpoint")
    elif cosine_state is not None and "weight" in cosine_state:
        # CosineHead and ArcFaceHead both have `weight` of shape
        # (num_classes, embedding_dim). Direct copy is safe and gives ArcFace
        # a head that already separates classes -> the margin term lands on a
        # geometry where it can actually help.
        with torch.no_grad():
            head2.weight.copy_(cosine_state["weight"])
        print("[stage2] ArcFace head initialized from Stage-1 cosine head")
    else:
        print("[stage2] ArcFace head random-initialized "
              "(no Stage-1 weights to inherit)")

    freeze_backbone(backbone, False)  # always full-model in stage 2
    params = list(backbone.parameters()) + list(head2.parameters())
    optimizer = SGD(params, lr=args.lr_stage2, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    total_steps_s2 = max(1, args.epochs_stage2 * len(train_loader))
    warmup_steps_s2 = max(1, int(args.warmup_frac * total_steps_s2))
    scheduler = build_scheduler(optimizer, total_steps_s2, warmup_steps_s2)

    global_step = 0
    for ep in range(start_epoch, args.epochs_stage2):
        # 3-phase margin schedule:
        #   ep < hold:                       margin = 0       (consolidate)
        #   hold <= ep < hold+warmup:        margin = m * (ep-hold)/warmup
        #   ep >= hold+warmup:               margin = m       (full)
        # The hold phase is critical when the LR is low: gives the optimizer
        # several epochs to settle into the cosine geometry under the ArcFace
        # head before any angular penalty is applied.
        hold = args.margin_hold_epochs
        warmup = args.margin_warmup_epochs
        if ep < hold:
            current_margin = 0.0
            phase = "hold"
        elif warmup > 0 and ep < hold + warmup:
            current_margin = args.margin * (ep - hold) / warmup
            phase = "warmup"
        else:
            current_margin = args.margin
            phase = "full"
        head2.set_margin(current_margin)
        writer.add_scalar("stage2/current_margin", current_margin, ep)
        if phase != "full":
            print(f"[stage2] epoch {ep}: margin = {current_margin:.3f} "
                  f"({phase}; target {args.margin})")

        ts, global_step = train_one_epoch(
            backbone, head2, train_loader, optimizer, scheduler, device,
            f"s2 ep{ep}", writer, global_step, args.amp, args.grad_clip,
            head_takes_labels=True,
        )
        vs = evaluate(backbone, head2, val_loader, device, head_takes_labels=True)
        print(f"[s2 ep {ep:02d}] "
              f"train_loss={ts.loss:.4f} train_acc={ts.acc * 100:.2f}% | "
              f"val_loss={vs.loss:.4f} val_acc={vs.acc * 100:.2f}% | "
              f"{ts.seconds:.1f}s")
        history.append({"stage": 2, "epoch": ep,
                        "train": asdict(ts), "val": asdict(vs)})

        writer.add_scalar("stage2/train_loss", ts.loss, ep)
        writer.add_scalar("stage2/train_acc", ts.acc, ep)
        writer.add_scalar("stage2/val_loss", vs.loss, ep)
        writer.add_scalar("stage2/val_acc", vs.acc, ep)
        mlu.log_metrics_flat({
            "stage2.train_loss": ts.loss, "stage2.train_acc": ts.acc,
            "stage2.val_loss": vs.loss, "stage2.val_acc": vs.acc,
            "stage2.epoch_seconds": ts.seconds,
            "stage2.current_margin": current_margin,
            "best_val_acc": max(best_val_acc, vs.acc),
        }, step=args.epochs_stage1 + ep)

        (log_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({
            "stage": 2,
            "epoch": ep,
            "model_state": backbone.state_dict(),
            "head_state": head2.state_dict(),
            "best_val_acc": max(best_val_acc, vs.acc),
            "args": vars(args),
            "num_classes": num_classes,
        }, args.output)
        if vs.acc > best_val_acc:
            best_val_acc = vs.acc
            torch.save(
                {"stage": 2, "epoch": ep, "model_state": backbone.state_dict(),
                 "head_state": head2.state_dict(),
                 "best_val_acc": best_val_acc, "args": vars(args),
                 "num_classes": num_classes},
                str(Path(args.output).with_suffix(".best.pt")),
            )

    writer.close()
    print(f"\n[train] DONE. best val_acc = {best_val_acc * 100:.2f}%. "
          f"latest -> {args.output} ; best -> "
          f"{Path(args.output).with_suffix('.best.pt')}")

    # MLflow: register best checkpoint + log latest checkpoint + tensorboard logs.
    best_path = Path(args.output).with_suffix(".best.pt")
    mlu.log_artifact_file(args.output, artifact_path="checkpoints")
    mlu.log_artifact_dir(log_dir, artifact_path="tensorboard")
    mlu.register_best_model(
        checkpoint=best_path,
        model_name=mlu._CFG.get("registered_model_name", "face-recognition-iresnet"),
        stage="Staging",
        description=f"{args.backbone} | best val_acc={best_val_acc * 100:.2f}% | "
                    f"data={Path(args.data).name}",
    )


if __name__ == "__main__":
    main()
