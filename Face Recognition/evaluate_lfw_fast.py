"""
Fast LFW evaluation that skips MTCNN (LFW funneled images are pre-centered)
and feeds them directly through the embedding model.

Use this when:
- You're benchmarking the embedding model (not the detector)
- You want a quick LFW number without 30+ minutes of MTCNN preprocessing

For an honest end-to-end number on un-cropped data, use `evaluate.py` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
from evaluate import kfold_evaluate, parse_pairs, roc_auc, tar_at_far
from model import build_embedding_model, load_checkpoint
from utils import get_device, l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="facenet_vggface2")
    p.add_argument("--input-size", type=int, default=160,
                   help="160 for FaceNet/VGGFace2, 112 for ArcFace iResNet")
    p.add_argument("--center-crop", type=int, default=160,
                   help="Center crop applied to the funneled 250x250 image "
                        "before resize. 160 -> face region only.")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    print(f"[fast-eval] {len(pairs)} pairs across {len(np.unique(folds))} folds")

    # Build model.
    if Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        print(f"[fast-eval] using {args.backbone} from {args.checkpoint}")
    else:
        model = build_embedding_model("facenet_vggface2")
        print("[fast-eval] using pretrained FaceNet (VGGFace2)")
    model.to(device).eval()

    # Preprocessing: center-crop the funneled image, resize to model input,
    # normalize per-channel to (x - 127.5) / 128 (the FaceNet recipe).
    tf = transforms.Compose([
        transforms.CenterCrop(args.center_crop),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
    ])

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"[fast-eval] embedding {len(unique_paths)} unique images")

    embeddings: dict[Path, np.ndarray] = {}
    missing = 0
    with torch.no_grad():
        batch_imgs: list[torch.Tensor] = []
        batch_paths: list[Path] = []

        def flush() -> None:
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs, dim=0).to(device)
            emb = model(x).detach().cpu().numpy()
            emb = l2_normalize(emb.astype(np.float32))
            for p, e in zip(batch_paths, emb):
                embeddings[p] = e
            batch_imgs.clear()
            batch_paths.clear()

        for p in tqdm(unique_paths, desc="embed"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                missing += 1
                continue
            batch_imgs.append(tf(img))
            batch_paths.append(p)
            if len(batch_imgs) >= args.batch_size:
                flush()
        flush()

    if missing:
        print(f"[fast-eval] warning: {missing} images failed to load")

    # Score every pair.
    scores: list[float] = []
    labels: list[int] = []
    keep_folds: list[int] = []
    for (a, b, label), fold in zip(pairs, folds):
        ea, eb = embeddings.get(a), embeddings.get(b)
        if ea is None or eb is None:
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(int(label))
        keep_folds.append(int(fold))
    scores_a = np.array(scores, dtype=np.float32)
    labels_a = np.array(labels, dtype=np.int32)
    folds_a = np.array(keep_folds, dtype=np.int32)
    print(f"[fast-eval] scored {len(scores_a)} pairs")

    grid = np.linspace(-0.1, 1.0, 222)
    kf = kfold_evaluate(scores_a, labels_a, folds_a, grid)
    auc = roc_auc(scores_a, labels_a)
    tar1e3, thr1e3 = tar_at_far(scores_a, labels_a, 1e-3)

    print()
    print("=" * 60)
    print("LFW verification results (fast path, no MTCNN)")
    print("=" * 60)
    print(f"  Backbone           : {args.backbone}")
    print(f"  Input size         : {args.input_size}x{args.input_size}")
    print(f"  Mean accuracy      : {kf['mean_accuracy']*100:.2f}% +/- {kf['std_accuracy']*100:.2f}%")
    print(f"  Per-fold accuracies:")
    for i, a in enumerate(kf['fold_accuracies']):
        print(f"    fold {i:2d}: {a*100:.2f}%   (threshold {kf['fold_thresholds'][i]:.3f})")
    print(f"  ROC AUC            : {auc:.4f}")
    print(f"  TAR @ FAR=1e-3     : {tar1e3*100:.2f}%   (threshold={thr1e3:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
