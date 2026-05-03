"""
LFW verification benchmark.

The standard LFW protocol: 6000 face pairs (3000 same, 3000 different) split
into 10 folds. For each fold we:
    1. Compute embeddings for all faces in the pairs
    2. Search a cosine-similarity threshold on the held-out 9 folds
    3. Score the remaining fold at that threshold

Reported metrics:
    - Mean accuracy +/- std across the 10 folds (the headline number)
    - Best-overall threshold on the full 6000 pairs (for sanity)
    - TAR @ FAR=1e-3 (true accept rate at 0.1% false accept rate)
    - ROC AUC

Usage:
    python evaluate.py --lfw-root data/lfw/lfw --pairs data/lfw/pairs.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config
from inference import FaceDetector
from model import build_embedding_model, load_checkpoint
from utils import batch_preprocess, get_device, l2_normalize


# ---------------------------------------------------------------------------
# Pairs file parser (LFW format)
# ---------------------------------------------------------------------------

def parse_pairs(pairs_file: Path, lfw_root: Path) -> tuple[list[tuple[Path, Path, int]], np.ndarray]:
    """
    LFW pairs.txt format:
        First line:  <num_folds>  <num_pairs_per_fold_per_class>
        Then for each fold:
            <num_pairs_per_fold_per_class> "same" lines:    name  i  j
            <num_pairs_per_fold_per_class> "diff" lines:    name1 i1 name2 j2
    """
    lines = pairs_file.read_text().splitlines()
    header = lines[0].split()
    num_folds, num_per_class = int(header[0]), int(header[1])
    fold_size = num_per_class * 2

    pairs: list[tuple[Path, Path, int]] = []
    folds: list[int] = []

    cursor = 1
    for fold_idx in range(num_folds):
        for _ in range(num_per_class):
            tok = lines[cursor].split()
            cursor += 1
            name, i, j = tok[0], int(tok[1]), int(tok[2])
            p1 = lfw_root / name / f"{name}_{i:04d}.jpg"
            p2 = lfw_root / name / f"{name}_{j:04d}.jpg"
            pairs.append((p1, p2, 1))
            folds.append(fold_idx)
        for _ in range(num_per_class):
            tok = lines[cursor].split()
            cursor += 1
            n1, i1, n2, i2 = tok[0], int(tok[1]), tok[2], int(tok[3])
            p1 = lfw_root / n1 / f"{n1}_{i1:04d}.jpg"
            p2 = lfw_root / n2 / f"{n2}_{i2:04d}.jpg"
            pairs.append((p1, p2, 0))
            folds.append(fold_idx)

    return pairs, np.array(folds, dtype=np.int32)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(
    paths: Iterable[Path],
    detector: FaceDetector,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> dict[Path, np.ndarray]:
    """Detect + align + embed every image. Returns {path: 512-dim L2 vec}."""
    paths = list(paths)
    aligned: list[np.ndarray] = []
    keep_paths: list[Path] = []
    fail = 0

    for p in tqdm(paths, desc="detect+align"):
        img = cv2.imread(str(p))
        if img is None:
            fail += 1
            continue
        dets = detector.detect(img, conf_thresh=0.5)
        if not dets:
            # LFW images are pre-cropped, so MTCNN sometimes rejects them;
            # fall back to a center-crop of the original image.
            h, w = img.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            crop = img[y0:y0 + side, x0:x0 + side]
            crop = cv2.resize(crop, (config.INPUT_SIZE, config.INPUT_SIZE))
            aligned.append(crop)
        else:
            # Pick the most central detected face.
            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2
            best = min(dets, key=lambda d: (
                ((d[1][0] + d[1][2]) / 2 - cx) ** 2 + ((d[1][1] + d[1][3]) / 2 - cy) ** 2
            ))
            aligned.append(best[0])
        keep_paths.append(p)

    if fail:
        print(f"[evaluate] warning: {fail} images failed to load")

    embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(aligned), batch_size), desc="embed"):
        batch = batch_preprocess(aligned[i:i + batch_size]).to(device)
        emb = model(batch).detach().cpu().numpy()
        embeddings.append(l2_normalize(emb))
    if not embeddings:
        return {}
    embeddings = np.concatenate(embeddings, axis=0)
    return dict(zip(keep_paths, embeddings))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = (scores >= threshold).astype(np.int32)
    return float((preds == labels).mean())


def kfold_evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    grid: np.ndarray,
) -> dict:
    """For each of the 10 folds: pick threshold on the other 9, score on this one."""
    fold_ids = np.unique(folds)
    accs: list[float] = []
    chosen_thr: list[float] = []
    for f in fold_ids:
        train_mask = folds != f
        test_mask = folds == f
        accs_grid = [_accuracy_at_threshold(scores[train_mask], labels[train_mask], t) for t in grid]
        best_t = float(grid[int(np.argmax(accs_grid))])
        chosen_thr.append(best_t)
        accs.append(_accuracy_at_threshold(scores[test_mask], labels[test_mask], best_t))
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "fold_accuracies": [float(a) for a in accs],
        "fold_thresholds": chosen_thr,
    }


def tar_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> tuple[float, float]:
    """True Accept Rate at a fixed False Accept Rate. Returns (TAR, threshold)."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    sorted_neg = np.sort(neg_scores)[::-1]
    if len(sorted_neg) == 0:
        return 0.0, 1.0
    idx = int(np.ceil(target_far * len(sorted_neg))) - 1
    idx = max(0, min(idx, len(sorted_neg) - 1))
    threshold = float(sorted_neg[idx])
    tar = float((pos_scores >= threshold).mean())
    return tar, threshold


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal ROC AUC. Avoids sklearn dependency at this layer."""
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    P = max(1, int(y.sum()))
    N = max(1, len(y) - P)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    # np.trapezoid was added in NumPy 2.0 (np.trapz removed). Fall back for 1.x.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", type=str, required=True,
                   help="Path to extracted LFW images (folder of folders)")
    p.add_argument("--pairs", type=str, required=True,
                   help="Path to LFW pairs.txt")
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    print(f"[evaluate] {len(pairs)} pairs across {len(np.unique(folds))} folds")

    # Build model: trained checkpoint if present, else pretrained FaceNet.
    if Path(args.checkpoint).is_file():
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        print(f"[evaluate] using {args.backbone} from {args.checkpoint}")
    else:
        model = build_embedding_model("facenet_vggface2")
        print("[evaluate] no checkpoint -> evaluating pretrained FaceNet (VGGFace2)")
    model.to(device).eval()

    detector = FaceDetector(device)

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings = compute_embeddings(unique_paths, detector, model, device, args.batch_size)

    # Compute pairwise cosine similarity for each pair (skip pairs we couldn't embed).
    scores: list[float] = []
    labels: list[int] = []
    keep_folds: list[int] = []
    for (a, b, label), fold in zip(pairs, folds):
        ea, eb = embeddings.get(a), embeddings.get(b)
        if ea is None or eb is None:
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(label)
        keep_folds.append(int(fold))
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    keep_folds = np.array(keep_folds, dtype=np.int32)

    print(f"[evaluate] scored {len(scores)} pairs")

    grid = np.linspace(-0.1, 1.0, 222)
    kfold = kfold_evaluate(scores, labels, keep_folds, grid)
    auc = roc_auc(scores, labels)
    tar1e3, thr1e3 = tar_at_far(scores, labels, 1e-3)

    print()
    print("=" * 60)
    print("LFW verification results")
    print("=" * 60)
    print(f"  Mean accuracy      : {kfold['mean_accuracy']*100:.2f}% "
          f"+/- {kfold['std_accuracy']*100:.2f}%")
    print(f"  Per-fold thresholds: "
          f"min={min(kfold['fold_thresholds']):.3f} "
          f"max={max(kfold['fold_thresholds']):.3f}")
    print(f"  ROC AUC            : {auc:.4f}")
    print(f"  TAR @ FAR=1e-3     : {tar1e3*100:.2f}%  (threshold={thr1e3:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
