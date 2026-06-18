"""
Recognition-quality benchmark: enroll N images, test on the rest.

Mirrors the realistic deployment scenario: the user enrolls themselves
from a few photos, then we measure whether the system recognizes them
across many other photos AND doesn't mistake other people for them.

Reports per-model:
    - True-positive rate (correct recognitions of the enrolled person)
    - False-positive rate (other people misidentified as enrolled)
    - Mean cosine similarity of true-positive vs true-negative pairs

Compares the from-scratch CASIA model against the pretrained FaceNet
baseline so we have an honest "how much SOTA gap is left" number.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import config
from inference import FaceRecognizer
from utils import FaceDatabase, l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", default="data/sklearn_lfw/lfw_home/lfw_funneled")
    p.add_argument("--enroll-name", default="Tony_Blair",
                   help="LFW identity to enroll. Pick someone with many "
                        "samples (Tony_Blair=144, Colin_Powell=236, etc).")
    p.add_argument("--n-enroll", type=int, default=5)
    p.add_argument("--n-test-pos", type=int, default=20,
                   help="How many positives (other photos of enrolled) to test")
    p.add_argument("--n-test-neg", type=int, default=100,
                   help="How many negatives (other identities) to test")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    return p.parse_args()


def benchmark(rec: FaceRecognizer, args, label: str) -> None:
    rng = random.Random(args.seed)
    lfw = Path(args.lfw_root)
    target_dir = lfw / args.enroll_name
    target_imgs = sorted(target_dir.glob("*.jpg"))
    if len(target_imgs) < args.n_enroll + args.n_test_pos:
        raise RuntimeError(
            f"{args.enroll_name} only has {len(target_imgs)} images; need "
            f"{args.n_enroll + args.n_test_pos}"
        )
    rng.shuffle(target_imgs)
    enroll_imgs = target_imgs[: args.n_enroll]
    pos_imgs = target_imgs[args.n_enroll : args.n_enroll + args.n_test_pos]

    # Negatives: random images from random other identities
    other_ids = [d for d in lfw.iterdir() if d.is_dir() and d.name != args.enroll_name]
    rng.shuffle(other_ids)
    neg_imgs: list[Path] = []
    for d in other_ids:
        files = list(d.glob("*.jpg"))
        if files:
            neg_imgs.append(rng.choice(files))
        if len(neg_imgs) >= args.n_test_neg:
            break

    # Reset DB and enroll.
    rec.db.records.clear()
    rec.db.save()
    n_enrolled = 0
    for p in enroll_imgs:
        n_enrolled += rec.enroll_from_image(cv2.imread(str(p)), args.enroll_name)

    # Score every positive image -> measure TP rate + similarity
    pos_sims: list[float] = []
    pos_correct = 0
    for p in pos_imgs:
        results = rec.recognize_image(cv2.imread(str(p)), threshold=args.threshold)
        if not results:
            pos_sims.append(0.0)
            continue
        # Pick the most central detected face.
        primary = max(results, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
        pos_sims.append(primary.similarity)
        if primary.name == args.enroll_name:
            pos_correct += 1

    neg_sims: list[float] = []
    neg_false_positive = 0
    for p in neg_imgs:
        results = rec.recognize_image(cv2.imread(str(p)), threshold=args.threshold)
        if not results:
            neg_sims.append(0.0)
            continue
        primary = max(results, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
        neg_sims.append(primary.similarity)
        if primary.name == args.enroll_name:
            neg_false_positive += 1

    tp_rate = pos_correct / max(1, len(pos_imgs))
    fp_rate = neg_false_positive / max(1, len(neg_imgs))
    pos_mean = float(np.mean(pos_sims)) if pos_sims else 0.0
    neg_mean = float(np.mean(neg_sims)) if neg_sims else 0.0
    margin = pos_mean - neg_mean

    print(f"\n{'=' * 72}")
    print(f"{label}")
    print(f"{'=' * 72}")
    print(f"  Enrolled {args.enroll_name} from {n_enrolled} embeddings "
          f"(across {args.n_enroll} images)")
    print(f"  Threshold = {args.threshold:.2f}")
    print(f"  TRUE POSITIVES  : {pos_correct}/{len(pos_imgs)} = {tp_rate * 100:.1f}%   "
          f"(mean cos = {pos_mean:.3f})")
    print(f"  FALSE POSITIVES : {neg_false_positive}/{len(neg_imgs)} = {fp_rate * 100:.1f}%   "
          f"(mean cos = {neg_mean:.3f})")
    print(f"  MARGIN          : {margin:.3f}   (gap between pos and neg cosines; bigger = better)")


def main() -> None:
    args = parse_args()

    # Run with from-scratch CASIA checkpoint.
    rec = FaceRecognizer(checkpoint=args.checkpoint, backbone=args.backbone)
    benchmark(rec, args, label=f"FROM-SCRATCH model ({args.backbone}, CASIA 1000 IDs / 13 epochs)")

    # Reset checkpoint path so the FaceRecognizer falls back to pretrained.
    rec_pre = FaceRecognizer(checkpoint="this_path_does_not_exist", backbone="iresnet18")
    benchmark(rec_pre, args, label="PRETRAINED baseline (FaceNet/VGGFace2, comparison only)")


if __name__ == "__main__":
    main()
