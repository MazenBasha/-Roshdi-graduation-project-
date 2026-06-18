"""
Offline face alignment + cleaning for a training set.

Reads an ImageFolder-style root, runs MTCNN on every image, applies a
5-point similarity-transform alignment to the canonical 112x112 ArcFace
template, and writes the aligned crop to a parallel output tree.

Doing this offline (vs. on-the-fly inside the DataLoader) is the right
trade-off for face recognition training:
  - MTCNN is the slowest part of the pipeline (~5x the embedding cost).
  - Aligned crops are deterministic; the DataLoader can hammer the
    augmented-but-aligned 112x112 jpegs at full speed.
  - Bad detections (no face / multiple ambiguous faces / tiny faces) are
    surfaced once at preprocessing time instead of polluting every epoch.

Cleaning rules:
  - Drop images where MTCNN finds no face above `--min-detection-conf`.
  - When MTCNN finds multiple faces, keep the most central one.
  - Drop images where the chosen face is smaller than `--min-face-size`.
  - Drop identities that end up with fewer than `--min-per-class` valid
    images after cleaning.

Usage:
    python preprocess_align.py \
        --in  data/casia-webface \
        --out data/casia-webface_aligned

Multiprocessing is intentionally NOT used here: MTCNN holds CUDA/MPS state
that doesn't share well across processes. The script is I/O- and detector-
bound; throughput is fine on a single worker (~10-20 imgs/sec on CPU,
30-60 imgs/sec on MPS). For huge datasets, shard the input root and run
multiple processes in parallel manually.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from inference import FaceDetector
from utils import get_device


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", type=str, required=True,
                   help="Input dataset root (folder per identity)")
    p.add_argument("--out", dest="dst", type=str, required=True,
                   help="Output dataset root (will be created)")
    p.add_argument("--size", type=int, default=config.INPUT_SIZE,
                   help="Output crop size in pixels")
    p.add_argument("--min-detection-conf", type=float, default=0.90)
    p.add_argument("--min-face-size", type=int, default=config.MIN_FACE_SIZE)
    p.add_argument("--min-per-class", type=int, default=2,
                   help="Drop identities with fewer aligned images than this.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process images that already exist in --out")
    return p.parse_args()


def pick_central(detections, image_shape):
    """When MTCNN finds several faces, keep the one closest to image center."""
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    return min(
        detections,
        key=lambda d: ((d[1][0] + d[1][2]) / 2 - cx) ** 2 + ((d[1][1] + d[1][3]) / 2 - cy) ** 2,
    )


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)

    device = get_device()
    detector = FaceDetector(device, min_face_size=args.min_face_size)
    print(f"[align] device={device} | size={args.size} | "
          f"min_conf={args.min_detection_conf}")

    identity_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    print(f"[align] {len(identity_dirs)} identities to scan")

    n_total = n_kept = n_skipped_nodet = n_skipped_small = 0
    dropped_identities: list[str] = []

    for id_dir in tqdm(identity_dirs, desc="identities"):
        out_dir = dst / id_dir.name
        out_dir.mkdir(exist_ok=True)
        files = sorted(f for f in id_dir.iterdir() if f.suffix.lower() in _VALID_EXTS)
        kept_for_id = 0
        for fpath in files:
            n_total += 1
            out_path = out_dir / fpath.name
            if out_path.exists() and not args.overwrite:
                kept_for_id += 1
                n_kept += 1
                continue

            img = cv2.imread(str(fpath))
            if img is None:
                continue
            dets = detector.detect(img, conf_thresh=args.min_detection_conf)
            if not dets:
                n_skipped_nodet += 1
                continue
            face, box, _lmk, _score = pick_central(dets, img.shape)
            face_w = box[2] - box[0]
            face_h = box[3] - box[1]
            if min(face_w, face_h) < args.min_face_size:
                n_skipped_small += 1
                continue
            if face.shape[:2] != (args.size, args.size):
                face = cv2.resize(face, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), face)
            kept_for_id += 1
            n_kept += 1

        if kept_for_id < args.min_per_class:
            shutil.rmtree(out_dir, ignore_errors=True)
            dropped_identities.append(id_dir.name)

    print()
    print("=" * 60)
    print(f"  Input images           : {n_total}")
    print(f"  Aligned + kept         : {n_kept}")
    print(f"  Skipped: no detection  : {n_skipped_nodet}")
    print(f"  Skipped: face too small: {n_skipped_small}")
    print(f"  Dropped identities     : {len(dropped_identities)} "
          f"(< {args.min_per_class} valid images each)")
    print(f"  Output -> {dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
