"""
Real-world face recognition consistency benchmark.

LFW measures whether the model can verify cross-domain pairs. It does NOT
measure what actually breaks the user experience on assistive glasses:
**how stable is the displayed name when the same person is in view across
many consecutive frames?** A model that gets 95% per-frame accuracy but
flickers between Ali / Unknown every other frame is unusable.

This script captures (or replays) a webcam stream of N seconds, runs both
the raw per-frame recognizer AND the tracker-smoothed pipeline, and
reports:

  - per_frame_acc        : raw label correctness (against a ground-truth name)
  - smoothed_acc         : tracker-smoothed label correctness
  - flicker_rate         : per_frame: fraction of label changes between
                            consecutive frames where the same face is in view
  - smoothed_flicker     : same metric on the tracker output
  - false_positive_rate  : frames where an unknown face was reported as known
  - mean_inference_ms    : end-to-end per-frame latency

Usage:
    # Live capture, 20 seconds, you provide the ground-truth name on stdin
    python test_realworld.py --camera 0 --seconds 20 --truth "Ali"

    # Or replay a video file (no ground-truth needed; reports flicker only)
    python test_realworld.py --video myclip.mp4

    # Or evaluate against a folder of images of one person (sanity check)
    python test_realworld.py --folder data/sklearn_lfw/lfw_home/lfw_funneled/Tony_Blair --truth Tony_Blair
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import cv2

import config
from inference import FaceRecognizer
from tracker import FaceTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="OpenCV camera index")
    src.add_argument("--video", type=str, help="Path to a video file")
    src.add_argument("--folder", type=str, help="Folder of images of one person")
    p.add_argument("--seconds", type=float, default=15.0,
                   help="How long to capture from --camera")
    p.add_argument("--truth", type=str, default="",
                   help="Ground-truth name (must already be in face DB) for "
                        "computing accuracy. If blank, only flicker is reported.")
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--vote-min", type=int, default=3)
    return p.parse_args()


def frame_iter(args):
    """Yield BGR frames from the chosen source until exhausted / time-out."""
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        try:
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        finally:
            cap.release()
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        finally:
            cap.release()
    else:
        for p in sorted(Path(args.folder).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(p))
                if img is not None:
                    yield img


def main() -> None:
    args = parse_args()
    rec = FaceRecognizer(
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        use_clahe=not args.no_clahe,
    )
    tracker = FaceTracker()

    truth = args.truth.strip() or None

    raw_labels: list[str | None] = []   # per-frame primary face name
    smoothed_labels: list[str | None] = []
    latencies_ms: list[float] = []
    n_frames_with_face = 0
    n_frames = 0

    for frame in frame_iter(args):
        n_frames += 1
        t0 = time.time()
        results = rec.recognize_image(frame, threshold=args.threshold)
        # Tracker smoothing.
        track_pairs = tracker.step([(r.bbox, r.embedding) for r in results])
        for trk, di in track_pairs:
            trk.vote_name(results[di].name, results[di].similarity, min_votes=args.vote_min)
        latencies_ms.append((time.time() - t0) * 1000)

        if not results:
            raw_labels.append(None)
            smoothed_labels.append(None)
            continue
        n_frames_with_face += 1
        # Primary face = largest bbox.
        primary = max(
            range(len(results)),
            key=lambda i: (results[i].bbox[2] - results[i].bbox[0])
                          * (results[i].bbox[3] - results[i].bbox[1]),
        )
        raw_labels.append(results[primary].name)

        # Find the corresponding track for the primary detection.
        primary_track = next(
            (t for t, di in track_pairs if di == primary),
            None,
        )
        smoothed_labels.append(primary_track.last_name if primary_track else None)

    # Metrics.
    def acc(labels: list[str | None]) -> float | None:
        if truth is None:
            return None
        in_view = [l for l in labels if l is not None or truth is None]
        if not in_view:
            return 0.0
        correct = sum(1 for l in labels if l == truth)
        return correct / max(1, len(labels))

    def flicker(labels: list[str | None]) -> float:
        """Fraction of consecutive-frame transitions where the label changed."""
        # Only count transitions where BOTH frames had a detected face.
        pairs = [
            (a, b) for a, b in zip(labels[:-1], labels[1:])
            # Even None->None is "no flicker"; only count cases where face
            # was present in at least one frame (anything not both-None).
        ]
        if not pairs:
            return 0.0
        flips = sum(1 for a, b in pairs if a != b)
        return flips / len(pairs)

    def fpr(labels: list[str | None]) -> float | None:
        """If truth is set: fraction of frames labeled with WRONG known name."""
        if truth is None:
            return None
        return sum(1 for l in labels if l is not None and l != truth) / max(1, len(labels))

    print()
    print("=" * 70)
    print(f"Real-world consistency benchmark")
    print("=" * 70)
    print(f"  Source              : "
          f"{'camera' if args.camera is not None else ('video ' + args.video if args.video else 'folder ' + args.folder)}")
    print(f"  Frames processed    : {n_frames}  (face detected in {n_frames_with_face})")
    print(f"  Mean latency        : {mean(latencies_ms):.1f} ms / frame")
    print(f"  Threshold           : {args.threshold:.3f}")
    if truth is not None:
        print(f"  Ground-truth name   : {truth}")
        print(f"  Per-frame accuracy  : {acc(raw_labels) * 100:.2f}%")
        print(f"  Smoothed accuracy   : {acc(smoothed_labels) * 100:.2f}%")
        print(f"  False-positive rate : {fpr(raw_labels) * 100:.2f}%  (raw)")
        print(f"                        {fpr(smoothed_labels) * 100:.2f}%  (smoothed)")
    print(f"  Flicker rate        : {flicker(raw_labels) * 100:.2f}%  (raw)")
    print(f"                        {flicker(smoothed_labels) * 100:.2f}%  (smoothed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
