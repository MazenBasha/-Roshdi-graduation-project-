"""
Evaluate a trained YOLO checkpoint on the test split.

Reports detection metrics (mAP50, mAP50-95, precision, recall) and saves a
confusion matrix plot via ultralytics' built-in val pipeline.

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights outputs/runs/train/weights/best.pt
    python src/evaluate.py --split val          # evaluate on val instead of test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained YOLO weights")
    p.add_argument("--weights", default=config.DEFAULT_WEIGHTS,
                   help="Path to trained .pt weights")
    p.add_argument("--data", default=config.DATA_YAML)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--device", default="")
    p.add_argument("--name", default="eval", help="Run subfolder under outputs/runs/")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    model = load_model(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=config.RUNS_DIR,
        name=args.name,
        exist_ok=True,
        plots=True,                # saves PR / confusion-matrix plots
    )

    # ultralytics returns a DetMetrics object. Pull the headline numbers.
    box = metrics.box
    print("\n" + "=" * 60)
    print(f"Detection metrics on '{args.split}' split")
    print("=" * 60)
    print(f"  mAP@0.5      : {box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {box.map:.4f}")
    print(f"  Precision    : {box.mp:.4f}")
    print(f"  Recall       : {box.mr:.4f}")

    print(f"\nPer-class mAP@0.5:")
    print(f"  {'Class':<12} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print(f"  {'-' * 50}")
    for i, name in enumerate(config.CLASS_NAMES):
        try:
            p, r, ap50, ap = box.class_result(i)
            print(f"  {name:<12} {ap50:>8.4f} {ap:>10.4f} {p:>8.4f} {r:>8.4f}")
        except Exception:
            # Class may have no instances in this split.
            print(f"  {name:<12} {'n/a':>8} {'n/a':>10} {'n/a':>8} {'n/a':>8}")

    print("\nPlots + confusion matrix saved under "
          f"{os.path.join(config.RUNS_DIR, args.name)}")


if __name__ == "__main__":
    main()
