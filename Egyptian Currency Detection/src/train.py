"""
Train a YOLOv8 detector on the Egyptian currency dataset.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch 8
    python src/train.py --model yolov8s.pt        # bigger backbone
    python src/train.py --resume                  # resume last run

Outputs:
    outputs/runs/train/weights/best.pt    # best checkpoint
    outputs/runs/train/weights/last.pt
    outputs/runs/train/results.png        # loss / mAP curves
    outputs/runs/train/confusion_matrix.png
"""

import argparse
import os
import sys

# Allow `python src/train.py` from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on currency data")
    p.add_argument("--data", default=config.DATA_YAML, help="Path to data.yaml")
    p.add_argument("--model", default=config.BASE_MODEL,
                   help="Base model: yolov8n.pt / yolov8s.pt / yolov8m.pt")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    p.add_argument("--resume", action="store_true", help="Resume last run")
    p.add_argument("--name", default="train", help="Run subfolder under outputs/runs/")
    p.add_argument("--low-mem", action="store_true",
                   help="Disable mosaic + mixup (cuts CPU RAM use roughly in half)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"data.yaml not found: {args.data}\n"
            "See README.md for the dataset labeling guide."
        )

    from ultralytics import YOLO

    os.makedirs(config.RUNS_DIR, exist_ok=True)

    model = YOLO(args.model)

    # Augmentation knobs tuned for hand-held currency photos:
    # - hsv_*  : lighting / camera color shifts
    # - degrees / perspective : rotated and tilted notes
    # - translate / scale : note can sit anywhere, at any distance
    # - mosaic : forces the model to learn multi-note scenes from single-note crops
    # - mixup  : extra regularization
    # - flipud=0 : currency is rarely upside-down in real life; keep low
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device or None,
        seed=config.SEED,
        project=config.RUNS_DIR,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.0 if args.low_mem else 1.0,
        mixup=0.0 if args.low_mem else 0.15,
        copy_paste=0.0 if args.low_mem else 0.1,
        # optimization
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
    )

    print(f"\nTraining done. Best weights: "
          f"{os.path.join(config.RUNS_DIR, args.name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
