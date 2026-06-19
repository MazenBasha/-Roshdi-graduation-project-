"""[OPTIONAL] Fine-tune YOLO11n on a custom YOLO-format dataset.

This script is **not part of the production pipeline**. Production inference
uses the pretrained `yolo11n.pt` directly (see server/inference.py and
README.md §1). Use this only if you want to specialise the detector to a
domain (e.g. Egyptian street scenes after user-feedback collection).

Usage:
    python train.py --data datasets/COCO-25/data.yaml --epochs 30

The data.yaml path is printed by download_dataset.py. On a single T4/A100
batch=16 and imgsz=640 is a good starting point. Reduce batch if you OOM.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Pretrained weights to fine-tune from")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto",
                        help="auto / 0 / 0,1 / cpu / mps")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="coco_yolo11n")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)

    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=20,
        save=True,
        plots=True,
    )

    val_results = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"mAP50:    {val_results.box.map50:.4f}")

    best = Path(train_results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
