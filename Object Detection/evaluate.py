"""Evaluate a YOLO checkpoint against a YOLO-format dataset.

Default behaviour matches the project's production stance: evaluate the
**pretrained** `yolo11n.pt` against the built-in `coco128.yaml` (auto-
downloaded by Ultralytics — no Roboflow required). Both flags can be overridden
to evaluate a custom fine-tune against any data.yaml.

Outputs:
  - results/eval_<model>.json     overall + per-class AP
  - results/eval_<model>.csv      per-class table for the report

Usage:
    # production-default evaluation
    python evaluate.py

    # evaluate a custom fine-tune against a custom split
    python evaluate.py --weights runs/detect/.../weights/best.pt \\
                       --data datasets/COCO-50/data.yaml --baseline yolo11n.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate_one(weights: str, data: str, imgsz: int) -> dict:
    model = YOLO(weights)
    r = model.val(data=data, imgsz=imgsz, plots=False, verbose=False)
    per_class = {}
    try:
        ap50 = r.box.ap50.tolist()
        ap = r.box.ap.tolist()
        names = r.names if hasattr(r, "names") else model.names
        for idx in range(len(ap)):
            per_class[names[idx]] = {"AP50": ap50[idx], "AP50_95": ap[idx]}
    except Exception:
        pass
    return {
        "weights": weights,
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "per_class": per_class,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt",
                   help="Checkpoint to evaluate (pretrained name or local path)")
    p.add_argument("--data", default="coco128.yaml",
                   help="data.yaml. Default 'coco128.yaml' auto-downloads via Ultralytics.")
    p.add_argument("--baseline", default=None,
                   help="Optional second checkpoint for side-by-side comparison")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.weights).stem

    print(f"Evaluating {args.weights} on {args.data} …")
    fine = evaluate_one(args.weights, args.data, args.imgsz)
    result = {"primary": fine}

    if args.baseline:
        print(f"Evaluating baseline {args.baseline} …")
        base = evaluate_one(args.baseline, args.data, args.imgsz)
        result["baseline"] = base
        result["delta"] = {
            "mAP50":    fine["mAP50"]    - base["mAP50"],
            "mAP50_95": fine["mAP50_95"] - base["mAP50_95"],
        }
        print(f"\nΔ mAP50:    {result['delta']['mAP50']:+.4f}")
        print(f"Δ mAP50-95: {result['delta']['mAP50_95']:+.4f}")

    json_path = out_dir / f"eval_{name}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {json_path}")

    csv_path = out_dir / f"eval_{name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "AP50", "AP50_95", "AP50_baseline", "AP50_95_baseline"])
        base_pc = result.get("baseline", {}).get("per_class", {})
        for cls, m in fine["per_class"].items():
            b = base_pc.get(cls, {})
            w.writerow([cls,
                        f"{m['AP50']:.4f}", f"{m['AP50_95']:.4f}",
                        f"{b.get('AP50', '')}", f"{b.get('AP50_95', '')}"])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
