"""Export YOLO weights to mobile-friendly formats.

Default exports the pretrained `yolo11n.pt` — no training required. Override
`--weights` to export a custom fine-tune.

Usage:
    # production-default: export pretrained
    python export.py --formats onnx tflite coreml

    # with INT8 quantization (needs --data for calibration images)
    python export.py --formats tflite coreml --int8 --data coco128.yaml --nms

INT8 quantization typically shrinks the model ~4x with a small mAP drop.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="Pretrained name (auto-downloaded) or path to a custom .pt")
    parser.add_argument("--formats", nargs="+",
                        default=["onnx", "tflite", "coreml"],
                        choices=["onnx", "tflite", "coreml", "torchscript", "ncnn"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantization (needs --data for calibration)")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--data", default=None,
                        help="data.yaml for INT8 calibration")
    parser.add_argument("--nms", action="store_true",
                        help="Embed NMS in the exported graph (CoreML/TFLite)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out_paths = []
    for fmt in args.formats:
        kwargs = dict(format=fmt, imgsz=args.imgsz)
        if args.int8 and fmt in {"tflite", "onnx", "coreml"}:
            kwargs["int8"] = True
            if args.data:
                kwargs["data"] = args.data
        if args.half and not args.int8:
            kwargs["half"] = True
        if args.nms and fmt in {"coreml", "tflite"}:
            kwargs["nms"] = True
        path = model.export(**kwargs)
        out_paths.append((fmt, path))

    print("\nExported:")
    for fmt, path in out_paths:
        size_mb = Path(path).stat().st_size / 1e6 if Path(path).exists() else 0
        print(f"  {fmt:10s} -> {path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
