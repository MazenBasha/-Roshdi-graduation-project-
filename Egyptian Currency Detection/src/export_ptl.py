"""
Export the trained YOLOv8 detector to TorchScript Lite (.ptl) for PyTorch
Mobile (Android / iOS).

Pipeline:
    best.pt
      └─> ultralytics export (TorchScript)        -> best.torchscript
            └─> torch.jit.load + mobile_optimizer  -> best.ptl   (Lite interpreter)

Usage:
    python src/export_ptl.py
    python src/export_ptl.py --weights outputs/runs/train2/weights/best.pt
    python src/export_ptl.py --imgsz 416     # match what was used at training
    python src/export_ptl.py --no-optimize   # skip the mobile_optimizer step

The exported .ptl outputs the YOLO head's raw tensor. Run NMS on-device
(or use ultralytics' included post-processing on the host).
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 weights to TorchScript Lite")
    p.add_argument("--weights", default=config.DEFAULT_WEIGHTS,
                   help="Path to trained .pt weights")
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE,
                   help="Input image size used at export time (must match runtime)")
    p.add_argument("--output", default=None,
                   help="Output .ptl path (default: alongside the .pt)")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip optimize_for_mobile()")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Train first: python src/train.py"
        )

    weights_dir = os.path.dirname(os.path.abspath(args.weights))
    stem = os.path.splitext(os.path.basename(args.weights))[0]

    # Default output: same folder as best.pt, named best.ptl
    out_ptl = args.output or os.path.join(weights_dir, f"{stem}.ptl")
    ts_path = os.path.join(weights_dir, f"{stem}.torchscript")

    print("=" * 60)
    print("Export YOLOv8 -> TorchScript Lite (.ptl)")
    print("=" * 60)
    print(f"  Source weights : {args.weights}")
    print(f"  Image size     : {args.imgsz}x{args.imgsz}")
    print(f"  TorchScript    : {ts_path}")
    print(f"  Output (.ptl)  : {out_ptl}")

    # ── Step 1: ultralytics → TorchScript ────────────────────────────────
    print("\n[1/3] Tracing model via ultralytics export(format='torchscript')...")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    # ultralytics writes <stem>.torchscript next to the .pt
    produced = model.export(format="torchscript", imgsz=args.imgsz, optimize=False)
    # `produced` is the path it wrote; copy/rename to our expected ts_path.
    if isinstance(produced, (list, tuple)):
        produced = produced[0]
    if str(produced) != ts_path and os.path.exists(produced):
        # ultralytics already wrote it under the canonical name
        ts_path = str(produced)
    print(f"  TorchScript saved: {ts_path}")

    # ── Step 2: load TorchScript + (optional) mobile optimization ────────
    print("\n[2/3] Loading TorchScript and applying mobile optimizations...")
    scripted = torch.jit.load(ts_path, map_location="cpu")
    scripted.eval()

    if not args.no_optimize:
        try:
            scripted = optimize_for_mobile(scripted)
            print("  optimize_for_mobile: applied")
        except Exception as e:
            print(f"  optimize_for_mobile: SKIPPED ({e})")
    else:
        print("  optimize_for_mobile: skipped (--no-optimize)")

    # ── Step 3: save for Lite interpreter ────────────────────────────────
    print("\n[3/3] Saving for PyTorch Mobile Lite interpreter...")
    scripted._save_for_lite_interpreter(out_ptl)
    size_mb = os.path.getsize(out_ptl) / 1024 / 1024
    print(f"  Saved: {out_ptl}")
    print(f"  Size:  {size_mb:.2f} MB")

    # ── Smoke test the exported .ptl ─────────────────────────────────────
    print("\n[verify] Running a forward pass on a dummy tensor...")
    loaded = torch.jit.load(out_ptl, map_location="cpu")
    loaded.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = loaded(dummy)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if isinstance(out, (list, tuple)):
        shapes = [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__ for x in out]
    else:
        shapes = tuple(out.shape)
    print(f"  Output shape(s): {shapes}")
    print(f"  CPU forward:     {elapsed_ms:.1f} ms")

    print("\n" + "=" * 60)
    print("Export complete.")
    print(f"  File:    {out_ptl}")
    print(f"  Size:    {size_mb:.2f} MB")
    print(f"  Classes: {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"  Input:   1 x 3 x {args.imgsz} x {args.imgsz}, normalized to [0,1]")
    print("\nOn mobile, run NMS on the raw output tensor "
          "(or use the ultralytics post-processing helpers).")
    print("=" * 60)


if __name__ == "__main__":
    main()
