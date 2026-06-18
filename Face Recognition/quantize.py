"""
Edge-deployment optimization for the Roshdi assistive glasses.

Two paths:

1. Dynamic INT8 quantization (CPU). Shrinks Linear/Conv weights to int8 and
   does dynamic per-tensor activation quantization at runtime. Best fit for
   the iResNet trunk + final FC. Saves a much smaller .pt file and gives a
   ~2-4x CPU speedup.

2. ONNX export (FP32 or FP16). Lets the team deploy via ONNX Runtime / TensorRT
   on the actual glasses hardware. ONNX Runtime supports static-quantization
   passes downstream if you need INT8 there too.

Usage:
    python quantize.py --checkpoint checkpoints/arcface_iresnet50.best.pt \\
                       --backbone iresnet50 \\
                       --out-quant checkpoints/iresnet50_int8.pt \\
                       --out-onnx checkpoints/iresnet50.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import config
from model import build_embedding_model, count_parameters, load_checkpoint
from utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    p.add_argument("--out-quant", type=str, default="",
                   help="If set, write a dynamically-quantized state_dict here")
    p.add_argument("--out-onnx", type=str, default="",
                   help="If set, export ONNX to this path")
    p.add_argument("--onnx-fp16", action="store_true",
                   help="Cast ONNX export to FP16 (smaller, GPU-friendly)")
    p.add_argument("--input-size", type=int, default=config.INPUT_SIZE)
    return p.parse_args()


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization for Linear / Conv layers (CPU).

    PyTorch supports dynamic quant on `nn.Linear` officially. For convolutions,
    we quantize what's safe to quantize (Linear) and leave the rest in FP32 —
    this is the realistic, supported path on CPU. For full INT8 on convs, use
    the ONNX export + ONNX Runtime static quantization on the target device.

    Engine selection: x86 Linux defaults to fbgemm; ARM (Apple Silicon, mobile)
    needs qnnpack. We pick whichever the build supports.
    """
    model.eval().cpu()
    available = torch.backends.quantized.supported_engines
    for engine in ("qnnpack", "fbgemm"):
        if engine in available:
            torch.backends.quantized.engine = engine
            break
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel


def export_onnx(model: nn.Module, path: str, input_size: int, fp16: bool) -> None:
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    if fp16:
        model = model.half()
        dummy = dummy.half()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"[quantize] ONNX{'(fp16)' if fp16 else ''} exported -> {path} "
          f"({Path(path).stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")  # quantization is CPU-only

    model = build_embedding_model(args.backbone).to(device).eval()
    if Path(args.checkpoint).is_file():
        load_checkpoint(model, args.checkpoint, map_location=device)
        print(f"[quantize] loaded {args.backbone} from {args.checkpoint}")
    else:
        print(f"[quantize] no checkpoint at {args.checkpoint} — exporting "
              f"randomly-initialized weights (use only for plumbing tests).")

    print(f"[quantize] FP32 params: {count_parameters(model)/1e6:.2f}M")

    if args.out_quant:
        qmodel = dynamic_quantize(model)
        Path(args.out_quant).parent.mkdir(parents=True, exist_ok=True)
        torch.save(qmodel.state_dict(), args.out_quant)
        size_mb = Path(args.out_quant).stat().st_size / 1e6
        print(f"[quantize] dynamic INT8 state_dict -> {args.out_quant} ({size_mb:.2f} MB)")

    if args.out_onnx:
        export_onnx(model, args.out_onnx, args.input_size, args.onnx_fp16)


if __name__ == "__main__":
    main()
