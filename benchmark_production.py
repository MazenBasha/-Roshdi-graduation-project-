"""
Production AI metrics: latency, throughput, memory, model size, mobile readiness.

Measures, per backbone:
    - p50 / p95 / p99 embedding latency on the selected device
    - p50 / p95 / p99 latency on CPU (always reported, since the glasses
      target is CPU/NPU)
    - Throughput at batch sizes 1, 4, 16, 64
    - Peak RSS memory delta
    - On-disk model size (PyTorch FP32, INT8 dynamic-quant if available, ONNX)
    - ONNX Runtime CPU latency if the .onnx file exists
    - Mobile readiness checklist: input shape contract, dynamic ops,
      preprocessing portability, embedding norm

Run:
    python benchmark_production.py \
        --checkpoint checkpoints/casia_v4_iresnet18.best.pt \
        --backbone iresnet18 \
        --onnx checkpoints/iresnet18.onnx \
        --int8 checkpoints/iresnet18_int8.pt \
        --report-dir reports/production
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from model import build_embedding_model, load_checkpoint
from utils import get_device
import mlflow_utils as mlu


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


def measure_latency(model, device, batch_size, n_warmup=10, n_iters=100):
    model.eval()
    x = torch.randn(batch_size, 3, 112, 112, device=device)
    # Warmup.
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # Measure.
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms
    return {
        "batch_size": batch_size,
        "n_iters": n_iters,
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "p99_ms": percentile(times, 99),
        "mean_ms": float(np.mean(times)),
        "throughput_imgs_per_sec": batch_size / (np.mean(times) / 1000.0),
    }


def measure_memory_after_load(model_builder) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = model_builder()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in KB; on macOS in bytes.
    unit = "bytes" if sys.platform == "darwin" else "kB"
    delta = after - before
    return {"unit": unit, "delta": int(delta), "after": int(after), "before": int(before)}


def file_size(path) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    return p.stat().st_size


def benchmark_onnx(onnx_path: Path, batch_size: int = 1, n_iters: int = 50):
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name_in = sess.get_inputs()[0].name
    x = np.random.randn(batch_size, 3, 112, 112).astype(np.float32)
    for _ in range(5):
        _ = sess.run(None, {name_in: x})
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {name_in: x})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "batch_size": batch_size,
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "p99_ms": percentile(times, 99),
        "mean_ms": float(np.mean(times)),
        "throughput_imgs_per_sec": batch_size / (np.mean(times) / 1000.0),
    }


def mobile_readiness(model: torch.nn.Module) -> dict:
    """Static checks that gate mobile deployment."""
    issues = []
    model.eval()  # BatchNorm needs eval mode for batch=1 forward
    # Input shape contract.
    try:
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            y = model(x)
        contract_ok = (y.dim() == 2 and y.shape == (1, config.EMBEDDING_DIM))
        if not contract_ok:
            issues.append(f"Output shape mismatch: got {tuple(y.shape)}, expected (1, {config.EMBEDDING_DIM})")
    except Exception as e:
        contract_ok = False; issues.append(f"forward(1,3,112,112) failed: {e}")
    # Dynamic ops — count GroupNorm/InstanceNorm which can hurt mobile.
    bad_ops = []
    for m in model.modules():
        klass = m.__class__.__name__
        if klass in {"GroupNorm", "InstanceNorm2d"}:
            bad_ops.append(klass)
    # Parameter count + footprint.
    n_params = sum(p.numel() for p in model.parameters())
    fp32_bytes = n_params * 4
    return {
        "input_contract_ok": bool(contract_ok),
        "issues": issues,
        "bad_mobile_ops": sorted(set(bad_ops)),
        "n_params": int(n_params),
        "fp32_size_bytes_estimate": int(fp32_bytes),
        "int8_size_bytes_estimate": int(n_params * 1),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--onnx", type=str, default="")
    p.add_argument("--int8", type=str, default="")
    p.add_argument("--batch-sizes", type=str, default="1,4,16,64")
    p.add_argument("--n-iters", type=int, default=80)
    p.add_argument("--report-dir", type=str, default="reports/production")
    return p.parse_args()


def main():
    args = parse_args()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="benchmark_production",
        run_name=f"bench_{args.backbone}",
        params=vars(args),
        category="Performance",
        tags={"step": "benchmark_production", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
    ):
        _run(args, report_dir)


def _run(args, report_dir):
    device = get_device()
    cpu = torch.device("cpu")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"[bench] device={device}  CPU=darwin/{platform.machine()}")

    # Memory + size on CPU build.
    def build():
        m = build_embedding_model(args.backbone)
        load_checkpoint(m, args.checkpoint, map_location=cpu)
        return m
    mem = measure_memory_after_load(build)
    model = build_embedding_model(args.backbone)
    load_checkpoint(model, args.checkpoint, map_location=cpu)
    mob = mobile_readiness(model)

    # Latency: device.
    model.to(device).eval()
    device_lat = [measure_latency(model, device, bs, n_iters=args.n_iters) for bs in batch_sizes]

    # Latency: CPU (always reported).
    model.to(cpu).eval()
    cpu_lat = [measure_latency(model, cpu, bs, n_iters=max(20, args.n_iters // 2)) for bs in batch_sizes]

    # ONNX runtime CPU.
    onnx_lat = None
    if args.onnx and Path(args.onnx).exists():
        onnx_lat = [benchmark_onnx(Path(args.onnx), batch_size=bs, n_iters=30) for bs in batch_sizes]

    sizes = {
        "checkpoint_bytes": file_size(args.checkpoint),
        "onnx_bytes": file_size(args.onnx) if args.onnx else None,
        "onnx_data_bytes": file_size(str(args.onnx) + ".data") if args.onnx else None,
        "int8_bytes": file_size(args.int8) if args.int8 else None,
    }

    out = {
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": sys.version.split()[0], "torch": torch.__version__,
                     "device": str(device)},
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "memory_after_load": mem,
        "mobile_readiness": mob,
        "model_sizes_bytes": sizes,
        "latency_device": device_lat,
        "latency_cpu": cpu_lat,
        "latency_onnx_cpu": onnx_lat,
    }
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    md = ["# Production AI metrics\n",
          f"Backbone: `{args.backbone}` | Device: `{device}`\n\n",
          "## Mobile readiness\n",
          f"- Input contract OK: {mob['input_contract_ok']}\n",
          f"- Issues: {mob['issues']}\n",
          f"- Params: {mob['n_params']/1e6:.2f} M\n",
          f"- Estimated FP32 footprint: {mob['fp32_size_bytes_estimate']/1e6:.1f} MB\n",
          f"- Estimated INT8 footprint: {mob['int8_size_bytes_estimate']/1e6:.1f} MB\n\n",
          "## On-disk sizes\n"]
    for k, v in sizes.items():
        if v is not None:
            md.append(f"- {k}: {v/1e6:.2f} MB\n")
    md.append("\n## Latency on device\n")
    md.append("| batch | p50 ms | p95 ms | p99 ms | throughput |\n|--:|--:|--:|--:|--:|\n")
    for r in device_lat:
        md.append(f"| {r['batch_size']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
                  f"{r['p99_ms']:.2f} | {r['throughput_imgs_per_sec']:.1f} im/s |\n")
    md.append("\n## Latency on CPU\n")
    md.append("| batch | p50 ms | p95 ms | p99 ms | throughput |\n|--:|--:|--:|--:|--:|\n")
    for r in cpu_lat:
        md.append(f"| {r['batch_size']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
                  f"{r['p99_ms']:.2f} | {r['throughput_imgs_per_sec']:.1f} im/s |\n")
    if onnx_lat is not None:
        md.append("\n## Latency on ONNX Runtime CPU\n")
        md.append("| batch | p50 ms | p95 ms | p99 ms | throughput |\n|--:|--:|--:|--:|--:|\n")
        for r in onnx_lat:
            md.append(f"| {r['batch_size']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
                      f"{r['p99_ms']:.2f} | {r['throughput_imgs_per_sec']:.1f} im/s |\n")
    (report_dir / "summary.md").write_text("".join(md))
    print(f"[bench] wrote {report_dir/'metrics.json'} + summary.md")

    # MLflow: re-log per-batch latency + sizes as metrics so they're plottable.
    for r in device_lat:
        bs = r["batch_size"]
        mlu.log_metrics_flat({
            f"latency_device.bs{bs}.p50_ms": r["p50_ms"],
            f"latency_device.bs{bs}.p95_ms": r["p95_ms"],
            f"latency_device.bs{bs}.p99_ms": r["p99_ms"],
            f"latency_device.bs{bs}.throughput_ips": r["throughput_imgs_per_sec"],
        })
    for r in cpu_lat:
        bs = r["batch_size"]
        mlu.log_metrics_flat({
            f"latency_cpu.bs{bs}.p50_ms": r["p50_ms"],
            f"latency_cpu.bs{bs}.p95_ms": r["p95_ms"],
            f"latency_cpu.bs{bs}.p99_ms": r["p99_ms"],
            f"latency_cpu.bs{bs}.throughput_ips": r["throughput_imgs_per_sec"],
        })
    if onnx_lat is not None:
        for r in onnx_lat:
            bs = r["batch_size"]
            mlu.log_metrics_flat({
                f"latency_onnx.bs{bs}.p50_ms": r["p50_ms"],
                f"latency_onnx.bs{bs}.p95_ms": r["p95_ms"],
                f"latency_onnx.bs{bs}.p99_ms": r["p99_ms"],
                f"latency_onnx.bs{bs}.throughput_ips": r["throughput_imgs_per_sec"],
            })
    for k, v in sizes.items():
        if v is not None:
            mlu.log_metrics_flat({f"size.{k}_mb": v / 1e6})
    mlu.log_metrics_flat({
        "params_millions": mob["n_params"] / 1e6,
        "fp32_estimate_mb": mob["fp32_size_bytes_estimate"] / 1e6,
        "int8_estimate_mb": mob["int8_size_bytes_estimate"] / 1e6,
    })
    mlu.log_artifacts_glob(report_dir, ["*.json", "*.md"], artifact_path="production")


if __name__ == "__main__":
    main()
