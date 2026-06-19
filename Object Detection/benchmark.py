"""Latency/throughput benchmark. Reports p50/p95/p99 + RPS for one worker.

Defaults to the pretrained yolo11n.pt — no training required.

Usage:
    python benchmark.py --image samples/test.jpg --warmup 10 --iters 200

    # to benchmark a custom fine-tune instead:
    python benchmark.py --weights path/to/custom.pt --image samples/test.jpg
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt",
                   help="Pretrained name (auto-downloaded) or path to a custom .pt")
    p.add_argument("--image", required=True, help="Sample input image")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model = YOLO(args.weights)
    img = np.array(Image.open(args.image).convert("RGB"))

    for _ in range(args.warmup):
        model.predict(source=img, imgsz=args.imgsz, verbose=False, device=args.device)

    latencies = []
    t_start = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=args.imgsz, verbose=False, device=args.device)
        latencies.append((time.perf_counter() - t0) * 1000)
    total_s = time.perf_counter() - t_start

    latencies.sort()
    def q(pct: float) -> float:
        return latencies[min(int(pct * len(latencies)), len(latencies) - 1)]

    report = {
        "iterations": args.iters,
        "imgsz": args.imgsz,
        "p50_ms": q(0.50),
        "p95_ms": q(0.95),
        "p99_ms": q(0.99),
        "mean_ms": statistics.fmean(latencies),
        "stdev_ms": statistics.pstdev(latencies),
        "rps_single_worker": args.iters / total_s,
    }
    print(json.dumps(report, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
