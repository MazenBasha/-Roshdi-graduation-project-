"""Drift detection.

Compares the *current* per-class detection-frequency distribution (read from
Prometheus metrics or a local JSON snapshot) against a baseline captured at
training time. Reports per-class deltas + a population stability index (PSI).

Usage:
    python drift.py --baseline results/baseline_distribution.json \
                    --current  results/current_distribution.json

If PSI > 0.25 for any class or > 0.10 overall, the script exits 2 — wire this
into a cron / CI job to page on drift.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def psi(expected: float, actual: float, eps: float = 1e-4) -> float:
    e = max(expected, eps)
    a = max(actual, eps)
    return (a - e) * math.log(a / e)


def normalise(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--threshold-class", type=float, default=0.25)
    p.add_argument("--threshold-overall", type=float, default=0.10)
    args = p.parse_args()

    base = normalise(json.loads(args.baseline.read_text()))
    curr = normalise(json.loads(args.current.read_text()))

    all_classes = sorted(set(base) | set(curr))
    rows = []
    overall = 0.0
    for c in all_classes:
        b, a = base.get(c, 0.0), curr.get(c, 0.0)
        s = psi(b, a)
        overall += s
        rows.append((c, b, a, s))

    print(f"{'class':25s} {'baseline':>10s} {'current':>10s} {'psi':>8s}")
    for c, b, a, s in sorted(rows, key=lambda r: -abs(r[3]))[:30]:
        print(f"{c:25s} {b:10.4f} {a:10.4f} {s:8.4f}")
    print(f"\nOverall PSI: {overall:.4f}")

    alarms = [(c, s) for c, _, _, s in rows if abs(s) > args.threshold_class]
    if alarms:
        print("\nDRIFT ALERT — classes exceeding per-class PSI threshold:")
        for c, s in alarms:
            print(f"  {c}: PSI={s:.3f}")
        sys.exit(2)
    if overall > args.threshold_overall:
        print(f"\nDRIFT ALERT — overall PSI {overall:.3f} > {args.threshold_overall}")
        sys.exit(2)


if __name__ == "__main__":
    main()
