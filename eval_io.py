"""
Shared I/O helpers for the evaluation suite.

Centralises the contract for passing a chosen threshold between stages so
no module hardcodes 0.378 (or any other magic number).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_threshold(path_or_dir: str | Path, fallback: float) -> float:
    """Load the canonical decision threshold written by `evaluate_full.py`.

    Accepts:
      - a path to a `threshold.json` file, OR
      - a directory containing one (e.g. `reports/lfw_full/`).

    Returns `fallback` if the file is missing or malformed (so downstream
    scripts still run when invoked standalone). Prints a one-line note so
    the chosen path is visible in logs.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "threshold.json"
    if not p.exists():
        print(f"[eval_io] no threshold.json at {p} — falling back to {fallback}")
        return float(fallback)
    try:
        data = json.loads(p.read_text())
        t = float(data["threshold"])
        print(f"[eval_io] loaded threshold={t:.4f} from {p}")
        return t
    except Exception as e:
        print(f"[eval_io] could not parse {p} ({e}) — falling back to {fallback}")
        return float(fallback)
