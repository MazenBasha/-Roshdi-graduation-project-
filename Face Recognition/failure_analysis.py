"""
Failure analysis on LFW pairs.

Given a trained model, embed all LFW pairs, then for each pair compute:
    - cosine similarity
    - prediction (>= threshold) vs ground truth label
    - failure category by simple heuristics:
        * different identities, accepted ("look-alike FP")
        * same identity, rejected ("missed FN")
        * sub-categorize the FN by mean brightness gap, image-size gap, and
          aspect-ratio gap of the two crops (proxies for illumination /
          resolution / pose)

Outputs:
    fp_gallery.png     — top-32 hardest false positives (highest cos, wrong)
    fn_gallery.png     — top-32 hardest false negatives (lowest cos, wrong)
    fp_hardest.html / fn_hardest.html — clickable HTML pages
    failure_metrics.json — per-category counts, mean cos, examples list
    summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
from evaluate import parse_pairs
from eval_io import load_threshold
from model import build_embedding_model, load_checkpoint
from utils import get_device, l2_normalize
import mlflow_utils as mlu


def _tf(input_size, center_crop=160):
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
    ])


def categorize_failure(img_a_path: Path, img_b_path: Path, kind: str) -> str:
    """Heuristic failure category."""
    a = cv2.imread(str(img_a_path)); b = cv2.imread(str(img_b_path))
    if a is None or b is None:
        return "io_error"
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).mean()
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).mean()
    brightness_gap = abs(ga - gb)
    aspect_gap = abs((a.shape[1] / a.shape[0]) - (b.shape[1] / b.shape[0]))
    res_gap = abs(a.shape[0] - b.shape[0])

    cats = []
    if brightness_gap > 35:
        cats.append("illumination")
    if aspect_gap > 0.10:
        cats.append("pose_or_aspect")
    if res_gap > 40:
        cats.append("resolution")
    if not cats:
        cats.append("ambiguous")
    if kind == "fp":
        cats.insert(0, "look_alike")
    return "+".join(cats)


def render_gallery(rows: list[dict], title: str, out_png: Path, n_cols: int = 8):
    n = min(32, len(rows))
    n_cols = min(n_cols, max(1, n))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2,
                             figsize=(2 * n_cols * 2, 2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for k in range(n):
        r = rows[k]
        rr = k // n_cols; cc = (k % n_cols) * 2
        for off, p in enumerate((r["a"], r["b"])):
            img = cv2.imread(str(p))
            ax = axes[rr][cc + off]
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        axes[rr][cc].set_title(f"cos={r['score']:.2f}", fontsize=7)
        axes[rr][cc + 1].set_title(r["category"], fontsize=7, color="red")
    # Hide unused axes.
    for k in range(n, n_rows * n_cols):
        rr = k // n_cols; cc = (k % n_cols) * 2
        axes[rr][cc].axis("off"); axes[rr][cc + 1].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def render_html(rows: list[dict], title: str, out_html: Path):
    html = ["<!doctype html><meta charset=utf-8>",
            f"<title>{title}</title>",
            "<style>body{font-family:sans-serif} .row{display:flex;align-items:center;"
            "gap:8px;padding:6px;border-bottom:1px solid #ccc} img{height:96px} "
            ".cat{color:#c00;font-weight:bold} .score{font-family:monospace}</style>",
            f"<h1>{title}</h1>"]
    for r in rows:
        html.append(
            f"<div class=row>"
            f"<img src='{r['a']}'/><img src='{r['b']}'/>"
            f"<span class=score>cos={r['score']:.3f}</span>"
            f"<span class=cat>{r['category']}</span>"
            f"<span>{Path(r['a']).parent.name} vs {Path(r['b']).parent.name}</span>"
            "</div>"
        )
    out_html.write_text("\n".join(html))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold. Use --threshold-from to load it "
                        "from a previous evaluate_full.py run instead.")
    p.add_argument("--threshold-from", type=str, default="",
                   help="Path to evaluate_full.py's threshold.json (or its "
                        "report dir). Overrides --threshold when present.")
    p.add_argument("--threshold-fallback", type=float, default=config.MATCH_THRESHOLD,
                   help="Fallback if neither --threshold nor --threshold-from "
                        "is usable.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--report-dir", type=str, default="reports/failures")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="failure_analysis",
        run_name=f"failures_{args.backbone}",
        params=vars(args),
        category="Reliability",
        tags={"step": "failure_analysis", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):
    # Resolve the operating threshold.
    if args.threshold_from:
        threshold = load_threshold(args.threshold_from, args.threshold_fallback)
    elif args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(args.threshold_fallback)
    args.threshold = threshold

    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    if Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        bb = args.backbone
    else:
        model = build_embedding_model("facenet_vggface2"); bb = "facenet_vggface2"
    model.to(device).eval()
    input_size = 112 if bb.startswith("iresnet") else 160
    tf = _tf(input_size)

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"[failures] embedding {len(unique_paths)} unique images")
    embeddings: dict[Path, np.ndarray] = {}
    imgs, ps = [], []
    with torch.no_grad():
        def flush():
            if not imgs:
                return
            x = torch.stack(imgs, dim=0).to(device)
            e = l2_normalize(model(x).cpu().numpy().astype(np.float32))
            for pp, ee in zip(ps, e):
                embeddings[pp] = ee
            imgs.clear(); ps.clear()
        for p in tqdm(unique_paths, desc="embed"):
            try:
                pil = Image.open(p).convert("RGB")
            except Exception:
                continue
            imgs.append(tf(pil)); ps.append(p)
            if len(imgs) >= args.batch_size:
                flush()
        flush()

    fps, fns = [], []
    for (a, b, label), _f in zip(pairs, folds):
        ea, eb = embeddings.get(a), embeddings.get(b)
        if ea is None or eb is None:
            continue
        s = float(np.dot(ea, eb))
        pred = int(s >= args.threshold)
        if pred == 1 and label == 0:
            fps.append({"a": str(a), "b": str(b), "score": s})
        elif pred == 0 and label == 1:
            fns.append({"a": str(a), "b": str(b), "score": s})

    # Hardest FPs = highest cosine (most-confident wrong accept).
    fps.sort(key=lambda r: r["score"], reverse=True)
    # Hardest FNs = lowest cosine (most-confident wrong reject).
    fns.sort(key=lambda r: r["score"])

    print(f"[failures] FP={len(fps)}  FN={len(fns)}  thr={args.threshold:.3f}")

    # Categorize.
    for r in fps:
        r["category"] = categorize_failure(Path(r["a"]), Path(r["b"]), kind="fp")
    for r in fns:
        r["category"] = categorize_failure(Path(r["a"]), Path(r["b"]), kind="fn")

    # Aggregate categories.
    def agg(rows):
        out = {}
        for r in rows:
            out[r["category"]] = out.get(r["category"], 0) + 1
        return out
    cat_fp = agg(fps); cat_fn = agg(fns)

    # Galleries (PNG + HTML).
    render_gallery(fps[:32], "Top 32 hardest false positives", report_dir / "fp_gallery.png")
    render_gallery(fns[:32], "Top 32 hardest false negatives", report_dir / "fn_gallery.png")
    render_html(fps[:64], "Hardest false positives", report_dir / "fp_hardest.html")
    render_html(fns[:64], "Hardest false negatives", report_dir / "fn_hardest.html")

    metrics = {
        "backbone": bb, "threshold": args.threshold,
        "n_fp": len(fps), "n_fn": len(fns),
        "fp_categories": cat_fp, "fn_categories": cat_fn,
        "fp_top10": fps[:10], "fn_top10": fns[:10],
    }
    # Canonical name `metrics.json` so aggregate_reports.py picks it up
    # automatically alongside every other evaluator.
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    md = ["# Failure analysis\n", f"Backbone: `{bb}`\n",
          f"Threshold: {args.threshold:.3f}\n\n",
          f"## False positives: {len(fps)}\n",
          json.dumps(cat_fp, indent=2), "\n\n",
          f"## False negatives: {len(fns)}\n",
          json.dumps(cat_fn, indent=2), "\n\n",
          "## Top-10 hardest FPs (highest similarity, wrong)\n"]
    for r in fps[:10]:
        md.append(f"- cos={r['score']:.3f}  `{r['category']}`  "
                  f"{Path(r['a']).parent.name} vs {Path(r['b']).parent.name}\n")
    md.append("\n## Top-10 hardest FNs (lowest similarity, wrong)\n")
    for r in fns[:10]:
        md.append(f"- cos={r['score']:.3f}  `{r['category']}`  "
                  f"{Path(r['a']).parent.name} vs {Path(r['b']).parent.name}\n")
    (report_dir / "summary.md").write_text("".join(md))
    print(f"[failures] wrote galleries + HTML + JSON + summary.md to {report_dir}")

    mlu.log_metrics_flat({
        "failures.n_fp": len(fps),
        "failures.n_fn": len(fns),
        "failures.threshold": args.threshold,
    })
    # Per-category FP/FN counts so the dashboard shows which failure modes
    # dominate (occlusion / pose / illumination / etc).
    mlu.log_metrics_flat({f"failures.fp_category.{k}": v for k, v in cat_fp.items()})
    mlu.log_metrics_flat({f"failures.fn_category.{k}": v for k, v in cat_fn.items()})
    mlu.log_artifacts_glob(report_dir, ["*.png", "*.json", "*.md", "*.html"],
                           artifact_path="failures")


if __name__ == "__main__":
    main()
