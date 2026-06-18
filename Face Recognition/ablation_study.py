"""
Ablation study on the inference-time pipeline.

Toggles independently:
    - CLAHE preprocessing on/off
    - Horizontal-flip TTA on/off
    - Center-crop tightness (none / 160 / 200 / 224 on the funneled image)
    - Threshold sweep around the per-fold cross-validated operating point

For each cell, runs LFW verification and reports k-fold accuracy + AUC.
Produces a 2^N comparison table.

NOTE: this uses the embedding-only fast path (no MTCNN) for speed —
CLAHE is applied directly to the funneled 250x250 image before resize.
That mirrors what the production pipeline does after alignment.

Outputs (under --report-dir):
    metrics.json
    ablation_table.csv
    ablation_heatmap.png
    summary.md
"""

from __future__ import annotations

import argparse
import io
import itertools
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
from evaluate import kfold_evaluate, parse_pairs, roc_auc, tar_at_far
from evaluate_full import equal_error_rate, far_frr_curve
from model import build_embedding_model, load_checkpoint
from utils import apply_clahe, get_device, l2_normalize
import mlflow_utils as mlu


@torch.no_grad()
def embed_path(model, device, tf, p: Path, use_clahe: bool, tta: bool):
    pil = Image.open(p).convert("RGB")
    if use_clahe:
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        bgr = apply_clahe(bgr)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x = tf(pil).unsqueeze(0).to(device)
    e = model(x)
    if tta:
        e2 = model(torch.flip(x, dims=(-1,)))
        e = torch.nn.functional.normalize(e, dim=1) + torch.nn.functional.normalize(e2, dim=1)
    e = e.cpu().numpy().astype(np.float32)
    return l2_normalize(e[0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--center-crops", type=str, default="0,160,200")
    p.add_argument("--clahe-options", type=str, default="off,on")
    p.add_argument("--tta-options", type=str, default="off,on")
    p.add_argument("--max-pairs", type=int, default=2000,
                   help="Cap for speed; -1 = all 6000")
    p.add_argument("--report-dir", type=str, default="reports/ablation")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="ablation_study",
        run_name=f"ablation_{args.backbone}",
        params=vars(args),
        category="Reliability",
        tags={"step": "ablation_study", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):
    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        idx = np.arange(len(pairs))
        rng = np.random.default_rng(42)
        rng.shuffle(idx); idx = idx[: args.max_pairs]
        pairs = [pairs[i] for i in idx]; folds = folds[idx]

    if Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        bb = args.backbone
    else:
        model = build_embedding_model("facenet_vggface2"); bb = "facenet_vggface2"
    model.to(device).eval()
    input_size = 112 if bb.startswith("iresnet") else 160

    crops = [int(c) for c in args.center_crops.split(",")]
    clahe_opts = [s == "on" for s in args.clahe_options.split(",")]
    tta_opts = [s == "on" for s in args.tta_options.split(",")]
    cells = list(itertools.product(crops, clahe_opts, tta_opts))
    print(f"[ablation] {len(cells)} cells x {len(pairs)} pairs")

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})

    results = []
    for crop, use_clahe, tta in cells:
        if crop > 0:
            tf = transforms.Compose([
                transforms.CenterCrop(crop),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
            ])
        embeddings: dict[Path, np.ndarray] = {}
        for p in tqdm(unique_paths, desc=f"crop={crop} clahe={use_clahe} tta={tta}",
                      leave=False):
            try:
                embeddings[p] = embed_path(model, device, tf, p, use_clahe, tta)
            except Exception:
                continue
        scores, labels, keep_folds = [], [], []
        for (a, b, lab), f in zip(pairs, folds):
            ea, eb = embeddings.get(a), embeddings.get(b)
            if ea is None or eb is None:
                continue
            scores.append(float(np.dot(ea, eb)))
            labels.append(int(lab)); keep_folds.append(int(f))
        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        keep_folds = np.array(keep_folds, dtype=np.int32)
        grid = np.linspace(-0.1, 1.0, 222)
        kf = kfold_evaluate(scores, labels, keep_folds, grid)
        auc = roc_auc(scores, labels)
        thr_sweep, far, frr = far_frr_curve(scores, labels, n_thresh=300)
        eer, eer_thr = equal_error_rate(thr_sweep, far, frr)
        results.append({
            "crop": crop, "clahe": use_clahe, "tta": tta,
            "accuracy": float(kf["mean_accuracy"]),
            "accuracy_std": float(kf["std_accuracy"]),
            "auc": float(auc), "eer": float(eer),
            "tar_at_far_1e3": float(tar_at_far(scores, labels, 1e-3)[0]),
            "n_pairs": int(len(scores)),
        })
        print(f"  crop={crop} clahe={use_clahe} tta={tta}: "
              f"acc={kf['mean_accuracy']*100:.2f}% AUC={auc:.4f} EER={eer*100:.2f}%")

    # CSV.
    csv_lines = ["crop,clahe,tta,accuracy,accuracy_std,auc,eer,tar_far_1e3,n_pairs\n"]
    for r in results:
        csv_lines.append(f"{r['crop']},{int(r['clahe'])},{int(r['tta'])},"
                         f"{r['accuracy']:.4f},{r['accuracy_std']:.4f},"
                         f"{r['auc']:.4f},{r['eer']:.4f},{r['tar_at_far_1e3']:.4f},"
                         f"{r['n_pairs']}\n")
    (report_dir / "ablation_table.csv").write_text("".join(csv_lines))

    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    # Heatmap: crop on x-axis, (clahe,tta) on y-axis, color = accuracy.
    crops_u = sorted(set(c for c, _, _ in cells))
    ct_u = sorted({(c, t) for _, c, t in cells})
    mat = np.full((len(ct_u), len(crops_u)), np.nan)
    for r in results:
        i = ct_u.index((r["clahe"], r["tta"]))
        j = crops_u.index(r["crop"])
        mat[i, j] = r["accuracy"] * 100
    fig, ax = plt.subplots(figsize=(1.2 * len(crops_u) + 2, 0.8 * len(ct_u) + 2))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(crops_u))); ax.set_xticklabels([str(c) for c in crops_u])
    ax.set_yticks(range(len(ct_u))); ax.set_yticklabels(
        [f"CLAHE={int(c)} TTA={int(t)}" for c, t in ct_u])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < np.nanmean(mat) else "black",
                        fontsize=9)
    ax.set_xlabel("Center crop"); ax.set_ylabel("Setting")
    ax.set_title(f"Ablation: accuracy on LFW — {bb}")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(report_dir / "ablation_heatmap.png", dpi=130)
    plt.close(fig)

    # Summary markdown.
    md = ["# Inference-pipeline ablation\n",
          f"Backbone: `{bb}` | Pairs: {len(pairs)}\n\n",
          "| crop | CLAHE | TTA | accuracy | AUC | EER | TAR@FAR=1e-3 |\n"
          "|--:|:-:|:-:|--:|--:|--:|--:|\n"]
    for r in results:
        md.append(f"| {r['crop']} | {r['clahe']} | {r['tta']} | "
                  f"{r['accuracy']*100:.2f}% | {r['auc']:.4f} | "
                  f"{r['eer']*100:.2f}% | {r['tar_at_far_1e3']*100:.2f}% |\n")
    best = max(results, key=lambda r: r["accuracy"])
    md.append(f"\n**Best**: crop={best['crop']}, CLAHE={best['clahe']}, "
              f"TTA={best['tta']} -> acc={best['accuracy']*100:.2f}%\n")
    (report_dir / "summary.md").write_text("".join(md))
    print(f"[ablation] wrote {report_dir/'metrics.json'} + heatmap + CSV + summary.md")

    # MLflow: one metric per ablation cell + the global best, plus the heatmap.
    for idx, r in enumerate(results):
        cell = f"crop{r['crop']}_clahe{int(r['clahe'])}_tta{int(r['tta'])}"
        mlu.log_metrics_flat({
            f"ablation.{cell}.accuracy": r["accuracy"],
            f"ablation.{cell}.auc": r["auc"],
            f"ablation.{cell}.eer": r["eer"],
            f"ablation.{cell}.tar_at_far_1e3": r["tar_at_far_1e3"],
        })
    mlu.log_metrics_flat({"ablation.best_accuracy": best["accuracy"]})
    mlu.log_artifacts_glob(report_dir, ["*.png", "*.csv", "*.json", "*.md"],
                           artifact_path="ablation")


if __name__ == "__main__":
    main()
