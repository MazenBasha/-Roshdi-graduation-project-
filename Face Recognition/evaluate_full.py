"""
Full evaluation suite for LFW verification.

Adds, on top of evaluate_lfw_fast.py:
    - Equal Error Rate (EER) + DET curve
    - ROC curve with AUC
    - Precision / Recall / F1 at the per-fold threshold
    - Full FAR / FRR sweep at fixed FAR points (1e-1, 1e-2, 1e-3, 1e-4)
    - Confusion matrix at the cross-validated threshold
    - 95% bootstrap confidence intervals on accuracy and AUC
    - Optional paired bootstrap test against a second model
    - All metrics dumped to reports/<run>/metrics.json + plots saved as PNGs

Two-model comparison (paired bootstrap on the same LFW pairs):
    python evaluate_full.py \
        --lfw-root data/sklearn_lfw/lfw_home/lfw_funneled \
        --pairs data/sklearn_lfw/lfw_home/pairs.txt \
        --checkpoint checkpoints/casia_v4_iresnet18.best.pt --backbone iresnet18 \
        --compare-backbone facenet_vggface2 \
        --report-dir reports/lfw_full

Outputs (created under --report-dir):
    metrics.json        — all numbers, machine-readable
    roc.png, det.png    — curves
    confusion.png       — confusion matrix heatmap
    threshold_sweep.png — FAR/FRR vs threshold
    summary.md          — human-readable report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats
from torchvision import transforms
from tqdm import tqdm

import config
from evaluate import kfold_evaluate, parse_pairs, roc_auc, tar_at_far
from model import build_embedding_model, load_checkpoint
from utils import get_device, l2_normalize
import mlflow_utils as mlu


# ---------------------------------------------------------------------------
# Embedding extraction (the "fast" no-MTCNN path)
# ---------------------------------------------------------------------------

def _build_transform(input_size: int, center_crop: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
    ])


def embed_unique_images(
    paths: Sequence[Path],
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    center_crop: int,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    tf = _build_transform(input_size, center_crop)
    embeddings: dict[Path, np.ndarray] = {}
    batch_imgs: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush() -> None:
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        emb = model(x).detach().cpu().numpy()
        emb = l2_normalize(emb.astype(np.float32))
        for p, e in zip(batch_paths, emb):
            embeddings[p] = e
        batch_imgs.clear()
        batch_paths.clear()

    with torch.no_grad():
        for p in tqdm(paths, desc="embed"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            batch_imgs.append(tf(img))
            batch_paths.append(p)
            if len(batch_imgs) >= batch_size:
                flush()
        flush()

    return embeddings


def score_pairs(
    pairs, folds, embeddings: dict[Path, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, labels, keep_folds = [], [], []
    for (a, b, label), fold in zip(pairs, folds):
        ea, eb = embeddings.get(a), embeddings.get(b)
        if ea is None or eb is None:
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(int(label))
        keep_folds.append(int(fold))
    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(keep_folds, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def far_frr_curve(scores: np.ndarray, labels: np.ndarray, n_thresh: int = 500):
    """Return thresholds, FAR, FRR over the full operating range."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresh)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    far = np.array([float((neg >= t).mean()) for t in thresholds])
    frr = np.array([float((pos < t).mean()) for t in thresholds])
    return thresholds, far, frr


def equal_error_rate(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray):
    """Equal Error Rate: the operating point where FAR = FRR."""
    diff = far - frr
    sign = np.sign(diff)
    # Find the index where sign of (FAR-FRR) flips. Interpolate linearly there.
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips) == 0:
        i = int(np.argmin(np.abs(diff)))
        return float((far[i] + frr[i]) / 2.0), float(thresholds[i])
    i = int(flips[0])
    # Linear interp between i and i+1.
    f0, f1 = diff[i], diff[i + 1]
    if f1 == f0:
        alpha = 0.0
    else:
        alpha = -f0 / (f1 - f0)
    eer = float(far[i] + alpha * (far[i + 1] - far[i]))
    thr = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    return eer, thr


def frr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float):
    """FRR (= 1 - TAR) at a fixed FAR. Returns (frr, threshold)."""
    tar, thr = tar_at_far(scores, labels, target_far)
    return 1.0 - tar, thr


def precision_recall_f1(scores: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (scores >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 2000, seed: int = 42):
    """Percentile bootstrap 95% CI on a scalar statistic of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(values[idx])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_acc_ci(scores, labels, threshold, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        preds = (scores[idx] >= threshold).astype(np.int32)
        samples[i] = (preds == labels[idx]).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_auc_ci(scores, labels, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = roc_auc(scores[idx], labels[idx])
        except Exception:
            samples[i] = float("nan")
    samples = samples[~np.isnan(samples)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_bootstrap(scores_a, scores_b, labels, threshold_a, threshold_b,
                     n_boot=2000, seed=42):
    """Paired bootstrap on accuracy: is model B significantly better than A
    on the SAME pairs? Returns (delta_acc, ci_lo, ci_hi, p_value_one_sided).
    """
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_a = ((scores_a[idx] >= threshold_a).astype(int) == labels[idx]).mean()
        acc_b = ((scores_b[idx] >= threshold_b).astype(int) == labels[idx]).mean()
        deltas[i] = acc_b - acc_a
    return (
        float(deltas.mean()),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        float((deltas <= 0).mean()),  # p-value: P(B not better than A)
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(scores, labels, out_path, title="ROC"):
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    P = max(1, int(y.sum())); N = max(1, len(y) - P)
    tpr = tps / P; fpr = fps / N
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    auc = roc_auc(scores, labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_det(thresholds, far, frr, eer, eer_thr, out_path, title="DET",
             axis="normal_deviate"):
    """Detection-Error Tradeoff plot.

    `axis="normal_deviate"`: the canonical NIST DET layout (Martin et al. 1997,
    NIST FRVT). Axes are Φ⁻¹(rate). Linear DET means the score distributions
    were Gaussian. This is the academically standard form.

    `axis="log"`: log-log axes. Common in some literature, easier to read for
    casual readers, but not the canonical DET.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    if axis == "normal_deviate":
        from scipy.stats import norm
        eps = 1e-5
        far_c = np.clip(far, eps, 1 - eps)
        frr_c = np.clip(frr, eps, 1 - eps)
        ax.plot(norm.ppf(far_c), norm.ppf(frr_c))
        ax.scatter([norm.ppf(np.clip(eer, eps, 1 - eps))],
                   [norm.ppf(np.clip(eer, eps, 1 - eps))],
                   color="red", zorder=3,
                   label=f"EER = {eer*100:.2f}% @ thr={eer_thr:.3f}")
        ticks_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
        tick_vals = [norm.ppf(r) for r in ticks_rate]
        tick_labels = [f"{r*100:g}%" for r in ticks_rate]
        ax.set_xticks(tick_vals); ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_vals); ax.set_yticklabels(tick_labels)
        ax.set_xlim(norm.ppf(0.0005), norm.ppf(0.5))
        ax.set_ylim(norm.ppf(0.0005), norm.ppf(0.5))
    else:
        ax.plot(far, frr)
        ax.scatter([eer], [eer], color="red", zorder=3,
                   label=f"EER = {eer*100:.2f}% @ thr={eer_thr:.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-4, 1.0); ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(title + (" (normal-deviate)" if axis == "normal_deviate" else " (log-log)"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_threshold_sweep(thresholds, far, frr, out_path, title="FAR/FRR vs threshold"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, far, label="FAR")
    ax.plot(thresholds, frr, label="FRR")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: dict, out_path, title="Confusion matrix"):
    mat = np.array([[cm["tn"], cm["fp"]],
                    [cm["fn"], cm["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="black" if mat[i, j] < mat.max() / 2 else "white",
                    fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: different", "pred: same"])
    ax.set_yticklabels(["true: different", "true: same"])
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_score_histogram(scores, labels, threshold, out_path,
                         title="Cosine similarity distribution"):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(neg, bins=60, alpha=0.6, label="different (neg)", color="tab:red")
    ax.hist(pos, bins=60, alpha=0.6, label="same (pos)", color="tab:green")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_one_model(
    label: str, backbone: str, checkpoint: str | None,
    pairs, folds, device, args, report_dir: Path,
):
    print(f"\n=== {label} ===")
    if checkpoint and Path(checkpoint).is_file():
        model = build_embedding_model(backbone)
        load_checkpoint(model, checkpoint, map_location=device)
    else:
        model = build_embedding_model("facenet_vggface2")
        backbone = "facenet_vggface2"
    model.to(device).eval()

    input_size = 112 if backbone.startswith("iresnet") else 160
    center_crop = 160 if backbone != "facenet_vggface2" else 160

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings = embed_unique_images(
        unique_paths, model, device, input_size, center_crop, args.batch_size,
    )
    scores, labels, keep_folds = score_pairs(pairs, folds, embeddings)

    # K-fold cross-validated threshold + accuracy.
    grid = np.linspace(-0.1, 1.0, 222)
    kf = kfold_evaluate(scores, labels, keep_folds, grid)
    chosen_threshold = float(np.mean(kf["fold_thresholds"]))

    # Bootstrap CIs.
    acc_lo, acc_hi = bootstrap_acc_ci(scores, labels, chosen_threshold,
                                      n_boot=args.bootstrap, seed=args.seed)
    auc = roc_auc(scores, labels)
    auc_lo, auc_hi = bootstrap_auc_ci(scores, labels,
                                      n_boot=args.bootstrap, seed=args.seed)

    # FAR/FRR sweep + EER.
    thr_sweep, far, frr = far_frr_curve(scores, labels, n_thresh=600)
    eer, eer_thr = equal_error_rate(thr_sweep, far, frr)

    # Operating-point table.
    op_points: dict[str, dict] = {}
    for target_far in (1e-1, 1e-2, 1e-3, 1e-4):
        frr_v, thr_v = frr_at_far(scores, labels, target_far)
        tar_v = 1 - frr_v
        op_points[f"FAR={target_far:g}"] = {
            "threshold": thr_v, "TAR": float(tar_v), "FRR": float(frr_v),
        }

    # Precision / Recall / F1 / confusion at the chosen threshold.
    cm_chosen = precision_recall_f1(scores, labels, chosen_threshold)
    # Same at threshold=0.40 (the production default).
    cm_prod = precision_recall_f1(scores, labels, config.MATCH_THRESHOLD)

    # Per-fold std as a proxy for variance.
    fold_acc_std = float(np.std(kf["fold_accuracies"]))

    # Save plots.
    model_dir = report_dir / label.replace(" ", "_").lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(scores, labels, model_dir / "roc.png", title=f"ROC — {label}")
    plot_det(thr_sweep, far, frr, eer, eer_thr, model_dir / "det.png",
             title=f"DET — {label}")
    plot_threshold_sweep(thr_sweep, far, frr, model_dir / "threshold_sweep.png",
                         title=f"FAR/FRR vs threshold — {label}")
    plot_confusion(cm_chosen, model_dir / "confusion.png",
                   title=f"Confusion @ thr={chosen_threshold:.3f} — {label}")
    plot_score_histogram(scores, labels, chosen_threshold,
                         model_dir / "score_histogram.png",
                         title=f"Score distribution — {label}")

    out = {
        "label": label,
        "backbone": backbone,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "n_pairs_scored": int(len(scores)),
        "kfold_mean_accuracy": float(kf["mean_accuracy"]),
        "kfold_std_accuracy": float(kf["std_accuracy"]),
        "fold_accuracies": [float(a) for a in kf["fold_accuracies"]],
        "fold_thresholds": [float(t) for t in kf["fold_thresholds"]],
        "fold_threshold_std": fold_acc_std,
        "chosen_threshold": chosen_threshold,
        "accuracy_95ci": [acc_lo, acc_hi],
        "roc_auc": float(auc),
        "roc_auc_95ci": [auc_lo, auc_hi],
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "operating_points": op_points,
        "metrics_at_chosen_threshold": cm_chosen,
        "metrics_at_production_threshold_0p40": cm_prod,
    }
    (model_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"  k-fold acc = {kf['mean_accuracy']*100:.2f}% ± {kf['std_accuracy']*100:.2f}%"
          f"  (95% CI {acc_lo*100:.2f}–{acc_hi*100:.2f})")
    print(f"  ROC AUC    = {auc:.4f}  (95% CI {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"  EER        = {eer*100:.2f}%  @ thr={eer_thr:.3f}")
    print(f"  P / R / F1 @ thr={chosen_threshold:.3f}: "
          f"{cm_chosen['precision']:.3f} / {cm_chosen['recall']:.3f} / {cm_chosen['f1']:.3f}")
    return out, scores, labels, keep_folds, chosen_threshold


def write_summary_md(report_dir: Path, primary, secondary, paired):
    lines = ["# Full LFW evaluation\n"]
    for m in (primary, secondary):
        if m is None:
            continue
        lines.append(f"## {m['label']}\n")
        lines.append(f"- Backbone: `{m['backbone']}`\n")
        lines.append(f"- Pairs scored: {m['n_pairs_scored']}\n")
        lines.append(f"- 10-fold accuracy: **{m['kfold_mean_accuracy']*100:.2f}% ± {m['kfold_std_accuracy']*100:.2f}%**"
                     f"  (95% CI {m['accuracy_95ci'][0]*100:.2f}–{m['accuracy_95ci'][1]*100:.2f})\n")
        lines.append(f"- ROC AUC: **{m['roc_auc']:.4f}**  (95% CI {m['roc_auc_95ci'][0]:.4f}–{m['roc_auc_95ci'][1]:.4f})\n")
        lines.append(f"- EER: **{m['eer']*100:.2f}%** @ threshold {m['eer_threshold']:.3f}\n")
        lines.append(f"- P/R/F1 @ chosen threshold {m['chosen_threshold']:.3f}: "
                     f"{m['metrics_at_chosen_threshold']['precision']:.3f} / "
                     f"{m['metrics_at_chosen_threshold']['recall']:.3f} / "
                     f"{m['metrics_at_chosen_threshold']['f1']:.3f}\n")
        lines.append("- Operating points:\n")
        for k, v in m["operating_points"].items():
            lines.append(f"  - {k}: thr={v['threshold']:.3f}  TAR={v['TAR']*100:.2f}%  FRR={v['FRR']*100:.2f}%\n")
        lines.append("\n")
    if paired is not None:
        d, lo, hi, p = paired
        lines.append("## Paired bootstrap comparison\n")
        lines.append(f"Δ accuracy (B − A) = **{d*100:+.2f}%**  (95% CI {lo*100:+.2f}–{hi*100:+.2f})\n")
        lines.append(f"One-sided p-value (B not better than A) = {p:.4f}\n")
    (report_dir / "summary.md").write_text("".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--label", type=str, default="from_scratch")
    p.add_argument("--compare-checkpoint", type=str, default="")
    p.add_argument("--compare-backbone", type=str, default="")
    p.add_argument("--compare-label", type=str, default="pretrained_baseline")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-dir", type=str, default="reports/lfw_full")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    with mlu.init_run(
        experiment="evaluate_full",
        run_name=f"lfw_full_{args.label}",
        params=vars(args),
        tags={"step": "evaluate_full", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
        category="Evaluation",
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):
    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    print(f"[eval-full] {len(pairs)} pairs across {len(np.unique(folds))} folds")

    primary, scores_a, labels_a, _, thr_a = evaluate_one_model(
        args.label, args.backbone, args.checkpoint, pairs, folds, device, args, report_dir,
    )
    secondary = None; paired = None
    if args.compare_backbone:
        secondary, scores_b, labels_b, _, thr_b = evaluate_one_model(
            args.compare_label, args.compare_backbone,
            args.compare_checkpoint or None, pairs, folds, device, args, report_dir,
        )
        # Sanity: pairs and labels must match.
        if len(scores_a) == len(scores_b) and np.all(labels_a == labels_b):
            paired = paired_bootstrap(scores_a, scores_b, labels_a, thr_a, thr_b,
                                      n_boot=args.bootstrap, seed=args.seed)
            print(f"\nPaired bootstrap: Δacc = {paired[0]*100:+.2f}%  "
                  f"(95% CI {paired[1]*100:+.2f}–{paired[2]*100:+.2f}%)  "
                  f"p={paired[3]:.4f}")
        else:
            print("[eval-full] WARNING: pair indices differ between models — paired test skipped")

    # Combined JSON.
    out = {"primary": primary, "secondary": secondary,
           "paired_bootstrap": (
               {"delta_accuracy": paired[0], "ci_lo": paired[1], "ci_hi": paired[2],
                "p_value_one_sided": paired[3]} if paired else None
           )}
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    # Tiny canonical threshold file consumed by downstream evaluators so
    # we never use a magic number for the operating point.
    (report_dir / "threshold.json").write_text(json.dumps({
        "label": primary["label"],
        "threshold": primary["chosen_threshold"],
        "source": "kfold mean of fold thresholds",
        "eer_threshold": primary["eer_threshold"],
        "eer": primary["eer"],
    }, indent=2))
    write_summary_md(report_dir, primary, secondary, paired)
    print(f"\n[eval-full] wrote {report_dir / 'metrics.json'} + "
          f"threshold.json + plots + summary.md")

    # MLflow: re-log key metrics so they appear in the dashboard, plus all plots.
    mlu.log_metrics_flat({
        "kfold_mean_accuracy": primary["kfold_mean_accuracy"],
        "kfold_std_accuracy": primary["kfold_std_accuracy"],
        "roc_auc": primary["roc_auc"],
        "eer": primary["eer"],
        "eer_threshold": primary["eer_threshold"],
        "chosen_threshold": primary["chosen_threshold"],
        "precision_at_chosen_thr": primary["metrics_at_chosen_threshold"]["precision"],
        "recall_at_chosen_thr": primary["metrics_at_chosen_threshold"]["recall"],
        "f1_at_chosen_thr": primary["metrics_at_chosen_threshold"]["f1"],
        "TAR_at_FAR_1e-3": primary["operating_points"].get("FAR=0.001", {}).get("TAR", float("nan")),
        "TAR_at_FAR_1e-4": primary["operating_points"].get("FAR=0.0001", {}).get("TAR", float("nan")),
        "FRR_at_FAR_1e-3": primary["operating_points"].get("FAR=0.001", {}).get("FRR", float("nan")),
    })
    if paired is not None:
        d, lo, hi, p = paired
        mlu.log_metrics_flat({"paired_delta_acc": d, "paired_ci_lo": lo,
                              "paired_ci_hi": hi, "paired_p_value": p})
    mlu.log_artifacts_glob(report_dir, ["*.json", "*.md"], artifact_path="evaluate_full")
    for sub in report_dir.iterdir():
        if sub.is_dir():
            mlu.log_artifact_dir(sub, artifact_path=f"evaluate_full/{sub.name}")


if __name__ == "__main__":
    main()
