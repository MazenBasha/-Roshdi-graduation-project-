"""
Robustness evaluation under synthetic perturbations.

Takes LFW pairs, applies a perturbation to the SECOND image of each pair
(simulating that the gallery is clean and the probe is degraded), and reports
how LFW accuracy, AUC and EER degrade as the perturbation strength grows.

Perturbations covered:
    - brightness / illumination (gamma)
    - blur (Gaussian)
    - JPEG compression
    - occlusion (random rectangle masking)
    - planar rotation (proxy for in-plane pose)
    - Gaussian noise

Output (under --report-dir):
    metrics.json         — all per-strength numbers, machine-readable
    robustness_<name>.png — accuracy curve per perturbation
    summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm

import config
from evaluate import kfold_evaluate, parse_pairs, roc_auc
from evaluate_full import equal_error_rate, far_frr_curve
from model import build_embedding_model, load_checkpoint
from utils import get_device, l2_normalize
import mlflow_utils as mlu


def _build_transform(input_size, center_crop):
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
    ])


# ---------------------------------------------------------------------------
# Perturbation functions (PIL -> PIL, parameterized by `strength` in [0,1])
# ---------------------------------------------------------------------------

def perturb_brightness(img: Image.Image, strength: float) -> Image.Image:
    # gamma in [1.0, 3.5] for strength in [0,1]; >1 = darker
    gamma = 1.0 + 2.5 * strength
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


def perturb_blur(img: Image.Image, strength: float) -> Image.Image:
    radius = 0.5 + 6.0 * strength
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def perturb_jpeg(img: Image.Image, strength: float) -> Image.Image:
    q = max(1, int(95 - 90 * strength))  # 95 -> 5
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def _image_rng(img: Image.Image, global_seed: int) -> np.random.Generator:
    """Per-image deterministic RNG.

    Reproducibility requires the same image+seed combination to yield the
    same perturbation across runs. Independence requires different images
    to yield independent perturbations. We get both by hashing the image
    bytes together with the global seed and feeding that into PCG64.
    """
    h = hashlib.blake2b(img.tobytes(), digest_size=8).digest()
    img_seed = int.from_bytes(h, "little", signed=False)
    return np.random.default_rng(np.uint64(img_seed ^ np.uint64(global_seed)))


def perturb_occlusion(img: Image.Image, strength: float,
                      global_seed: int = 0) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    side = int(0.1 * min(h, w) + 0.5 * min(h, w) * strength)
    rng = _image_rng(img, global_seed)
    y0 = int(rng.integers(0, max(1, h - side)))
    x0 = int(rng.integers(0, max(1, w - side)))
    arr[y0:y0 + side, x0:x0 + side] = 0
    return Image.fromarray(arr)


def perturb_rotation(img: Image.Image, strength: float,
                     global_seed: int = 0) -> Image.Image:
    angle = 45.0 * strength
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def perturb_noise(img: Image.Image, strength: float,
                  global_seed: int = 0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    sigma = 60.0 * strength
    rng = _image_rng(img, global_seed)
    arr = arr + rng.normal(0, sigma, size=arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _wrap_seed(fn: Callable) -> Callable:
    """Adapter so perturbations that don't take a seed still match the signature."""
    def wrapped(img: Image.Image, strength: float, global_seed: int = 0) -> Image.Image:
        return fn(img, strength)
    return wrapped


PERTURBATIONS: dict[str, Callable[..., Image.Image]] = {
    "brightness": _wrap_seed(perturb_brightness),
    "blur":        _wrap_seed(perturb_blur),
    "jpeg":        _wrap_seed(perturb_jpeg),
    "occlusion":   perturb_occlusion,
    "rotation":    perturb_rotation,
    "noise":       perturb_noise,
}


# ---------------------------------------------------------------------------
# Embedding cache (per perturbation strength) + scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_all(
    paths: list[Path], model, device, input_size, center_crop, batch_size,
    perturb_fn: Callable | None = None, strength: float = 0.0,
    global_seed: int = 0,
) -> dict[Path, np.ndarray]:
    tf = _build_transform(input_size, center_crop)
    embeddings: dict[Path, np.ndarray] = {}
    batch_imgs, batch_paths = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        emb = model(x).detach().cpu().numpy()
        emb = l2_normalize(emb.astype(np.float32))
        for p, e in zip(batch_paths, emb):
            embeddings[p] = e
        batch_imgs.clear(); batch_paths.clear()

    for p in tqdm(paths, desc=f"embed s={strength:.2f}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        if perturb_fn is not None and strength > 0:
            img = perturb_fn(img, strength, global_seed)
        batch_imgs.append(tf(img))
        batch_paths.append(p)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return embeddings


def score_with_split_embeddings(pairs, folds, emb_first, emb_second):
    scores, labels, keep_folds = [], [], []
    for (a, b, label), fold in zip(pairs, folds):
        ea, eb = emb_first.get(a), emb_second.get(b)
        if ea is None or eb is None:
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(int(label))
        keep_folds.append(int(fold))
    return (np.array(scores, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            np.array(keep_folds, dtype=np.int32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--strengths", type=str, default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated strength values in [0,1]")
    p.add_argument("--perturbations", type=str, default=",".join(PERTURBATIONS.keys()))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42,
                   help="Global seed; per-image RNGs are derived from "
                        "blake2b(image_bytes) XOR seed so every image gets "
                        "an independent noise/occlusion realisation while "
                        "the whole run is reproducible.")
    p.add_argument("--report-dir", type=str, default="reports/robustness")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="robustness_eval",
        run_name=f"robustness_{args.backbone}",
        params=vars(args),
        category="Reliability",
        tags={"step": "robustness_eval", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):
    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    print(f"[robustness] {len(pairs)} pairs across {len(np.unique(folds))} folds")

    if args.checkpoint and Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        bb = args.backbone
    else:
        model = build_embedding_model("facenet_vggface2"); bb = "facenet_vggface2"
    model.to(device).eval()
    input_size = 112 if bb.startswith("iresnet") else 160
    center_crop = 160

    # Embed the FIRST image of every pair once, clean. The "gallery side."
    first_paths = sorted({a for a, _, _ in pairs})
    second_paths = sorted({b for _, b, _ in pairs})

    emb_first = embed_all(first_paths, model, device, input_size, center_crop, args.batch_size)

    strengths = [float(s) for s in args.strengths.split(",")]
    perturbations = [s.strip() for s in args.perturbations.split(",") if s.strip()]

    all_results = {}
    grid = np.linspace(-0.1, 1.0, 222)

    for pname in perturbations:
        if pname not in PERTURBATIONS:
            print(f"[robustness] unknown perturbation '{pname}' — skipped"); continue
        print(f"\n--- perturbation: {pname} ---")
        per_strength = []
        for s in strengths:
            emb_second = embed_all(
                second_paths, model, device, input_size, center_crop, args.batch_size,
                perturb_fn=PERTURBATIONS[pname], strength=s, global_seed=args.seed,
            )
            scores, labels, keep_folds = score_with_split_embeddings(
                pairs, folds, emb_first, emb_second,
            )
            kf = kfold_evaluate(scores, labels, keep_folds, grid)
            auc = roc_auc(scores, labels)
            thr_sweep, far, frr = far_frr_curve(scores, labels, n_thresh=400)
            eer, eer_thr = equal_error_rate(thr_sweep, far, frr)
            per_strength.append({
                "strength": s,
                "accuracy": float(kf["mean_accuracy"]),
                "accuracy_std": float(kf["std_accuracy"]),
                "auc": float(auc),
                "eer": float(eer),
                "n_pairs": int(len(scores)),
            })
            print(f"  s={s:.2f}  acc={kf['mean_accuracy']*100:.2f}%  AUC={auc:.4f}  EER={eer*100:.2f}%")
            # Step = strength * 100, so MLflow shows a degradation curve per metric.
            mlu.log_metrics_flat({
                f"{pname}.accuracy": kf["mean_accuracy"],
                f"{pname}.auc": auc,
                f"{pname}.eer": eer,
            }, step=int(s * 100))
        all_results[pname] = per_strength

        # Plot per-perturbation degradation.
        ss = [r["strength"] for r in per_strength]
        accs = [r["accuracy"] * 100 for r in per_strength]
        aucs = [r["auc"] for r in per_strength]
        eers = [r["eer"] * 100 for r in per_strength]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(ss, accs, "o-", label="Accuracy (%)")
        ax1.plot(ss, eers, "s-", label="EER (%)")
        ax1.set_xlabel("Perturbation strength")
        ax1.set_ylabel("Percent")
        ax1.set_title(f"Robustness — {pname}")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(ss, aucs, "^--", color="gray", label="AUC")
        ax2.set_ylabel("ROC AUC")
        ax1.legend(loc="upper right"); ax2.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(report_dir / f"robustness_{pname}.png", dpi=130)
        plt.close(fig)

    (report_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))

    # Summary markdown.
    md = ["# Robustness evaluation\n", f"Backbone: `{bb}`\n", f"Checkpoint: `{args.checkpoint}`\n\n"]
    for pname, rows in all_results.items():
        md.append(f"## {pname}\n")
        md.append("| strength | accuracy | AUC | EER |\n|--:|--:|--:|--:|\n")
        for r in rows:
            md.append(f"| {r['strength']:.2f} | {r['accuracy']*100:.2f}% | "
                      f"{r['auc']:.4f} | {r['eer']*100:.2f}% |\n")
        md.append("\n")
    (report_dir / "summary.md").write_text("".join(md))
    print(f"\n[robustness] wrote {report_dir/'metrics.json'} + plots + summary.md")

    # MLflow: log all degradation plots + tables + a per-perturbation summary metric
    # (accuracy at the strongest perturbation strength, useful for run comparison).
    for pname, rows in all_results.items():
        if rows:
            worst = rows[-1]
            mlu.log_metrics_flat({
                f"summary.{pname}.acc_at_strongest": worst["accuracy"],
                f"summary.{pname}.eer_at_strongest": worst["eer"],
            })
    mlu.log_artifacts_glob(report_dir, ["*.png", "*.json", "*.md"],
                           artifact_path="robustness")


if __name__ == "__main__":
    main()
