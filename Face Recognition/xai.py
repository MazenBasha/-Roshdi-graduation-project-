"""
Explainability (XAI) for the face recognition pipeline.

Face embedding models are not classifiers — there are no class probabilities
or attention heads tied to identities. The right notion of "explanation" for
this system is therefore:

  1. Top-K nearest gallery embeddings + cosine scores. Reveals not only
     "who we think this is" but also the runner-ups, so a reviewer can see
     whether the decision was confident or borderline.
  2. Per-decision threshold explanation: how far above / below the cosine
     threshold the match landed, the margin to the runner-up, and the
     calibrated confidence (softmax over top-K scores).
  3. Embedding-space visualization (t-SNE) over a held-out gallery, so a
     reviewer can see cluster geometry and how confused identities sit
     near each other.
  4. Grad-CAM++ over the iResNet backbone for spatial saliency on the
     INPUT pixel grid. Justification at the end of this module.

Outputs (under --report-dir):
    explain_<i>.json    — JSON top-K + decision metadata per probe image
    explain_<i>.png     — visualization (probe + top-K thumbnails + scores)
    tsne_gallery.png    — embedding-space t-SNE (if --tsne)
    gradcam_<i>.png     — saliency overlay (if --gradcam)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm

import config
from model import build_embedding_model, build_iresnet, load_checkpoint
from utils import get_device, l2_normalize
import mlflow_utils as mlu


_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5019607, 0.5019607, 0.5019607)


def _tf(input_size: int, center_crop: int = 160):
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Per-decision explanation
# ---------------------------------------------------------------------------

def explain_decision(
    probe_emb: np.ndarray,
    gallery_embeds: dict[str, np.ndarray],  # identity -> centroid
    threshold: float,
    top_k: int = 5,
):
    """Return a JSON-serializable explanation of the match decision."""
    if not gallery_embeds:
        return {
            "decision": "unknown",
            "top_k": [],
            "above_threshold": [],
            "below_threshold": [],
            "threshold": threshold,
            "explanation": "Gallery is empty.",
        }
    q = l2_normalize(probe_emb.astype(np.float32))
    scores = [(name, float(np.dot(q, e))) for name, e in gallery_embeds.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    best_name, best_score = top[0]

    above = [(n, s) for n, s in top if s >= threshold]
    below = [(n, s) for n, s in top if s < threshold]
    runner_up = top[1] if len(top) > 1 else (None, -1.0)
    margin = best_score - runner_up[1]

    # Softmax pseudo-confidence over top-K (temperature 10 to spread the mass).
    raw = np.array([s for _, s in top]); raw_t = raw * 10.0
    sm = np.exp(raw_t - raw_t.max()); sm /= sm.sum()

    explanation = (
        f"Best match: {best_name!r} with cosine={best_score:.3f}. "
        f"Threshold={threshold:.3f}, so the system "
        f"{'accepts' if best_score >= threshold else 'rejects (returns unknown)'} this match. "
        f"Margin to runner-up ({runner_up[0]!r}, cos={runner_up[1]:.3f}) is {margin:.3f} "
        f"({'comfortable' if margin > 0.1 else 'thin'}). "
        f"Top-K softmax confidence on {best_name!r} = {sm[0]*100:.1f}%."
    )
    return {
        "decision": best_name if best_score >= threshold else "unknown",
        "best_name": best_name,
        "best_score": best_score,
        "runner_up_name": runner_up[0],
        "runner_up_score": runner_up[1],
        "margin_to_runner_up": margin,
        "softmax_topk_confidence": [
            {"name": n, "cosine": s, "softmax": float(sm[i])}
            for i, (n, s) in enumerate(top)
        ],
        "above_threshold": [{"name": n, "cosine": s} for n, s in above],
        "below_threshold": [{"name": n, "cosine": s} for n, s in below],
        "threshold": threshold,
        "explanation": explanation,
    }


def render_explanation_image(
    probe_img: np.ndarray,
    explanation: dict,
    gallery_thumbnails: dict[str, np.ndarray],
    out_path: Path,
):
    """Save a panel: probe on left, top-K thumbnails on the right, with scores."""
    top = explanation["softmax_topk_confidence"]
    n = len(top)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3))
    axes[0].imshow(cv2.cvtColor(probe_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("probe"); axes[0].axis("off")
    for i, entry in enumerate(top):
        ax = axes[i + 1]
        thumb = gallery_thumbnails.get(entry["name"])
        if thumb is not None:
            ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        accepted = entry["cosine"] >= explanation["threshold"]
        color = "green" if accepted and i == 0 else "red" if i == 0 else "gray"
        ax.set_title(f"{entry['name']}\ncos={entry['cosine']:.2f}",
                     color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle(explanation["explanation"], fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# t-SNE / embedding visualization
# ---------------------------------------------------------------------------

def tsne_visualize(embeddings: dict[str, list[np.ndarray]], out_path: Path,
                   max_classes: int = 30, max_per_class: int = 20,
                   perplexity: float = 30.0, seed: int = 42) -> bool:
    """t-SNE of gallery embeddings, color-coded by identity.

    Sklearn's TSNE requires `perplexity < n_samples`. The original t-SNE
    paper (van der Maaten & Hinton, 2008) recommends 5 ≤ perplexity ≤ 50.
    For tiny galleries we clamp adaptively and fall back to PCA when n is
    too small for t-SNE to be meaningful (n < 8).

    Returns True if a plot was written, False if skipped.
    """
    names = sorted(embeddings.keys())[:max_classes]
    X, y = [], []
    for i, n in enumerate(names):
        es = embeddings[n][:max_per_class]
        X.extend(es); y.extend([i] * len(es))
    if not X:
        print("[xai] t-SNE skipped: no embeddings provided")
        return False
    X = np.stack(X, axis=0); y = np.array(y)
    n_samples = len(X)
    if n_samples < 2:
        print(f"[xai] t-SNE skipped: only {n_samples} sample")
        return False

    # Sklearn requires perplexity < n_samples (strict). The paper's lower
    # bound is 5; below that the conditional probabilities are too uniform
    # for the optimisation to find useful structure.
    perp = float(min(perplexity, max(2.0, n_samples - 1)))

    used_pca = False
    if n_samples < 8 or perp < 5.0:
        # t-SNE is not meaningful here. Fall back to PCA so we still produce
        # a useful embedding-space sanity plot rather than crash or lie.
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
        used_pca = True
        print(f"[xai] PCA fallback (n={n_samples} too small for t-SNE)")
    else:
        print(f"[xai] running t-SNE on {n_samples} embeddings, "
              f"{len(names)} identities, perplexity={perp}")
        Z = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=seed, learning_rate="auto").fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20", max(1, len(names)))
    for i, n in enumerate(names):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], s=20, color=cmap(i),
                   label=n if i < 20 else None, alpha=0.8)
    if len(names) <= 20:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    method = "PCA" if used_pca else "t-SNE"
    ax.set_title(f"{method} of embeddings ({len(names)} identities, n={n_samples})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Grad-CAM++ for iResNet
# ---------------------------------------------------------------------------
#
# Justification for using Grad-CAM here:
#
#   Face embeddings have no class logit to back-propagate from, so we
#   back-propagate from the EMBEDDING-CHANNEL projection onto a reference
#   direction (the gallery centroid for the predicted identity). This
#   produces a saliency map of "which input pixels most influenced the
#   embedding direction towards this person."
#
#   Adds value: yes for sanity-checking (it should highlight the face,
#   not the background; failure to do so flags a broken alignment or
#   embedding head).
#   Does NOT replace causal interpretability — Grad-CAM is a coarse,
#   well-known approximation. We expose it as one decision aid, not a
#   guarantee. SHAP / LIME on raw pixels for a 24M-parameter convnet are
#   far too expensive per-decision to justify here (LIME would need
#   thousands of perturbed forward passes), so we do not implement them.

class GradCAMpp:
    """Grad-CAM family for face-embedding networks.

    Two variants are supported via `variant`:

      * `"vanilla"` — Grad-CAM (Selvaraju et al. 2017):
            w_k = (1/Z) Σ_ij ∂Y/∂A_kij
            CAM = ReLU(Σ_k w_k · A_k)
        Y can be any differentiable scalar (we use cosine similarity to a
        gallery centroid). This is the academically-strict default for
        face embeddings, since the Grad-CAM derivation does not require Y
        to be a softmax probability.

      * `"plusplus"` — Grad-CAM++ (Chattopadhyay et al. 2018):
            α_kij = grad² / (2·grad² + Σ_ab A_kab · grad³)
            w_k = Σ_ij α_kij · ReLU(grad_kij)
            CAM = ReLU(Σ_k w_k · A_k)
        The α derivation in the original paper assumes Y is a positive
        class score (softmax probability). Applied here to cosine
        similarity ∈ [-1,1], it is an *extension* — common in face-XAI
        practice but not strictly justified by the original proof.
    """

    def __init__(self, backbone: torch.nn.Module,
                 target_layer_name: str = "layer4",
                 variant: str = "vanilla"):
        if variant not in ("vanilla", "plusplus"):
            raise ValueError(f"variant must be 'vanilla' or 'plusplus', got {variant!r}")
        self.variant = variant
        self.backbone = backbone.eval()
        self.target_layer = dict(backbone.named_modules())[target_layer_name]
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = self.target_layer.register_forward_hook(self._save_act)
        self._h2 = self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        self.activations = output.detach()

    def _save_grad(self, _m, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self._h1.remove(); self._h2.remove()

    def __call__(self, input_tensor: torch.Tensor, ref_direction: torch.Tensor):
        """input_tensor: (1,3,H,W). ref_direction: (D,). Returns HxW saliency in [0,1]."""
        self.backbone.zero_grad(set_to_none=True)
        emb = self.backbone(input_tensor)
        ref = ref_direction.to(emb.device).unsqueeze(0)
        ref = F.normalize(ref, dim=1)
        emb_n = F.normalize(emb, dim=1)
        target = (emb_n * ref).sum()
        target.backward()
        A = self.activations[0]                  # (C, h, w)
        G = self.gradients[0]                    # (C, h, w)
        if self.variant == "vanilla":
            # Selvaraju et al. 2017: weight each channel by mean gradient.
            weights = G.mean(dim=(1, 2))         # (C,)
        else:
            G2 = G * G; G3 = G2 * G
            sum_act = A.sum(dim=(1, 2), keepdim=True)
            denom = 2 * G2 + sum_act * G3
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = G2 / denom
            weights = (alpha * F.relu(G)).sum(dim=(1, 2))
        cam = F.relu((weights.view(-1, 1, 1) * A).sum(dim=0))
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (cam * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    if image_bgr.shape[:2] != heat_color.shape[:2]:
        image_bgr = cv2.resize(image_bgr, (heat_color.shape[1], heat_color.shape[0]))
    return cv2.addWeighted(image_bgr, 1 - alpha, heat_color, alpha, 0)


# ---------------------------------------------------------------------------
# CLI: explain a folder of probe images against a folder-of-folders gallery
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gallery", required=True, type=str,
                   help="LFW-style folder: gallery/<identity>/img.jpg")
    p.add_argument("--probes", required=True, type=str,
                   help="Folder of probe images (any flat folder of jpgs)")
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--max-identities", type=int, default=30)
    p.add_argument("--max-per-identity", type=int, default=10)
    p.add_argument("--tsne", action="store_true")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--gradcam-layer", type=str, default="layer4")
    p.add_argument("--gradcam-variant", type=str, default="vanilla",
                   choices=["vanilla", "plusplus"],
                   help="Vanilla = Selvaraju 2017 (academically strict for "
                        "non-softmax Y); plusplus = Chattopadhyay 2018 "
                        "(extension to cosine similarity).")
    p.add_argument("--report-dir", type=str, default="reports/xai")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="xai",
        run_name=f"xai_{args.backbone}",
        params=vars(args),
        category="Explainability",
        tags={"step": "xai", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint),
              "gradcam_variant": args.gradcam_variant},
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):

    if Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        bb = args.backbone
    else:
        model = build_embedding_model("facenet_vggface2"); bb = "facenet_vggface2"
    model.to(device).eval()
    input_size = 112 if bb.startswith("iresnet") else 160
    tf = _tf(input_size)

    # Build gallery: identity -> list of embeddings + thumbnail.
    print(f"[xai] building gallery from {args.gallery}")
    gallery_path = Path(args.gallery)
    identity_dirs = sorted([d for d in gallery_path.iterdir() if d.is_dir()])
    identity_dirs = identity_dirs[: args.max_identities]
    per_identity: dict[str, list[np.ndarray]] = {}
    thumbs: dict[str, np.ndarray] = {}
    centroids: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for d in tqdm(identity_dirs, desc="gallery"):
            imgs = sorted(d.glob("*.jpg"))[: args.max_per_identity]
            if not imgs:
                continue
            embs = []
            for ip in imgs:
                pil = Image.open(ip).convert("RGB")
                x = tf(pil).unsqueeze(0).to(device)
                e = model(x).cpu().numpy().astype(np.float32)
                embs.append(l2_normalize(e[0]))
            per_identity[d.name] = embs
            centroids[d.name] = l2_normalize(np.mean(embs, axis=0))
            # Thumb from the first image.
            thumb = cv2.imread(str(imgs[0]))
            if thumb is not None:
                thumbs[d.name] = cv2.resize(thumb, (112, 112))

    # Run probes.
    probe_files = sorted(Path(args.probes).glob("*.jpg"))[:50]
    cam = None
    if args.gradcam and bb.startswith("iresnet"):
        cam = GradCAMpp(model, target_layer_name=args.gradcam_layer,
                        variant=args.gradcam_variant)

    all_explanations = []
    for i, pf in enumerate(tqdm(probe_files, desc="probes")):
        pil = Image.open(pf).convert("RGB")
        x = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model(x).cpu().numpy().astype(np.float32)
        emb = l2_normalize(e[0])
        explanation = explain_decision(emb, centroids, args.threshold)
        explanation["probe_path"] = str(pf)
        all_explanations.append(explanation)

        probe_bgr = cv2.imread(str(pf))
        if probe_bgr is None:
            continue
        probe_bgr = cv2.resize(probe_bgr, (112, 112))
        render_explanation_image(probe_bgr, explanation, thumbs,
                                 report_dir / f"explain_{i:03d}.png")

        if cam is not None and explanation["decision"] != "unknown":
            ref = torch.from_numpy(centroids[explanation["best_name"]].astype(np.float32))
            saliency = cam(x.clone().requires_grad_(True), ref)
            overlay = overlay_cam(probe_bgr, saliency)
            cv2.imwrite(str(report_dir / f"gradcam_{i:03d}.png"), overlay)

    if cam is not None:
        cam.close()

    # Aggregate JSON.
    (report_dir / "explanations.json").write_text(json.dumps(all_explanations, indent=2))

    if args.tsne and per_identity:
        tsne_visualize(per_identity, report_dir / "tsne_gallery.png")

    # Quick markdown index.
    md = ["# Explainability report\n",
          f"Backbone: `{bb}` | Threshold: {args.threshold:.3f}\n\n",
          f"Probes processed: {len(all_explanations)}\n",
          f"Decisions: " +
          ", ".join(f"{lbl}={sum(1 for e in all_explanations if e['decision']==lbl)}"
                    for lbl in {"unknown"} | {e['best_name'] for e in all_explanations}) + "\n",
          "\nSee `explain_*.png` and `explanations.json` for per-probe details.\n"]
    (report_dir / "summary.md").write_text("".join(md))
    print(f"[xai] wrote {len(all_explanations)} explanations to {report_dir}")

    # MLflow: log all XAI artifacts + aggregate decision counts as metrics.
    n_accept = sum(1 for e in all_explanations if e["decision"] != "unknown")
    n_unknown = sum(1 for e in all_explanations if e["decision"] == "unknown")
    mean_best = float(np.mean([e["best_score"] for e in all_explanations])) if all_explanations else 0.0
    mean_margin = float(np.mean([e["margin_to_runner_up"] for e in all_explanations])) if all_explanations else 0.0
    mlu.log_metrics_flat({
        "xai.n_probes": len(all_explanations),
        "xai.n_accept": n_accept,
        "xai.n_unknown": n_unknown,
        "xai.mean_best_cosine": mean_best,
        "xai.mean_margin_to_runner_up": mean_margin,
        "xai.threshold": args.threshold,
    })
    # Cap PNG flood: log first 10 explain_/gradcam_ + tsne + json + md.
    mlu.log_artifact_file(report_dir / "explanations.json", artifact_path="xai")
    mlu.log_artifact_file(report_dir / "summary.md", artifact_path="xai")
    mlu.log_artifact_file(report_dir / "tsne_gallery.png", artifact_path="xai")
    for prefix in ("explain_", "gradcam_"):
        for f in sorted(report_dir.glob(f"{prefix}*.png"))[:10]:
            mlu.log_artifact_file(f, artifact_path=f"xai/{prefix.rstrip('_')}")


if __name__ == "__main__":
    main()
