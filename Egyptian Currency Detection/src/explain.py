"""
Visual explainability for the currency detector.

Answers "why did the model output this note?" by producing a saliency heatmap
over the image (the *supporting evidence*) plus a per-detection trust signal and
a plain-language reason string (the *reasoning*).  Two methods are supported:

- "eigencam" (default): the first principal component of the feature maps that
  feed the YOLOv8 Detect head.  No gradients, no NMS backprop -> robust, fast,
  and cleanly localized on the salient object.  This is the de-facto standard
  for YOLO explainability and is what we recommend for reports/screenshots.
- "gradcam": class-discriminative Grad-CAM, tied to each detection via the
  IoU-matched anchors that actually produced its box.  More faithful to "why
  THIS class" but, on YOLO's anchor-free head with large receptive fields, the
  heatmap can be diffuse.  Opt in with --explain-method gradcam.

Per detection we report `salience_ratio` = mean(CAM inside the box) /
mean(CAM over the whole image).  >1 means the model attends to this region more
than average (good supporting evidence); ~1 is expected when a note fills the
frame; <1 means it leaned on surrounding context.  Unlike a raw energy fraction,
this ratio is robust to box size and to multiple notes splitting the attention.

Heatmaps target the three Detect-head inputs (P3/P4/P5 -> 80/40/20 px) and are
averaged across scales.  Everything runs on CPU.

This module is best-effort: explain_image() degrades to (original frame, []) on
any error so the optional explainability path can never break a detection run.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # torch is only needed when explainability is actually requested
    import torch
except Exception:  # pragma: no cover - torch is a hard dep in practice
    torch = None

import config
from logging_config import setup_logging

logger = setup_logging()

# Feature maps feeding the Detect head, highest -> lowest spatial resolution.
# Read from the model when possible; this is the YOLOv8n fallback.
DEFAULT_TARGET_LAYERS = [15, 18, 21]
DEFAULT_METHOD = "eigencam"
DEFAULT_ALPHA = 0.5   # heatmap blend weight over the original image
TOP_K_ANCHORS = 10    # anchors used as the Grad-CAM target per detection
NORM_PERCENTILE = 99  # clip outliers here when normalizing for display


# ─── Model introspection ─────────────────────────────────────────────────────

def _unwrap(model):
    """Return the underlying torch nn.Module from an ultralytics YOLO wrapper."""
    return getattr(model, "model", model)


def _head_input_layers(net) -> List[int]:
    """Best-effort discovery of the Detect head's input layer indices."""
    try:
        f = net.model[-1].f
        if isinstance(f, (list, tuple)) and all(isinstance(i, int) for i in f):
            return list(f)
    except Exception:
        pass
    return DEFAULT_TARGET_LAYERS


def _class_index(model, cls_name: str) -> Optional[int]:
    """Map a class name to the model's class index (None if unknown)."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        for idx, name in names.items():
            if name == cls_name:
                return int(idx)
    if cls_name in config.CLASS_NAMES:
        return config.CLASS_NAMES.index(cls_name)
    return None


# ─── Activation / gradient capture ───────────────────────────────────────────

class _Activations:
    """Forward/backward hooks that stash activations and gradients per layer."""

    def __init__(self, net, layer_indices: List[int], grads: bool = False):
        self.acts: Dict[int, "torch.Tensor"] = {}
        self.grads: Dict[int, "torch.Tensor"] = {}
        self._handles = []
        for i in layer_indices:
            layer = net.model[i]
            self._handles.append(layer.register_forward_hook(self._save_act(i)))
            if grads:
                self._handles.append(
                    layer.register_full_backward_hook(self._save_grad(i)))

    def _save_act(self, i):
        def hook(_m, _inp, out):
            self.acts[i] = out
        return hook

    def _save_grad(self, i):
        def hook(_m, _gin, gout):
            self.grads[i] = gout[0].detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ─── CAM kernels ─────────────────────────────────────────────────────────────

def _eigen_cam(act: "torch.Tensor") -> np.ndarray:
    """Eigen-CAM: project activations onto their first principal component.

    Sign is aligned to per-pixel activation energy so the salient object (not its
    complement) comes out positive — the principal component's sign is otherwise
    arbitrary.
    """
    a = act[0].detach().cpu().numpy().astype(np.float32)   # [C, h, w]
    c, h, w = a.shape
    flat = a.reshape(c, h * w).T                           # [hw, C]
    flat = flat - flat.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(flat, full_matrices=False)
    proj = (flat @ vt[0]).reshape(h, w)
    energy = (a ** 2).sum(axis=0)                          # [h, w]
    if np.corrcoef(proj.ravel(), energy.ravel())[0, 1] < 0:
        proj = -proj
    return proj


def _grad_cam(act: "torch.Tensor", grad: "torch.Tensor") -> np.ndarray:
    """Grad-CAM for one feature map: ReLU(sum_k mean(grad_k) * act_k)."""
    weights = grad.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1)                       # [1, h, w]
    return torch.relu(cam)[0].detach().cpu().numpy().astype(np.float32)


def _anchor_iou(box_xywh: "torch.Tensor", box_xyxy) -> "torch.Tensor":
    """IoU of every anchor's decoded box (center xywh, [4, A]) with one box."""
    cx, cy, w, h = box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]
    ax1, ay1, ax2, ay2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    bx1, by1, bx2, by2 = box_xyxy
    iw = (torch.clamp(ax2, max=bx2) - torch.clamp(ax1, min=bx1)).clamp(min=0)
    ih = (torch.clamp(ay2, max=by2) - torch.clamp(ay1, min=by1)).clamp(min=0)
    inter = iw * ih
    union = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0) \
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union.clamp(min=1e-9)


# ─── Map post-processing ─────────────────────────────────────────────────────

def _normalize(cam: np.ndarray) -> np.ndarray:
    """Min/percentile normalize to [0, 1], clipping outliers so a single spike
    can't flatten the rest of the map to ~0."""
    cam = cam - cam.min()
    hi = np.percentile(cam, NORM_PERCENTILE)
    if hi > 1e-8:
        cam = np.clip(cam / hi, 0.0, 1.0)
    return cam


def _overlay(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a [0,1] CAM (frame-sized) onto the image as a JET heatmap."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, frame_bgr, 1 - alpha, 0)


def _salience_ratio(cam: np.ndarray, bbox, frame_shape) -> float:
    """mean(CAM in box) / mean(CAM overall).  Robust to box size + note count."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    overall = float(cam.mean())
    if overall <= 1e-8 or x2 <= x1 or y2 <= y1:
        return 0.0
    return round(float(cam[y1:y2, x1:x2].mean()) / overall, 3)


def _explanation_text(cls: str, conf: float, risk: Optional[str],
                      ratio: float, warning: Optional[str]) -> str:
    """Plain-language 'why' string for a single detection."""
    if ratio >= 1.25:
        focus = "strongly on this note (well above the image-average saliency)"
    elif ratio >= 1.0:
        focus = "on this note (at or above the image-average saliency)"
    elif ratio >= 0.7:
        focus = "partly on the surrounding context as well as the note"
    else:
        focus = "largely on context outside the note's box"
    parts = [
        f"Predicted {cls} with {conf:.0%} confidence.",
        f"The saliency heatmap shows the model concentrated {focus} "
        f"(salience ratio {ratio:.2f}x the image average).",
    ]
    if risk:
        parts.append(f"Confidence risk band: {risk}.")
    if ratio < 0.7:
        parts.append("Low in-box saliency - treat this detection with caution.")
    if warning:
        parts.append(warning)
    return " ".join(parts)


# ─── Forward helpers ─────────────────────────────────────────────────────────

def _to_input(frame_bgr: np.ndarray, imgsz: int) -> "torch.Tensor":
    """Square-resize a BGR frame to a normalized CHW tensor (keeps CAM->image a
    simple per-axis scale)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz)).astype(np.float32) / 255.0
    return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).contiguous()


def _eigen_maps(net, hooks, x, layers, H, W) -> np.ndarray:
    """Single global Eigen-CAM map (averaged across scales), frame-sized."""
    with torch.no_grad():
        net(x)
    acc = np.zeros((H, W), dtype=np.float32)
    used = 0
    for li in layers:
        if li in hooks.acts:
            acc += cv2.resize(_normalize(_eigen_cam(hooks.acts[li])), (W, H))
            used += 1
    return _normalize(acc / used) if used else acc


def _grad_maps(model, net, hooks, x, detections, layers, H, W, imgsz):
    """One Grad-CAM map per detection (frame-sized), tied to its anchors."""
    with torch.enable_grad():
        out = net(x)
    preds = (out[0] if isinstance(out, (list, tuple)) else out)[0]  # [4+nc, A]
    box_xywh, scores = preds[:4, :], preds[4:, :]
    sx, sy = imgsz / W, imgsz / H

    cams: List[np.ndarray] = []
    for det in detections:
        c = _class_index(model, det.get("class"))
        if c is None:
            c = int(scores.sum(dim=1).argmax().item())
        cls_row = scores[c]
        bbox = det.get("bbox", [0, 0, 0, 0])
        box_xyxy = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
        relevance = _anchor_iou(box_xywh, box_xyxy) * cls_row.detach()
        k = min(TOP_K_ANCHORS, int((relevance > 0).sum().item()))
        target = cls_row[torch.topk(relevance, k).indices].sum() if k > 0 \
            else cls_row.max()

        net.zero_grad(set_to_none=True)
        hooks.grads.clear()
        target.backward(retain_graph=True)

        acc = np.zeros((H, W), dtype=np.float32)
        used = 0
        for li in layers:
            if li in hooks.acts and li in hooks.grads:
                acc += cv2.resize(
                    _normalize(_grad_cam(hooks.acts[li], hooks.grads[li])), (W, H))
                used += 1
        cams.append(_normalize(acc / used) if used else acc)
    return cams


# ─── Public API ──────────────────────────────────────────────────────────────

def explain_image(
    model,
    frame_bgr: np.ndarray,
    detections: List[Dict],
    method: str = DEFAULT_METHOD,
    target_layers: Optional[List[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    imgsz: int = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Compute a saliency overlay + per-detection explanations.

    Returns (overlay_bgr, explanations) index-aligned with `detections`; each
    entry carries `salience_ratio`, `explain_method`, and a human-readable
    `explanation`.  On any failure returns (frame_bgr, []) — best-effort only.
    """
    if torch is None:
        logger.warning("torch unavailable; skipping explainability")
        return frame_bgr, []
    if not detections:
        return frame_bgr, []

    method = (method or DEFAULT_METHOD).lower()
    imgsz = imgsz or config.IMG_SIZE
    H, W = frame_bgr.shape[:2]

    try:
        net = _unwrap(model)
        net.eval()
        layers = target_layers or _head_input_layers(net)
        x = _to_input(frame_bgr, imgsz)

        if method == "gradcam":
            x.requires_grad_(True)  # ensures the autograd graph is built
            hooks = _Activations(net, layers, grads=True)
            try:
                det_cams = _grad_maps(
                    model, net, hooks, x, detections, layers, H, W, imgsz)
            finally:
                hooks.remove()
            ratio_maps = det_cams
            combined = np.max(np.stack(det_cams), axis=0)
        else:  # eigencam (default)
            method = "eigencam"
            hooks = _Activations(net, layers, grads=False)
            try:
                global_map = _eigen_maps(net, hooks, x, layers, H, W)
            finally:
                hooks.remove()
            ratio_maps = [global_map] * len(detections)
            combined = global_map

        explanations: List[Dict] = []
        for det, cam in zip(detections, ratio_maps):
            ratio = _salience_ratio(cam, det.get("bbox", [0, 0, 0, 0]),
                                    frame_bgr.shape)
            explanations.append({
                "explain_method": method,
                "salience_ratio": ratio,
                "explanation": _explanation_text(
                    det.get("class"), float(det.get("confidence", 0.0)),
                    det.get("risk"), ratio, det.get("warning")),
            })

        overlay = _overlay(frame_bgr, _normalize(combined), alpha)
        return overlay, explanations

    except Exception as e:  # explainability must never break a detection run
        logger.exception(f"Explainability ({method}) failed: {e}")
        return frame_bgr, []
