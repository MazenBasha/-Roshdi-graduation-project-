"""
Shared helpers: model loading, detection post-processing, drawing, JSON output.
"""

import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

OUTPUT_SCHEMA_VERSION = "2.0"


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(weights_path: str):
    """Load a YOLOv8 model from a .pt file. Imports lazily so the project
    can be inspected without ultralytics installed."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Train first: python src/train.py"
        )
    from ultralytics import YOLO
    return YOLO(weights_path)


# ─── Result post-processing ──────────────────────────────────────────────────

def parse_yolo_result(result, class_names: List[str]) -> List[Dict]:
    """Convert one ultralytics Results object into our detection dict list."""
    detections: List[Dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        detections.append({
            "class": cls_name,
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        })
    return detections


def summarize(detections: List[Dict]) -> Tuple[Dict[str, int], int]:
    """Count notes per class and compute the total EGP value."""
    counts = Counter(d["class"] for d in detections)
    total = sum(config.DENOMINATIONS.get(cls, 0) * n for cls, n in counts.items())
    return dict(counts), int(total)


# ─── Visualization ───────────────────────────────────────────────────────────

# Distinct BGR colors per class (matplotlib tab10-ish, picked for visibility).
_PALETTE = [
    (255, 87, 51),   # 1
    (51, 255, 87),   # 5
    (51, 87, 255),   # 10
    (255, 51, 221),  # 20
    (51, 221, 255),  # 50
    (255, 221, 51),  # 100
    (170, 51, 255),  # 200
]


def _color_for(cls_name: str) -> Tuple[int, int, int]:
    if cls_name in config.CLASS_NAMES:
        return _PALETTE[config.CLASS_NAMES.index(cls_name) % len(_PALETTE)]
    return (200, 200, 200)


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes + per-box labels on a copy of the frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = _color_for(det["class"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class']}  {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 8, ly + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def draw_summary_panel(frame: np.ndarray, counts: Dict[str, int], total: int) -> np.ndarray:
    """Overlay a top-left panel showing per-class counts and the total."""
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [f"{cls}: {n}" for cls, n in sorted(counts.items())]
    lines.append(f"TOTAL: {total} EGP")

    # Panel sizing
    pad = 10
    line_h = 24
    panel_w = 220
    panel_h = pad * 2 + line_h * len(lines)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 8
        is_total = i == len(lines) - 1
        color = (0, 220, 255) if is_total else (240, 240, 240)
        scale = 0.65 if is_total else 0.55
        thickness = 2 if is_total else 1
        cv2.putText(out, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return out


# ─── JSON output ─────────────────────────────────────────────────────────────

def _confidence_stats(detections: List[Dict]) -> Dict:
    confs = [float(d.get("confidence", 0.0)) for d in detections]
    if not confs:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    n = len(confs)
    mean = sum(confs) / n
    var = sum((c - mean) ** 2 for c in confs) / n
    return {
        "min": round(min(confs), 4),
        "max": round(max(confs), 4),
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
    }


def build_output_dict(
    detections: List[Dict],
    image_path: Optional[str] = None,
    inference_ms: Optional[float] = None,
    model_version: Optional[str] = None,
    image_size: Optional[Tuple[int, int]] = None,
    confidence_report: Optional[Dict] = None,
) -> Dict:
    """Build the schema-v2 output document.

    Backward compatible: the legacy top-level fields (`detections`, `counts`,
    `total`) are still present so existing parsers keep working.  New fields
    are additive.
    """
    counts, total = summarize(detections)

    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None
    area_full = (width * height) if (width and height) else None

    enriched: List[Dict] = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        new_det = {
            "id": idx,
            "class": det.get("class"),
            "confidence": float(det.get("confidence", 0.0)),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "area_pixels": int(max(0, (x2 - x1)) * max(0, (y2 - y1))),
        }
        if "risk" in det:
            new_det["risk"] = det["risk"]
        if "warning" in det:
            new_det["warning"] = det["warning"]
        # Explainability fields (present only when --explain was used).
        for key in ("explain_method", "salience_ratio", "explanation"):
            if key in det:
                new_det[key] = det[key]
        if width and height:
            new_det["bbox_normalized"] = [
                round(float(x1) / width, 4),
                round(float(y1) / height, 4),
                round(float(x2) / width, 4),
                round(float(y2) / height, 4),
            ]
        enriched.append(new_det)

    high_risk_count = sum(
        1 for d in enriched if d.get("risk") in ("high", "reject")
    )

    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image": image_path,
            "model_version": model_version,
            "inference_time_ms": round(float(inference_ms), 3) if inference_ms is not None else None,
            "image_size": [width, height] if width and height else None,
        },
        "detections": enriched,
        "summary": {
            "num_detections": len(enriched),
            "counts": counts,
            "total_egp": total,
            "confidence": _confidence_stats(enriched),
            "high_risk_count": high_risk_count,
        },
        # Legacy fields - preserved for backward compatibility.
        "counts": counts,
        "total": total,
        "success": True,
    }
    if confidence_report is not None:
        output["confidence_report"] = confidence_report
    return output


def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
