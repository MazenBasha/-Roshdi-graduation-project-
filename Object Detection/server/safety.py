"""Output filters.

Roshdi's users are visually impaired; the model only sees their environment.
But the *cloud* server might process a stranger's face. We:

  1. Optionally pixelate the upper third of any 'person' box before returning
     a visualisation (PII).
  2. Suppress detections whose label is on a deny list (none by default — COCO
     is non-sensitive — but the hook exists for future custom labels).
"""
from __future__ import annotations

from PIL import Image

from .config import Settings
from .schemas import Detection


SENSITIVE_LABELS: set[str] = set()  # populated by ops if a model adds sensitive classes


def filter_detections(detections: list[Detection]) -> list[Detection]:
    if not SENSITIVE_LABELS:
        return detections
    return [d for d in detections if d.class_name not in SENSITIVE_LABELS]


def pixelate_persons(img: Image.Image, detections: list[Detection], cfg: Settings) -> Image.Image:
    """Pixelate the top third of each 'person' box (approx. head region)."""
    if not cfg.blur_persons:
        return img
    img = img.copy()
    for d in detections:
        if d.class_name != "person":
            continue
        x1, y1, x2, y2 = int(d.box.x1), int(d.box.y1), int(d.box.x2), int(d.box.y2)
        head_y2 = y1 + int((y2 - y1) / 3)
        crop = img.crop((x1, y1, x2, head_y2))
        if crop.size[0] < 8 or crop.size[1] < 8:
            continue
        small = crop.resize((max(crop.size[0] // 16, 1), max(crop.size[1] // 16, 1)), Image.BILINEAR)
        crop = small.resize(crop.size, Image.NEAREST)
        img.paste(crop, (x1, y1, x2, head_y2))
    return img
