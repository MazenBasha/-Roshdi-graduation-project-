"""
Reliability layer for the face recognition pipeline.

Provides:
    - SafeRecognizer: a wrapper around FaceRecognizer.recognize_image_dicts
      that enforces input safety, rejection logic, and unknown-person
      detection with a calibrated confidence channel.
    - InputValidationError: raised for malformed inputs at the API boundary.

Three rejection / safety mechanisms layered on top of cosine similarity:

  1. Hard input validation: shape, dtype, range, NaN/Inf, blank-image
     short-circuit.
  2. Confidence thresholding: if best cos < `match_threshold`, return
     {"name": "Unknown"} unconditionally.
  3. Margin-based abstain: when best cos is above threshold but the gap
     to the runner-up is small, return {"name": "Uncertain"} so the
     caller can ask for another frame. This is critical for assistive
     applications where misidentification has higher cost than abstaining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from inference import FaceRecognizer
from utils import l2_normalize


class InputValidationError(ValueError):
    pass


@dataclass
class RejectionPolicy:
    match_threshold: float = config.MATCH_THRESHOLD
    abstain_margin: float = 0.05      # require best - runner_up >= margin
    abstain_softmax_min: float = 0.55 # top-K softmax must be >= this
    blank_std_threshold: float = 2.0  # std in pixel space below this -> blank
    min_face_score: float = 0.92      # MTCNN confidence floor


def validate_image(image_bgr) -> np.ndarray:
    if image_bgr is None:
        raise InputValidationError("Input image is None")
    if not isinstance(image_bgr, np.ndarray):
        raise InputValidationError(f"Expected np.ndarray, got {type(image_bgr).__name__}")
    if image_bgr.dtype != np.uint8:
        raise InputValidationError(f"Expected dtype=uint8, got {image_bgr.dtype}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise InputValidationError(
            f"Expected HxWx3 BGR image, got shape {image_bgr.shape}"
        )
    h, w = image_bgr.shape[:2]
    if h < 32 or w < 32:
        raise InputValidationError(f"Image too small: {h}x{w} (min 32x32)")
    if h > 4096 or w > 4096:
        raise InputValidationError(f"Image too large: {h}x{w} (max 4096 per side)")
    if not np.isfinite(image_bgr).all():
        raise InputValidationError("Image contains NaN/Inf")
    return image_bgr


def is_blank(image_bgr: np.ndarray, std_threshold: float = 2.0) -> bool:
    """Reject images that are essentially constant — covered lens, black
    frame, all-white scan, etc."""
    return float(image_bgr.std()) < std_threshold


def softmax(x: np.ndarray, temperature: float = 10.0) -> np.ndarray:
    z = x * temperature
    z = z - z.max()
    e = np.exp(z); return e / e.sum()


class SafeRecognizer:
    def __init__(self, recognizer: FaceRecognizer | None = None,
                 policy: RejectionPolicy | None = None):
        self.rec = recognizer or FaceRecognizer()
        self.policy = policy or RejectionPolicy()

    def recognize(self, image_bgr: np.ndarray) -> dict:
        """Top-level safe entry point. Returns a dict with:
            ok: bool
            error: str | None
            faces: list of {name, confidence, reason, top_k}
        """
        try:
            image_bgr = validate_image(image_bgr)
        except InputValidationError as e:
            return {"ok": False, "error": str(e), "faces": []}
        if is_blank(image_bgr, self.policy.blank_std_threshold):
            return {"ok": True, "error": None, "faces": [],
                    "note": "blank or near-uniform input — short-circuited"}

        results = self.rec.recognize_image(image_bgr,
                                           threshold=self.policy.match_threshold)
        faces = []
        for r in results:
            if r.detection_score < self.policy.min_face_score:
                faces.append({"name": "Rejected", "confidence": float(r.detection_score),
                              "reason": "low detection score",
                              "bbox": [int(v) for v in r.bbox],
                              "top_k": []})
                continue
            # Top-K against the DB (centroid match).
            q = l2_normalize(r.embedding.astype(np.float32))
            scores = []
            for name, rec in self.rec.db.records.items():
                if rec.centroid is None:
                    continue
                scores.append((name, float(np.dot(q, rec.centroid))))
            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[: max(2, 5)]
            if not top:
                faces.append({"name": "Unknown", "confidence": 0.0,
                              "reason": "empty gallery", "top_k": [],
                              "bbox": [int(v) for v in r.bbox]})
                continue
            best_name, best_score = top[0]
            runner_score = top[1][1] if len(top) > 1 else -1.0
            margin = best_score - runner_score
            sm = softmax(np.array([s for _, s in top]))
            label, reason = best_name, "match"
            if best_score < self.policy.match_threshold:
                label, reason = "Unknown", "below match threshold"
            elif margin < self.policy.abstain_margin or sm[0] < self.policy.abstain_softmax_min:
                label, reason = "Uncertain", "thin margin to runner-up"
            faces.append({
                "name": label,
                "confidence": float(best_score),
                "softmax_confidence": float(sm[0]),
                "margin_to_runner_up": float(margin),
                "reason": reason,
                "bbox": [int(v) for v in r.bbox],
                "top_k": [{"name": n, "cosine": float(s)} for n, s in top],
            })
        return {"ok": True, "error": None, "faces": faces}
