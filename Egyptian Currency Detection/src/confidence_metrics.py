"""
Confidence + uncertainty analysis on YOLO detections.

The model is over-confident on the 50/100 EGP pair (similar color palette)
and weakest on 1 EGP (rare class in our dataset).  This module attaches
risk flags + per-class warnings so callers can decide when a detection
needs human review.
"""

import math
from typing import Dict, List

# Empirically tuned bands.  Tweak in one place; the rest of the project reads them.
RISK_LOW = 0.90
RISK_MEDIUM = 0.75
RISK_HIGH = 0.50

CONFUSION_WARNINGS = {
    "100_EGP": (0.95, "May be confused with 50_EGP (similar color)"),
    "50_EGP": (0.95, "May be confused with 100_EGP (similar color)"),
    "1_EGP": (0.90, "Rare class, fewer training examples"),
}


def _risk_label(confidence: float) -> str:
    if confidence >= RISK_LOW:
        return "low"
    if confidence >= RISK_MEDIUM:
        return "medium"
    if confidence >= RISK_HIGH:
        return "high"
    return "reject"


class ConfidenceAnalyzer:
    @staticmethod
    def filter_by_confidence(detections: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Keep only detections with confidence >= threshold."""
        return [d for d in detections if float(d.get("confidence", 0.0)) >= threshold]

    @staticmethod
    def add_uncertainty_flags(detections: List[Dict]) -> List[Dict]:
        """Annotate each detection with `risk` and optional `warning` fields.

        Operates in place on a shallow copy so callers can keep the original
        list intact if they need it.
        """
        annotated: List[Dict] = []
        for det in detections:
            new_det = dict(det)
            conf = float(new_det.get("confidence", 0.0))
            new_det["risk"] = _risk_label(conf)

            cls = new_det.get("class")
            if cls in CONFUSION_WARNINGS:
                thresh, msg = CONFUSION_WARNINGS[cls]
                if conf < thresh:
                    new_det["warning"] = msg
            annotated.append(new_det)
        return annotated

    @staticmethod
    def confidence_report(detections: List[Dict]) -> Dict:
        """Summary statistics over a list of detections."""
        if not detections:
            return {
                "num_detections": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "std_confidence": 0.0,
                "high_confidence_count": 0,
                "medium_confidence_count": 0,
                "low_confidence_count": 0,
                "high_risk_detections": [],
            }

        confs = [float(d.get("confidence", 0.0)) for d in detections]
        n = len(confs)
        mean = sum(confs) / n
        variance = sum((c - mean) ** 2 for c in confs) / n
        std = math.sqrt(variance)

        high = sum(1 for c in confs if c >= RISK_LOW)
        medium = sum(1 for c in confs if RISK_MEDIUM <= c < RISK_LOW)
        low = sum(1 for c in confs if c < RISK_MEDIUM)

        risky = []
        for d in detections:
            risk = d.get("risk") or _risk_label(float(d.get("confidence", 0.0)))
            if risk in ("high", "reject"):
                entry = {
                    "class": d.get("class"),
                    "confidence": float(d.get("confidence", 0.0)),
                }
                if "warning" in d:
                    entry["warning"] = d["warning"]
                risky.append(entry)

        return {
            "num_detections": n,
            "avg_confidence": round(mean, 4),
            "min_confidence": round(min(confs), 4),
            "max_confidence": round(max(confs), 4),
            "std_confidence": round(std, 4),
            "high_confidence_count": high,
            "medium_confidence_count": medium,
            "low_confidence_count": low,
            "high_risk_detections": risky,
        }