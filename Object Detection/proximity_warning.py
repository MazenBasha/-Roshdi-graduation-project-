"""Proximity warning logic for Roshdi Object Detection.

This module converts YOLO-style detections into voice-friendly proximity warnings.
It does NOT estimate real distance in meters. It estimates visual proximity using
normalized bounding-box area/height inside the camera frame.

Expected detection input format:
{
    "class_name": "chair",
    "confidence": 0.88,
    "bbox": [x1, y1, x2, y2]
}
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


# All values are starting points. Tune them using your real phone/webcam.
THRESHOLD_CONFIG: Dict[str, Any] = {
    "groups": {
        "large_danger": {
            "classes": {"person", "car", "bus", "truck", "motorcycle", "bicycle"},
            "near": {"area": 0.18, "height": 0.45, "mode": "or"},
            "very_near": {"area": 0.35, "height": 0.65, "mode": "or"},
        },
        "medium_obstacle": {
            "classes": {"chair", "bench", "couch", "dining table", "suitcase", "backpack"},
            "near": {"area": 0.12, "height": 0.35, "mode": "or"},
            "very_near": {"area": 0.25, "height": 0.55, "mode": "or"},
        },
        "small_object": {
            "classes": {"bottle", "cup", "cell phone", "book", "remote", "mouse", "keyboard"},
            "near": {"area": 0.08, "height": None, "mode": "center_area_only"},
            "very_near": {"area": 0.16, "height": None, "mode": "center_area_only"},
        },
    },
    "default": {
        "near": {"area": 0.15, "height": 0.40, "mode": "or"},
        "very_near": {"area": 0.30, "height": 0.60, "mode": "or"},
    },
}

# Simple Arabic names for common COCO classes used in warnings.
ARABIC_CLASS_NAMES: Dict[str, str] = {
    "person": "شخص",
    "bicycle": "عجلة",
    "car": "عربية",
    "motorcycle": "موتوسيكل",
    "bus": "أتوبيس",
    "truck": "شاحنة",
    "chair": "كرسي",
    "bench": "دكة",
    "couch": "كنبة",
    "dining table": "ترابيزة",
    "suitcase": "شنطة",
    "backpack": "شنطة ظهر",
    "bottle": "زجاجة",
    "cup": "كوباية",
    "cell phone": "موبايل",
    "book": "كتاب",
    "remote": "ريموت",
    "mouse": "ماوس",
    "keyboard": "كيبورد",
}

POSITION_EN = {
    "left": "on your left",
    "center": "in front of you",
    "right": "on your right",
}

POSITION_AR = {
    "left": "على الشمال",
    "center": "قدامك",
    "right": "على اليمين",
}

DANGER_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
OBSTACLE_CLASSES = {"chair", "bench", "couch", "dining table", "suitcase", "backpack"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _article(name: str) -> str:
    """Return a simple English article for speech."""
    if not name:
        return "an object"
    return ("an " if name[0].lower() in "aeiou" else "a ") + name


@dataclass
class ProximityWarningSystem:
    """Stateful proximity-warning engine.

    The class keeps a short frame history for temporal smoothing and a cooldown
    dictionary to avoid repeating the same warning too frequently.
    """

    min_confidence: float = 0.40
    cooldown_seconds: float = 2.0
    history_size: int = 3
    required_hits: int = 2
    config: Dict[str, Any] = field(default_factory=lambda: THRESHOLD_CONFIG)

    history: deque = field(init=False)
    last_alert_time: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.history_size)

    def process_frame(
        self,
        frame_width: int,
        frame_height: int,
        detections: Iterable[Dict[str, Any]],
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process one frame and return a voice-friendly result dictionary."""
        now = time.time() if now is None else now
        analyzed = []

        for det in detections:
            item = self._analyze_detection(frame_width, frame_height, det)
            if item is not None:
                analyzed.append(item)

        # Only near/very_near objects are candidates for speech.
        warning_candidates = [d for d in analyzed if d["distance_hint"] in {"near", "very_near"}]

        # Temporal smoothing: save current frame candidates before checking stable keys.
        frame_keys = {(d["class_name"], d["horizontal_position"], d["distance_hint"]) for d in warning_candidates}
        self.history.append(frame_keys)

        stable_candidates = [d for d in warning_candidates if self._is_stable(d)]
        main_object = self._select_priority_object(stable_candidates)

        if main_object is None:
            return {
                "should_speak": False,
                "message_en": "",
                "message_ar": "",
                "main_object": None,
                "all_detections": analyzed,
            }

        key = (main_object["class_name"], main_object["horizontal_position"])
        if not self._cooldown_passed(key, now):
            return {
                "should_speak": False,
                "message_en": "",
                "message_ar": "",
                "main_object": main_object,
                "all_detections": analyzed,
            }

        self.last_alert_time[key] = now
        message_en, message_ar = self._generate_messages(main_object)

        return {
            "should_speak": True,
            "message_en": message_en,
            "message_ar": message_ar,
            "main_object": main_object,
            "all_detections": analyzed,
        }

    def _analyze_detection(self, frame_width: int, frame_height: int, det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter and enrich one detection with ratios, position, and distance hint."""
        confidence = _safe_float(det.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None

        class_name = str(det.get("class_name", "object")).strip().lower()
        bbox = det.get("bbox") or det.get("box")
        if isinstance(bbox, dict):
            bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
        if not bbox or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [_safe_float(v) for v in bbox]
        x1, x2 = max(0.0, min(x1, frame_width)), max(0.0, min(x2, frame_width))
        y1, y2 = max(0.0, min(y1, frame_height)), max(0.0, min(y2, frame_height))
        if x2 <= x1 or y2 <= y1 or frame_width <= 0 or frame_height <= 0:
            return None

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        frame_area = float(frame_width * frame_height)
        area_ratio = bbox_area / frame_area
        height_ratio = bbox_height / float(frame_height)
        center_x = (x1 + x2) / 2.0
        horizontal_position = self._horizontal_position(center_x, frame_width)
        distance_hint = self._distance_hint(class_name, area_ratio, height_ratio, horizontal_position)

        return {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "distance_hint": distance_hint,
            "horizontal_position": horizontal_position,
            "area_ratio": round(area_ratio, 4),
            "height_ratio": round(height_ratio, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        }

    @staticmethod
    def _horizontal_position(center_x: float, frame_width: int) -> str:
        if center_x < 0.33 * frame_width:
            return "left"
        if center_x > 0.66 * frame_width:
            return "right"
        return "center"

    def _distance_hint(self, class_name: str, area_ratio: float, height_ratio: float, horizontal_position: str) -> str:
        thresholds = self._thresholds_for_class(class_name)
        if self._condition_met(thresholds["very_near"], area_ratio, height_ratio, horizontal_position):
            return "very_near"
        if self._condition_met(thresholds["near"], area_ratio, height_ratio, horizontal_position):
            return "near"
        return "far"

    def _thresholds_for_class(self, class_name: str) -> Dict[str, Dict[str, Any]]:
        for group in self.config["groups"].values():
            if class_name in group["classes"]:
                return {"near": group["near"], "very_near": group["very_near"]}
        return self.config["default"]

    @staticmethod
    def _condition_met(rule: Dict[str, Any], area_ratio: float, height_ratio: float, horizontal_position: str) -> bool:
        mode = rule.get("mode", "or")
        area_thr = rule.get("area")
        height_thr = rule.get("height")

        area_ok = area_thr is not None and area_ratio > float(area_thr)
        height_ok = height_thr is not None and height_ratio > float(height_thr)

        if mode == "center_area_only":
            return horizontal_position == "center" and area_ok
        if mode == "and":
            return area_ok and height_ok
        return area_ok or height_ok

    def _is_stable(self, detection: Dict[str, Any]) -> bool:
        """Return true if same class + position is near/very_near in enough recent frames."""
        class_name = detection["class_name"]
        position = detection["horizontal_position"]
        count = 0
        for frame_keys in self.history:
            for key_class, key_pos, key_dist in frame_keys:
                if key_class == class_name and key_pos == position and key_dist in {"near", "very_near"}:
                    count += 1
                    break
        return count >= self.required_hits

    @staticmethod
    def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
        """Optional IoU helper for future tracking across frames."""
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return 0.0 if union <= 0 else inter_area / union

    def _select_priority_object(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        def priority(det: Dict[str, Any]) -> Tuple[int, float]:
            name = det["class_name"]
            if det["distance_hint"] == "very_near":
                group_priority = 0
            elif name in DANGER_CLASSES:
                group_priority = 1
            elif name in OBSTACLE_CLASSES:
                group_priority = 2
            elif name == "person":
                group_priority = 3
            else:
                group_priority = 4
            # Sort by lower priority first, then larger area_ratio first.
            return (group_priority, -float(det["area_ratio"]))

        return sorted(candidates, key=priority)[0]

    def _cooldown_passed(self, key: Tuple[str, str], now: float) -> bool:
        last = self.last_alert_time.get(key)
        return last is None or (now - last) >= self.cooldown_seconds

    @staticmethod
    def _generate_messages(det: Dict[str, Any]) -> Tuple[str, str]:
        name = det["class_name"]
        pos = det["horizontal_position"]
        dist = det["distance_hint"]

        en_obj = _article(name)
        en_pos = POSITION_EN.get(pos, "near you")
        ar_obj = ARABIC_CLASS_NAMES.get(name, "جسم")
        ar_pos = POSITION_AR.get(pos, "قريب منك")

        if dist == "very_near":
            message_en = f"Danger, {en_obj} is very close {en_pos}."
            message_ar = f"خطر، {ar_obj} قريب جدًا {ar_pos}"
        else:
            if name in DANGER_CLASSES:
                message_en = f"Warning, {en_obj} is close {en_pos}."
                message_ar = f"تحذير، {ar_obj} قريب {ar_pos}"
            else:
                message_en = f"Be careful, {en_obj} is close {en_pos}."
                message_ar = f"خلي بالك، {ar_obj} قريب {ar_pos}"

        return message_en, message_ar


if __name__ == "__main__":
    # Minimal standalone example.
    system = ProximityWarningSystem(min_confidence=0.4)
    sample_detections = [
        {"class_name": "chair", "confidence": 0.88, "bbox": [120, 180, 420, 640]},
        {"class_name": "bottle", "confidence": 0.74, "bbox": [520, 360, 570, 450]},
    ]

    # Call 3 times to demonstrate temporal smoothing.
    for _ in range(3):
        output = system.process_frame(640, 720, sample_detections)
        print(output)
        time.sleep(0.2)
