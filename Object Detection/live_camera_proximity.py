"""Live webcam demo with YOLO + Roshdi proximity warnings.

Usage examples:
    python live_camera_proximity.py
    python live_camera_proximity.py --camera 1 --weights yolo11n.pt --conf 0.40
    python live_camera_proximity.py --speak --lang ar

Press Q in the OpenCV window to quit.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from proximity_warning import ProximityWarningSystem


class Speaker:
    """Optional non-blocking text-to-speech wrapper.

    If pyttsx3 is not installed or fails, the app will still print warnings.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._engine = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        if not enabled:
            return

        try:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except Exception as exc:  # pragma: no cover - depends on local OS audio
            print(f"WARNING: text-to-speech disabled because pyttsx3 failed: {exc}")
            self.enabled = False

    def say(self, text: str) -> None:
        if not text:
            return
        print(f"VOICE: {text}")
        if self.enabled and self._engine is not None:
            self._queue.put(text)

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"WARNING: speech failed: {exc}")
            finally:
                self._queue.task_done()


def load_model(weights: str, imgsz: int) -> YOLO:
    print(f"Loading YOLO model: {weights}")
    model = YOLO(weights)
    # Warm up with a black frame.
    _ = model.predict(source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, verbose=False)
    print(f"Model ready. Number of classes: {len(getattr(model, 'names', {}) or {})}")
    return model


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    # CAP_DSHOW often opens faster on Windows. If it fails, fallback to default.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. Try --camera 1 or check camera permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def yolo_result_to_detections(result: Any, names: Dict[int, str]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if result.boxes is None or result.boxes.shape[0] == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        detections.append(
            {
                "class_name": names.get(int(cls_id), str(int(cls_id))),
                "confidence": float(conf),
                "bbox": [float(v) for v in box],
            }
        )
    return detections


def color_for_class(class_name: str) -> tuple[int, int, int]:
    seed = abs(hash(class_name)) % (2**32)
    rng = np.random.default_rng(seed)
    return tuple(int(v) for v in rng.integers(80, 255, size=3))


def draw_detection(frame: np.ndarray, det: Dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    color = color_for_class(det["class_name"])
    label = f"{det['class_name']} {det['confidence']:.2f} | {det['distance_hint']} | {det['horizontal_position']}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_label = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_label), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(frame: np.ndarray, text: str, fps: float, processed_fps: float) -> None:
    overlay = f"{text} | camera FPS: {fps:.1f} | processed FPS target: {processed_fps:.1f}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, overlay, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roshdi live camera object proximity warning demo")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO .pt weights path or ultralytics model name")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Use 0 for default webcam")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO and proximity min confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--process-fps", type=float, default=5.0, help="Run YOLO/proximity logic at this FPS")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown in seconds for same warning")
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech using pyttsx3 if available")
    parser.add_argument("--lang", choices=["en", "ar"], default="en", help="Speech language field to use")
    parser.add_argument("--print-json", action="store_true", help="Print full proximity output JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_interval = 1.0 / max(args.process_fps, 0.1)

    try:
        model = load_model(args.weights, args.imgsz)
    except Exception as exc:
        print(f"FATAL: model loading failed: {exc}", file=sys.stderr)
        print("Tip: make sure ultralytics is installed and you have internet once if yolo11n.pt is not cached.", file=sys.stderr)
        return 1

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    names = {int(k): v for k, v in getattr(model, "names", {}).items()} if hasattr(model, "names") else {}
    proximity = ProximityWarningSystem(min_confidence=args.conf, cooldown_seconds=args.cooldown)
    speaker = Speaker(enabled=args.speak)

    window_name = "Roshdi Object Detection - Proximity Warning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_process_time = 0.0
    last_output: Dict[str, Any] = {"all_detections": []}
    frames = 0
    fps = 0.0
    fps_timer = time.perf_counter()

    print("Running. Press Q in the camera window to quit.")
    print(f"Processing YOLO/proximity at {args.process_fps:.1f} FPS, using latest camera frame only.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame from camera")
            time.sleep(0.03)
            continue

        now = time.perf_counter()
        frames += 1
        if now - fps_timer >= 1.0:
            fps = frames / (now - fps_timer)
            frames = 0
            fps_timer = now

        # Run expensive inference only every 200 ms by default. The displayed
        # frame is always the newest camera frame, so there is no old-frame queue.
        if now - last_process_time >= process_interval:
            last_process_time = now
            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            yolo_detections = yolo_result_to_detections(results[0], names)
            frame_h, frame_w = frame.shape[:2]
            last_output = proximity.process_frame(frame_w, frame_h, yolo_detections)

            if args.print_json:
                print(json.dumps(last_output, ensure_ascii=False, indent=2))

            if last_output.get("should_speak"):
                message = last_output["message_ar"] if args.lang == "ar" else last_output["message_en"]
                speaker.say(message)

        display = frame.copy()
        for det in last_output.get("all_detections", []):
            draw_detection(display, det)

        status_text = "No warning"
        if last_output.get("main_object"):
            obj = last_output["main_object"]
            status_text = f"{obj['class_name']} {obj['distance_hint']} {obj['horizontal_position']}"
        if last_output.get("should_speak"):
            status_text = "SPEAK: " + (last_output["message_ar"] if args.lang == "ar" else last_output["message_en"])

        draw_status(display, status_text, fps, args.process_fps)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
