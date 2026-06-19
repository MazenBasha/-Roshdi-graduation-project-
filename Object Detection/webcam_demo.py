"""Real-time webcam object detection using the pretrained YOLO11n model.

A standalone demo — does NOT train or fine-tune. Opens the default webcam,
runs YOLO11n per frame, draws boxes + labels + an FPS overlay, and prints
newly-detected classes to the terminal.

Usage:
    python webcam_demo.py
    python webcam_demo.py --camera 1 --conf 0.35 --imgsz 640
    python webcam_demo.py --weights path/to/custom.pt

Press 'q' in the video window to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(weights: str, imgsz: int) -> YOLO:
    """Verify the model loads and is callable before the camera is opened."""
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    # Warmup on a black frame so the first real frame isn't slow / errors early.
    _ = model.predict(
        source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
        imgsz=imgsz, verbose=False,
    )
    n = len(getattr(model, "names", {}) or {})
    print(f"Model ready — {n} classes")
    return model


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        msg = (
            f"Cannot open camera index {index}.\n"
            "On macOS the terminal needs Camera permission:\n"
            "  System Settings → Privacy & Security → Camera → enable Terminal / iTerm."
        )
        raise RuntimeError(msg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def draw(frame: np.ndarray, box, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def color_for(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id)
    return tuple(int(v) for v in rng.integers(64, 256, size=3))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt",
                   help="Pretrained name (auto-downloaded) or path to a custom .pt")
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV camera index (0 = default)")
    p.add_argument("--conf", type=float, default=0.40,
                   help="Confidence threshold for displayed detections")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-runtime", type=float, default=0.0,
                   help="If >0, auto-exit after this many seconds (used by tests).")
    args = p.parse_args()

    try:
        model = load_model(args.weights, args.imgsz)
    except Exception as e:
        print(f"FATAL: model load failed — {e}", file=sys.stderr)
        return 1

    try:
        cap = open_camera(args.camera)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    names = model.names if hasattr(model, "names") else {}
    window = "YOLO11n - press 'q' to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_dt = deque(maxlen=30)
    last_seen: set[str] = set()
    consecutive_read_failures = 0
    t_started = time.perf_counter()
    print("Streaming. Press 'q' in the window to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures > 30:
                    print("FATAL: camera read failed repeatedly — aborting.", file=sys.stderr)
                    return 3
                time.sleep(0.03)
                continue
            consecutive_read_failures = 0

            results = model.predict(
                source=frame, imgsz=args.imgsz,
                conf=args.conf, iou=args.iou, verbose=False,
            )
            r = results[0]
            current: set[str] = set()
            if r.boxes is not None and r.boxes.shape[0] > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for box, c, k in zip(xyxy, confs, cls):
                    name = names.get(int(k), str(int(k)))
                    draw(frame, box, f"{name} {c:.2f}", color_for(int(k)))
                    current.add(name)

            new_classes = current - last_seen
            for name in sorted(new_classes):
                print(f"[detected] {name}", flush=True)
            last_seen = current

            frame_dt.append(time.perf_counter() - t0)
            if sum(frame_dt) > 0:
                fps = len(frame_dt) / sum(frame_dt)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if args.max_runtime > 0 and time.perf_counter() - t_started > args.max_runtime:
                print(f"max-runtime ({args.max_runtime}s) reached, exiting")
                break
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
