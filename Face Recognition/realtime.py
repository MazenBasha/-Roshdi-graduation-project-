"""
Production real-time face recognition for the Roshdi assistive glasses.

What's different from a naive recognize-each-frame loop:

  * Multi-face per frame — every visible face is detected, embedded,
    matched, and annotated independently.
  * IoU + EMA tracking — a `Track` object persists across frames so the
    same person keeps the same on-screen ID even if MTCNN momentarily
    drops them. Embeddings are smoothed with EMA over recent frames.
  * Majority-vote display name — single mis-frames don't change the
    label; the displayed name only flips after several consistent
    matches. Kills the "Ali / Unknown / Ali / Unknown" flicker.
  * Novelty-gated online enrollment — when the user enrolls a track, we
    keep adding NEW views (different pose / lighting) to that person's
    rolling window over subsequent frames as long as each new embedding
    is sufficiently different from the centroid. The DB grows toward a
    pose-diverse template instead of N copies of the same frame.
  * CLAHE in the embedding pipeline — equalizes brightness on the L
    channel of LAB so harsh backlight / dim rooms don't shift the
    embedding distribution.

Keyboard controls (focus the OpenCV window):
  q  -- quit
  e  -- toggle enrollment mode for the next unknown track in view
  r  -- toggle continuous re-enrollment mode (novel views auto-added to
        the currently-shown known tracks)
  s  -- save the current annotated frame
  +  -- raise match threshold (more conservative)
  -  -- lower match threshold (more permissive)
  c  -- toggle CLAHE preprocessing
  t  -- toggle test-time augmentation (TTA)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from inference import FaceRecognizer
from tracker import FaceTracker


# Distinct BGR colors recycled across track IDs.
_PALETTE = [
    (0, 200, 0), (0, 165, 255), (255, 100, 0), (255, 50, 200),
    (0, 220, 220), (180, 180, 0), (50, 200, 255), (220, 100, 220),
]


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default=config.BACKBONE)
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--every", type=int, default=1,
                   help="Run recognition every N frames (>=1)")
    p.add_argument("--no-clahe", action="store_true",
                   help="Disable CLAHE preprocessing in the embedding pipeline")
    p.add_argument("--tta", action="store_true",
                   help="Enable horizontal-flip test-time augmentation")
    p.add_argument("--track-iou", type=float, default=0.30)
    p.add_argument("--track-max-missed", type=int, default=8)
    p.add_argument("--track-ema", type=float, default=0.30)
    p.add_argument("--vote-min", type=int, default=3,
                   help="Min consistent name matches (over last 10 frames) "
                        "before a track's display name is allowed to switch")
    return p.parse_args()


def annotate_tracks(frame: np.ndarray, tracks_with_dets: list, threshold: float) -> np.ndarray:
    out = frame.copy()
    for track, _di in tracks_with_dets:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        color = color_for_track(track.track_id)
        if track.last_name is None:
            label = f"#{track.track_id}  Unknown  ({track.last_similarity:.2f})"
            box_color = (0, 0, 220)
        else:
            label = f"#{track.track_id}  {track.last_name}  ({track.last_similarity:.2f})"
            box_color = color
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        # Solid label background for legibility on busy frames.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    use_clahe = not args.no_clahe

    rec = FaceRecognizer(
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        use_clahe=use_clahe,
        tta=args.tta,
    )
    tracker = FaceTracker(
        iou_threshold=args.track_iou,
        max_missed=args.track_max_missed,
        ema_alpha=args.track_ema,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    threshold = args.threshold
    enroll_next_unknown = False
    auto_reenroll = False
    fps_smooth = 0.0
    t_prev = time.time()
    frame_idx = 0
    last_render: list = []

    print("[realtime] running. q quit | e enroll next unknown | r toggle auto-reenroll")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[realtime] camera read failed")
            break

        if frame_idx % max(1, args.every) == 0:
            results = rec.recognize_image(frame, threshold=threshold)
            # Tracker input: (bbox, embedding) per detection.
            track_inputs = [(r.bbox, r.embedding) for r in results]
            track_pairs = tracker.step(track_inputs)
            # Re-vote each updated track using its raw frame match.
            for trk, di in track_pairs:
                trk.vote_name(results[di].name, results[di].similarity, min_votes=args.vote_min)
                # Online novelty-gated enrollment of known tracks: add this
                # frame's view if it's a substantially different pose/lighting.
                if auto_reenroll and trk.last_name is not None:
                    rec.enroll(trk.last_name, results[di].embedding, novelty_threshold=0.92)

            # Enrollment trigger: capture the next unknown track.
            if enroll_next_unknown:
                target = next(
                    (t for t, _ in track_pairs if t.last_name is None and t.age >= 3),
                    None,
                )
                if target is not None:
                    cv2.imshow("enroll-this-face",
                               results[next(di for t, di in track_pairs if t is target)].aligned_face)
                    cv2.waitKey(1)
                    name = input("  Name for highlighted track "
                                 f"#{target.track_id} (blank to skip): ").strip()
                    if name:
                        rec.enroll(name, target.embedding)
                        # Force the track to immediately reflect the new ID.
                        target.last_name = name
                        target.name_history.clear()
                        for _ in range(args.vote_min):
                            target.name_history.append(name)
                        print(f"  -> enrolled '{name}' (DB size = {len(rec.db)})")
                    try:
                        cv2.destroyWindow("enroll-this-face")
                    except cv2.error:
                        pass
                    enroll_next_unknown = False

            last_render = track_pairs

        annotated = annotate_tracks(frame, last_render, threshold)

        # FPS + status overlay.
        now = time.time()
        instant_fps = 1.0 / max(1e-6, now - t_prev)
        t_prev = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth else instant_fps
        n_visible = sum(1 for t, _ in last_render)
        status = (
            f"FPS {fps_smooth:4.1f} | thr {threshold:.2f} | "
            f"DB {len(rec.db)} | tracks {n_visible} | "
            f"clahe={'on' if rec.use_clahe else 'off'} | "
            f"tta={'on' if rec.tta else 'off'} | "
            f"enroll={'WAITING' if enroll_next_unknown else 'off'} | "
            f"auto={'ON' if auto_reenroll else 'off'}"
        )
        cv2.putText(annotated, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Roshdi Face Recognition", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            enroll_next_unknown = not enroll_next_unknown
            print(f"[realtime] enroll_next_unknown = {enroll_next_unknown}")
        elif key == ord("r"):
            auto_reenroll = not auto_reenroll
            print(f"[realtime] auto_reenroll = {auto_reenroll}")
        elif key == ord("s"):
            out_path = Path(config.LOG_DIR) / f"frame_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"[realtime] saved {out_path}")
        elif key in (ord("+"), ord("=")):
            threshold = min(1.0, threshold + 0.02)
        elif key == ord("-"):
            threshold = max(0.0, threshold - 0.02)
        elif key == ord("c"):
            rec.use_clahe = not rec.use_clahe
        elif key == ord("t"):
            rec.tta = not rec.tta

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
