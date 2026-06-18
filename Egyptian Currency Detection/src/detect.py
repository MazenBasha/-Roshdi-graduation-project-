"""
Run multi-note detection on images, folders, video files, or a webcam.

For each input the script produces:
- An annotated image / video (boxes, labels, confidence, count panel)
- A JSON file with detections, per-class counts, and the total EGP value:

    {
      "detections": [
        {"class": "100_EGP", "confidence": 0.94, "bbox": [x1, y1, x2, y2]},
        ...
      ],
      "counts": {"100_EGP": 2, "50_EGP": 1},
      "total": 250
    }

Usage:
    python src/detect.py --image path/to/img.jpg
    python src/detect.py --image-dir path/to/folder
    python src/detect.py --video path/to/clip.mp4
    python src/detect.py --webcam            # camera 0
    python src/detect.py --webcam --camera 1
    python src/detect.py --image x.jpg --weights outputs/runs/train/weights/best.pt
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    load_model,
    parse_yolo_result,
    draw_detections,
    draw_summary_panel,
    build_output_dict,
    save_json,
    summarize,
)
from validators import InputValidator, build_error_json
from logging_config import setup_logging
from confidence_metrics import ConfidenceAnalyzer
from explain import explain_image

logger = setup_logging()


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Egyptian currency multi-note detection")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image")
    src.add_argument("--image-dir", help="Path to a folder of images")
    src.add_argument("--video", help="Path to a video file")
    src.add_argument("--webcam", action="store_true", help="Use a live camera")

    p.add_argument("--weights", default=config.DEFAULT_WEIGHTS,
                   help="Path to trained YOLO weights (.pt)")
    p.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                   help="Min box confidence")
    p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD,
                   help="NMS IoU threshold")
    p.add_argument("--agnostic-nms", dest="agnostic", action="store_true",
                   default=config.AGNOSTIC_NMS,
                   help="Class-agnostic NMS (suppress overlapping boxes across classes)")
    p.add_argument("--no-agnostic-nms", dest="agnostic", action="store_false",
                   help="Disable class-agnostic NMS")
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--output-dir", default=config.DETECT_OUT_DIR,
                   help="Where to save annotated outputs + JSON")
    p.add_argument("--no-show", action="store_true",
                   help="Don't open an OpenCV preview window (use for headless runs)")
    p.add_argument("--explain", action="store_true",
                   help="Generate a saliency heatmap (<stem>_explain.jpg) and add "
                        "per-detection salience + reasoning to the JSON")
    p.add_argument("--explain-method", dest="explain_method",
                   choices=["eigencam", "gradcam"], default="eigencam",
                   help="Explainability method: eigencam (default, robust) or "
                        "gradcam (class-discriminative)")
    return p.parse_args()


# ─── Single-image inference ──────────────────────────────────────────────────

def _save_error(output: Dict, img_path: str, out_dir: str) -> Dict:
    """Persist an error JSON next to where the success JSON would have been."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(img_path or "unknown"))[0]
        save_json(output, os.path.join(out_dir, f"{stem}.json"))
    except Exception as e:
        logger.error(f"Could not write error JSON: {e}",
                     extra={"image_path": img_path})
    return output


def run_on_image(model, img_path: str, args, out_dir: str) -> Dict:
    logger.info("Starting detection", extra={"image_path": img_path})

    # ── Validation ───────────────────────────────────────────────────────
    is_valid, msg, meta = InputValidator.validate_image_content(img_path)
    if not is_valid:
        logger.error(f"Validation failed: {msg}",
                     extra={"image_path": img_path, "error_type": "validation"})
        return _save_error(build_error_json(msg, img_path), img_path, out_dir)

    frame = cv2.imread(img_path)
    if frame is None:
        msg = f"OpenCV could not decode image: {img_path}"
        logger.error(msg, extra={"image_path": img_path, "error_type": "decode"})
        return _save_error(build_error_json(msg, img_path), img_path, out_dir)

    # ── Inference ────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        results = model.predict(
            source=frame, conf=args.conf, iou=args.iou, agnostic_nms=args.agnostic,
            imgsz=args.imgsz, max_det=config.MAX_DETECTIONS, verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.exception("Inference failed",
                         extra={"image_path": img_path, "error_type": "inference"})
        return _save_error(
            build_error_json(f"Inference failed: {e}", img_path), img_path, out_dir,
        )

    detections = parse_yolo_result(results[0], config.CLASS_NAMES)
    detections = ConfidenceAnalyzer.add_uncertainty_flags(detections)
    conf_report = ConfidenceAnalyzer.confidence_report(detections)

    annotated = draw_detections(frame, detections)
    counts, total = summarize(detections)
    annotated = draw_summary_panel(annotated, counts, total)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_out = os.path.join(out_dir, f"{stem}_annotated.jpg")
    json_out = os.path.join(out_dir, f"{stem}.json")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(img_out, annotated)

    # ── Explainability (optional) ─────────────────────────────────────────
    explain_out = None
    if getattr(args, "explain", False):
        overlay, explanations = explain_image(
            model, frame, detections,
            method=getattr(args, "explain_method", "eigencam"),
            imgsz=args.imgsz)
        # Merge per-detection reasoning back so it lands in the output JSON.
        for det, exp in zip(detections, explanations):
            det.update(exp)
        if explanations:
            explain_out = os.path.join(out_dir, f"{stem}_explain.jpg")
            cv2.imwrite(explain_out, overlay)
            logger.info("Grad-CAM written", extra={"image_path": img_path})

    h, w = frame.shape[:2]
    output = build_output_dict(
        detections,
        image_path=img_path,
        inference_ms=elapsed_ms,
        model_version=getattr(args, "weights", None),
        image_size=(w, h),
        confidence_report=conf_report,
    )
    save_json(output, json_out)

    confs = [d["confidence"] for d in detections]
    logger.info(
        "Detection complete",
        extra={
            "image_path": img_path,
            "num_detections": len(detections),
            "inference_ms": round(elapsed_ms, 3),
            "confidence_min": round(min(confs), 4) if confs else None,
            "confidence_max": round(max(confs), 4) if confs else None,
        },
    )

    print(f"\n[{os.path.basename(img_path)}]  ({elapsed_ms:.1f} ms)")
    print(f"  Detections: {len(detections)}")
    for cls, n in sorted(counts.items()):
        print(f"    {cls}: {n}")
    print(f"  TOTAL: {total} EGP")
    print(f"  Saved: {img_out}")
    print(f"         {json_out}")
    if explain_out:
        print(f"         {explain_out}  ({args.explain_method} heatmap)")

    if not args.no_show:
        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output


def run_on_folder(model, folder: str, args) -> List[Dict]:
    ok, msg, images = InputValidator.validate_folder(folder)
    if not ok:
        logger.error(f"Folder validation failed: {msg}",
                     extra={"image_path": folder, "error_type": "validation"})
        return [build_error_json(msg, folder)]

    out_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(folder)))
    results = []
    for img_path in images:
        # Folder mode never opens windows, even if --no-show was off.
        prev_no_show = args.no_show
        args.no_show = True
        try:
            results.append(run_on_image(model, img_path, args, out_dir))
        finally:
            args.no_show = prev_no_show
    return results


# ─── Video / webcam inference ────────────────────────────────────────────────

def run_on_stream(model, source, args, save_path: str = None):
    """`source` is either an int (webcam index) or a video file path."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print("Press 'q' to quit." if not args.no_show else "Running headless...")

    last_log = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame, conf=args.conf, iou=args.iou, agnostic_nms=args.agnostic,
                imgsz=args.imgsz, max_det=config.MAX_DETECTIONS, verbose=False,
            )
            detections = parse_yolo_result(results[0], config.CLASS_NAMES)
            counts, total = summarize(detections)

            annotated = draw_detections(frame, detections)
            annotated = draw_summary_panel(annotated, counts, total)

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("Egyptian Currency Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_log > 1.0:
                last_log = now
                print(f"  detections={len(detections):2d}  total={total} EGP")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {save_path}")
        if not args.no_show:
            cv2.destroyAllWindows()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading weights: {args.weights}")
    model = load_model(args.weights)

    if args.image:
        run_on_image(model, args.image, args, args.output_dir)
    elif args.image_dir:
        run_on_folder(model, args.image_dir, args)
    elif args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        stem = os.path.splitext(os.path.basename(args.video))[0]
        save_path = os.path.join(args.output_dir, f"{stem}_annotated.mp4")
        run_on_stream(model, args.video, args, save_path=save_path)
    elif args.webcam:
        run_on_stream(model, args.camera, args, save_path=None)


if __name__ == "__main__":
    main()
