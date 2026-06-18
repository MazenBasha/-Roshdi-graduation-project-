"""
Live camera inference for Egyptian Currency Classification.

Detects rectangular currency note regions using OpenCV contour analysis,
draws bounding boxes, and classifies each detected note.

Features:
- Automatic currency region detection (edge + contour based)
- Bounding box overlay with class label and confidence
- Prediction smoothing over time for stability
- Falls back to center-crop if no region is detected
- Test-time augmentation (TTA) for higher accuracy

Usage:
    python camera.py
    python camera.py --model outputs/model.ptl
    python camera.py --camera 1
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Live Camera Currency Detection")
    parser.add_argument(
        "--model", type=str, default=config.BEST_MODEL_PATH,
        help="Path to model (.pth or .ptl)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K predictions to show")
    parser.add_argument("--min-confidence", type=float, default=0.40,
                        help="Min confidence to show a detection")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    if model_path.endswith(".ptl") or model_path.endswith(".pt"):
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = build_model()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
    model.eval()
    return model


# ─── Currency Region Detection ──────────────────────────────────────────────

def detect_currency_regions(frame):
    """
    Detect rectangular regions in the frame that are likely currency notes.
    Uses edge detection + contour approximation to find rectangles.
    Returns list of (x, y, w, h) bounding boxes.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    # Preprocess: blur to reduce noise, then detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold handles uneven lighting better than Canny alone
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # Morphological close to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Also try Canny for cleaner edges
    edges = cv2.Canny(blurred, 30, 120)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Combine both approaches
    combined = cv2.bitwise_or(closed, edges_dilated)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Currency note should be at least 3% of frame and at most 85%
        if area < frame_area * 0.03 or area > frame_area * 0.85:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Currency notes are roughly rectangular (4 corners),
        # but allow 4-8 vertices for perspective distortion
        if 4 <= len(approx) <= 8:
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            # Currency notes have aspect ratio roughly 1.5-2.5
            # But allow wider range for tilted/partial views
            if 1.2 <= aspect_ratio <= 4.0:
                # Add some padding (10% on each side)
                pad_x = int(bw * 0.1)
                pad_y = int(bh * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                regions.append((x1, y1, x2 - x1, y2 - y1))

    # Sort by area (largest first) and limit to top 3
    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions[:3]


# ─── Classification ─────────────────────────────────────────────────────────

def get_inference_transform():
    """Transform for camera input — direct resize without aggressive crop."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


def get_tta_transforms():
    """
    Test-Time Augmentation: multiple views of the same crop.
    Average predictions across all views for more stable results.
    """
    base = [
        # Original
        transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ]),
        # Slight center crop
        transforms.Compose([
            transforms.Resize(int(config.IMG_SIZE * 1.15)),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ]),
    ]
    return base


@torch.no_grad()
def classify_region(model, pil_img, transforms_list, device):
    """
    Classify a PIL image using TTA (average predictions over multiple views).
    Returns (class_index, confidence, all_probs).
    """
    all_probs = None
    for t in transforms_list:
        tensor = t(pil_img).unsqueeze(0).to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu()
        if all_probs is None:
            all_probs = probs
        else:
            all_probs += probs
        del tensor, output

    all_probs /= len(transforms_list)
    conf, pred = all_probs.max(dim=0)
    return pred.item(), conf.item(), all_probs


# ─── Prediction Smoother ────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Smooths predictions over recent frames to avoid flickering.
    Uses exponential moving average of class probabilities.
    """

    def __init__(self, num_classes: int, window: int = 8, alpha: float = 0.4):
        self.num_classes = num_classes
        self.alpha = alpha
        self.smoothed = None
        self.history = deque(maxlen=window)

    def update(self, probs):
        """Update with new frame probabilities, return smoothed result."""
        if isinstance(probs, torch.Tensor):
            probs = probs.numpy()

        self.history.append(probs)

        if self.smoothed is None:
            self.smoothed = probs.copy()
        else:
            self.smoothed = self.alpha * probs + (1 - self.alpha) * self.smoothed

        pred = int(np.argmax(self.smoothed))
        conf = float(self.smoothed[pred])
        return pred, conf, self.smoothed

    def reset(self):
        self.smoothed = None
        self.history.clear()


# ─── Drawing Helpers ─────────────────────────────────────────────────────────

def draw_detection(frame, x, y, w, h, label, confidence, color):
    """Draw a bounding box with label on the frame."""
    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    # Corner accents (makes the box look more professional)
    corner_len = min(25, w // 4, h // 4)
    for cx, cy, dx, dy in [
        (x, y, 1, 1), (x + w, y, -1, 1),
        (x, y + h, 1, -1), (x + w, y + h, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + corner_len * dx, cy), color, 5)
        cv2.line(frame, (cx, cy), (cx, cy + corner_len * dy), color, 5)

    # Label background
    text = f"{label} EGP  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_y = max(y - 10, th + 10)
    cv2.rectangle(frame, (x, label_y - th - 8), (x + tw + 10, label_y + 4), color, -1)
    cv2.putText(frame, text, (x + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_sidebar(frame, probs, top_k=3):
    """Draw top-K class probabilities as a sidebar."""
    indices = np.argsort(probs)[::-1][:top_k]
    y_start = 80
    for i, idx in enumerate(indices):
        prob = probs[idx]
        if prob < 0.05:
            break
        cls_name = config.CLASS_NAMES[idx]
        bar_len = int(prob * 180)
        y = y_start + i * 30

        # Background bar
        cv2.rectangle(frame, (8, y - 2), (220, y + 22), (40, 40, 40), -1)
        # Confidence bar
        bar_color = (0, 200, 0) if prob > 0.7 else (0, 200, 255) if prob > 0.4 else (150, 150, 150)
        cv2.rectangle(frame, (10, y), (10 + bar_len, y + 18), bar_color, -1)
        # Text
        cv2.putText(frame, f"{cls_name}: {prob:.0%}",
                    (15, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_status_bar(frame, fps, mode, device_name):
    """Draw bottom status bar."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS: {fps:.0f} | Mode: {mode} | {device_name}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, "Press 'q' to quit",
                (w - 150, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = f"GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "CPU"
    print(f"Device: {device_name}")

    # Load model
    model = load_model(args.model, device)
    tta_transforms = get_tta_transforms()
    simple_transform = get_inference_transform()
    print("Model loaded.")

    # Prediction smoother
    smoother = PredictionSmoother(config.NUM_CLASSES, window=10, alpha=0.35)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened. Press 'q' to quit.")

    fps_list = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # Step 1: Try to detect currency regions
        regions = detect_currency_regions(frame)
        mode = "DETECTING"

        if regions:
            # Classify the largest detected region
            x, y, w, h = regions[0]
            crop_bgr = frame[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            pred_idx, conf, probs = classify_region(model, pil_crop, tta_transforms, device)
            smoothed_pred, smoothed_conf, smoothed_probs = smoother.update(probs.numpy())

            label = config.CLASS_NAMES[smoothed_pred]

            if smoothed_conf >= args.min_confidence:
                mode = "DETECTED"
                # Color based on confidence
                if smoothed_conf > 0.8:
                    color = (0, 220, 0)      # green = confident
                elif smoothed_conf > 0.5:
                    color = (0, 200, 255)    # orange = moderate
                else:
                    color = (0, 100, 255)    # red = uncertain

                draw_detection(display, x, y, w, h, label, smoothed_conf, color)

                # Draw additional detected regions (smaller ones)
                for rx, ry, rw, rh in regions[1:]:
                    sub_bgr = frame[ry:ry+rh, rx:rx+rw]
                    sub_rgb = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB)
                    pil_sub = Image.fromarray(sub_rgb)
                    sub_tensor = simple_transform(pil_sub).unsqueeze(0).to(device)
                    with torch.no_grad():
                        sub_out = model(sub_tensor)
                    sub_probs = torch.softmax(sub_out, dim=1)[0]
                    sub_conf, sub_pred = sub_probs.max(dim=0)
                    if sub_conf.item() >= args.min_confidence:
                        sub_label = config.CLASS_NAMES[sub_pred.item()]
                        draw_detection(display, rx, ry, rw, rh,
                                       sub_label, sub_conf.item(), (200, 200, 0))
                    del sub_tensor, sub_out, sub_probs

                draw_sidebar(display, smoothed_probs, args.top_k)
            else:
                mode = "LOW CONF"
                cv2.rectangle(display, (x, y), (x+w, y+h), (100, 100, 100), 2)
        else:
            # Fallback: classify the center crop of the frame
            center_size = min(h_frame, w_frame) * 2 // 3
            cx, cy = w_frame // 2, h_frame // 2
            x1 = cx - center_size // 2
            y1 = cy - center_size // 2
            crop_bgr = frame[y1:y1+center_size, x1:x1+center_size]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            pred_idx, conf, probs = classify_region(model, pil_crop, [simple_transform], device)
            smoothed_pred, smoothed_conf, smoothed_probs = smoother.update(probs.numpy())

            if smoothed_conf >= args.min_confidence:
                label = config.CLASS_NAMES[smoothed_pred]
                # Draw a guide rectangle showing what area is being classified
                cv2.rectangle(display, (x1, y1), (x1+center_size, y1+center_size),
                              (150, 150, 150), 1)
                cv2.putText(display, f"Center: {label} EGP {smoothed_conf:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)
                draw_sidebar(display, smoothed_probs, args.top_k)
            mode = "CENTER CROP"

        # Title bar
        cv2.rectangle(display, (0, 0), (w_frame, 35), (20, 20, 20), -1)
        cv2.putText(display, "Egyptian Currency Detector",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        elapsed = (time.perf_counter() - t0) * 1000
        fps_list.append(1000.0 / max(elapsed, 1))
        avg_fps = sum(fps_list) / len(fps_list)
        draw_status_bar(display, avg_fps, mode, device_name)

        cv2.imshow("Egyptian Currency Detector", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()
