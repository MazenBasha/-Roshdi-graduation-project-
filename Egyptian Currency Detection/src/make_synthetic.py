"""
Generate a synthetic multi-note detection dataset from the single-note photos.

The original dataset only has full-image pseudo-boxes, so the model learned
"a note fills the whole frame" and at inference splits one note into multiple
partial boxes (wrong counts + cross-class confusion). This script composites
the existing single-note crops onto larger canvases at random positions,
scales, and rotations, writing EXACT YOLO boxes. Training on these teaches
real localization and counting.

Output: data_synth/images/{train,val}/*.jpg + data_synth/labels/{train,val}/*.txt
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SRC_IMAGES = os.path.join(config.DATA_DIR, "images")
SRC_LABELS = os.path.join(config.DATA_DIR, "labels")
OUT_DIR = os.path.join(config.PROJECT_ROOT, "data_synth")

# Notes per canvas and their sampling weights (favor 1-3, the realistic range).
NOTE_COUNTS = [1, 2, 3, 4]
NOTE_WEIGHTS = [0.30, 0.32, 0.26, 0.12]

CANVAS_RANGE = (640, 960)        # random square-ish canvas side
NOTE_FRAC_RANGE = (0.28, 0.55)   # note long-side as fraction of canvas
MAX_ROT = 18                     # degrees
MAX_IOU = 0.35                   # cap overlap so boxes stay meaningful
PLACE_TRIES = 40


def load_index(split):
    """Return dict {class_id: [image_paths]} for a split."""
    by_cls = {}
    for img in glob.glob(os.path.join(SRC_IMAGES, split, "*.jpg")):
        lf = os.path.join(SRC_LABELS, split,
                          os.path.splitext(os.path.basename(img))[0] + ".txt")
        try:
            cid = int(open(lf).read().split()[0])
        except Exception:
            continue
        by_cls.setdefault(cid, []).append(img)
    return by_cls


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def make_canvas(side, rng):
    """Textured background: a heavily blurred, dim random note photo or a flat color."""
    if rng.random() < 0.5 and _BG_POOL:
        bg = Image.open(rng.choice(_BG_POOL)).convert("RGB").resize((side, side))
        from PIL import ImageFilter, ImageEnhance
        bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(8, 20)))
        bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.4, 0.8))
        return bg
    c = tuple(rng.randint(30, 200) for _ in range(3))
    return Image.new("RGB", (side, side), c)


def place_one(canvas, note_img, side, placed_boxes, rng):
    """Try to paste one rotated note without exceeding MAX_IOU. Returns box or None.

    The note is rotated in RGBA with expand so transparent corners don't paste
    as black; the box is the rotated image's extent — the tight axis-aligned
    bounding box of the rotated note, exactly what YOLO needs.
    """
    note = Image.open(note_img).convert("RGBA")
    frac = rng.uniform(*NOTE_FRAC_RANGE)
    target = int(side * frac)
    w, h = note.size
    scale = target / max(w, h)
    note = note.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    note = note.rotate(rng.uniform(-MAX_ROT, MAX_ROT), expand=True,
                       resample=Image.BICUBIC)

    nw, nh = note.size
    if nw >= side or nh >= side:
        return None

    for _ in range(PLACE_TRIES):
        px = rng.randint(0, side - nw)
        py = rng.randint(0, side - nh)
        box = (px, py, px + nw, py + nh)
        if all(iou(box, b) <= MAX_IOU for b in placed_boxes):
            canvas.paste(note, (px, py), note)  # use alpha as paste mask
            return box
    return None


def gen_split(split, n_images, rng):
    by_cls = load_index(split)
    classes = sorted(by_cls)
    # inverse-frequency weights so rare classes (1_EGP) appear more often
    freq = {c: len(by_cls[c]) for c in classes}
    inv = {c: 1.0 / freq[c] for c in classes}
    s = sum(inv.values())
    cls_w = [inv[c] / s for c in classes]

    img_out = os.path.join(OUT_DIR, "images", split)
    lbl_out = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    written = 0
    for i in range(n_images):
        side = rng.randint(*CANVAS_RANGE)
        canvas = make_canvas(side, rng)
        n_notes = rng.choices(NOTE_COUNTS, NOTE_WEIGHTS)[0]
        placed_boxes, lines = [], []
        for _ in range(n_notes):
            cid = rng.choices(classes, cls_w)[0]
            note_path = rng.choice(by_cls[cid])
            box = place_one(canvas, note_path, side, placed_boxes, rng)
            if box is None:
                continue
            placed_boxes.append(box)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / side
            cy = (y1 + y2) / 2 / side
            bw = (x2 - x1) / side
            bh = (y2 - y1) / side
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        name = f"synth_{split}_{i:05d}"
        canvas.save(os.path.join(img_out, name + ".jpg"), quality=88)
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1
        if written % 200 == 0:
            print(f"  [{split}] {written}/{n_images}")
    print(f"[{split}] wrote {written} composites")


_BG_POOL = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1800)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    global _BG_POOL
    _BG_POOL = glob.glob(os.path.join(SRC_IMAGES, "train", "*.jpg"))[:400]

    gen_split("train", args.train, rng)
    gen_split("val", args.val, rng)
    print(f"\nDone. Synthetic dataset at: {OUT_DIR}")


if __name__ == "__main__":
    main()
