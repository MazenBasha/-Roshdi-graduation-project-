"""
Bootstrap a YOLO dataset from a classification-style folder layout.

Source layout (one note per image, label = folder name):
    src/<split>/<class_name>/<image>.jpg

Destination layout (YOLO):
    data_yolo/images/<split>/<image>.jpg
    data_yolo/labels/<split>/<image>.txt   # one line: <class_id> 0.5 0.5 1.0 1.0

Because the source dataset has no bounding boxes, every image is given a
PSEUDO-LABEL: a full-image box. This works for single-note photos with a
clean background. For multi-note scenes you must hand-label real boxes.

Class folder names (with old/new variants) are mapped to our 7-class scheme:
    "1"        -> 1_EGP
    "5"        -> 5_EGP
    "10"       -> 10_EGP
    "10 (new)" -> 10_EGP
    "20"       -> 20_EGP
    "20 (new)" -> 20_EGP
    "50"       -> 50_EGP
    "100"      -> 100_EGP
    "200"      -> 200_EGP

Usage:
    python src/bootstrap_yolo_dataset.py
    python src/bootstrap_yolo_dataset.py --src "e:/data/egyptian_currency/data"
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


# Source folder name (classification dataset) -> YOLO class id (7-class scheme)
FOLDER_TO_ID = {
    "1": 0,
    "5": 1,
    "10": 2,
    "10 (new)": 2,
    "20": 3,
    "20 (new)": 3,
    "50": 4,
    "100": 5,
    "200": 6,
}

# Source split name -> YOLO split name
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join("e:", os.sep, "data",
                                                  "egyptian_currency", "data"),
                   help="Root of the classification-style dataset")
    p.add_argument("--dst", default=config.DATA_DIR,
                   help="Destination YOLO data root")
    p.add_argument("--copy", action="store_true", default=True,
                   help="Copy images (default). Use --link to symlink instead.")
    p.add_argument("--link", action="store_true",
                   help="Symlink images instead of copying (saves ~417 MB)")
    return p.parse_args()


def write_pseudo_label(label_path: str, class_id: int) -> None:
    # Full-image box: x_center=0.5, y_center=0.5, w=1.0, h=1.0
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def main():
    args = parse_args()
    if not os.path.isdir(args.src):
        raise FileNotFoundError(f"Source not found: {args.src}")

    use_link = args.link
    img_root = os.path.join(args.dst, "images")
    lbl_root = os.path.join(args.dst, "labels")

    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for src_split, dst_split in SPLIT_MAP.items():
        src_split_dir = os.path.join(args.src, src_split)
        if not os.path.isdir(src_split_dir):
            print(f"[skip] missing split: {src_split_dir}")
            continue

        out_img_dir = os.path.join(img_root, dst_split)
        out_lbl_dir = os.path.join(lbl_root, dst_split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for cls_folder in sorted(os.listdir(src_split_dir)):
            if cls_folder not in FOLDER_TO_ID:
                print(f"  [skip] unknown class folder: {cls_folder}")
                continue
            class_id = FOLDER_TO_ID[cls_folder]
            cls_dir = os.path.join(src_split_dir, cls_folder)
            if not os.path.isdir(cls_dir):
                continue

            # Folder slug for filename uniqueness across "10" and "10 (new)" etc.
            slug = cls_folder.replace(" ", "_").replace("(", "").replace(")", "")

            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMG_EXT:
                    skipped += 1
                    continue

                stem = os.path.splitext(fname)[0]
                # Prefix filename with the class slug to avoid collisions
                # between "10/img1.jpg" and "10 (new)/img1.jpg".
                new_stem = f"{slug}__{stem}"

                src_img = os.path.join(cls_dir, fname)
                dst_img = os.path.join(out_img_dir, new_stem + ext)
                dst_lbl = os.path.join(out_lbl_dir, new_stem + ".txt")

                if os.path.exists(dst_img):
                    pass  # already done — overwrite label only
                elif use_link:
                    try:
                        os.symlink(src_img, dst_img)
                    except OSError:
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

                write_pseudo_label(dst_lbl, class_id)
                counts[dst_split] += 1

        print(f"[{dst_split}] {counts[dst_split]} images "
              f"-> {out_img_dir}")

    print(f"\nSkipped non-image files: {skipped}")
    print(f"Total: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"\nNOTE: every label is a PSEUDO full-image box. Suitable only for")
    print(f"single-note training images. Re-label multi-note photos by hand.")


if __name__ == "__main__":
    main()
