"""
Dataset download / preparation helpers.

Datasets handled here:
    lfw          -- Labeled Faces in the Wild (evaluation only)
    casia        -- CASIA-WebFace (training, ~10K identities, ~500K images)
    vggface2     -- VGGFace2 (training, very large, requires manual download)
    digiface     -- already on disk; only normalizes the layout

Why no automatic VGGFace2 / MS-Celeb download:
    VGGFace2 is hosted behind academic terms-of-use (Oxford VGG) and
    MS-Celeb-1M was withdrawn by Microsoft. We cannot legitimately download
    them automatically. This script gives you the exact URL to register on
    and the expected on-disk layout afterwards.

LFW IS auto-downloaded -- it's freely hosted by U. Mass.
CASIA-WebFace is auto-downloaded from a community Google Drive mirror
(Tim Esler / facenet-pytorch). Confirm the license before training.

Usage:
    python download_datasets.py --dataset lfw
    python download_datasets.py --dataset casia
    python download_datasets.py --dataset vggface2     # prints instructions
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import config

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
LFW_PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"

# CASIA-WebFace is best fetched through the community-maintained mirror that
# ships with facenet-pytorch's docs. We surface the gdown URL here; the user
# may override with their own mirror if the link goes stale.
CASIA_GDRIVE_ID = "1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l"   # Casia-WebFace (aligned 112x112)
CASIA_FILENAME = "casia-webface.zip"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 14) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] already present: {dest}")
        return
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract_archive(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix in {".tgz", ".gz"} or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")
    print(f"[extract] {archive.name} -> {dest_dir}")


# ---------------------------------------------------------------------------
# LFW (evaluation)
# ---------------------------------------------------------------------------

def fetch_lfw() -> Path:
    out_root = config.DATA_DIR / "lfw"
    out_root.mkdir(parents=True, exist_ok=True)

    archive = out_root / "lfw.tgz"
    download_file(LFW_URL, archive)

    images_dir = out_root / "lfw"
    if not images_dir.exists():
        extract_archive(archive, out_root)

    pairs_path = out_root / "pairs.txt"
    download_file(LFW_PAIRS_URL, pairs_path)

    n_people = sum(1 for p in images_dir.iterdir() if p.is_dir())
    n_images = sum(1 for _ in images_dir.rglob("*.jpg"))
    print(f"[lfw] ready: {n_people} identities, {n_images} images at {images_dir}")
    print(f"[lfw] pairs file: {pairs_path}")
    return out_root


# ---------------------------------------------------------------------------
# CASIA-WebFace (training)
# ---------------------------------------------------------------------------

def fetch_casia() -> Path:
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise ImportError("Install gdown: pip install gdown") from e

    out_root = config.DATA_DIR / "casia-webface"
    out_root.mkdir(parents=True, exist_ok=True)
    archive = out_root / CASIA_FILENAME

    if not archive.exists():
        print(f"[casia] downloading via gdown id={CASIA_GDRIVE_ID}")
        gdown.download(id=CASIA_GDRIVE_ID, output=str(archive), quiet=False)

    images_dir = out_root / "casia-webface"
    if not images_dir.exists():
        extract_archive(archive, out_root)

    n_people = sum(1 for p in images_dir.iterdir() if p.is_dir())
    print(f"[casia] ready: {n_people} identities at {images_dir}")
    print(f"[casia] train with: python train.py --data {images_dir}")
    return out_root


# ---------------------------------------------------------------------------
# VGGFace2 (training) — manual instructions only
# ---------------------------------------------------------------------------

def fetch_vggface2() -> None:
    print("""
[vggface2] VGGFace2 is hosted by Oxford VGG behind an academic-use license
that requires a Google account and acceptance of terms. We can't download
it programmatically without violating the terms.

    1. Register at:  https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
    2. Download `vggface2_train.tar.gz` (~36 GB).
    3. Extract so you have:
           {data}/vggface2/train/n000001/0001_01.jpg
           {data}/vggface2/train/n000001/0002_01.jpg
           ...
    4. (Optional but recommended) align faces to 112x112 with MTCNN:
           python -c "from inference import FaceDetector; ..."
       or use insightface's `align_faces.py` script.
    5. Then train:
           python train.py --data {data}/vggface2/train --epochs 30
""".format(data=config.DATA_DIR))


# ---------------------------------------------------------------------------
# DigiFace (already on disk on the user's Desktop)
# ---------------------------------------------------------------------------

def link_digiface(src: Path) -> Path:
    """Symlink the existing DigiFace subset into data/ for convenience."""
    if not src.exists():
        raise FileNotFoundError(src)
    dest = config.DATA_DIR / "digiface_subset"
    if dest.exists():
        print(f"[digiface] already linked: {dest}")
        return dest
    try:
        dest.symlink_to(src.resolve(), target_is_directory=True)
        print(f"[digiface] symlinked {src} -> {dest}")
    except OSError:
        # Filesystems without symlink support: fall back to copy.
        shutil.copytree(src, dest)
        print(f"[digiface] copied {src} -> {dest}")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lfw", "casia", "vggface2", "digiface"], required=True)
    p.add_argument("--digiface-src", type=str, default="/Users/jilan/Desktop/data_subset/train",
                   help="Source path for an existing DigiFace subset (only for --dataset digiface)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "lfw":
        fetch_lfw()
    elif args.dataset == "casia":
        fetch_casia()
    elif args.dataset == "vggface2":
        fetch_vggface2()
    elif args.dataset == "digiface":
        link_digiface(Path(args.digiface_src))


if __name__ == "__main__":
    main()
