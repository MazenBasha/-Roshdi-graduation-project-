"""
Convert an InsightFace MXNet RecordIO dataset (`train.rec` + `train.idx`)
into an ImageFolder layout that `dataset.FaceFolderDataset` can load.

Why:
    The official CASIA-WebFace / VGGFace2 / MS1M packs are distributed as
    MXNet RecordIO. Our training pipeline is PyTorch-native and expects
    one-folder-per-identity. This script bridges the two without
    requiring an mxnet install (mxnet has no Py3.13 wheels).

Format reference:
    https://mxnet.apache.org/api/architecture/note_data_loading
    Each record:
        magic     : uint32, 0xced7230a
        cflag/len : uint32, top 2 bits = continuation flag, low 30 bits = length
        payload   : `length` bytes, padded to 4-byte alignment
            IRHeader (first 24 bytes):
                flag  : int32     (label count if multi-label, else 0)
                label : float32   (single label when flag == 0)
                id    : uint64
                id2   : uint64
            image_bytes : JPEG/PNG-encoded (length - 24 bytes)

Usage:
    # Convert the whole dataset
    python convert_recordio.py \
        --rec  data/casia-webface/faces_webface_112x112/train.rec \
        --idx  data/casia-webface/faces_webface_112x112/train.idx \
        --out  data/casia-webface_imgs

    # Convert a subset (good for short experiments)
    python convert_recordio.py --rec ... --idx ... --out ... \
        --max-identities 1000 \
        --max-per-identity 30
"""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

MAGIC = 0xCED7230A
HEADER_FMT = "<ifQQ"
HEADER_SIZE = 24  # int32 + float32 + uint64 + uint64


def read_idx_map(idx_path: Path) -> dict[int, int]:
    """Parse the .idx companion: each line is `<record_id>\\t<byte_offset>`."""
    out: dict[int, int] = {}
    with idx_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[int(parts[0])] = int(parts[1])
    return out


def read_record(rec_file, offset: int) -> tuple[int, bytes]:
    """Return (label, raw_image_bytes) for the record stored at `offset`."""
    rec_file.seek(offset)
    magic = struct.unpack("<I", rec_file.read(4))[0]
    if magic != MAGIC:
        raise ValueError(f"bad magic at offset {offset}: {hex(magic)}")
    cflag_length = struct.unpack("<I", rec_file.read(4))[0]
    length = cflag_length & 0x3FFFFFFF
    payload = rec_file.read(length)
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"short payload at offset {offset}")
    _flag, label_f, _id, _id2 = struct.unpack(HEADER_FMT, payload[:HEADER_SIZE])
    return int(label_f), payload[HEADER_SIZE:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", type=str, required=True, help="Path to train.rec")
    p.add_argument("--idx", type=str, required=True, help="Path to train.idx")
    p.add_argument("--out", type=str, required=True, help="Output ImageFolder root")
    p.add_argument("--max-identities", type=int, default=0,
                   help="Keep only the first N identities (0 = all)")
    p.add_argument("--max-per-identity", type=int, default=0,
                   help="Keep only the first N images per identity (0 = all)")
    p.add_argument("--ext", type=str, default=".jpg",
                   help="Output file extension (the bytes are written verbatim, "
                        "so this should match the encoded image format)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[convert] reading idx ...")
    idx_map = read_idx_map(Path(args.idx))
    record_ids = sorted(idx_map.keys())
    print(f"[convert] {len(record_ids):,} records in idx")

    # First pass (cheap): label -> [record_ids] mapping. The InsightFace
    # convention is that record_id 0 is a special "header" containing
    # per-identity ranges, so we read records on the fly and bucket them.
    # The first record in the index is metadata; we skip if it has label
    # -1 or otherwise looks non-image.
    by_label: dict[int, list[int]] = defaultdict(list)
    with Path(args.rec).open("rb") as f:
        for rec_id in tqdm(record_ids, desc="indexing"):
            try:
                label, img = read_record(f, idx_map[rec_id])
            except Exception:
                continue
            # The very first record(s) in some packs are metadata blocks
            # with label fields that point to identity ranges; their
            # "image_bytes" don't decode as images. We accept them and
            # they'll fail the JPEG sanity check below, harmlessly.
            if not img or len(img) < 16:
                continue
            by_label[label].append(rec_id)

    # Apply identity / per-identity caps.
    sorted_labels = sorted(by_label.keys())
    if args.max_identities > 0:
        sorted_labels = sorted_labels[: args.max_identities]
    print(f"[convert] keeping {len(sorted_labels):,} identities")

    n_total_kept = 0
    with Path(args.rec).open("rb") as f:
        for label in tqdm(sorted_labels, desc="extract"):
            rec_ids = by_label[label]
            if args.max_per_identity > 0:
                rec_ids = rec_ids[: args.max_per_identity]
            id_dir = out / f"{label:07d}"
            id_dir.mkdir(exist_ok=True)
            for rid in rec_ids:
                _label, img_bytes = read_record(f, idx_map[rid])
                # Sanity: most pack contents are JPEG (FFD8 FF) or PNG (89 50).
                if not (img_bytes[:2] in (b"\xff\xd8", b"\x89P")):
                    continue
                out_path = id_dir / f"{rid:09d}{args.ext}"
                if not out_path.exists():
                    out_path.write_bytes(img_bytes)
                    n_total_kept += 1

    print(f"[convert] wrote {n_total_kept:,} images across {len(sorted_labels):,} "
          f"identities to {out}")


if __name__ == "__main__":
    main()
