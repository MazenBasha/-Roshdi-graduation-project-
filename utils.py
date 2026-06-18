"""
Utility module: face database, alignment, similarity, image preprocessing.

The face database is a persistent dictionary stored as pickle. Each person
holds a rolling window of L2-normalized embeddings plus a centroid that is
recomputed on every update. Centroid matching is more robust to pose/lighting
variation than nearest-neighbor on a single embedding.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

import config


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

# ArcFace normalization: x = (x - 127.5) / 128.0  (range roughly [-1, 1])
_ARC_MEAN = 127.5
_ARC_STD = 128.0


def apply_clahe(face_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the LUMA channel only.

    Why: real-world wearable-glasses footage has wildly varying exposure
    (going indoors, backlight, etc.). Applying CLAHE on the L channel of LAB
    color space lifts shadow detail and tames highlights without shifting
    color balance — and it's cheap (~0.2 ms for a 112x112 crop). Doing it
    only on the brightness channel preserves the embedding network's color
    statistics, which is much safer than full RGB equalization.
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_face(
    face_bgr: np.ndarray,
    size: int = config.INPUT_SIZE,
    use_clahe: bool = False,
) -> torch.Tensor:
    """BGR uint8 face crop -> CHW float tensor ready for the embedding model.

    `use_clahe=True` applies adaptive histogram equalization first; useful in
    real-world inference where lighting varies. Disable for training-time
    preprocessing so the model sees the same statistics it was trained on.
    """
    if face_bgr.shape[:2] != (size, size):
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if use_clahe:
        face_bgr = apply_clahe(face_bgr)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - _ARC_MEAN) / _ARC_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw)


def batch_preprocess(
    face_list: Iterable[np.ndarray],
    size: int = config.INPUT_SIZE,
    use_clahe: bool = False,
) -> torch.Tensor:
    return torch.stack([preprocess_face(f, size, use_clahe=use_clahe) for f in face_list], dim=0)


# ---------------------------------------------------------------------------
# Face alignment via 5-point landmarks (similarity transform)
# ---------------------------------------------------------------------------

# Canonical 5-point template for 112x112 ArcFace.
_REFERENCE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a face crop so the 5 landmarks land on the canonical positions.

    `landmarks` must be a (5, 2) array in the SAME image coordinates as `image_bgr`.
    Returns a `size x size` BGR aligned crop.
    """
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    dst = _REFERENCE_5PTS_112.copy()
    if size != 112:
        dst *= size / 112.0
    # estimateAffinePartial2D = similarity transform (scale + rotation + translation)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        # Fallback: just resize the bounding crop.
        return cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(image_bgr, M, (size, size), borderValue=0.0)


# ---------------------------------------------------------------------------
# Embedding math
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(np.asarray(a, dtype=np.float32))
    b = l2_normalize(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Face database (persistent, dynamic enrollment)
# ---------------------------------------------------------------------------

@dataclass
class PersonRecord:
    """Per-identity entry in the face DB.

    Holds a rolling window of L2-normalized embeddings (deque) plus a centroid
    that is recomputed on every update. The window cap (MAX_EMBEDDINGS_PER_PERSON)
    is critical — without it, online enrollment from a webcam quickly accumulates
    hundreds of nearly-identical embeddings per person, the centroid stops
    moving, and adapting to lighting / pose changes becomes impossible.
    """
    name: str
    embeddings: deque = field(default_factory=lambda: deque(maxlen=config.MAX_EMBEDDINGS_PER_PERSON))
    centroid: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    n_recognitions: int = 0

    def add(self, embedding: np.ndarray) -> None:
        """Add a new embedding to the rolling window and recompute the centroid."""
        self.embeddings.append(l2_normalize(embedding.astype(np.float32)))
        stacked = np.stack(list(self.embeddings), axis=0)
        self.centroid = l2_normalize(stacked.mean(axis=0))
        self.updated_at = time.time()

    def maybe_add(self, embedding: np.ndarray, novelty_threshold: float = 0.85) -> bool:
        """Add `embedding` only if it's sufficiently novel vs the current centroid.

        Avoids polluting the window with near-duplicates from consecutive webcam
        frames (cosine sim ~0.99 against the centroid). A new embedding is only
        kept if cos(emb, centroid) < `novelty_threshold` — i.e. it represents
        a genuinely different view (new pose / lighting / expression). Returns
        True if added.
        """
        if self.centroid is None:
            self.add(embedding)
            return True
        sim = float(np.dot(l2_normalize(embedding.astype(np.float32)), self.centroid))
        if sim < novelty_threshold:
            self.add(embedding)
            return True
        # Even when not added, refresh last_seen so identity isn't pruned.
        self.last_seen = time.time()
        return False

    def mark_seen(self) -> None:
        self.last_seen = time.time()
        self.n_recognitions += 1


class FaceDatabase:
    """Persistent dynamic face database."""

    def __init__(self, path: Path | str = config.FACE_DB_PATH):
        self.path = Path(path)
        self.records: dict[str, PersonRecord] = {}
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                self.records = pickle.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    # ---- mutators ----
    def enroll(self, name: str, embedding: np.ndarray) -> PersonRecord:
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be empty")
        rec = self.records.get(name)
        if rec is None:
            rec = PersonRecord(name=name)
            self.records[name] = rec
        rec.add(embedding)
        self.save()
        return rec

    def remove(self, name: str) -> bool:
        if name in self.records:
            del self.records[name]
            self.save()
            return True
        return False

    # ---- queries ----
    def __len__(self) -> int:
        return len(self.records)

    def names(self) -> list[str]:
        return sorted(self.records.keys())

    def identify(
        self,
        embedding: np.ndarray,
        threshold: float = config.MATCH_THRESHOLD,
        top_k: int = 5,
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Return (best_name_or_None, best_score, top_k_scores)."""
        if not self.records:
            return None, 0.0, []
        q = l2_normalize(embedding.astype(np.float32))
        scores: list[tuple[str, float]] = []
        for name, rec in self.records.items():
            if rec.centroid is None:
                continue
            scores.append((name, float(np.dot(q, rec.centroid))))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        if not top:
            return None, 0.0, []
        best_name, best_score = top[0]
        return (best_name if best_score >= threshold else None), best_score, top


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
