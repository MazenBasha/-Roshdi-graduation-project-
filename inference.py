"""
End-to-end face recognition inference.

Pipeline for one image:
    1. Detect faces with MTCNN -> bounding boxes + 5 landmarks + score
    2. Align each face to the 112x112 ArcFace canonical template
    3. Forward through the embedding model -> 512-dim L2-normalized vector
    4. Match against the persistent face DB (cosine similarity vs centroids)
    5. If best score >= threshold -> return the name
       Else -> "unknown"; calling code can prompt for a name and enroll

This module exposes:
    - FaceRecognizer: callable end-to-end recognizer
    - recognize_image_cli: CLI for single-image identification + enrollment
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

import config
from model import build_embedding_model, load_checkpoint
from utils import FaceDatabase, align_face, batch_preprocess, get_device, l2_normalize


# ---------------------------------------------------------------------------
# Detector wrapper (MTCNN via facenet-pytorch)
# ---------------------------------------------------------------------------

class FaceDetector:
    """MTCNN wrapper that returns aligned 112x112 BGR crops + bboxes + scores."""

    def __init__(self, device: torch.device, min_face_size: int = config.MIN_FACE_SIZE):
        try:
            from facenet_pytorch import MTCNN  # type: ignore
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch is required for MTCNN. Install with: pip install facenet-pytorch"
            ) from e
        # MTCNN uses adaptive_avg_pool2d on arbitrary input sizes, which the
        # MPS backend doesn't support yet (pytorch/pytorch#96056). Pin the
        # detector to CPU; the embedding model still uses the requested device.
        det_device = torch.device("cpu") if device.type == "mps" else device
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.85],   # P, R, O-net thresholds
            factor=0.709,
            device=det_device,
            post_process=False,             # we handle normalization ourselves
        )

    def detect(self, image_bgr: np.ndarray, conf_thresh: float = config.DETECTION_CONFIDENCE):
        """Returns list of (aligned_face_bgr, bbox_xyxy, landmarks_5x2, score)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes, probs, landmarks = self.mtcnn.detect(pil, landmarks=True)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        if boxes is None:
            return results
        for box, prob, lmk in zip(boxes, probs, landmarks):
            if prob is None or prob < conf_thresh:
                continue
            aligned = align_face(image_bgr, lmk, size=config.INPUT_SIZE)
            results.append((aligned, np.asarray(box, dtype=np.float32),
                            np.asarray(lmk, dtype=np.float32), float(prob)))
        return results


# ---------------------------------------------------------------------------
# End-to-end recognizer
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    landmarks: np.ndarray     # (5, 2)
    detection_score: float
    embedding: np.ndarray     # (512,) L2-normalized
    name: str | None          # None if unknown
    similarity: float         # best similarity vs DB (0 if DB empty)
    aligned_face: np.ndarray  # 112x112 BGR aligned crop (kept for enrollment)

    def to_dict(self) -> dict:
        """Serializable summary for the multi-face API."""
        return {
            "name": self.name or "Unknown",
            "confidence": round(float(self.similarity), 3),
            "bbox": [int(v) for v in self.bbox],
            "detection_score": round(float(self.detection_score), 3),
        }


class FaceRecognizer:
    def __init__(
        self,
        backbone: str = config.BACKBONE,
        checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        db_path: str | Path = config.FACE_DB_PATH,
        use_clahe: bool = True,
        tta: bool = False,
    ):
        self.device = device or get_device()
        self.detector = FaceDetector(self.device)
        # Production preprocessing for real-world wearable footage.
        self.use_clahe = use_clahe
        # Test-time augmentation: average embeddings of original + horizontal
        # flip. Doubles inference cost; helps a lot under partial occlusion.
        self.tta = tta

        # Decide which embedding model to load.
        if checkpoint and Path(checkpoint).is_file():
            self.model = build_embedding_model(backbone)
            load_checkpoint(self.model, str(checkpoint), map_location=self.device)
            print(f"[inference] loaded {backbone} from {checkpoint}")
        elif backbone == "facenet_vggface2":
            self.model = build_embedding_model("facenet_vggface2")
            print("[inference] loaded pretrained FaceNet (VGGFace2)")
        else:
            # No trained checkpoint -> fall back to pretrained FaceNet so the
            # system works out-of-the-box on real faces.
            print(f"[inference] no checkpoint at '{checkpoint}', falling back to "
                  f"pretrained FaceNet (VGGFace2). Train with `python train.py` to "
                  f"use a custom iResNet checkpoint.")
            self.model = build_embedding_model("facenet_vggface2")

        self.model.to(self.device).eval()
        self.db = FaceDatabase(db_path)

    # ---- core ----

    @torch.no_grad()
    def embed(self, aligned_faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Batched embedding with optional CLAHE + test-time augmentation."""
        if not aligned_faces_bgr:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        batch = batch_preprocess(aligned_faces_bgr, use_clahe=self.use_clahe).to(self.device)
        emb = self.model(batch)
        if self.tta:
            # Horizontal flip TTA: average the L2-normalized originals and flips.
            emb_flip = self.model(torch.flip(batch, dims=(-1,)))
            emb = (
                torch.nn.functional.normalize(emb, dim=1)
                + torch.nn.functional.normalize(emb_flip, dim=1)
            ) / 2.0
        emb = emb.detach().cpu().numpy()
        return l2_normalize(emb.astype(np.float32))

    def recognize_image(
        self,
        image_bgr: np.ndarray,
        threshold: float = config.MATCH_THRESHOLD,
    ) -> list[FaceResult]:
        """Detect every face in `image_bgr`, embed them, and match each
        against the persistent face DB. Returns one FaceResult per detected
        face. Empty list if no faces are detected.
        """
        detections = self.detector.detect(image_bgr)
        if not detections:
            return []
        aligned = [d[0] for d in detections]
        embeddings = self.embed(aligned)

        results: list[FaceResult] = []
        for (face, box, lmk, det_score), emb in zip(detections, embeddings):
            name, sim, _topk = self.db.identify(emb, threshold=threshold)
            if name is not None:
                self.db.records[name].mark_seen()
            results.append(FaceResult(
                bbox=box, landmarks=lmk, detection_score=det_score,
                embedding=emb, name=name, similarity=sim, aligned_face=face,
            ))
        return results

    def recognize_image_dicts(
        self,
        image_bgr: np.ndarray,
        threshold: float = config.MATCH_THRESHOLD,
    ) -> list[dict]:
        """Same as `recognize_image` but returns a list of dicts:
            [{"name": "Ali", "confidence": 0.82, ...}, ...]
        Matches the JSON shape requested by the multi-person API contract.
        """
        return [r.to_dict() for r in self.recognize_image(image_bgr, threshold)]

    # ---- dynamic enrollment ----

    def enroll(self, name: str, embedding: np.ndarray, novelty_threshold: float = 0.85) -> bool:
        """Add `embedding` to person `name`'s rolling window if it's novel.

        Returns True if the embedding was actually stored, False if it was
        considered too similar to the existing centroid (cos > novelty_threshold).
        Use False to detect when we're seeing redundant frames.
        """
        record = self.db.records.get(name)
        if record is None:
            self.db.enroll(name, embedding)
            return True
        added = record.maybe_add(embedding, novelty_threshold=novelty_threshold)
        self.db.save()
        return added

    def enroll_from_image(self, image_bgr: np.ndarray, name: str) -> int:
        """Detect faces in an image and enroll all of them under `name`.
        Returns how many faces were enrolled (counting only novel embeddings)."""
        detections = self.detector.detect(image_bgr)
        if not detections:
            return 0
        embeddings = self.embed([d[0] for d in detections])
        n_added = 0
        for emb in embeddings:
            if self.enroll(name, emb):
                n_added += 1
        return n_added


# ---------------------------------------------------------------------------
# Drawing helper (used by realtime.py and CLI)
# ---------------------------------------------------------------------------

def annotate(image_bgr: np.ndarray, results: list[FaceResult]) -> np.ndarray:
    out = image_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        if r.name is None:
            color, label = (0, 0, 255), f"unknown ({r.similarity:.2f})"
        else:
            color, label = (0, 200, 0), f"{r.name} ({r.similarity:.2f})"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CLI: recognize one image; on unknown faces, prompt for a name and enroll
# ---------------------------------------------------------------------------

def recognize_image_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT),
                        help="Path to trained backbone checkpoint")
    parser.add_argument("--backbone", type=str, default=config.BACKBONE)
    parser.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    parser.add_argument("--enroll", action="store_true",
                        help="Prompt for a name and enroll any unknown faces")
    parser.add_argument("--save", type=str, default="",
                        help="Optional path to save annotated image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    rec = FaceRecognizer(backbone=args.backbone, checkpoint=args.checkpoint)
    results = rec.recognize_image(image, threshold=args.threshold)

    if not results:
        print("[inference] no faces detected.")
        return

    for i, r in enumerate(results):
        if r.name:
            print(f"[face {i}] IDENTIFIED as '{r.name}'  (similarity={r.similarity:.3f})")
        else:
            print(f"[face {i}] UNKNOWN  (best similarity={r.similarity:.3f})")
            if args.enroll:
                # Simulated user prompt for the dynamic enrollment flow.
                name = input(f"  Enter name for face {i} (or blank to skip): ").strip()
                if name:
                    rec.enroll(name, r.embedding)
                    print(f"  -> enrolled '{name}'. DB now has {len(rec.db)} people.")

    if args.save:
        out = annotate(image, rec.recognize_image(image, threshold=args.threshold))
        cv2.imwrite(args.save, out)
        print(f"[inference] annotated image saved to {args.save}")


if __name__ == "__main__":
    recognize_image_cli()
