"""Model wrapper. Owns the Ultralytics model, predicts, post-processes."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .schemas import BBox, Detection

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    version: str          # weights file SHA-256 (first 12 chars)
    loaded_at: float
    class_names: dict[int, str]


class Detector:
    """Wraps a YOLO model. Thread-safe enough for FastAPI (Ultralytics serialises
    predict internally); for true parallelism run multiple workers."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None
        self._info: ModelInfo | None = None

    def load(self) -> None:
        """Load YOLO weights.

        Resolution order:
          1. If `weights_path` points to an existing file, load it.
          2. Else, pass the string to Ultralytics (e.g. "yolo11n.pt") and let it
             auto-download the pretrained checkpoint to the working directory.

        The resulting *resolved* file's SHA-256 prefix is used as the model
        version reported via /healthz and every detection response.
        """
        from ultralytics import YOLO  # heavy import deferred
        requested = Path(self.cfg.weights_path)
        if requested.exists():
            self._model = YOLO(str(requested))
            resolved = requested
            source = "local"
        else:
            # Ultralytics treats this as a hub identifier and downloads it.
            self._model = YOLO(str(requested))
            resolved = Path(getattr(self._model, "ckpt_path", None) or requested.name)
            if not resolved.exists():
                # Fall back: file may be in CWD after download.
                cwd_candidate = Path.cwd() / requested.name
                resolved = cwd_candidate if cwd_candidate.exists() else resolved
            source = "auto-downloaded"
        digest = _sha256(resolved)[:12] if resolved.exists() else "unknown"
        names = self._model.names if hasattr(self._model, "names") else {}
        self._info = ModelInfo(version=digest, loaded_at=time.time(), class_names=dict(names))
        log.info(
            "model loaded (%s): %s — %d classes, version %s",
            source, resolved, len(names), digest,
        )

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("model not loaded")
        return self._info

    async def predict(self, image: Image.Image) -> tuple[list[Detection], list[str]]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        loop = asyncio.get_running_loop()
        try:
            dets = await asyncio.wait_for(
                loop.run_in_executor(None, self._predict_sync, image),
                timeout=self.cfg.inference_timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise InferenceTimeout(self.cfg.inference_timeout_s) from e
        warnings: list[str] = []
        return dets, warnings

    def _predict_sync(self, image: Image.Image) -> list[Detection]:
        cfg = self.cfg
        results = self._model.predict(
            source=np.array(image),
            imgsz=cfg.imgsz,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            max_det=cfg.max_det,
            device=None if cfg.device == "auto" else cfg.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.shape[0] == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        out: list[Detection] = []
        W, H = image.size
        area = float(W * H)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses):
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / area < cfg.safety_min_box_area_frac:
                continue
            cx = (x1 + x2) / 2
            horizontal = "left" if cx < W / 3 else "right" if cx > 2 * W / 3 else "center"
            ratio = box_area / area
            if ratio > 0.40:
                distance = "near"
            elif ratio > 0.10:
                distance = "medium"
            else:
                distance = "far"
            out.append(Detection(
                class_id=int(k),
                class_name=self._info.class_names.get(int(k), str(int(k))),
                confidence=float(c),
                box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                horizontal_position=horizontal,
                distance_hint=distance,
            ))
        return out


class InferenceTimeout(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"inference exceeded {seconds}s budget")
        self.seconds = seconds


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
