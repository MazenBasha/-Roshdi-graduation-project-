"""Runtime configuration loaded from env vars or .env.

All knobs live here so a deploy can change behaviour without code changes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ROSHDI_OD_",
        extra="ignore",
    )

    # --- Model ---
    # Default is the Ultralytics-hosted pretrained YOLO11n checkpoint, which is
    # auto-downloaded on first use. To deploy a custom fine-tune, point this at
    # a local .pt / .onnx / .tflite file.
    weights_path: Path = Field(
        default=Path("yolo11n.pt"),
        description=(
            "YOLO weights to load. A bare 'yolo11n.pt' (or other Ultralytics "
            "name) is auto-downloaded; an existing absolute/relative path is "
            "used as-is."
        ),
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 100

    # --- API ---
    api_key: str | None = Field(
        default=None,
        description="If set, clients must send X-API-Key header.",
    )
    max_image_bytes: int = 8 * 1024 * 1024   # 8 MiB
    max_image_pixels: int = 25 * 1_000_000   # 25 MP — guards against decompression bombs
    inference_timeout_s: float = 5.0
    request_id_header: str = "X-Request-ID"

    # --- Safety / privacy ---
    blur_persons: bool = Field(
        default=False,
        description=(
            "If True, faces inside 'person' detections are pixelated before the "
            "image is returned. Off by default — Roshdi is single-user offline."
        ),
    )
    safety_min_box_area_frac: float = 0.0005  # drop boxes <0.05% of image area

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    metrics_enabled: bool = True
    sample_predictions: float = 0.01  # 1% of predictions are logged in full

    # --- Fallback ---
    fallback_classes_only: list[str] = Field(
        default_factory=lambda: ["person", "car", "chair", "cup", "bottle"],
        description=(
            "If the primary model fails, return an empty result but mark these "
            "classes as 'previously seen recently' so the caller can speak a "
            "graceful 'still searching' message instead of crashing."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
