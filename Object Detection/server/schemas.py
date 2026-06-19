"""Wire schemas. Stable across versions — bump /v1 if you change them."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates of the *input* image."""
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x_ordered(cls, v: float, info) -> float:
        if v < info.data.get("x1", 0):
            raise ValueError("x2 must be >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def y_ordered(cls, v: float, info) -> float:
        if v < info.data.get("y1", 0):
            raise ValueError("y2 must be >= y1")
        return v


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BBox
    # Position hint useful for spatial-audio guidance to a blind user.
    # "left" | "center" | "right" derived from box centre x.
    horizontal_position: str
    # Rough size cue based on box area — "near" if area>40% of image, etc.
    distance_hint: str


class DetectResponse(BaseModel):
    request_id: str
    model_version: str
    inference_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]
    degraded: bool = False  # True if served from fallback path
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_s: float


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
