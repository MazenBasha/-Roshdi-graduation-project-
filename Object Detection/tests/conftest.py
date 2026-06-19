"""Shared fixtures.

Tests are split by what they need:

  * pure-Python tests (validation, schemas, safety) run anywhere.
  * model-loading tests (`-m model`) skip unless YOLO weights exist.
  * server tests (`-m server`) skip unless both weights *and* fastapi are
    installed.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

from server.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Settings:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not a real model")
    monkeypatch.setenv("ROSHDI_OD_WEIGHTS_PATH", str(weights))
    return Settings(weights_path=weights)


def _png_bytes(size=(64, 64), color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(64, 64), color=(128, 128, 128), quality=80) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return _png_bytes


@pytest.fixture
def jpeg_bytes():
    return _jpeg_bytes


def _model_reachable() -> bool:
    """Pretrained yolo11n.pt auto-downloads on first use, so the only reason
    we'd skip is no network. Honour OFFLINE=1 as an explicit opt-out (used in
    pure-unit CI jobs that don't want the ~5 MB download)."""
    if os.environ.get("OFFLINE") == "1":
        return False
    # Custom weights path already present?
    p = os.environ.get("ROSHDI_OD_WEIGHTS_PATH", "yolo11n.pt")
    if Path(p).is_file() and Path(p).stat().st_size > 1024:
        return True
    # Otherwise we expect Ultralytics to download — assume reachable unless
    # OFFLINE was set.
    return True


needs_weights = pytest.mark.skipif(
    not _model_reachable(),
    reason="model unavailable (OFFLINE=1 or no network)",
)
