"""Inference tests — load a real YOLO11n model and run predictions.

Marked `needs_weights`: skipped when no weights are present (e.g. fresh clone,
CI before training). Run with: pytest -m model
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image

from server.config import Settings
from server.inference import Detector, InferenceTimeout
from tests.conftest import needs_weights


@needs_weights
def test_loads_and_predicts():
    cfg = Settings()
    d = Detector(cfg)
    d.load()
    assert d.info.version
    assert d.info.class_names

    img = Image.new("RGB", (640, 640), (128, 128, 128))
    detections, warnings = asyncio.run(d.predict(img))
    # A grey square shouldn't trigger any detection but must not crash.
    assert isinstance(detections, list)
    assert isinstance(warnings, list)


@needs_weights
def test_predict_respects_timeout(monkeypatch):
    cfg = Settings(inference_timeout_s=0.0001)
    d = Detector(cfg)
    d.load()
    img = Image.new("RGB", (640, 640), (0, 0, 0))
    with pytest.raises(InferenceTimeout):
        asyncio.run(d.predict(img))


@needs_weights
def test_horizontal_position_classification():
    """Synthesize a 'left' object and verify spatial annotation is correct."""
    cfg = Settings()
    d = Detector(cfg)
    d.load()
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    # paint a black box on the far left third
    img[200:400, 20:180] = 0
    detections, _ = asyncio.run(d.predict(Image.fromarray(img)))
    # We don't assert detection happens (model may or may not see the patch),
    # but if it does, the helper logic must label it 'left'.
    for det in detections:
        cx = (det.box.x1 + det.box.x2) / 2
        if cx < 640 / 3:
            assert det.horizontal_position == "left"
