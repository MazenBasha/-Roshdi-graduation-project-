"""Edge-case + adversarial input tests.

Goal: the service never 500s on hostile or weird inputs — every failure mode
should be a 4xx with a sensible reason, not a crash.
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from server.config import Settings
from server.validation import InvalidImage, validate_and_open


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_all_black_image():
    img, _ = validate_and_open(_png(np.zeros((128, 128, 3), np.uint8)), "image/png", Settings())
    assert img.size == (128, 128)


def test_all_white_image():
    img, _ = validate_and_open(_png(np.full((128, 128, 3), 255, np.uint8)), "image/png", Settings())
    assert img.size == (128, 128)


def test_uniform_noise_image():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    img, _ = validate_and_open(_png(arr), "image/png", Settings())
    assert img.size == (256, 256)


def test_extreme_aspect_ratio():
    img, _ = validate_and_open(_png(np.zeros((32, 4000, 3), np.uint8)), "image/png", Settings())
    assert img.size == (4000, 32)


def test_truncated_jpeg_header():
    # First 50 bytes of a JPEG, then garbage — header valid, body broken.
    real = io.BytesIO()
    Image.new("RGB", (64, 64)).save(real, format="JPEG")
    truncated = real.getvalue()[:50] + b"\x00" * 200
    with pytest.raises(InvalidImage):
        validate_and_open(truncated, "image/jpeg", Settings())


def test_fake_png_magic_only():
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256
    with pytest.raises(InvalidImage):
        validate_and_open(payload, "image/png", Settings())


def test_zip_bomb_disguised_as_image():
    # A nested ZIP magic dressed up — should fail the Pillow header check.
    payload = b"PK\x03\x04" + b"\x00" * 1024
    with pytest.raises(InvalidImage):
        validate_and_open(payload, "image/png", Settings())


def test_polyglot_struct_payload():
    # 16 bytes of pathological lengths — exercises the OSError branch.
    payload = struct.pack(">IIII", 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)
    with pytest.raises(InvalidImage):
        validate_and_open(payload, "image/png", Settings())
