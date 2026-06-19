"""Input-validation tests. Run without a model — purely Pillow + Pydantic."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from server.config import Settings
from server.validation import InvalidImage, validate_and_open


def test_accepts_normal_png(png_bytes):
    img, w = validate_and_open(png_bytes(), "image/png", Settings())
    assert img.size == (64, 64)
    assert img.mode == "RGB"
    assert w == []


def test_accepts_normal_jpeg(jpeg_bytes):
    img, w = validate_and_open(jpeg_bytes(), "image/jpeg", Settings())
    assert img.size == (64, 64)


def test_rejects_empty_body():
    with pytest.raises(InvalidImage):
        validate_and_open(b"", "image/png", Settings())


def test_rejects_oversize_body(png_bytes):
    cfg = Settings(max_image_bytes=10)
    with pytest.raises(InvalidImage, match="larger than"):
        validate_and_open(png_bytes(), "image/png", cfg)


def test_rejects_wrong_mime(png_bytes):
    with pytest.raises(InvalidImage, match="unsupported"):
        validate_and_open(png_bytes(), "application/pdf", Settings())


def test_rejects_garbage_bytes():
    with pytest.raises(InvalidImage):
        validate_and_open(b"not an image at all" * 100, "image/png", Settings())


def test_rejects_too_small_image(png_bytes):
    with pytest.raises(InvalidImage, match="too small"):
        validate_and_open(png_bytes(size=(8, 8)), "image/png", Settings())


def test_strips_alpha_channel():
    img = Image.new("RGBA", (64, 64), (1, 2, 3, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, _ = validate_and_open(buf.getvalue(), "image/png", Settings())
    assert out.mode == "RGB"


def test_decompression_bomb_guard():
    # build a 1x1 png and inflate the pixel limit guard below its area
    cfg = Settings(max_image_pixels=10)
    big = Image.new("RGB", (1000, 1000), (0, 0, 0))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    with pytest.raises(InvalidImage):
        validate_and_open(buf.getvalue(), "image/png", cfg)
