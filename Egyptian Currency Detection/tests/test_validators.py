"""
Unit tests for src/validators.py.

All fixtures are built in-memory via PIL or tempfile - no external test
images required, so this runs identically on CI and the dev box.
"""

import os
import tempfile

import pytest
from PIL import Image

from validators import InputValidator


def _tmp_image(width=416, height=416, fmt="JPEG", color=(128, 128, 128)):
    """Create a real image on disk and return its path.  Caller deletes it."""
    suffix = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Image.new("RGB", (width, height), color).save(path, fmt)
    return path


class TestValidation:
    def test_validate_missing_file(self):
        ok, msg = InputValidator.validate_image_path(
            "/no/such/file/here_definitely_missing.jpg"
        )
        assert ok is False
        assert "not found" in msg.lower()

    def test_validate_empty_path(self):
        ok, msg = InputValidator.validate_image_path("")
        assert ok is False
        assert msg  # non-empty error message

    def test_validate_unsupported_format(self, tmp_path):
        bad = tmp_path / "note.txt"
        bad.write_text("not an image")
        ok, msg = InputValidator.validate_image_path(str(bad))
        assert ok is False
        assert "unsupported" in msg.lower()

    def test_validate_directory_passed_as_image(self, tmp_path):
        ok, msg = InputValidator.validate_image_path(str(tmp_path))
        assert ok is False

    def test_validate_corrupt_jpeg(self, tmp_path):
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"NOT_A_JPEG_AT_ALL")
        ok, msg, _ = InputValidator.validate_image_content(str(bad))
        assert ok is False
        assert "corrupt" in msg.lower() or "unreadable" in msg.lower()

    def test_validate_oversized_image(self, tmp_path):
        # 8000x7000 = 56M pixels > 50M cap
        path = tmp_path / "huge.jpg"
        Image.new("RGB", (8000, 7000), (200, 200, 200)).save(path, "JPEG")
        ok, msg, _ = InputValidator.validate_image_content(str(path))
        assert ok is False
        assert "too large" in msg.lower()

    def test_validate_undersized_image(self, tmp_path):
        # 5x5 = 25 pixels < 100 cap
        path = tmp_path / "tiny.png"
        Image.new("RGB", (5, 5), (0, 0, 0)).save(path, "PNG")
        ok, msg, _ = InputValidator.validate_image_content(str(path))
        assert ok is False
        assert "too small" in msg.lower()

    def test_validate_valid_image(self):
        path = _tmp_image(416, 416)
        try:
            ok, msg, meta = InputValidator.validate_image_content(path)
            assert ok is True, msg
            assert meta["width"] == 416
            assert meta["height"] == 416
            assert meta["pixels"] == 416 * 416
            assert meta["mode"] == "RGB"
            assert pytest.approx(meta["aspect_ratio"], rel=1e-3) == 1.0
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("ext,fmt", [
        (".jpg", "JPEG"),
        (".png", "PNG"),
        (".bmp", "BMP"),
    ])
    def test_validate_supported_formats(self, tmp_path, ext, fmt):
        path = tmp_path / f"sample{ext}"
        Image.new("RGB", (200, 200)).save(path, fmt)
        ok, msg = InputValidator.validate_image_path(str(path))
        assert ok is True, msg


class TestFolder:
    def test_validate_missing_folder(self):
        ok, msg, imgs = InputValidator.validate_folder(
            "/no/such/folder/definitely_missing"
        )
        assert ok is False
        assert imgs == []

    def test_validate_empty_folder(self, tmp_path):
        ok, msg, imgs = InputValidator.validate_folder(str(tmp_path))
        assert ok is False
        assert "no supported images" in msg.lower()
        assert imgs == []

    def test_validate_folder_ignores_non_images(self, tmp_path):
        (tmp_path / "readme.txt").write_text("notes")
        ok, msg, imgs = InputValidator.validate_folder(str(tmp_path))
        assert ok is False
        assert imgs == []

    def test_validate_folder_with_images(self, tmp_path):
        for i in range(3):
            Image.new("RGB", (200, 200)).save(tmp_path / f"img_{i}.jpg", "JPEG")
        # decoy non-image file should be ignored
        (tmp_path / "README.txt").write_text("x")
        ok, msg, imgs = InputValidator.validate_folder(str(tmp_path))
        assert ok is True, msg
        assert len(imgs) == 3
        assert all(p.endswith(".jpg") for p in imgs)
