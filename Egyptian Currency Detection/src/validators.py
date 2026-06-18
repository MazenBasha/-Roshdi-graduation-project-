"""
Input validation for the Egyptian Currency Detection pipeline.

The validator is defensive: it never raises, it returns (is_valid, error_msg, ...)
tuples that callers can convert into structured error JSON.  This keeps the
single-CLI entry point crash-free even on corrupted, oversized, or
unsupported inputs.
"""

import os
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


class InputValidator:
    MAX_IMAGE_SIZE = 50_000_000   # pixels (width * height)
    MIN_IMAGE_SIZE = 100          # pixels (width * height)
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    @staticmethod
    def validate_image_path(path: str) -> Tuple[bool, str]:
        """Check that `path` points to a readable, supported image file."""
        if not path:
            return False, "Empty image path"
        if not os.path.exists(path):
            return False, f"File not found: {path}"
        if not os.path.isfile(path):
            return False, f"Path is not a file: {path}"

        ext = os.path.splitext(path)[1].lower()
        if ext not in InputValidator.SUPPORTED_FORMATS:
            return False, (
                f"Unsupported format '{ext}'. "
                f"Allowed: {sorted(InputValidator.SUPPORTED_FORMATS)}"
            )

        if not os.access(path, os.R_OK):
            return False, f"File is not readable: {path}"
        return True, ""

    @staticmethod
    def validate_image_content(image_path: str) -> Tuple[bool, str, Dict]:
        """Load the image with PIL to catch corruption / oversized / undersized."""
        ok, msg = InputValidator.validate_image_path(image_path)
        if not ok:
            return False, msg, {}

        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
        except (UnidentifiedImageError, OSError, ValueError) as e:
            return False, f"Corrupted or unreadable image: {e}", {}

        pixels = int(width) * int(height)
        if pixels < InputValidator.MIN_IMAGE_SIZE:
            return False, (
                f"Image too small: {width}x{height} ({pixels} pixels, "
                f"min {InputValidator.MIN_IMAGE_SIZE})"
            ), {}
        if pixels > InputValidator.MAX_IMAGE_SIZE:
            return False, (
                f"Image too large: {width}x{height} "
                f"({pixels // 1_000_000}M pixels, "
                f"max {InputValidator.MAX_IMAGE_SIZE // 1_000_000}M)"
            ), {}

        metadata = {
            "width": int(width),
            "height": int(height),
            "pixels": pixels,
            "aspect_ratio": round(float(width) / float(height), 4) if height else 0.0,
            "mode": mode,
        }
        return True, "", metadata

    @staticmethod
    def validate_folder(folder_path: str) -> Tuple[bool, str, List[str]]:
        """Find supported images in a folder."""
        if not folder_path:
            return False, "Empty folder path", []
        if not os.path.exists(folder_path):
            return False, f"Folder not found: {folder_path}", []
        if not os.path.isdir(folder_path):
            return False, f"Path is not a directory: {folder_path}", []

        images: List[str] = []
        for name in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(name)[1].lower()
            if ext in InputValidator.SUPPORTED_FORMATS:
                full = os.path.join(folder_path, name)
                if os.path.isfile(full):
                    images.append(full)

        if not images:
            return False, (
                f"No supported images in: {folder_path} "
                f"(allowed: {sorted(InputValidator.SUPPORTED_FORMATS)})"
            ), []
        return True, "", images


def build_error_json(error_msg: str, image_path: str = "") -> Dict:
    """Build the canonical error response used when validation fails."""
    return {
        "success": False,
        "error": error_msg,
        "image": image_path,
        "detections": [],
        "counts": {},
        "total": 0,
    }