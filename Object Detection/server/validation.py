"""Input validation. Runs *before* the image touches the model."""
from __future__ import annotations

import io
from typing import Tuple

from PIL import Image, UnidentifiedImageError

from .config import Settings


class InvalidImage(ValueError):
    """Raised for any client-supplied image we refuse to process."""


ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def validate_and_open(data: bytes, content_type: str | None, cfg: Settings) -> Tuple[Image.Image, list[str]]:
    """Return a decoded RGB Pillow image, or raise InvalidImage.

    Defends against: empty body, oversize body, wrong mime, decompression bombs,
    transparent / palette images, corrupt JPEG, EXIF rotation, animated GIFs.
    """
    warnings: list[str] = []
    if not data:
        raise InvalidImage("empty body")
    if len(data) > cfg.max_image_bytes:
        raise InvalidImage(f"image larger than {cfg.max_image_bytes} bytes")
    if content_type and content_type.split(";")[0].strip() not in ALLOWED_MIME:
        raise InvalidImage(f"unsupported content-type: {content_type}")

    # Pillow's bomb guard — snapshot and restore to avoid global-state leakage.
    prev_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = cfg.max_image_pixels
    try:
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()  # parse header without decoding pixels
            img = Image.open(io.BytesIO(data))  # reopen — verify() invalidates
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
            raise InvalidImage(f"cannot decode image: {e}") from e
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max

    if img.width < 32 or img.height < 32:
        raise InvalidImage("image too small (min 32x32)")
    if img.width * img.height > cfg.max_image_pixels:
        raise InvalidImage("image exceeds pixel limit")

    if getattr(img, "is_animated", False):
        warnings.append("animated input: only first frame used")

    # Honour EXIF orientation, then drop alpha / palette
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, warnings
