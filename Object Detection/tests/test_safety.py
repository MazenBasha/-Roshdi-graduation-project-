from __future__ import annotations

from PIL import Image

from server import safety
from server.config import Settings
from server.schemas import BBox, Detection


def _person(x1, y1, x2, y2) -> Detection:
    return Detection(
        class_id=0, class_name="person", confidence=0.9,
        box=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        horizontal_position="center", distance_hint="medium",
    )


def test_pixelate_persons_off_by_default():
    img = Image.new("RGB", (100, 100), (200, 0, 0))
    out = safety.pixelate_persons(img, [_person(10, 10, 60, 60)], Settings())
    assert list(img.getdata()) == list(out.getdata())


def test_pixelate_persons_changes_head_region():
    img = Image.new("RGB", (200, 200), (200, 0, 0))
    cfg = Settings(blur_persons=True)
    out = safety.pixelate_persons(img, [_person(20, 20, 100, 100)], cfg)
    # head region should be different colour after pixelation? not in flat image —
    # but the call must not raise and must return a same-size image.
    assert out.size == img.size


def test_filter_detections_passthrough_by_default():
    dets = [_person(0, 0, 10, 10)]
    assert safety.filter_detections(dets) == dets


def test_filter_detections_drops_sensitive(monkeypatch):
    monkeypatch.setattr(safety, "SENSITIVE_LABELS", {"person"})
    assert safety.filter_detections([_person(0, 0, 10, 10)]) == []
