"""Schema-level tests — wire stability."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.schemas import BBox, Detection


def test_bbox_ordering_enforced():
    with pytest.raises(ValidationError):
        BBox(x1=10, y1=0, x2=5, y2=10)


def test_detection_confidence_bounded():
    with pytest.raises(ValidationError):
        Detection(
            class_id=0, class_name="person", confidence=1.5,
            box=BBox(x1=0, y1=0, x2=1, y2=1),
            horizontal_position="left", distance_hint="far",
        )


def test_detection_round_trips():
    d = Detection(
        class_id=0, class_name="person", confidence=0.9,
        box=BBox(x1=0, y1=0, x2=10, y2=10),
        horizontal_position="center", distance_hint="medium",
    )
    raw = d.model_dump_json()
    assert Detection.model_validate_json(raw) == d
