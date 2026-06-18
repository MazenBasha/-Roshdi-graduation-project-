"""
Unit tests for the safety / input-validation layer of reliability.py.

These do NOT require model weights or MTCNN — they exercise validate_image,
is_blank, softmax, and RejectionPolicy logic directly.

Run with:
    pytest tests/test_safety.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from reliability import (
    InputValidationError, RejectionPolicy, is_blank, softmax, validate_image,
)


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

def make_img(h=128, w=128, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_validate_image_happy_path():
    img = make_img()
    out = validate_image(img)
    assert out is img


def test_validate_image_rejects_none():
    with pytest.raises(InputValidationError):
        validate_image(None)


def test_validate_image_rejects_non_ndarray():
    with pytest.raises(InputValidationError):
        validate_image([[1, 2, 3]])


def test_validate_image_rejects_wrong_dtype():
    img = make_img().astype(np.float32)
    with pytest.raises(InputValidationError):
        validate_image(img)


def test_validate_image_rejects_grayscale():
    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(InputValidationError):
        validate_image(img)


def test_validate_image_rejects_4_channels():
    img = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(InputValidationError):
        validate_image(img)


def test_validate_image_rejects_too_small():
    with pytest.raises(InputValidationError):
        validate_image(make_img(h=16, w=16))


def test_validate_image_rejects_too_large():
    with pytest.raises(InputValidationError):
        validate_image(make_img(h=8192, w=128))


# ---------------------------------------------------------------------------
# is_blank
# ---------------------------------------------------------------------------

def test_is_blank_detects_constant_image():
    assert is_blank(make_img(value=100))
    assert is_blank(make_img(value=0))


def test_is_blank_passes_textured_image():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    assert not is_blank(img)


def test_is_blank_threshold_controllable():
    rng = np.random.default_rng(0)
    img = rng.integers(120, 136, size=(128, 128, 3), dtype=np.uint8)  # low variance
    # With default threshold (2.0) this textured-but-flat image may pass.
    # With threshold 10, it should be flagged blank.
    assert not is_blank(img, std_threshold=2.0)
    assert is_blank(img, std_threshold=10.0)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    s = softmax(np.array([0.1, 0.2, 0.3]))
    assert abs(s.sum() - 1.0) < 1e-6


def test_softmax_largest_value_wins():
    s = softmax(np.array([0.1, 0.9, 0.2]))
    assert int(s.argmax()) == 1


def test_softmax_temperature_sharpens():
    raw = np.array([0.5, 0.6])
    cold = softmax(raw, temperature=1.0)
    hot = softmax(raw, temperature=20.0)
    # Higher temperature => sharper, larger spread between top two.
    assert (hot.max() - hot.min()) > (cold.max() - cold.min())


# ---------------------------------------------------------------------------
# RejectionPolicy defaults
# ---------------------------------------------------------------------------

def test_rejection_policy_has_sane_defaults():
    p = RejectionPolicy()
    assert 0.0 < p.match_threshold < 1.0
    assert p.abstain_margin > 0
    assert 0.0 < p.abstain_softmax_min < 1.0
    assert p.min_face_score > 0.5


def test_rejection_policy_is_dataclass_overridable():
    p = RejectionPolicy(match_threshold=0.5, abstain_margin=0.1)
    assert p.match_threshold == 0.5
    assert p.abstain_margin == 0.1
