"""
Unit tests for the explainability module (src/explain.py).

These cover the pure post-processing helpers, the salience metric, the
reason-string logic, and the graceful no-op paths — without running the model,
so they stay fast and CPU-only.
"""

import numpy as np
import pytest

import explain
from utils import build_output_dict


# ─── _normalize ──────────────────────────────────────────────────────────────

class TestNormalize:
    def test_range_is_unit(self):
        out = explain._normalize(np.array([[0.0, 5.0], [10.0, 2.0]], np.float32))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_single_spike_preserves_structure(self):
        # A structured (gradient) map with one huge outlier: naive max-norm would
        # divide by the spike and flatten the gradient to ~0 (the all-blue-heatmap
        # bug). Percentile clipping must keep the bright end of the gradient hot.
        cam = np.tile(np.linspace(0.0, 1.0, 100, dtype=np.float32), (100, 1))
        cam[0, 0] = 1000.0
        out = explain._normalize(cam)
        assert out.max() <= 1.0
        assert (out > 0.5).mean() > 0.2  # upper half of the gradient survives

    def test_constant_map_is_safe(self):
        out = explain._normalize(np.full((4, 4), 3.0, np.float32))
        assert np.isfinite(out).all()


# ─── _salience_ratio ─────────────────────────────────────────────────────────

class TestSalienceRatio:
    def test_hot_box_scores_above_one(self):
        cam = np.zeros((100, 100), np.float32)
        cam[10:40, 10:40] = 1.0  # all energy inside this box
        assert explain._salience_ratio(cam, [10, 10, 40, 40], (100, 100)) > 1.0

    def test_cold_box_scores_below_one(self):
        cam = np.zeros((100, 100), np.float32)
        cam[60:90, 60:90] = 1.0  # energy elsewhere
        assert explain._salience_ratio(cam, [0, 0, 20, 20], (100, 100)) < 1.0

    def test_uniform_map_scores_near_one(self):
        cam = np.ones((100, 100), np.float32)
        assert explain._salience_ratio(cam, [10, 10, 50, 50], (100, 100)) == pytest.approx(1.0, abs=1e-3)

    def test_degenerate_box_is_zero(self):
        cam = np.ones((50, 50), np.float32)
        assert explain._salience_ratio(cam, [10, 10, 10, 10], (50, 50)) == 0.0

    def test_box_clipped_to_frame(self):
        cam = np.ones((50, 50), np.float32)
        # Out-of-bounds box must not raise and stays finite.
        assert explain._salience_ratio(cam, [-5, -5, 999, 999], (50, 50)) >= 0.0


# ─── _explanation_text ───────────────────────────────────────────────────────

class TestExplanationText:
    def test_high_ratio_reads_as_focused(self):
        txt = explain._explanation_text("100_EGP", 0.95, "low", 1.4, None)
        assert "100_EGP" in txt and "95%" in txt and "strongly" in txt

    def test_low_ratio_adds_caution(self):
        txt = explain._explanation_text("50_EGP", 0.6, "high", 0.4, None)
        assert "caution" in txt.lower()

    def test_warning_is_appended(self):
        warn = "May be confused with 50_EGP (similar color)"
        txt = explain._explanation_text("100_EGP", 0.8, "medium", 1.0, warn)
        assert warn in txt


# ─── _class_index ────────────────────────────────────────────────────────────

class TestClassIndex:
    class _FakeModel:
        names = {0: "1_EGP", 5: "100_EGP"}

    def test_resolves_from_model_names(self):
        assert explain._class_index(self._FakeModel(), "100_EGP") == 5

    def test_unknown_class_returns_none(self):
        assert explain._class_index(self._FakeModel(), "nope_EGP") is None


# ─── explain_image graceful paths ────────────────────────────────────────────

class TestExplainImageNoop:
    def test_empty_detections_returns_frame_unchanged(self):
        frame = np.zeros((20, 20, 3), np.uint8)
        overlay, exps = explain.explain_image(model=None, frame_bgr=frame,
                                              detections=[])
        assert exps == []
        assert overlay is frame


# ─── build_output_dict passthrough ───────────────────────────────────────────

class TestOutputPassthrough:
    def test_explain_fields_flow_into_json(self):
        det = {
            "class": "100_EGP", "confidence": 0.9, "bbox": [0, 0, 10, 10],
            "explain_method": "eigencam", "salience_ratio": 1.23,
            "explanation": "because reasons",
        }
        out = build_output_dict([det])
        d0 = out["detections"][0]
        assert d0["explain_method"] == "eigencam"
        assert d0["salience_ratio"] == 1.23
        assert d0["explanation"] == "because reasons"

    def test_fields_absent_when_not_explained(self):
        det = {"class": "100_EGP", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
        d0 = build_output_dict([det])["detections"][0]
        assert "salience_ratio" not in d0
        assert "explanation" not in d0
