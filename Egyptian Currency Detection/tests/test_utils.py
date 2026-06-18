"""
Unit tests for summarize() / build_output_dict() / JSON schema v2.
"""

import pytest

from utils import build_output_dict, summarize, OUTPUT_SCHEMA_VERSION
from confidence_metrics import ConfidenceAnalyzer


def _det(cls, conf=0.9, bbox=(0, 0, 100, 100)):
    return {"class": cls, "confidence": conf, "bbox": list(bbox)}


class TestUtils:
    def test_summarize_empty(self):
        counts, total = summarize([])
        assert counts == {}
        assert total == 0

    def test_summarize_single_note(self):
        counts, total = summarize([_det("100_EGP")])
        assert counts == {"100_EGP": 1}
        assert total == 100

    def test_summarize_multiple_notes(self):
        counts, total = summarize([
            _det("100_EGP"),
            _det("100_EGP"),
            _det("50_EGP"),
            _det("20_EGP"),
            _det("1_EGP"),
        ])
        assert counts == {"100_EGP": 2, "50_EGP": 1, "20_EGP": 1, "1_EGP": 1}
        assert total == 271

    def test_summarize_unknown_class(self):
        # Unknown class names contribute to count but 0 to total
        counts, total = summarize([_det("999_EGP"), _det("100_EGP")])
        assert counts.get("999_EGP") == 1
        assert total == 100


class TestJsonSchema:
    def test_output_has_legacy_fields(self):
        out = build_output_dict([_det("100_EGP")])
        # Legacy fields preserved
        assert "detections" in out
        assert "counts" in out
        assert "total" in out
        assert out["total"] == 100

    def test_output_has_required_fields(self):
        dets = ConfidenceAnalyzer.add_uncertainty_flags(
            [_det("100_EGP", 0.94), _det("50_EGP", 0.72)]
        )
        report = ConfidenceAnalyzer.confidence_report(dets)
        out = build_output_dict(
            dets,
            image_path="x.jpg",
            inference_ms=27.3,
            model_version="best.pt",
            image_size=(1920, 1080),
            confidence_report=report,
        )
        assert out["schema_version"] == OUTPUT_SCHEMA_VERSION
        for key in ("metadata", "detections", "summary",
                    "counts", "total", "confidence_report"):
            assert key in out, f"missing {key}"

    def test_metadata_fields(self):
        out = build_output_dict(
            [], image_path="a.jpg", inference_ms=10.0,
            model_version="m.pt", image_size=(640, 480),
        )
        meta = out["metadata"]
        assert meta["image"] == "a.jpg"
        assert meta["model_version"] == "m.pt"
        assert meta["inference_time_ms"] == 10.0
        assert meta["image_size"] == [640, 480]
        assert meta["timestamp"]  # ISO string present

    def test_normalized_bbox(self):
        out = build_output_dict(
            [_det("100_EGP", bbox=(96, 108, 192, 216))],
            image_size=(960, 1080),
        )
        det = out["detections"][0]
        assert det["bbox_normalized"] == [0.1, 0.1, 0.2, 0.2]
        assert det["area_pixels"] == (192 - 96) * (216 - 108)

    def test_confidence_report_structure(self):
        dets = ConfidenceAnalyzer.add_uncertainty_flags(
            [_det("100_EGP", 0.94), _det("50_EGP", 0.72)]
        )
        report = ConfidenceAnalyzer.confidence_report(dets)
        for key in ("num_detections", "avg_confidence", "min_confidence",
                    "max_confidence", "std_confidence",
                    "high_confidence_count", "medium_confidence_count",
                    "low_confidence_count", "high_risk_detections"):
            assert key in report, f"missing {key}"
        assert report["num_detections"] == 2
        assert report["min_confidence"] == pytest.approx(0.72)
        assert report["max_confidence"] == pytest.approx(0.94)

    def test_summary_confidence_stats(self):
        out = build_output_dict([_det("100_EGP", 0.9), _det("50_EGP", 0.7)])
        conf = out["summary"]["confidence"]
        assert conf["min"] == pytest.approx(0.7)
        assert conf["max"] == pytest.approx(0.9)
        assert conf["mean"] == pytest.approx(0.8)


class TestConfidenceAnalyzer:
    def test_filter_by_confidence(self):
        dets = [_det("100_EGP", 0.4), _det("50_EGP", 0.6), _det("20_EGP", 0.9)]
        out = ConfidenceAnalyzer.filter_by_confidence(dets, threshold=0.5)
        assert len(out) == 2
        assert all(d["confidence"] >= 0.5 for d in out)

    def test_risk_labels(self):
        dets = ConfidenceAnalyzer.add_uncertainty_flags([
            _det("100_EGP", 0.95),  # low risk
            _det("100_EGP", 0.80),  # medium
            _det("100_EGP", 0.60),  # high
            _det("100_EGP", 0.40),  # reject
        ])
        assert dets[0]["risk"] == "low"
        assert dets[1]["risk"] == "medium"
        assert dets[2]["risk"] == "high"
        assert dets[3]["risk"] == "reject"

    def test_warning_for_100_50_confusion(self):
        dets = ConfidenceAnalyzer.add_uncertainty_flags([_det("100_EGP", 0.80)])
        assert "warning" in dets[0]
        assert "50_EGP" in dets[0]["warning"]

    def test_no_warning_above_threshold(self):
        dets = ConfidenceAnalyzer.add_uncertainty_flags([_det("100_EGP", 0.98)])
        assert "warning" not in dets[0]

    def test_empty_report(self):
        report = ConfidenceAnalyzer.confidence_report([])
        assert report["num_detections"] == 0
        assert report["high_risk_detections"] == []
