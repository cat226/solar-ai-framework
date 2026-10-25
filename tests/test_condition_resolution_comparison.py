"""tests/test_condition_resolution_comparison.py — Phase 15D-A: unit tests
for the deterministic v2(640) vs v3_960 comparison logic in
training/detection/ground_pv/condition_resolution_comparison.py. Pure
logic only — no torch, no ultralytics, no dataset/checkpoint required."""
from __future__ import annotations

import pytest

from training.detection.ground_pv.condition_resolution_comparison import (
    ACTIVE_CLASSES,
    official_overall_delta,
    official_per_class_delta,
    size_bin_recall,
    size_bin_recall_delta,
    small_object_recall_improved,
)


def _manifest(precision, recall, mAP50, mAP50_95, per_class_overrides=None):
    per_class = {}
    for cls in ACTIVE_CLASSES:
        base = dict(precision=precision, recall=recall, AP50=mAP50, AP50_95=mAP50_95)
        if per_class_overrides and cls in per_class_overrides:
            base.update(per_class_overrides[cls])
        per_class[cls] = base
    return {
        "test_default_metrics": dict(precision=precision, recall=recall, mAP50=mAP50, mAP50_95=mAP50_95),
        "per_class": {"test_default": per_class},
    }


class TestOfficialOverallDelta:
    def test_basic_delta(self):
        base = _manifest(0.50, 0.40, 0.45, 0.20)
        cand = _manifest(0.55, 0.42, 0.46, 0.19)
        d = official_overall_delta(base, cand)
        assert d["precision"]["delta"] == pytest.approx(0.05)
        assert d["recall"]["delta"] == pytest.approx(0.02)
        assert d["mAP50"]["delta"] == pytest.approx(0.01)
        assert d["mAP50_95"]["delta"] == pytest.approx(-0.01)

    def test_baseline_and_candidate_preserved(self):
        base = _manifest(0.50, 0.40, 0.45, 0.20)
        cand = _manifest(0.55, 0.42, 0.46, 0.19)
        d = official_overall_delta(base, cand)
        assert d["precision"]["baseline"] == 0.50
        assert d["precision"]["candidate"] == 0.55

    def test_identical_manifests_zero_delta(self):
        m = _manifest(0.6, 0.5, 0.5, 0.3)
        d = official_overall_delta(m, m)
        assert all(row["delta"] == 0.0 for row in d.values())

    def test_custom_eval_key(self):
        base = {"val_operating_metrics": dict(precision=0.1, recall=0.2, mAP50=0.3, mAP50_95=0.4)}
        cand = {"val_operating_metrics": dict(precision=0.2, recall=0.3, mAP50=0.4, mAP50_95=0.5)}
        d = official_overall_delta(base, cand, eval_key="val_operating_metrics")
        assert d["precision"]["delta"] == pytest.approx(0.1)


class TestOfficialPerClassDelta:
    def test_per_class_delta_all_classes_present(self):
        base = _manifest(0.5, 0.4, 0.4, 0.2, per_class_overrides={"PHYSICAL_DAMAGE": dict(recall=0.6)})
        cand = _manifest(0.5, 0.4, 0.4, 0.2, per_class_overrides={"PHYSICAL_DAMAGE": dict(recall=0.65)})
        d = official_per_class_delta(base, cand)
        assert set(d.keys()) == set(ACTIVE_CLASSES)
        assert d["PHYSICAL_DAMAGE"]["recall"]["delta"] == pytest.approx(0.05)
        assert d["CLEAN"]["recall"]["delta"] == pytest.approx(0.0)


class TestSizeBinRecall:
    def test_basic_recall(self):
        counts = {"tiny": {"tp": 5, "fn": 15}, "small": {"tp": 10, "fn": 10},
                  "medium": {"tp": 8, "fn": 2}, "large": {"tp": 9, "fn": 1}}
        r = size_bin_recall(counts)
        assert r["tiny"] == pytest.approx(0.25)
        assert r["small"] == pytest.approx(0.5)
        assert r["medium"] == pytest.approx(0.8)
        assert r["large"] == pytest.approx(0.9)

    def test_missing_bin_is_none(self):
        counts = {"tiny": {"tp": 1, "fn": 1}}
        r = size_bin_recall(counts)
        assert r["small"] is None and r["medium"] is None and r["large"] is None

    def test_zero_instances_bin_is_none(self):
        counts = {"tiny": {"tp": 0, "fn": 0}}
        r = size_bin_recall(counts)
        assert r["tiny"] is None

    def test_empty_dict(self):
        r = size_bin_recall({})
        assert all(v is None for v in r.values())


def _summary_with_size_bins(per_class_bins: dict[str, dict[str, dict[str, int]]]) -> dict:
    per_class = {}
    for cls in ACTIVE_CLASSES:
        per_class[cls] = {"size_bin_recall": per_class_bins.get(cls, {})}
    return {"per_class": per_class}


class TestSizeBinRecallDelta:
    def test_improvement_detected(self):
        base = _summary_with_size_bins({
            "DUSTY": {"tiny": {"tp": 1, "fn": 9}},  # recall 0.10
        })
        cand = _summary_with_size_bins({
            "DUSTY": {"tiny": {"tp": 3, "fp": 0, "fn": 7}},  # recall 0.30
        })
        d = size_bin_recall_delta(base, cand)
        assert d["DUSTY"]["tiny"]["baseline"] == pytest.approx(0.10)
        assert d["DUSTY"]["tiny"]["candidate"] == pytest.approx(0.30)
        assert d["DUSTY"]["tiny"]["delta"] == pytest.approx(0.20)

    def test_missing_data_yields_none_delta(self):
        base = _summary_with_size_bins({})
        cand = _summary_with_size_bins({})
        d = size_bin_recall_delta(base, cand)
        for cls in ACTIVE_CLASSES:
            assert d[cls]["tiny"]["delta"] is None

    def test_all_classes_and_bins_present(self):
        base = _summary_with_size_bins({})
        cand = _summary_with_size_bins({})
        d = size_bin_recall_delta(base, cand)
        assert set(d.keys()) == set(ACTIVE_CLASSES)
        for cls in ACTIVE_CLASSES:
            assert set(d[cls].keys()) == {"tiny", "small", "medium", "large"}


class TestSmallObjectRecallImproved:
    def test_no_classes_improve_when_all_flat(self):
        size_delta = {
            cls: {"tiny": {"delta": 0.0}, "small": {"delta": 0.0}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}}
            for cls in ACTIVE_CLASSES
        }
        result = small_object_recall_improved(size_delta)
        assert result["classes_improved"] == 0
        assert result["material_improvement"] is False

    def test_material_improvement_when_enough_classes_improve(self):
        size_delta = {
            cls: {"tiny": {"delta": 0.10}, "small": {"delta": 0.10}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}}
            for cls in ACTIVE_CLASSES
        }
        result = small_object_recall_improved(size_delta, min_improving_classes=3)
        assert result["classes_improved"] == 4
        assert result["material_improvement"] is True

    def test_mixed_result_not_material(self):
        # Only 1 of 4 classes shows a real small-bin improvement — matches
        # Phase 15D-A's actual observed pattern (CLEAN tiny up, others flat/down).
        size_delta = {
            "CLEAN": {"tiny": {"delta": 0.09}, "small": {"delta": -0.05}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}},
            "DUSTY": {"tiny": {"delta": -0.04}, "small": {"delta": -0.07}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}},
            "BIRD_DROP": {"tiny": {"delta": 0.0}, "small": {"delta": -0.03}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}},
            "PHYSICAL_DAMAGE": {"tiny": {"delta": 0.0}, "small": {"delta": -0.09}, "medium": {"delta": 0.0}, "large": {"delta": 0.0}},
        }
        result = small_object_recall_improved(size_delta, min_improving_classes=3)
        assert result["classes_improved"] <= 1
        assert result["material_improvement"] is False

    def test_none_deltas_ignored_not_treated_as_zero(self):
        size_delta = {
            cls: {"tiny": {"delta": None}, "small": {"delta": None}, "medium": {"delta": None}, "large": {"delta": None}}
            for cls in ACTIVE_CLASSES
        }
        result = small_object_recall_improved(size_delta)
        for cls in ACTIVE_CLASSES:
            assert result["per_class"][cls]["avg_small_bin_delta"] is None
            assert result["per_class"][cls]["improved"] is False
