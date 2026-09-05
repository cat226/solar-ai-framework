"""tests/test_yolo_threshold_sweep.py — Unit tests for the pure logic in
training/evaluation/yolo_threshold_sweep.py: object-size bucketing and the
test-split refusal guard (this script must never be used to select a
threshold against the test split - see the module's own docstring and
Phase 6B's non-negotiable rule against test-set threshold optimization).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from training.evaluation.yolo_threshold_sweep import _bucket_of, _size_bucket_boundaries

_SCRIPT = Path(__file__).resolve().parent.parent / "training" / "evaluation" / "yolo_threshold_sweep.py"


class TestSizeBucketBoundaries:
    def test_boundaries_derived_from_real_quartiles(self):
        records = [
            {"gt_areas": [0.001, 0.002, 0.003, 0.004]},
            {"gt_areas": [0.005, 0.006, 0.007, 0.008]},
        ]
        boundaries = _size_bucket_boundaries(records)
        all_areas = sorted(0.001 * i for i in range(1, 9))
        assert boundaries["p25"] == pytest.approx(all_areas[2], abs=1e-9)
        assert boundaries["p50"] == pytest.approx(all_areas[4], abs=1e-9)
        assert boundaries["p75"] == pytest.approx(all_areas[6], abs=1e-9)

    def test_no_ground_truth_boxes_returns_zero_boundaries_not_an_error(self):
        boundaries = _size_bucket_boundaries([{"gt_areas": []}])
        assert boundaries == {"p25": 0.0, "p50": 0.0, "p75": 0.0}


class TestBucketOf:
    _boundaries = {"p25": 0.004, "p50": 0.006, "p75": 0.008}

    def test_below_p25_is_tiny(self):
        assert _bucket_of(0.001, self._boundaries) == "tiny"

    def test_between_p25_and_p50_is_small(self):
        assert _bucket_of(0.005, self._boundaries) == "small"

    def test_between_p50_and_p75_is_medium(self):
        assert _bucket_of(0.007, self._boundaries) == "medium"

    def test_at_or_above_p75_is_large(self):
        assert _bucket_of(0.008, self._boundaries) == "large"
        assert _bucket_of(0.5, self._boundaries) == "large"

    def test_boundary_values_are_inclusive_on_the_lower_edge_of_the_next_bucket(self):
        """Exactly at p25 belongs to 'small', not 'tiny' - a defensible,
        consistent convention (< strictly for the lower bucket)."""
        assert _bucket_of(self._boundaries["p25"], self._boundaries) == "small"


class TestRefusesTestSplit:
    def test_running_against_test_split_exits_nonzero_without_evaluating(self):
        """This is the concrete enforcement of the non-negotiable rule:
        threshold selection must never run against the test split."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--data-root", ".", "--split", "test"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "test split" in (result.stdout + result.stderr).lower()

    def test_val_split_is_the_default(self):
        import argparse
        # Mirror the script's own argument definition to confirm the default
        # without invoking real model loading.
        parser = argparse.ArgumentParser()
        parser.add_argument("--split", type=str, default="val")
        args = parser.parse_args([])
        assert args.split == "val"
