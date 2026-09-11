"""tests/test_phase15e_augmentation_experiment.py — Phase 15E-A regression
guard for the committed augmentation-ablation artifacts (v2 baseline vs
v4_aug candidate). Locks in both the artifact SCHEMA (so a future refactor
of `condition_resolution_comparison.py` / `analyze_condition_4class_errors.py`
can't silently change the committed report's shape) and the ACTUAL
reported finding (0/4 classes materially improved small-object recall),
so a future change to the decision-rule thresholds is caught explicitly
rather than silently flipping this phase's conclusion.

Skips cleanly if the committed report files are absent (they are small
JSON, always committed — see .gitignore — but this mirrors the same
skip-don't-fail pattern as test_ground_pv_frozen_artifacts_integrity.py
for any machine/checkout where they might not be present)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMPARISON_PATH = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "condition_4class_v1" / "reports"
    / "condition_yolo11n_4class_aug_v1" / "augmentation_comparison.json"
)
_TRAINING_MANIFEST_PATH = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "condition_4class_v1" / "reports"
    / "condition_yolo11n_4class_aug_v1" / "training_manifest.json"
)
_ERROR_SUMMARY_PATH = (
    _REPO_ROOT / "datasets" / "ground_pv_research" / "condition_4class_v1" / "reports"
    / "condition_error_analysis_aug" / "error_analysis_summary.json"
)

pytestmark = pytest.mark.skipif(
    not _COMPARISON_PATH.is_file(),
    reason="Phase 15E-A committed comparison report not present on this checkout",
)


@pytest.fixture(scope="module")
def comparison() -> dict:
    return json.loads(_COMPARISON_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def training_manifest() -> dict:
    return json.loads(_TRAINING_MANIFEST_PATH.read_text(encoding="utf-8"))


class TestComparisonReportSchema:
    def test_top_level_schema(self, comparison):
        expected_keys = {
            "schema_version", "phase", "baseline", "candidate",
            "environment_match", "dataset_hashes_match",
            "official_overall_delta_test_default", "official_per_class_delta_test_default",
            "size_bin_recall_delta", "small_object_recall_verdict",
            "note_on_matching_methodology",
        }
        assert expected_keys.issubset(comparison.keys())

    def test_overall_delta_has_all_four_metrics(self, comparison):
        assert set(comparison["official_overall_delta_test_default"].keys()) == {
            "precision", "recall", "mAP50", "mAP50_95",
        }

    def test_per_class_delta_covers_all_active_classes(self, comparison):
        assert set(comparison["official_per_class_delta_test_default"].keys()) == {
            "CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE",
        }

    def test_size_bin_delta_has_all_bins_per_class(self, comparison):
        for cls, bins in comparison["size_bin_recall_delta"].items():
            assert set(bins.keys()) == {"tiny", "small", "medium", "large"}


class TestComparisonReportReproducibility:
    def test_environment_and_dataset_matched(self, comparison):
        """This phase asserted before training that only the augmentation
        recipe changed — the comparison must confirm zero environment or
        dataset drift, not just claim it in prose."""
        assert comparison["environment_match"] is True
        assert comparison["dataset_hashes_match"] is True

    def test_baseline_is_v2_candidate_is_v4_aug(self, comparison):
        assert comparison["baseline"]["imgsz"] == 640
        assert comparison["baseline"]["epochs"] == 100
        assert comparison["candidate"]["imgsz"] == 640
        assert comparison["candidate"]["epochs"] == 100
        assert comparison["candidate"]["checkpoint_sha256"] == (
            "eb4d6151a53ded76049f7469946e3016384b48ba5bdbcbda2845efca7fdf618b"
        )


class TestReportedFinding:
    """Locks in the ACTUAL Phase 15E-A conclusion — a future change to the
    comparison module's decision-rule constants should fail this test
    loudly rather than silently reversing the reported recommendation."""

    def test_zero_classes_show_material_small_object_improvement(self, comparison):
        verdict = comparison["small_object_recall_verdict"]
        assert verdict["classes_improved"] == 0
        assert verdict["classes_total"] == 4
        assert verdict["material_improvement"] is False

    def test_physical_damage_is_the_largest_regression(self, comparison):
        """PHYSICAL_DAMAGE regressed on all four official metrics and had
        the largest negative avg small-bin delta of the four classes —
        this is the "substantial regression" the task asked to check for."""
        per_class_official = comparison["official_per_class_delta_test_default"]["PHYSICAL_DAMAGE"]
        for metric in ("precision", "recall", "AP50", "AP50_95"):
            assert per_class_official[metric]["delta"] < 0

        small_object = comparison["small_object_recall_verdict"]["per_class"]
        worst_class = min(
            small_object, key=lambda c: small_object[c]["avg_small_bin_delta"] or 0.0
        )
        assert worst_class == "PHYSICAL_DAMAGE"


class TestAugmentationInterventionManifestField:
    """The one genuinely NEW manifest field this phase introduces —
    locks in its shape so a future notebook-generator edit can't silently
    drop or malform it."""

    def test_augmentation_intervention_present_and_shaped(self, training_manifest):
        interv = training_manifest["augmentation_intervention"]
        assert interv["changed"] == {"mosaic": [0.5, 1.0], "copy_paste": [0.0, 0.5]}
        assert isinstance(interv["rationale"], str) and len(interv["rationale"]) > 0

    def test_only_the_two_declared_augmentation_params_differ_from_v2(self, training_manifest):
        v2_augmentation = {
            "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.4, "degrees": 5.0, "translate": 0.1,
            "scale": 0.3, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0,
        }
        aug_augmentation = training_manifest["augmentation"]
        differing = {k for k in v2_augmentation if v2_augmentation[k] != aug_augmentation[k]}
        assert differing == {"mosaic", "copy_paste"}

    def test_non_augmentation_training_settings_unchanged_from_v2(self, training_manifest):
        assert training_manifest["imgsz"] == 640
        assert training_manifest["epochs"] == 100
        assert training_manifest["patience"] == 20
        assert training_manifest["seed"] == 42
        assert training_manifest["optimizer"] == "auto"
        assert training_manifest["cos_lr"] is True
        assert training_manifest["base_model"] == "yolo11n.pt"


class TestErrorAnalysisSummaryPresent:
    def test_error_analysis_summary_covers_all_active_classes(self):
        if not _ERROR_SUMMARY_PATH.is_file():
            pytest.skip("Phase 15E-A committed error-analysis summary not present on this checkout")
        summary = json.loads(_ERROR_SUMMARY_PATH.read_text(encoding="utf-8"))
        assert set(summary["per_class"].keys()) == {"CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE"}
        assert summary["checkpoint_sha256"] == "eb4d6151a53ded76049f7469946e3016384b48ba5bdbcbda2845efca7fdf618b"
