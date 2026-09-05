"""tests/test_dataset_candidate_gate.py — Unit tests for
training/evaluation/dataset_candidate_gate.py, the reusable dataset-vetting
tooling introduced in Phase 6C.

Real dataset candidates investigated this phase (BDAPPV-adjacent PV01/03/08,
the Afroz-lineage Roboflow "Solar Panel Defects" family, OpenStat
Madagascar) are used as concrete regression fixtures below, so this suite
doubles as a machine-checked record of the actual Phase 6C gate decisions -
if the gate logic ever changed in a way that would flip one of these real,
already-documented verdicts, a test here would fail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from training.evaluation.dataset_candidate_gate import (
    DatasetCandidate,
    PROTECTED_ARTIFACT_NAMES,
    assert_candidate_path_is_safe,
    gate_verdict_is_defensible,
    promote_conditional_to_accept,
    validate_candidate,
    validate_yolo_annotation_line,
    validate_yolo_label_file,
)


def _bdappv_pv01_03_08() -> DatasetCandidate:
    """Real Phase 6C finding: correct license, wrong domain (all overhead/
    nadir viewpoints despite being drone-sourced for the 0.1m tier)."""
    return DatasetCandidate(
        name="Multi-resolution PV segmentation (PV01/PV03/PV08)",
        source_url="https://zenodo.org/records/5171712",
        license_id="CC-BY-4.0",
        acquisition_date="2026-09-05",
        image_type="satellite/aerial/UAV-orthophoto (all overhead/nadir)",
        has_bounding_boxes=True,
        target_domain_relevant=False,
        verdict="REJECT",
        verdict_reason="Wrong viewpoint domain (all three tiers overhead/nadir) despite good license/provenance",
    )


def _afroz_lineage_roboflow() -> DatasetCandidate:
    """Real Phase 6C finding: unverifiable provenance overrides a
    self-declared permissive license tag."""
    return DatasetCandidate(
        name="Roboflow Solar Panel Defects (Bird-Drop/Dusty/Electrical-Damage taxonomy)",
        source_url="https://universe.roboflow.com/solar-panel-defects/defect-detection-xessg",
        license_id="UNKNOWN",  # the real underlying Afroz Kaggle source's actual status
        acquisition_date="2026-09-05",
        image_type="ground-level close-up",
        has_bounding_boxes=True,
        target_domain_relevant=True,
        verdict="REJECT",
        verdict_reason="Traces to pythonafroz/solar-panel-images (Kaggle, License: Unknown) - already rejected for the classification pipeline",
    )


def _openstat_madagascar() -> DatasetCandidate:
    """Real Phase 6C finding: legitimate institutional provenance, but
    unresolved license variant and unconfirmed viewing angle -
    CONDITIONAL, not ACCEPT."""
    return DatasetCandidate(
        name="OpenStat Madagascar PV detection dataset",
        source_url="https://lacunafund.org/datasets/climate/",
        license_id="UNKNOWN",  # exact CC variant for this specific release not confirmed
        acquisition_date="2026-09-05",
        image_type="drone imagery of rooftop installations (viewing angle unconfirmed)",
        has_bounding_boxes=True,
        target_domain_relevant=True,
        verdict="CONDITIONAL",
        verdict_reason="Real institutional provenance (MAIDI/Lacuna Fund) but exact license variant and viewing angle (nadir vs. oblique) not directly verified",
    )


class TestValidateCandidate:
    def test_real_candidates_from_this_phase_are_valid_records(self):
        for candidate in (_bdappv_pv01_03_08(), _afroz_lineage_roboflow(), _openstat_madagascar()):
            assert validate_candidate(candidate) == []

    def test_empty_name_is_a_problem(self):
        c = _bdappv_pv01_03_08()
        c.name = "  "
        assert any("name" in p for p in validate_candidate(c))

    def test_empty_verdict_reason_is_a_problem(self):
        c = _bdappv_pv01_03_08()
        c.verdict_reason = ""
        assert any("verdict_reason" in p for p in validate_candidate(c))

    def test_invalid_verdict_value_is_a_problem(self):
        c = _bdappv_pv01_03_08()
        c.verdict = "MAYBE"
        assert any("verdict must be one of" in p for p in validate_candidate(c))


class TestGateVerdictIsDefensible:
    def test_reject_is_always_defensible(self):
        ok, _ = gate_verdict_is_defensible(_bdappv_pv01_03_08())
        assert ok is True

    def test_conditional_is_always_defensible(self):
        ok, _ = gate_verdict_is_defensible(_openstat_madagascar())
        assert ok is True

    def test_accept_with_unknown_license_is_not_defensible(self):
        """The concrete regression case: a candidate exactly like the
        Afroz-lineage Roboflow family, if someone tried to mark it ACCEPT
        despite the unknown license, must be rejected by the gate logic
        itself - not just by human review."""
        c = _afroz_lineage_roboflow()
        c.verdict = "ACCEPT"
        ok, reason = gate_verdict_is_defensible(c)
        assert ok is False
        assert "license" in reason.lower()

    def test_accept_without_bounding_boxes_is_not_defensible(self):
        c = DatasetCandidate(
            name="x", source_url="https://example.org", license_id="CC-BY-4.0",
            acquisition_date="2026-09-05", image_type="ground-level close-up",
            has_bounding_boxes=False, target_domain_relevant=True,
            verdict="ACCEPT", verdict_reason="test",
        )
        ok, reason = gate_verdict_is_defensible(c)
        assert ok is False
        assert "bounding" in reason.lower()

    def test_accept_with_everything_satisfied_is_defensible(self):
        c = DatasetCandidate(
            name="Hypothetical future ACCEPT candidate", source_url="https://example.org/doi",
            license_id="CC-BY-4.0", acquisition_date="2026-09-05",
            image_type="ground-level close-up", has_bounding_boxes=True,
            target_domain_relevant=True, verdict="ACCEPT", verdict_reason="all criteria met",
        )
        ok, _ = gate_verdict_is_defensible(c)
        assert ok is True


class TestPromoteConditionalToAccept:
    def test_cannot_promote_a_reject(self):
        with pytest.raises(ValueError, match="CONDITIONAL"):
            promote_conditional_to_accept(_bdappv_pv01_03_08(), new_evidence="doesn't matter")

    def test_cannot_promote_without_new_evidence(self):
        with pytest.raises(ValueError, match="new_evidence"):
            promote_conditional_to_accept(_openstat_madagascar(), new_evidence="")

    def test_promoting_openstat_without_a_permissive_license_still_fails(self):
        """Even with new evidence text, promotion must still pass the real
        license/annotation/domain checks - evidence text alone cannot
        silently grant ACCEPT (this is the concrete enforcement of "do not
        silently convert CONDITIONAL into ACCEPT")."""
        with pytest.raises(ValueError, match="license"):
            promote_conditional_to_accept(
                _openstat_madagascar(),
                new_evidence="Visually confirmed oblique ground-level imagery in a manual sample review",
            )

    def test_promoting_with_a_confirmed_permissive_license_succeeds(self):
        promoted = promote_conditional_to_accept(
            _openstat_madagascar(),
            new_evidence="Visually confirmed oblique ground-level imagery; license page confirms CC-BY-4.0 for this release",
            new_license_id="CC-BY-4.0",
        )
        assert promoted.verdict == "ACCEPT"
        assert "Visually confirmed" in promoted.verdict_reason


class TestYoloAnnotationValidation:
    def test_valid_line_passes(self):
        assert validate_yolo_annotation_line("0 0.5 0.5 0.2 0.2") is None

    def test_wrong_class_id_is_flagged(self):
        problem = validate_yolo_annotation_line("1 0.5 0.5 0.2 0.2")
        assert problem is not None and "class id" in problem

    def test_out_of_range_coordinate_is_flagged(self):
        problem = validate_yolo_annotation_line("0 1.5 0.5 0.2 0.2")
        assert problem is not None and "cx" in problem

    def test_degenerate_box_is_flagged(self):
        problem = validate_yolo_annotation_line("0 0.5 0.5 0.0 0.2")
        assert problem is not None and "degenerate" in problem

    def test_wrong_field_count_is_flagged(self):
        problem = validate_yolo_annotation_line("0 0.5 0.5")
        assert problem is not None and "5 fields" in problem

    def test_non_numeric_field_is_flagged(self):
        problem = validate_yolo_annotation_line("0 abc 0.5 0.2 0.2")
        assert problem is not None

    def test_empty_label_file_is_valid_negative_image(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        assert validate_yolo_label_file(p) == []

    def test_missing_label_file_is_valid(self, tmp_path):
        assert validate_yolo_label_file(tmp_path / "nope.txt") == []

    def test_label_file_with_one_bad_line_reports_it(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
        problems = validate_yolo_label_file(p)
        assert len(problems) == 1
        assert "line 2" in problems[0]


class TestProtectedArtifactGuard:
    def test_v1_yolo_weights_name_is_protected(self):
        assert "yolo_solar.pt" in PROTECTED_ARTIFACT_NAMES

    def test_v1_mobilenet_weights_name_is_protected(self):
        assert "mobilenet_solar_v1.pth" in PROTECTED_ARTIFACT_NAMES

    def test_writing_to_the_real_v1_yolo_path_raises(self):
        with pytest.raises(ValueError, match="protected"):
            assert_candidate_path_is_safe(Path("weights/yolo_solar.pt"))

    def test_writing_to_the_real_v1_mobilenet_path_raises(self):
        with pytest.raises(ValueError, match="protected"):
            assert_candidate_path_is_safe(Path("weights/mobilenet_solar_v1.pth"))

    def test_a_clearly_named_candidate_path_is_accepted(self):
        assert_candidate_path_is_safe(Path("weights/yolo_solar_domain_candidate.pt"))  # no raise

    def test_an_ambiguously_named_path_raises_even_if_not_literally_protected(self):
        with pytest.raises(ValueError, match="does not clearly identify"):
            assert_candidate_path_is_safe(Path("weights/yolo_solar_v2.pt"))
