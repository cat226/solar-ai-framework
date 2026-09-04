"""Tests for services/storage.py — local inspection history persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from services import storage
from services.pipeline import PipelineResult


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point storage at a throwaway DB file per test, never the real one."""
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


@pytest.fixture
def success_result(detection_single_panel, classification_clean, default_weather, default_physics, prediction_normal):
    from services.recommendation import generate_recommendations

    recs = generate_recommendations(
        classification=classification_clean,
        physics=default_physics,
        prediction=prediction_normal,
    )
    return PipelineResult(
        detection_result=detection_single_panel,
        classification_result=classification_clean,
        weather_data=default_weather,
        physics_data=default_physics,
        efficiency_prediction=prediction_normal,
        recommendations=recs,
        processing_time=1.23,
        status="SUCCESS",
        city="Testville",
        classifier_source="production",
        xgboost_available=True,  # this fixture represents a full run with a real prediction
    )


class TestRecordInspection:
    def test_rejects_non_success_result(self):
        bad = PipelineResult(status="ERROR", error_message="boom", error_type="ValueError")
        with pytest.raises(ValueError, match="SUCCESS"):
            storage.record_inspection(bad, city="Nowhere")

    def test_records_and_reads_back(self, success_result):
        row_id = storage.record_inspection(success_result, city="Testville", image_bytes=b"fake-image-bytes")
        row = storage.get_inspection(row_id)
        assert row is not None
        assert row["city"] == "Testville"
        assert row["fault_label"] == "Clean"
        assert row["panel_count"] == 1
        assert row["image_sha256"] == storage.image_sha256(b"fake-image-bytes")

    def test_image_bytes_never_persisted_raw(self, success_result):
        row_id = storage.record_inspection(success_result, city="Testville", image_bytes=b"secret-pixels")
        row = storage.get_inspection(row_id)
        assert b"secret-pixels" not in row["result_json"].encode()
        assert "secret-pixels" not in str(row.values())


class TestReadHelpers:
    def test_empty_summary_stats(self):
        stats = storage.get_summary_stats()
        assert stats["total_inspections"] == 0
        assert stats["fault_distribution"] == {}

    def test_summary_stats_after_recording(self, success_result):
        storage.record_inspection(success_result, city="Testville")
        stats = storage.get_summary_stats()
        assert stats["total_inspections"] == 1
        assert stats["fault_distribution"] == {"Clean": 1}

    def test_recent_inspections_newest_first(self, success_result):
        id1 = storage.record_inspection(success_result, city="First")
        id2 = storage.record_inspection(success_result, city="Second")
        rows = storage.get_recent_inspections(limit=10)
        assert [r["id"] for r in rows][:2] == [id2, id1]

    def test_get_inspection_missing_returns_none(self):
        assert storage.get_inspection(99999) is None

    def test_alerts_empty_when_no_critical_or_warning(self, success_result):
        # classification_clean + prediction_normal should be OK/INFO severity, not an alert.
        storage.record_inspection(success_result, city="Testville")
        assert storage.get_alerts() == []


class TestInterimAndXgboostTracking:
    """classifier_source / xgboost_available columns, added alongside the
    interim-classifier and graceful-XGBoost-degradation pipeline work."""

    def test_classifier_source_and_xgboost_available_persisted(self, success_result):
        row_id = storage.record_inspection(success_result, city="Testville")
        row = storage.get_inspection(row_id)
        assert row["classifier_source"] == "production"
        assert row["xgboost_available"] == 1

    def test_interim_and_unavailable_prediction_recorded_honestly(self, success_result):
        from dataclasses import replace
        from models.predictor import PredictionResult

        interim_result = replace(
            success_result,
            classifier_source="interim",
            xgboost_available=False,
            efficiency_prediction=PredictionResult(prediction_successful=False),
        )
        row_id = storage.record_inspection(interim_result, city="Testville")
        row = storage.get_inspection(row_id)
        assert row["classifier_source"] == "interim"
        assert row["xgboost_available"] == 0

    def test_summary_stats_counts_xgboost_unavailable(self, success_result):
        from dataclasses import replace

        storage.record_inspection(success_result, city="A")  # xgboost_available=True
        storage.record_inspection(replace(success_result, xgboost_available=False), city="B")
        stats = storage.get_summary_stats()
        assert stats["xgboost_unavailable_count"] == 1

    def test_avg_efficiency_loss_excludes_unavailable_predictions(self, success_result):
        """A row with no real prediction (efficiency_loss_pct defaults to
        0.0) must not drag down the average - it was never measured, not
        genuinely zero."""
        from dataclasses import replace
        from models.predictor import PredictionResult

        storage.record_inspection(success_result, city="A")  # real prediction, xgboost_available=True
        storage.record_inspection(
            replace(success_result, xgboost_available=False, efficiency_prediction=PredictionResult(prediction_successful=False)),
            city="B",
        )
        stats = storage.get_summary_stats()
        # Only the one real prediction should count toward the average.
        assert stats["avg_efficiency_loss_pct"] == success_result.efficiency_prediction.efficiency_loss_pct

    def test_summary_stats_class_counts_from_fault_distribution(self, success_result):
        storage.record_inspection(success_result, city="Testville")  # classification_clean -> "Clean"
        stats = storage.get_summary_stats()
        assert stats["clean_count"] == 1
        assert stats["dusty_count"] == 0
        assert stats["hotspot_count"] == 0

    def test_total_panels_analyzed_sums_panel_count(self, success_result):
        storage.record_inspection(success_result, city="A")
        storage.record_inspection(success_result, city="B")
        stats = storage.get_summary_stats()
        assert stats["total_panels_analyzed"] == 2 * success_result.detection_result.panel_count

    def test_panels_and_site_summary_stored_in_result_json(self, success_result):
        import json
        from dataclasses import replace
        from services.pipeline import PanelResult, SiteSummary

        result_with_panels = replace(
            success_result,
            panels=[PanelResult(panel_index=1, classification=success_result.classification_result)],
            site_summary=SiteSummary(total_panels=1, class_counts={"Clean": 1}),
        )
        row_id = storage.record_inspection(result_with_panels, city="Testville")
        row = storage.get_inspection(row_id)
        stored = json.loads(row["result_json"])
        assert len(stored["panels"]) == 1
        assert stored["site_summary"]["total_panels"] == 1
