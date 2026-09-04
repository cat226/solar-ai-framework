"""Tests for the per-panel analysis and graceful-XGBoost-degradation
additions to services/pipeline.py (PanelResult, SiteSummary,
_build_site_summary, and run_pipeline's new step 6/6b/7 behavior).

Follows the same mocking conventions as tests/test_pipeline.py: no real
model weights, no network, real domain dataclasses where practical.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from services.pipeline import (
    PanelResult,
    PipelineResult,
    SiteSummary,
    _build_site_summary,
    run_pipeline,
)
from services.recommendation import RecommendationReport, Severity
from models.classifier import ClassificationResult
from models.predictor import PredictionResult
from utils.exceptions import ModelLoadError


def _make_mock_model_manager():
    mm = MagicMock()
    mm.get_detector.return_value = MagicMock()
    mm.get_classifier.return_value = MagicMock()
    mm.get_predictor.return_value = MagicMock()
    mm.classifier_labels = ["Clean", "Dusty", "Hotspot"]
    mm.classifier_source = "interim"
    return mm


def _make_detector():
    detector = MagicMock()
    detector.detect.return_value = MagicMock()
    return detector


def _make_classifier():
    clf = MagicMock()
    clf.classify.return_value = MagicMock()
    return clf


def _make_predictor():
    pred = MagicMock()
    pred.predict.return_value = MagicMock()
    return pred


def _wire_common(monkeypatch, *, mm, detector, clf, predictor, weather, physics, features, recommendation):
    monkeypatch.setattr("services.pipeline.model_manager", mm)
    monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
    monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
    monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
    monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather)
    monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: physics)
    monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: features)
    monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: recommendation)


def _weather_mock():
    return MagicMock(
        fetch_successful=True, city="Chennai", ambient_temp_c=25.0, wind_speed_ms=2.0,
        cloud_cover_pct=30.0, latitude=13.0, longitude=80.0, timestamp=None,
    )


class TestBuildSiteSummary:
    def test_empty_panel_list_returns_all_zero_summary(self):
        summary = _build_site_summary([])
        assert summary == SiteSummary()
        assert summary.total_panels == 0
        assert summary.class_counts == {}
        assert summary.clean_pct == 0.0

    def test_counts_and_clean_pct(self):
        panels = [
            PanelResult(classification=ClassificationResult(label="Clean")),
            PanelResult(classification=ClassificationResult(label="Clean")),
            PanelResult(classification=ClassificationResult(label="Dusty")),
        ]
        summary = _build_site_summary(panels)
        assert summary.total_panels == 3
        assert summary.class_counts == {"Clean": 2, "Dusty": 1}
        assert summary.clean_pct == pytest.approx(66.67, abs=0.01)

    def test_averages_only_over_panels_with_successful_prediction(self):
        panels = [
            PanelResult(
                classification=ClassificationResult(label="Clean"),
                prediction=PredictionResult(efficiency_loss_pct=10.0, estimated_output_w=360.0, prediction_successful=True),
            ),
            PanelResult(
                classification=ClassificationResult(label="Dusty"),
                prediction=PredictionResult(prediction_successful=False),  # unavailable - must not pollute the average
            ),
        ]
        summary = _build_site_summary(panels)
        assert summary.panels_with_prediction == 1
        assert summary.average_efficiency_loss_pct == 10.0
        assert summary.average_estimated_output_w == 360.0

    def test_no_successful_predictions_gives_zero_averages_not_fabricated(self):
        panels = [PanelResult(classification=ClassificationResult(label="Clean"))]  # default prediction unsuccessful
        summary = _build_site_summary(panels)
        assert summary.panels_with_prediction == 0
        assert summary.average_efficiency_loss_pct == 0.0
        assert summary.average_estimated_output_w == 0.0


class TestPerPanelOrchestration:
    def test_panels_populated_one_per_detection_box(self, monkeypatch):
        img = Image.new("RGB", (640, 480))
        detection = MagicMock(
            panel_count=2, best_confidence=0.9,
            boxes=[[10.0, 10.0, 100.0, 100.0], [200.0, 150.0, 300.0, 250.0]],
            confidences=[0.9, 0.8],
        )
        whole_image_cls = MagicMock(label="Clean")
        panel1_cls = ClassificationResult(label="Dusty", class_id=1, confidence=0.7)
        panel2_cls = ClassificationResult(label="Hotspot", class_id=2, confidence=0.6)

        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        clf.classify.side_effect = [whole_image_cls, panel1_cls, panel2_cls]
        predictor = _make_predictor()
        prediction = PredictionResult(efficiency_loss_pct=5.0, estimated_output_w=380.0, prediction_successful=True)
        predictor.predict.return_value = prediction

        _wire_common(
            monkeypatch, mm=mm, detector=detector, clf=clf, predictor=predictor,
            weather=_weather_mock(), physics=MagicMock(), features=MagicMock(),
            recommendation=MagicMock(overall_severity=MagicMock(value="OK")),
        )

        result = run_pipeline(image=img, city="Chennai")

        assert result.status == "SUCCESS"
        assert len(result.panels) == 2
        assert [p.panel_index for p in result.panels] == [1, 2]
        assert result.panels[0].classification is panel1_cls
        assert result.panels[1].classification is panel2_cls
        assert result.panels[0].detection_confidence == 0.9
        assert result.panels[1].detection_confidence == 0.8
        assert result.panels[0].box == [10.0, 10.0, 100.0, 100.0]
        assert result.site_summary.total_panels == 2
        # whole-image classification result is untouched by the per-panel loop
        assert result.classification_result is whole_image_cls

    def test_zero_detections_yields_empty_panels_not_an_error(self, monkeypatch):
        img = Image.new("RGB", (640, 480))
        detection = MagicMock(panel_count=0, best_confidence=0.0, boxes=[], confidences=[])

        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        predictor = _make_predictor()
        predictor.predict.return_value = PredictionResult(prediction_successful=True)

        _wire_common(
            monkeypatch, mm=mm, detector=detector, clf=clf, predictor=predictor,
            weather=_weather_mock(), physics=MagicMock(), features=MagicMock(),
            recommendation=MagicMock(overall_severity=MagicMock(value="OK")),
        )

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.panels == []
        assert result.site_summary == SiteSummary()

    def test_classifier_source_propagated_from_model_manager(self, monkeypatch):
        img = Image.new("RGB", (640, 480))
        detection = MagicMock(panel_count=0, best_confidence=0.0, boxes=[], confidences=[])
        mm = _make_mock_model_manager()
        mm.classifier_source = "interim"
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        predictor = _make_predictor()
        predictor.predict.return_value = PredictionResult(prediction_successful=True)

        _wire_common(
            monkeypatch, mm=mm, detector=detector, clf=clf, predictor=predictor,
            weather=_weather_mock(), physics=MagicMock(), features=MagicMock(),
            recommendation=MagicMock(overall_severity=MagicMock(value="OK")),
        )

        result = run_pipeline(image=img, city="Chennai")
        assert result.classifier_source == "interim"


class TestXGBoostGracefulDegradation:
    def test_missing_xgboost_completes_with_success_and_marks_unavailable(self, monkeypatch):
        img = Image.new("RGB", (640, 480))
        detection = MagicMock(panel_count=0, best_confidence=0.0, boxes=[], confidences=[])

        mm = _make_mock_model_manager()
        mm.get_predictor.side_effect = ModelLoadError("XGBoost", "Pipeline not found at configured path.")
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        predictor = _make_predictor()

        _wire_common(
            monkeypatch, mm=mm, detector=detector, clf=clf, predictor=predictor,
            weather=_weather_mock(), physics=MagicMock(), features=MagicMock(),
            recommendation=MagicMock(overall_severity=MagicMock(value="OK")),
        )

        result = run_pipeline(image=img, city="Chennai")

        assert result.status == "SUCCESS"
        assert result.xgboost_available is False
        assert result.efficiency_prediction.prediction_successful is False
        # Recommendations must stay at the honest, unmodified default - never
        # fabricated from a 0.0 efficiency-loss placeholder.
        assert result.recommendations == RecommendationReport()
        assert result.recommendations.overall_severity == Severity.OK
        assert result.recommendations.summary == "No issues detected."
        predictor.predict.assert_not_called()

    def test_present_xgboost_still_runs_prediction_and_recommendations(self, monkeypatch):
        img = Image.new("RGB", (640, 480))
        detection = MagicMock(panel_count=0, best_confidence=0.0, boxes=[], confidences=[])
        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        predictor = _make_predictor()
        prediction = PredictionResult(efficiency_loss_pct=12.0, estimated_output_w=350.0, prediction_successful=True)
        predictor.predict.return_value = prediction
        recommendation = MagicMock(overall_severity=MagicMock(value="WARNING"))

        _wire_common(
            monkeypatch, mm=mm, detector=detector, clf=clf, predictor=predictor,
            weather=_weather_mock(), physics=MagicMock(), features=MagicMock(),
            recommendation=recommendation,
        )

        result = run_pipeline(image=img, city="Chennai")

        assert result.status == "SUCCESS"
        assert result.xgboost_available is True
        assert result.efficiency_prediction is prediction
        assert result.recommendations is recommendation

    def test_other_model_load_errors_still_abort(self, monkeypatch):
        """Only a missing XGBoost artifact degrades gracefully - a missing
        YOLO/MobileNet artifact still means there's nothing real to show."""
        img = Image.new("RGB", (640, 480))
        mm = _make_mock_model_manager()
        mm.get_detector.side_effect = ModelLoadError("YOLO", "Model weights not found.")

        monkeypatch.setattr("services.pipeline.model_manager", mm)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert result.error_type == "ModelLoadError"

    def test_missing_mobilenet_also_aborts(self, monkeypatch):
        """Same contract as a missing YOLO checkpoint: no production or
        interim MobileNet checkpoint at all means there is nothing real to
        classify with, so the pipeline must not proceed on a fabricated or
        default label."""
        img = Image.new("RGB", (640, 480))
        mm = _make_mock_model_manager()
        mm.get_classifier.side_effect = ModelLoadError("MobileNet", "No production or interim checkpoint present.")

        monkeypatch.setattr("services.pipeline.model_manager", mm)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert result.error_type == "ModelLoadError"
