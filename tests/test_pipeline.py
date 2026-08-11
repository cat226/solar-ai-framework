"""tests/test_pipeline.py - Deterministic tests for services/pipeline.py.

Covers the orchestration layer as a coordinator of mocked stages:

A. Entry-point validation
B. Normal orchestration / happy path
C. Data propagation between stages
D. PipelineResult structure
E. Failure isolation per stage
F. Weather degradation passthrough
G. Execution order
H. Determinism

Design rules honoured:
- no real model weights, no network, no API keys
- mocks patch where services/pipeline.py looks up symbols
- real domain dataclasses from the repository are used
- existing conftest fixtures are preferred
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from PIL import Image

from services.pipeline import PipelineResult, _validate_scalar_inputs, run_pipeline
from utils.exceptions import (
    FeatureValidationError,
    ImageValidationError,
    InputValidationError,
    ModelLoadError,
    PredictionError,
    SolarAIError,
)


# ---------------------------------------------------------------------------
# Helpers / sentinels
# ---------------------------------------------------------------------------

class SentinelException(SolarAIError):
    """Distinctive SolarAIError subclass for identifying specific stage failures."""
    pass


def _make_mock_model_manager():
    """Return a mock model_manager with get_* methods."""
    mm = MagicMock()
    mm.get_detector.return_value = MagicMock()
    mm.get_classifier.return_value = MagicMock()
    mm.get_predictor.return_value = MagicMock()
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


# ---------------------------------------------------------------------------
# A. Entry-point validation
# ---------------------------------------------------------------------------

class TestEntryPointValidation:
    """run_pipeline validates inputs and returns PipelineResult(status='ERROR')."""

    def test_none_image_returns_error_result(self):
        result = run_pipeline(image=None)
        assert result.status == "ERROR"
        assert "No image provided" in result.error_message
        assert result.error_type == "ImageValidationError"

    def test_non_pil_image_returns_error_result(self):
        result = run_pipeline(image="not-an-image")
        assert result.status == "ERROR"
        assert "Pipeline input must be a PIL image" in result.error_message
        assert result.error_type == "ImageValidationError"

    def test_negative_panel_age_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age=-1.0)
        assert result.status == "ERROR"
        assert "panel_age" in result.error_message
        assert result.error_type == "InputValidationError"

    def test_panel_age_above_100_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age=101.0)
        assert result.status == "ERROR"
        assert "panel_age" in result.error_message

    def test_panel_age_at_boundary_100_is_accepted(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age=100.0)
        # Validation passes; failure comes from missing mocks (ModelLoadError).
        assert result.status == "ERROR"
        assert result.error_type == "ModelLoadError"

    def test_negative_maintenance_count_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, maintenance_count=-1)
        assert result.status == "ERROR"
        assert "maintenance_count" in result.error_message

    def test_boolean_maintenance_count_rejected(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, maintenance_count=True)
        assert result.status == "ERROR"
        assert "maintenance_count" in result.error_message

    def test_negative_voltage_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, voltage=-1.0)
        assert result.status == "ERROR"
        assert "voltage" in result.error_message

    def test_negative_current_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, current=-1.0)
        assert result.status == "ERROR"
        assert "current" in result.error_message

    def test_blank_installation_type_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, installation_type="")
        assert result.status == "ERROR"
        assert "installation_type" in result.error_message

    def test_none_installation_type_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, installation_type=None)
        assert result.status == "ERROR"
        assert "installation_type" in result.error_message

    def test_nan_panel_age_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age=math.nan)
        assert result.status == "ERROR"
        assert "finite" in result.error_message

    def test_infinity_voltage_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, voltage=math.inf)
        assert result.status == "ERROR"
        assert "finite" in result.error_message

<<<<<<< HEAD
    def test_string_panel_age_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age="abc")
        assert result.status == "ERROR"
        assert "panel_age" in result.error_message
        assert result.error_type == "InputValidationError"

    def test_none_voltage_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, voltage=None)
        assert result.status == "ERROR"
        assert "voltage" in result.error_message
        assert result.error_type == "InputValidationError"

    def test_boolean_current_returns_error_result(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, current=True)
        assert result.status == "ERROR"
        assert "current" in result.error_message
        assert result.error_type == "InputValidationError"

=======
>>>>>>> 9e2fe55 (TEST-003: Add pipeline orchestration test coverage)
    def test_validation_stops_before_models(self, monkeypatch):
        img = Image.new("RGB", (10, 10))
        called = []

        def fake_get(*a, **kw):
            called.append("model_manager")
            return MagicMock()

        monkeypatch.setattr("services.pipeline.model_manager", MagicMock(get_detector=fake_get))
        result = run_pipeline(image=img, panel_age=-1.0)
        assert result.status == "ERROR"
        assert "model_manager" not in called


# ---------------------------------------------------------------------------
# B. Normal orchestration
# ---------------------------------------------------------------------------

class TestNormalOrchestration:
    """Full mocked happy path through all 7 stages."""

    def test_successful_pipeline_returns_success_status(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        detector = _make_detector()
        clf = _make_classifier()
        predictor = _make_predictor()

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"

    def test_successful_pipeline_populates_all_fields(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        detection = MagicMock(panel_count=1, best_confidence=0.9)
        classification = MagicMock(label="Clean")
        weather = MagicMock(fetch_successful=True, city="Chennai", ambient_temp_c=25.0,
                           wind_speed_ms=2.0, cloud_cover_pct=30.0, latitude=13.0, longitude=80.0, timestamp=None)
        physics = MagicMock()
        features = MagicMock()
        prediction = MagicMock()
        recommendation = MagicMock(overall_severity=MagicMock(value="OK"))

        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        clf.classify.return_value = classification
        predictor = _make_predictor()
        predictor.predict.return_value = prediction

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather)
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: physics)
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: features)
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: recommendation)

        result = run_pipeline(image=img, city="Chennai")
        assert result.detection_result is detection
        assert result.classification_result is classification
        assert result.weather_data is weather
        assert result.physics_data is physics
        assert result.feature_dataframe is features
        assert result.efficiency_prediction is prediction
        assert result.recommendations is recommendation

    def test_successful_pipeline_sets_city(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Mumbai")
        assert result.city == "Mumbai"

    def test_successful_pipeline_falls_back_to_default_city(self, monkeypatch, project_config):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img)
        assert result.city == project_config["weather"]["default_city"]

    def test_successful_pipeline_sets_processing_time(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

<<<<<<< HEAD
        result = run_pipeline(image=img, city="Chennai")
        assert result.processing_time >= 0.0

    def test_rgba_image_converted_to_rgb(self, monkeypatch):
        img = Image.new("RGBA", (224, 224), (255, 0, 0, 128))
        original_mode = img.mode
        mm = _make_mock_model_manager()
        detector = _make_detector()
        clf = _make_classifier()
        predictor = _make_predictor()

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert img.mode == original_mode
        detector.detect.assert_called_once()
        clf.classify.assert_called_once()
        assert detector.detect.call_args[0][0].mode == "RGB"
        assert clf.classify.call_args[0][0].mode == "RGB"

    def test_grayscale_image_converted_to_rgb(self, monkeypatch):
        img = Image.new("L", (224, 224), 128)
        original_mode = img.mode
        mm = _make_mock_model_manager()
        detector = _make_detector()
        clf = _make_classifier()
        predictor = _make_predictor()

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert img.mode == original_mode
        detector.detect.assert_called_once()
        clf.classify.assert_called_once()
        assert detector.detect.call_args[0][0].mode == "RGB"
        assert clf.classify.call_args[0][0].mode == "RGB"

=======
        result = run_pipeline(image=img)
        assert result.processing_time >= 0.0

>>>>>>> 9e2fe55 (TEST-003: Add pipeline orchestration test coverage)

# ---------------------------------------------------------------------------
# C. Execution order
# ---------------------------------------------------------------------------

class TestExecutionOrder:
    """Pipeline invokes stages in the expected sequence."""

    def test_stages_invoked_in_order(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        call_order = []

        def track(name):
            def wrapper(*a, **kw):
                call_order.append(name)
                return MagicMock()
            return wrapper

        mm = _make_mock_model_manager()
        detector = _make_detector()
        clf = _make_classifier()
        predictor = _make_predictor()

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", track("weather"))
        monkeypatch.setattr("services.pipeline.compute_physics", track("physics"))
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", track("features"))
        monkeypatch.setattr("services.pipeline.generate_recommendations", track("recommendations"))

        run_pipeline(image=img, city="Chennai")

        assert call_order == ["weather", "physics", "features", "recommendations"]


# ---------------------------------------------------------------------------
# D. Data propagation
# ---------------------------------------------------------------------------

class TestDataPropagation:
    """Outputs from each stage reach the expected downstream consumers."""

    def test_detection_result_passed_to_feature_engineering(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        detection = MagicMock(panel_count=1, best_confidence=0.9)
        classification = MagicMock(label="Clean")
        weather = MagicMock(fetch_successful=True, city="Chennai", ambient_temp_c=25.0,
                           wind_speed_ms=2.0, cloud_cover_pct=30.0, latitude=13.0, longitude=80.0, timestamp=None)
        physics = MagicMock()
        features = MagicMock()
        prediction = MagicMock()
        recommendation = MagicMock(overall_severity=MagicMock(value="OK"))

        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.return_value = detection
        clf = _make_classifier()
        clf.classify.return_value = classification
        predictor = _make_predictor()
        predictor.predict.return_value = prediction

        build_mock = MagicMock(return_value=features)
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather)
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: physics)
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", build_mock)
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: recommendation)

        run_pipeline(image=img, city="Chennai")
        build_mock.assert_called_once_with(
            weather=weather,
            physics=physics,
            classification=classification,
            detection=detection,
        )

    def test_feature_dataframe_passed_to_predictor(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        features = MagicMock()
        prediction = MagicMock()

        mm = _make_mock_model_manager()
        predictor = _make_predictor()
        predictor.predict.return_value = prediction

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: features)
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        run_pipeline(image=img, city="Chennai")
        predictor.predict.assert_called_once_with(features)

    def test_prediction_passed_to_recommendations(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        classification = MagicMock(label="Clean")
        prediction = MagicMock()
        recommendation_mock = MagicMock(overall_severity=MagicMock(value="OK"))

        mm = _make_mock_model_manager()
        clf = _make_classifier()
        clf.classify.return_value = classification
        predictor = _make_predictor()
        predictor.predict.return_value = prediction

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", recommendation_mock)

        run_pipeline(image=img, city="Chennai")
        recommendation_mock.assert_called_once()
        call_kwargs = recommendation_mock.call_args[1]
        assert call_kwargs["classification"] is classification
        assert call_kwargs["prediction"] is prediction

    def test_weather_data_passed_to_physics(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        weather = MagicMock(
            fetch_successful=True, city="Chennai",
            ambient_temp_c=25.0, wind_speed_ms=2.0,
            cloud_cover_pct=30.0, latitude=13.0, longitude=80.0,
            timestamp=None
        )
        physics = MagicMock()

        mm = _make_mock_model_manager()
        physics_mock = MagicMock(return_value=physics)

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather)
        monkeypatch.setattr("services.pipeline.compute_physics", physics_mock)
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        run_pipeline(image=img, city="Chennai")
        physics_mock.assert_called_once()
        call_kwargs = physics_mock.call_args[1]
        assert call_kwargs["ambient_temp_c"] == 25.0
        assert call_kwargs["wind_speed_ms"] == 2.0
        assert call_kwargs["cloud_cover_pct"] == 30.0
        assert call_kwargs["latitude"] == 13.0
        assert call_kwargs["longitude"] == 80.0
        assert call_kwargs["observation_time"] is None


# ---------------------------------------------------------------------------
# E. Failure isolation
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    """Each stage failure produces a controlled PipelineResult."""

    def test_detector_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        detector = _make_detector()
        detector.detect.side_effect = SentinelException("detector boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "detector boom" in result.error_message
        assert result.error_type == "SentinelException"

    def test_classifier_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        clf = _make_classifier()
        clf.classify.side_effect = SentinelException("classifier boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "classifier boom" in result.error_message

    def test_weather_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: (_ for _ in ()).throw(SentinelException("weather boom")))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "weather boom" in result.error_message

    def test_physics_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: (_ for _ in ()).throw(SentinelException("physics boom")))
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "physics boom" in result.error_message

    def test_feature_engineering_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: (_ for _ in ()).throw(FeatureValidationError("bad features")))
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "bad features" in result.error_message
        assert result.error_type == "FeatureValidationError"

    def test_predictor_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        predictor = _make_predictor()
        predictor.predict.side_effect = SentinelException("predictor boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "predictor boom" in result.error_message

    def test_recommendation_failure_returns_error_result(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: (_ for _ in ()).throw(SentinelException("rec boom")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "rec boom" in result.error_message

    def test_unexpected_exception_generic_message(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: (_ for _ in ()).throw(Exception("internal secret")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "unexpected error occurred" in result.error_message.lower()
        assert "internal secret" not in result.error_message
        assert result.error_type == "Exception"


# ---------------------------------------------------------------------------
# F. Weather degradation
# ---------------------------------------------------------------------------

class TestWeatherDegradation:
    """Pipeline continues when weather returns fallback/defaults."""

    def test_weather_fallback_does_not_crash_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        weather_fallback = MagicMock(fetch_successful=False, city="Chennai",
                                     ambient_temp_c=25.0, wind_speed_ms=2.0,
                                     cloud_cover_pct=0.0, latitude=0.0, longitude=0.0, timestamp=None)

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather_fallback)
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.weather_data is weather_fallback


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated pipeline runs with the same mocks produce equivalent results."""

    def test_repeated_runs_are_equivalent(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        r1 = run_pipeline(image=img, city="Chennai")
        r2 = run_pipeline(image=img, city="Chennai")
        assert r1.status == r2.status
        assert r1.city == r2.city
        assert r1.error_type == r2.error_type
        assert r1.error_message == r2.error_message

    def test_successful_run_does_not_mutate_input_image(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (120, 130, 140))
        original_mode = img.mode

        mm = _make_mock_model_manager()
        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: _make_predictor())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: MagicMock(fetch_successful=True, city=city))
        monkeypatch.setattr("services.pipeline.compute_physics", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.build_feature_dataframe", lambda **kw: MagicMock())
        monkeypatch.setattr("services.pipeline.generate_recommendations", lambda **kw: MagicMock(overall_severity=MagicMock(value="OK")))

        run_pipeline(image=img, city="Chennai")
        assert img.mode == original_mode