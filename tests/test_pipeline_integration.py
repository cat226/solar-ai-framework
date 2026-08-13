"""tests/test_pipeline_integration.py - Integration tests for services/pipeline.py.

These tests verify the REAL pipeline orchestration with real lightweight
production stages and mocked external/heavy boundaries.

Real implementations used:
- Input validation (_validate_scalar_inputs)
- Image validation and RGB conversion
- services.physics.compute_physics
- services.feature_engineering.build_feature_dataframe
- services.recommendation.generate_recommendations
- Pipeline orchestration and PipelineResult assembly

Mocked boundaries:
- Model inference (detector, classifier, predictor)
- External weather API (fetch_weather)

Design rules:
- deterministic
- no network
- no real model weights
- no GPU required
- realistic domain objects from production code
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from services.pipeline import run_pipeline
from services.recommendation import Severity
from utils.exceptions import SolarAIError


# ---------------------------------------------------------------------------
# Sentinel exception for integration tests
# ---------------------------------------------------------------------------

class SentinelIntegrationException(SolarAIError):
    """Distinctive SolarAIError subclass for identifying stage failures."""
    pass


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

def _make_detection(panel_count=1, confidence=0.9):
    from models.detector import DetectionResult
    return DetectionResult(
        boxes=[[10.0, 10.0, 210.0, 190.0]],
        confidences=[confidence],
        class_ids=[0],
        panel_count=panel_count,
        best_confidence=confidence,
        detection_successful=True,
    )


def _make_classification(label="Clean", confidence=0.95):
    from models.classifier import ClassificationResult
    return ClassificationResult(
        label=label,
        class_id=0,
        confidence=confidence,
        probabilities={
            "Clean": confidence,
            "Dusty": 0.02,
            "Bird-Drop": 0.01,
            "Electrical-Damage": 0.01,
            "Physical-Damage": 0.005,
            "Hotspot": 0.005,
        },
        classification_successful=True,
    )


def _make_weather(city="Chennai", fetch_successful=True, **kwargs):
    from services.weather import WeatherData
    defaults = {
        "ambient_temp_c": 25.0,
        "humidity_pct": 50.0,
        "wind_speed_ms": 2.0,
        "cloud_cover_pct": 30.0,
        "pressure_hpa": 1013.25,
        "latitude": 13.08,
        "longitude": 80.27,
        "timestamp": None,
        "description": "clear sky",
    }
    defaults.update(kwargs)
    return WeatherData(
        city=city,
        fetch_successful=fetch_successful,
        **defaults,
    )


def _make_prediction(loss_pct=5.0, output_w=380.0):
    from models.predictor import PredictionResult
    return PredictionResult(
        efficiency_loss_pct=loss_pct,
        estimated_output_w=output_w,
        prediction_successful=True,
    )


def _make_recommendation(severity="OK", summary="Panel is healthy"):
    from services.recommendation import RecommendationReport
    return RecommendationReport(
        recommendations=[],
        overall_severity=Severity(severity),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Mock factories with set_model support
# ---------------------------------------------------------------------------

def _make_mock_detector(detection_result=None):
    detector = MagicMock()
    detector.detect.return_value = detection_result or _make_detection()
    return detector


def _make_mock_classifier(classification_result=None):
    clf = MagicMock()
    clf.classify.return_value = classification_result or _make_classification()
    return clf


def _make_mock_predictor(prediction_result=None):
    predictor = MagicMock()
    predictor.predict.return_value = prediction_result or _make_prediction()
    return predictor


# ---------------------------------------------------------------------------
# Patching helper
# ---------------------------------------------------------------------------

def _patch_external(monkeypatch, detection=None, classification=None,
                    weather=None, prediction=None):
    """Patch only external/heavy boundaries; real stages use production code."""
    mm = MagicMock()
    mm.get_detector.return_value = MagicMock()
    mm.get_classifier.return_value = MagicMock()
    mm.get_predictor.return_value = MagicMock()

    detector = _make_mock_detector(detection)
    clf = _make_mock_classifier(classification)
    predictor = _make_mock_predictor(prediction)
    weather_data = weather or _make_weather()

    monkeypatch.setattr("services.pipeline.model_manager", mm)
    monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
    monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
    monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
    monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: weather_data)
    # Real implementations for physics, features, recommendations are used


# ---------------------------------------------------------------------------
# A. Happy path — real multi-stage integration
# ---------------------------------------------------------------------------

class TestPipelineHappyPath:
    """End-to-end pipeline with real physics/features/recommendations."""

    def test_successful_pipeline_returns_success_status(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"

    def test_successful_pipeline_populates_all_real_stages(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        detection = _make_detection(panel_count=2, confidence=0.88)
        classification = _make_classification(label="Dusty", confidence=0.85)
        weather = _make_weather(city="Chennai")
        prediction = _make_prediction(loss_pct=15.0, output_w=340.0)

        _patch_external(monkeypatch, detection=detection, classification=classification,
                        weather=weather, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.detection_result is detection
        assert result.classification_result is classification
        assert result.weather_data is weather
        assert result.physics_data is not None
        assert result.feature_dataframe is not None
        assert result.efficiency_prediction is prediction
        assert result.recommendations is not None
        assert result.recommendations.overall_severity in (Severity.WARNING, Severity.CRITICAL)

    def test_real_physics_propagates_to_features(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        weather = _make_weather(city="Chennai")
        _patch_external(monkeypatch, weather=weather)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.physics_data is not None
        assert result.feature_dataframe is not None
        assert "irradiance_wm2" in result.feature_dataframe.columns
        assert "module_temp_c" in result.feature_dataframe.columns
        assert result.feature_dataframe["irradiance_wm2"].iloc[0] == pytest.approx(
            result.physics_data.irradiance_wm2
        )

    def test_real_recommendations_reflect_prediction(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        prediction = _make_prediction(loss_pct=35.0, output_w=260.0)
        _patch_external(monkeypatch, prediction=prediction)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.recommendations is not None
        assert result.recommendations.overall_severity in (Severity.WARNING, Severity.CRITICAL)

    def test_deterministic_runs_produce_equivalent_results(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        _patch_external(monkeypatch)

        r1 = run_pipeline(image=img, city="Chennai")
        r2 = run_pipeline(image=img, city="Chennai")
        assert r1.status == r2.status == "SUCCESS"
        assert r1.physics_data.irradiance_wm2 == r2.physics_data.irradiance_wm2
        assert r1.physics_data.module_temp_c == r2.physics_data.module_temp_c
        assert r1.feature_dataframe.equals(r2.feature_dataframe)

    def test_input_image_not_mutated(self, monkeypatch):
        img = Image.new("RGB", (224, 224), (120, 130, 140))
        original_mode = img.mode
        _patch_external(monkeypatch)

        run_pipeline(image=img, city="Chennai")
        assert img.mode == original_mode


# ---------------------------------------------------------------------------
# B. Weather degradation with real downstream stages
# ---------------------------------------------------------------------------

class TestPipelineWeatherDegradation:
    """Pipeline continues with real stages when weather falls back to defaults."""

    def test_weather_fallback_runs_real_physics_and_features(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        weather_fallback = _make_weather(fetch_successful=False)
        _patch_external(monkeypatch, weather=weather_fallback)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.weather_data is weather_fallback
        assert result.physics_data is not None
        assert result.feature_dataframe is not None
        assert result.recommendations is not None


# ---------------------------------------------------------------------------
# C. Failure propagation with real orchestration
# ---------------------------------------------------------------------------

class TestPipelineFailurePropagation:
    """Stage failures produce controlled PipelineResult through real orchestration."""

    def test_detection_failure_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        detector = MagicMock()
        detector.detect.side_effect = SentinelIntegrationException("detector boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "detector boom" in result.error_message
        assert result.error_type == "SentinelIntegrationException"

    def test_classification_failure_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        clf = MagicMock()
        clf.classify.side_effect = SentinelIntegrationException("classifier boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_mock_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "classifier boom" in result.error_message

    def test_prediction_failure_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        predictor = MagicMock()
        predictor.predict.side_effect = SentinelIntegrationException("predictor boom")

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_mock_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_mock_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
        assert "predictor boom" in result.error_message


# ---------------------------------------------------------------------------
# D. Data-flow verification
# ---------------------------------------------------------------------------

class TestPipelineDataFlow:
    """Realistic values propagate correctly between real stages."""

    def test_weather_label_flows_into_physics_soiling(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        weather = _make_weather(city="Chennai")
        classification = _make_classification(label="Dusty")
        _patch_external(monkeypatch, weather=weather, classification=classification)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.physics_data.soiling_ratio < 1.0

    def test_physics_values_flow_into_features(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        weather = _make_weather(city="Chennai")
        _patch_external(monkeypatch, weather=weather)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        features = result.feature_dataframe
        assert "irradiance_wm2" in features.columns
        assert "module_temp_c" in features.columns
        assert features["irradiance_wm2"].iloc[0] == pytest.approx(
            result.physics_data.irradiance_wm2
        )
        assert features["module_temp_c"].iloc[0] == pytest.approx(
            result.physics_data.module_temp_c
        )

    def test_detection_confidence_flows_into_features(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        detection = _make_detection(confidence=0.95)
        _patch_external(monkeypatch, detection=detection)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"
        assert result.feature_dataframe["detection_confidence"].iloc[0] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# E. Input validation through real entry point
# ---------------------------------------------------------------------------

class TestPipelineInputValidation:
    """Invalid inputs rejected through run_pipeline() with real validation."""

    def test_none_image_returns_error(self):
        result = run_pipeline(image=None)
        assert result.status == "ERROR"
        assert result.error_type == "ImageValidationError"

    def test_non_pil_image_returns_error(self):
        result = run_pipeline(image="not-an-image")
        assert result.status == "ERROR"
        assert result.error_type == "ImageValidationError"

    def test_invalid_scalar_inputs_returns_error(self):
        img = Image.new("RGB", (10, 10))
        result = run_pipeline(image=img, panel_age=-1.0)
        assert result.status == "ERROR"
        assert result.error_type == "InputValidationError"

    def test_rgba_image_converts_and_succeeds(self, monkeypatch):
        img = Image.new("RGBA", (224, 224), (255, 0, 0, 128))
        _patch_external(monkeypatch)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"


# ---------------------------------------------------------------------------
# F. Complete cross-stage data-flow contract
# ---------------------------------------------------------------------------

class TestPipelineCrossStageDataFlow:
    """Every upstream field reaches the expected downstream consumer."""

    def test_complete_feature_vector_matches_upstream_sources(self, monkeypatch):
        img = Image.new("RGB", (224, 224))

        weather = _make_weather(
            city="Chennai",
            ambient_temp_c=30.0,
            humidity_pct=65.0,
            wind_speed_ms=3.5,
            cloud_cover_pct=40.0,
        )
        detection = _make_detection(panel_count=3, confidence=0.92)
        classification = _make_classification(label="Dusty", confidence=0.88)

        _patch_external(monkeypatch,
                        weather=weather,
                        detection=detection,
                        classification=classification)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"

        features = result.feature_dataframe
        row = features.iloc[0]

        assert row["irradiance_wm2"] == pytest.approx(result.physics_data.irradiance_wm2)
        assert row["module_temp_c"] == pytest.approx(result.physics_data.module_temp_c)
        assert row["ambient_temp_c"] == pytest.approx(weather.ambient_temp_c)
        assert row["humidity_pct"] == pytest.approx(weather.humidity_pct)
        assert row["wind_speed_ms"] == pytest.approx(weather.wind_speed_ms)
        assert row["cloud_cover_pct"] == pytest.approx(weather.cloud_cover_pct)
        assert row["soiling_ratio"] == pytest.approx(result.physics_data.soiling_ratio)
        assert row["detection_confidence"] == pytest.approx(detection.best_confidence)

        expected_fault_id = {"Clean": 0, "Dusty": 1, "Bird-Drop": 2,
                             "Electrical-Damage": 3, "Physical-Damage": 4, "Hotspot": 5}
        assert row["fault_class_id"] == pytest.approx(expected_fault_id[classification.label])

    def test_derived_features_computed_before_strict_schema_strip(self, monkeypatch):
        """Intermediate feature DataFrame includes derived columns; final output is strict schema."""
        from services.feature_engineering import build_features

        img = Image.new("RGB", (224, 224))
        weather = _make_weather(
            city="Chennai",
            ambient_temp_c=30.0,
            humidity_pct=65.0,
            wind_speed_ms=3.5,
            cloud_cover_pct=40.0,
        )
        detection = _make_detection(panel_count=3, confidence=0.92)
        classification = _make_classification(label="Dusty", confidence=0.88)

        _patch_external(monkeypatch,
                        weather=weather,
                        detection=detection,
                        classification=classification)

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "SUCCESS"

        raw_features = build_features(
            weather=result.weather_data,
            physics=result.physics_data,
            classification=result.classification_result,
            detection=result.detection_result,
        )
        assert "temperature_difference_c" in raw_features.columns
        assert "cloud_factor" in raw_features.columns
        assert "wind_cooling_factor" in raw_features.columns

        row = raw_features.iloc[0]
        assert row["temperature_difference_c"] == pytest.approx(
            result.physics_data.module_temp_c - weather.ambient_temp_c
        )
        assert row["cloud_factor"] == pytest.approx(result.physics_data.cloud_factor)
        assert row["wind_cooling_factor"] == pytest.approx(result.physics_data.wind_cooling_factor)


# ---------------------------------------------------------------------------
# G. Malformed model output propagation
# ---------------------------------------------------------------------------

class TestMalformedModelOutputPropagation:
    """Malformed model outputs are caught by validation and stop the pipeline."""

    def test_detector_nan_confidence_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        detector = MagicMock()
        from models.detector import DetectionResult
        detector.detect.return_value = DetectionResult(
            boxes=[[10.0, 10.0, 210.0, 190.0]],
            confidences=[float("nan")],
            class_ids=[0],
            panel_count=1,
            best_confidence=float("nan"),
            detection_successful=True,
        )

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: detector)
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"

    def test_classifier_nan_probability_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        clf = MagicMock()
        from models.classifier import ClassificationResult
        clf.classify.return_value = ClassificationResult(
            label="Clean",
            class_id=0,
            confidence=float("nan"),
            probabilities={"Clean": float("nan")},
            classification_successful=True,
        )

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_mock_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: clf)
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: MagicMock())
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"

    def test_predictor_nan_output_stops_pipeline(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        mm = MagicMock()
        predictor = MagicMock()
        from models.predictor import PredictionResult
        predictor.predict.return_value = PredictionResult(
            efficiency_loss_pct=float("nan"),
            estimated_output_w=float("nan"),
            prediction_successful=True,
        )

        monkeypatch.setattr("services.pipeline.model_manager", mm)
        monkeypatch.setattr("services.pipeline.SolarPanelDetector", lambda: _make_mock_detector())
        monkeypatch.setattr("services.pipeline.SolarFaultClassifier", lambda: _make_mock_classifier())
        monkeypatch.setattr("services.pipeline.EnergyPredictor", lambda: predictor)
        monkeypatch.setattr("services.pipeline.fetch_weather", lambda city: _make_weather())

        result = run_pipeline(image=img, city="Chennai")
        assert result.status == "ERROR"
