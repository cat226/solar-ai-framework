"""tests/test_predictor.py - Deterministic tests for models/predictor.py.

Covers the XGBoost regression wrapper:

A. Initialization and model injection
B. Missing model handling
C. Successful prediction
D. Efficiency loss clamping
E. Estimated output calculation
F. Configuration behavior
G. Inference exception handling
H. Logging behavior
I. Determinism

Design rules honoured:
- no real XGBoost weights
- no network access
- deterministic mocks for joblib pipeline
- existing conftest fixtures preferred
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.predictor import EnergyPredictor, PredictionResult
from utils.config import CFG
from utils.exceptions import ModelLoadError, PredictionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pipeline():
    """Create a mock joblib pipeline with a predict method."""
    pipeline = MagicMock()
    pipeline.predict = MagicMock()
    return pipeline


def _run_prediction(predictor, features, mock_prediction):
    """Helper to run prediction with a controlled mock return value."""
    pipeline = _make_mock_pipeline()
    pipeline.predict.return_value = mock_prediction
    predictor.set_model(pipeline)
    return predictor.predict(features)


# ---------------------------------------------------------------------------
# A. Initialization and model injection
# ---------------------------------------------------------------------------

class TestInitializationAndInjection:
    """EnergyPredictor lifecycle basics."""

    def test_fresh_predictor_has_no_pipeline(self):
        predictor = EnergyPredictor()
        assert predictor._pipeline is None

    def test_set_model_stores_pipeline(self):
        predictor = EnergyPredictor()
        pipeline = _make_mock_pipeline()
        predictor.set_model(pipeline)
        assert predictor._pipeline is pipeline

    def test_set_model_none_raises_model_load_error(self):
        predictor = EnergyPredictor()
        with pytest.raises(ModelLoadError, match="XGBoost"):
            predictor.set_model(None)

    def test_set_model_none_does_not_corrupt_state(self):
        predictor = EnergyPredictor()
        with pytest.raises(ModelLoadError):
            predictor.set_model(None)
        assert predictor._pipeline is None


# ---------------------------------------------------------------------------
# B. Missing model handling
# ---------------------------------------------------------------------------

class TestMissingModelHandling:
    """predict() validates pipeline is set before inference."""

    def test_predict_before_set_model_raises_model_load_error(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        with pytest.raises(ModelLoadError, match="XGBoost"):
            predictor.predict(features)


# ---------------------------------------------------------------------------
# C. Successful prediction
# ---------------------------------------------------------------------------

class TestSuccessfulPrediction:
    """predict() returns PredictionResult with correct structure."""

    def test_predict_returns_prediction_result(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([5.0]))
        assert isinstance(result, PredictionResult)

    def test_predict_sets_successful_flag(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([5.0]))
        assert result.prediction_successful is True

    def test_predict_zero_loss_gives_full_rated_power(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([0.0]))
        
        rated_power = CFG["physics"]["panel_rated_power_wp"]
        assert result.estimated_output_w == pytest.approx(rated_power)

    def test_predict_100_percent_loss_gives_zero_output(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([100.0]))
        
        assert result.estimated_output_w == pytest.approx(0.0)

    def test_predict_calculates_output_proportionally(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([50.0]))
        
        rated_power = CFG["physics"]["panel_rated_power_wp"]
        expected_output = rated_power * (1.0 - 50.0 / 100.0)
        assert result.estimated_output_w == pytest.approx(expected_output)

    def test_predict_uses_configured_panel_rating(self, project_config):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([10.0]))
        
        rated_power = project_config["physics"]["panel_rated_power_wp"]
        expected_output = rated_power * (1.0 - 10.0 / 100.0)
        assert result.estimated_output_w == pytest.approx(expected_output)


# ---------------------------------------------------------------------------
# D. Efficiency loss clamping
# ---------------------------------------------------------------------------

class TestEfficiencyLossClamping:
    """predict() clamps efficiency loss to [0, 100]."""

    def test_negative_efficiency_loss_clamped_to_zero(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([-10.0]))
        
        assert result.efficiency_loss_pct == pytest.approx(0.0)
        assert result.estimated_output_w == pytest.approx(CFG["physics"]["panel_rated_power_wp"])

    def test_efficiency_loss_above_100_clamped_to_100(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([150.0]))
        
        assert result.efficiency_loss_pct == pytest.approx(100.0)
        assert result.estimated_output_w == pytest.approx(0.0)

    def test_efficiency_loss_within_range_preserved(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        result = _run_prediction(predictor, features, np.array([25.5]))
        
        assert result.efficiency_loss_pct == pytest.approx(25.5)


# ---------------------------------------------------------------------------
# E. Inference exception handling
# ---------------------------------------------------------------------------

class TestInferenceExceptionHandling:
    """Inference and post-processing errors are wrapped in PredictionError."""

    def test_pipeline_predict_exception_raises_prediction_error(self):
        predictor = EnergyPredictor()
        pipeline = _make_mock_pipeline()
        pipeline.predict.side_effect = RuntimeError("XGBoost inference failed")
        predictor.set_model(pipeline)
        
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        with pytest.raises(PredictionError, match="XGBoost") as exc_info:
            predictor.predict(features)
        
        assert "XGBoost inference failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    @pytest.mark.parametrize("bad_output,expected_match", [
        ("not_an_array", "Invalid prediction output"),
        (None, "Invalid prediction output"),
        ([], "Invalid prediction output"),
        ([float("nan")], "Prediction output must be finite"),
        ([float("inf")], "Prediction output must be finite"),
        ([float("-inf")], "Prediction output must be finite"),
    ])
    def test_invalid_prediction_output_raises_prediction_error(self, bad_output, expected_match):
        """Invalid prediction outputs are caught and wrapped in PredictionError.
        
        Covers:
        - Non-numeric string (ValueError from float())
        - None (TypeError from float())
        - Empty array (IndexError from [0])
        - NaN (silent corruption → explicit rejection)
        - +/-inf (silent corruption → explicit rejection)
        """
        predictor = EnergyPredictor()
        pipeline = _make_mock_pipeline()
        pipeline.predict.return_value = bad_output
        predictor.set_model(pipeline)
        
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        with pytest.raises(PredictionError, match=expected_match):
            predictor.predict(features)


# ---------------------------------------------------------------------------
# F. Logging behavior
# ---------------------------------------------------------------------------

class TestLoggingBehavior:
    """predict() emits expected log messages."""

    def test_info_log_before_prediction(self, caplog):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        
        import logging
        with caplog.at_level(logging.INFO):
            _run_prediction(predictor, features, np.array([5.0]))
        
        assert "Running XGBoost prediction" in caplog.text

    def test_info_log_after_successful_prediction(self, caplog):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        
        import logging
        with caplog.at_level(logging.INFO):
            _run_prediction(predictor, features, np.array([15.5]))
        
        assert "Prediction complete" in caplog.text
        assert "efficiency_loss=15.50%" in caplog.text
        assert "output=" in caplog.text


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated predictions with same input produce equivalent results."""

    def test_repeated_predictions_are_equivalent(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({"irradiance_wm2": [500.0]})
        
        r1 = _run_prediction(predictor, features, np.array([10.0]))
        r2 = _run_prediction(predictor, features, np.array([10.0]))
        
        assert r1.efficiency_loss_pct == r2.efficiency_loss_pct
        assert r1.estimated_output_w == r2.estimated_output_w
        assert r1.prediction_successful == r2.prediction_successful


# ---------------------------------------------------------------------------
# H. Feature DataFrame behavior
# ---------------------------------------------------------------------------

class TestFeatureDataFrameBehavior:
    """predict() accepts and logs the feature DataFrame."""

    def test_predict_accepts_feature_dataframe(self):
        predictor = EnergyPredictor()
        features = pd.DataFrame({
            "irradiance_wm2": [500.0],
            "module_temp_c": [45.0],
            "ambient_temp_c": [25.0],
            "humidity_pct": [50.0],
            "wind_speed_ms": [2.0],
            "cloud_cover_pct": [30.0],
            "soiling_ratio": [0.95],
            "fault_class_id": [0],
            "detection_confidence": [0.92],
        })
        
        result = _run_prediction(predictor, features, np.array([5.0]))
        assert result.prediction_successful is True

    def test_predict_with_different_feature_values(self):
        predictor = EnergyPredictor()
        
        # Test with different feature values to ensure they don't affect mock result
        features1 = pd.DataFrame({"irradiance_wm2": [300.0]})
        features2 = pd.DataFrame({"irradiance_wm2": [800.0]})
        
        r1 = _run_prediction(predictor, features1, np.array([5.0]))
        r2 = _run_prediction(predictor, features2, np.array([5.0]))
        
        # Same mock prediction should give same result regardless of features
        assert r1.efficiency_loss_pct == r2.efficiency_loss_pct