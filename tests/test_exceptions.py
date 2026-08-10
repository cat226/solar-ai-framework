"""tests/test_exceptions.py - Direct tests for utils/exceptions.py.

Covers the domain-specific exception hierarchy:

A. Base SolarAIError
B. ModelLoadError
C. PredictionError
D. InputValidationError
E. ImageValidationError
F. FeatureValidationError
G. WeatherAPIError

Design rules honoured:
- no external dependencies
- deterministic constructor tests
- verifies exception hierarchy, attributes, and string representation
"""

from __future__ import annotations

import pytest

from utils.exceptions import (
    FeatureValidationError,
    ImageValidationError,
    InputValidationError,
    ModelLoadError,
    PredictionError,
    SolarAIError,
    WeatherAPIError,
)


# ---------------------------------------------------------------------------
# A. Base SolarAIError
# ---------------------------------------------------------------------------

class TestSolarAIError:
    """Base exception behaviour."""

    def test_is_exception_subclass(self):
        assert issubclass(SolarAIError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(SolarAIError):
            raise SolarAIError("base error")

    def test_message_preserved(self):
        exc = SolarAIError("something went wrong")
        assert str(exc) == "something went wrong"


# ---------------------------------------------------------------------------
# B. ModelLoadError
# ---------------------------------------------------------------------------

class TestModelLoadError:
    """Model loading failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(ModelLoadError, SolarAIError)

    def test_attributes_set(self):
        exc = ModelLoadError("YOLO", "weights missing")
        assert exc.model_name == "YOLO"
        assert exc.reason == "weights missing"

    def test_message_format(self):
        exc = ModelLoadError("MobileNet", "bad architecture")
        assert str(exc) == "[MobileNet] Model load failed: bad architecture"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ModelLoadError, match="YOLO"):
            raise ModelLoadError("YOLO", "boom")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise ModelLoadError("XGBoost", "no weights")


# ---------------------------------------------------------------------------
# C. PredictionError
# ---------------------------------------------------------------------------

class TestPredictionError:
    """Inference failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(PredictionError, SolarAIError)

    def test_attributes_set(self):
        exc = PredictionError("XGBoost", "NaN in output")
        assert exc.model_name == "XGBoost"
        assert exc.reason == "NaN in output"

    def test_message_format(self):
        exc = PredictionError("XGBoost", "inference crashed")
        assert str(exc) == "[XGBoost] Prediction failed: inference crashed"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PredictionError, match="XGBoost"):
            raise PredictionError("XGBoost", "boom")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise PredictionError("XGBoost", "boom")


# ---------------------------------------------------------------------------
# D. InputValidationError
# ---------------------------------------------------------------------------

class TestInputValidationError:
    """Input validation failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(InputValidationError, SolarAIError)

    def test_attributes_set(self):
        exc = InputValidationError("panel_age out of range")
        assert exc.reason == "panel_age out of range"

    def test_message_format(self):
        exc = InputValidationError("negative voltage")
        assert str(exc) == "Input validation failed: negative voltage"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(InputValidationError, match="panel_age"):
            raise InputValidationError("panel_age out of range")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise InputValidationError("bad input")


# ---------------------------------------------------------------------------
# E. ImageValidationError
# ---------------------------------------------------------------------------

class TestImageValidationError:
    """Image validation failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(ImageValidationError, SolarAIError)

    def test_attributes_set(self):
        exc = ImageValidationError("image is None")
        assert exc.reason == "image is None"

    def test_message_format(self):
        exc = ImageValidationError("unsupported format")
        assert str(exc) == "Image validation failed: unsupported format"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ImageValidationError, match="None"):
            raise ImageValidationError("image is None")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise ImageValidationError("bad image")


# ---------------------------------------------------------------------------
# F. FeatureValidationError
# ---------------------------------------------------------------------------

class TestFeatureValidationError:
    """Feature DataFrame validation failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(FeatureValidationError, SolarAIError)

    def test_attributes_set(self):
        exc = FeatureValidationError("missing column 'irradiance'")
        assert exc.reason == "missing column 'irradiance'"

    def test_message_format(self):
        exc = FeatureValidationError("NaN in features")
        assert str(exc) == "Feature validation failed: NaN in features"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(FeatureValidationError, match="missing"):
            raise FeatureValidationError("missing column 'irradiance'")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise FeatureValidationError("bad features")


# ---------------------------------------------------------------------------
# G. WeatherAPIError
# ---------------------------------------------------------------------------

class TestWeatherAPIError:
    """External weather API failure exception."""

    def test_is_solar_ai_error_subclass(self):
        assert issubclass(WeatherAPIError, SolarAIError)

    def test_attributes_set(self):
        exc = WeatherAPIError("Chennai", "HTTP 500")
        assert exc.city == "Chennai"
        assert exc.reason == "HTTP 500"

    def test_message_format(self):
        exc = WeatherAPIError("Mumbai", "connection timeout")
        assert str(exc) == "Weather API error for city 'Mumbai': connection timeout"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(WeatherAPIError, match="Chennai"):
            raise WeatherAPIError("Chennai", "API down")

    def test_caught_as_solar_ai_error(self):
        with pytest.raises(SolarAIError):
            raise WeatherAPIError("Chennai", "API down")

    @pytest.mark.parametrize(
        "city,reason",
        [
            ("Chennai", "HTTP 500"),
            ("Mumbai", "connection timeout"),
            ("Delhi", ""),
            ("Bangalore", "invalid API key"),
        ],
    )
    def test_various_cities_and_reasons(self, city, reason):
        exc = WeatherAPIError(city, reason)
        assert exc.city == city
        assert exc.reason == reason
        expected = f"Weather API error for city '{city}': {reason}"
        assert str(exc) == expected


# ---------------------------------------------------------------------------
# H. Hierarchy integrity
# ---------------------------------------------------------------------------

class TestHierarchy:
    """All exceptions derive from SolarAIError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ModelLoadError,
            PredictionError,
            InputValidationError,
            ImageValidationError,
            FeatureValidationError,
            WeatherAPIError,
        ],
    )
    def test_all_are_solar_ai_errors(self, exc_class):
        assert issubclass(exc_class, SolarAIError)
        assert issubclass(exc_class, Exception)
