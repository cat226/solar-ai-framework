"""utils/exceptions.py — Domain-specific exception hierarchy.

All application-level errors derive from :class:`SolarAIError` so callers
can catch the base class when a broad safety net is needed, or the specific
subclass when precise error handling is required.

Usage example::

    from utils.exceptions import WeatherAPIError, ModelLoadError

    try:
        data = fetch_weather(city)
    except WeatherAPIError as exc:
        logger.warning("Weather unavailable: %s", exc)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class SolarAIError(Exception):
    """Base class for all Solar AI Framework exceptions."""


# ---------------------------------------------------------------------------
# Model exceptions
# ---------------------------------------------------------------------------

class ModelLoadError(SolarAIError):
    """Raised when a model file cannot be loaded.

    Covers missing weight files, corrupt serialisation, incompatible
    architectures, or missing ML library dependencies.

    Args:
        model_name: Human-readable model identifier (e.g. ``"YOLO"``).
        reason: Underlying cause description.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"[{model_name}] Model load failed: {reason}")


# ---------------------------------------------------------------------------
# Inference / prediction exceptions
# ---------------------------------------------------------------------------

class PredictionError(SolarAIError):
    """Raised when ML inference fails at runtime.

    This is distinct from :class:`ModelLoadError`; the model loaded
    successfully but the forward pass or prediction call raised an error.

    Args:
        model_name: Human-readable model identifier.
        reason: Underlying cause description.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"[{model_name}] Prediction failed: {reason}")


# ---------------------------------------------------------------------------
# Input validation exceptions
# ---------------------------------------------------------------------------

class ImageValidationError(SolarAIError):
    """Raised when an uploaded image fails validation checks.

    Args:
        reason: Human-readable description of the validation failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Image validation failed: {reason}")


class FeatureValidationError(SolarAIError):
    """Raised when a feature DataFrame fails schema or range validation.

    Args:
        reason: Human-readable description of the validation failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Feature validation failed: {reason}")


# ---------------------------------------------------------------------------
# External service exceptions
# ---------------------------------------------------------------------------

class WeatherAPIError(SolarAIError):
    """Raised when the OpenWeatherMap API call fails unrecoverably.

    For graceful degradation (e.g. timeout → use defaults), the weather
    service catches lower-level errors internally.  This exception is raised
    only when the failure is hard enough that the pipeline cannot continue.

    Args:
        city: The city name that was queried.
        reason: Underlying HTTP or parsing error description.
    """

    def __init__(self, city: str, reason: str) -> None:
        self.city = city
        self.reason = reason
        super().__init__(f"Weather API error for city '{city}': {reason}")
