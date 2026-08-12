"""tests/conftest.py - Shared deterministic fixtures for the pytest suite.

Design principles
-----------------
* **Deterministic** - fixed clock and fixed values, so results are
  reproducible across runs and machines (no date/time, weather, network or
  randomness).
* **Lazy imports** - heavy/optional libraries (e.g. torch) are imported
  *inside* fixture bodies so test collection stays lightweight and works in
  environments without every runtime dependency.
* **Reusable** - fixtures are generic building blocks for the Sprint 3.3
  module suites (physics, configuration, recommendation, images, features,
  weather, pipeline). Nothing here modifies production behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path of the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def project_config():
    """Parsed ``configs/settings.yaml`` (the CFG singleton)."""
    from utils.config import CFG

    return CFG


# ---------------------------------------------------------------------------
# Fixed clock
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_utc_datetime() -> datetime:
    """Fixed daytime UTC instant (2026-07-14 06:00 UTC)."""
    return datetime(2026, 7, 14, 6, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_utc_midnight() -> datetime:
    """Fixed night-time UTC instant (2026-07-14 00:00 UTC)."""
    return datetime(2026, 7, 14, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_utc_noon() -> datetime:
    """Fixed solar-noon UTC instant (2026-07-14 12:00 UTC)."""
    return datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Deterministic E2E weather
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def deterministic_e2e_weather(monkeypatch, request):
    """Give E2E weather factories a fixed observation time.

    ``tests/test_pipeline_e2e.py`` intentionally exercises the real physics
    stage. Its weather factory previously left ``timestamp`` as ``None``,
    which made ``calculate_irradiance`` depend on the CI runner's current
    clock. Keep the test deterministic without changing production behavior.

    The fixed 04:00 UTC observation corresponds to a realistic morning solar
    position in Chennai and, together with the 90% cloud cover scenario,
    remains below the test's low-irradiance threshold.
    """
    path = getattr(request, "path", None)
    if path is None or path.name != "test_pipeline_e2e.py":
        return

    module = request.module
    original_make_weather = module._make_weather
    fixed_observation = datetime(2026, 7, 14, 4, 0, 0, tzinfo=timezone.utc)

    def _deterministic_make_weather(city="Chennai", fetch_successful=True, **kwargs):
        kwargs.setdefault("timestamp", fixed_observation)
        return original_make_weather(
            city=city,
            fetch_successful=fetch_successful,
            **kwargs,
        )

    monkeypatch.setattr(module, "_make_weather", _deterministic_make_weather)


# ---------------------------------------------------------------------------
# Deterministic domain data objects
# ---------------------------------------------------------------------------


@pytest.fixture
def default_weather(fixed_utc_datetime):
    """A representative, fully-successful WeatherData observation."""
    from services.weather import WeatherData

    return WeatherData(
        city="Chennai",
        ambient_temp_c=25.0,
        humidity_pct=50.0,
        wind_speed_ms=2.0,
        cloud_cover_pct=30.0,
        pressure_hpa=1013.25,
        latitude=13.08,
        longitude=80.27,
        timestamp=fixed_utc_datetime,
        description="clear sky",
        fetch_successful=True,
    )


@pytest.fixture
def default_physics():
    """A deterministic, healthy PhysicsResult for pure downstream tests."""
    from services.physics import PhysicsResult

    return PhysicsResult(
        irradiance_wm2=888.0,
        module_temp_c=49.75,
        soiling_ratio=1.0,
        temp_loss_pct=9.9,
        effective_efficiency=0.901,
        cloud_factor=0.925,
        wind_cooling_factor=3.0,
    )


@pytest.fixture
def classification_clean():
    """A confident ``Clean`` classification result."""
    from models.classifier import ClassificationResult

    return ClassificationResult(
        label="Clean",
        class_id=0,
        confidence=0.95,
        probabilities={
            "Clean": 0.95,
            "Dusty": 0.02,
            "Bird-Drop": 0.01,
            "Electrical-Damage": 0.01,
            "Physical-Damage": 0.005,
            "Hotspot": 0.005,
        },
        classification_successful=True,
    )


@pytest.fixture
def detection_single_panel():
    """One detected panel with high confidence."""
    from models.detector import DetectionResult

    return DetectionResult(
        boxes=[[10.0, 10.0, 210.0, 190.0]],
        confidences=[0.92],
        class_ids=[0],
        panel_count=1,
        best_confidence=0.92,
        detection_successful=True,
    )


@pytest.fixture
def prediction_normal():
    """A normal (low-loss) prediction result."""
    from models.predictor import PredictionResult

    return PredictionResult(
        efficiency_loss_pct=2.0,
        estimated_output_w=392.0,
        prediction_successful=True,
    )


# ---------------------------------------------------------------------------
# Deterministic image
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pil_image():
    """A small deterministic RGB image (no randomness, no network)."""
    from PIL import Image

    return Image.new("RGB", (224, 224), (120, 130, 140))
