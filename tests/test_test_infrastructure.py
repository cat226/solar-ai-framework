"""tests/test_test_infrastructure.py - Smoke tests for the pytest foundation.

These tests prove that pytest is configured correctly and that the shared
fixtures load in the canonical Python 3.12 environment.  They intentionally
exercise **infrastructure only** - not production behaviour.

No model weights, no network access and no external services are required.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pytest_configuration_present():
    """The pytest configuration and dev requirements files exist at root."""
    assert (PROJECT_ROOT / "pytest.ini").is_file()
    assert (PROJECT_ROOT / "requirements-dev.txt").is_file()


def test_project_layout_available():
    """The expected top-level project layout is present."""
    for rel in (
        "app.py",
        "configs/settings.yaml",
        "services",
        "models",
        "utils",
        "tests",
    ):
        assert (PROJECT_ROOT / rel).exists(), f"missing: {rel}"


def test_shared_fixtures_are_importable(
    project_root,
    project_config,
    fixed_utc_datetime,
    fixed_utc_midnight,
    default_weather,
    default_physics,
    classification_clean,
    detection_single_panel,
    prediction_normal,
    sample_pil_image,
):
    """Every shared fixture can be created from the canonical environment."""
    # Fixed clock
    assert fixed_utc_datetime.tzinfo is not None
    assert fixed_utc_midnight < fixed_utc_datetime

    # Parsed configuration carries the expected top-level sections
    assert "weather" in project_config
    assert "physics" in project_config

    # Domain data objects are well-formed
    assert default_weather.fetch_successful is True
    assert default_weather.ambient_temp_c == 25.0
    assert default_physics.soiling_ratio == 1.0
    assert classification_clean.label == "Clean"
    assert detection_single_panel.panel_count == 1
    assert prediction_normal.prediction_successful is True

    # Deterministic image
    assert sample_pil_image.mode == "RGB"
    assert sample_pil_image.size == (224, 224)

    # project_root resolves to the repository root
    assert (project_root / "app.py").is_file()


def test_coverage_runtime_is_available():
    """pytest-cov's coverage runtime is installed in this environment."""
    import coverage

    assert coverage.__version__
