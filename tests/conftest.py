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
# Deterministic image
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pil_image():
    """A small deterministic RGB image (no randomness, no network)."""
    from PIL import Image

    return Image.new("RGB", (224, 224), (120, 130, 140))
