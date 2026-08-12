"""utils/config.py — YAML configuration loader and secret resolver.

Loads ``configs/settings.yaml`` once at import time and exposes the parsed
dictionary as :data:`CFG`.  All other modules should import ``CFG`` from
here instead of reading the YAML themselves.

Secret resolution
-----------------
Sensitive values (API keys, credentials) must **not** live in YAML.
Use :func:`get_secret` to read them in priority order:

1. ``st.secrets["OPENWEATHER_API_KEY"]``  — if running inside Streamlit
2. ``os.environ["OPENWEATHER_API_KEY"]``  — from a ``.env`` or shell export
3. *fallback* — optional default value supplied by the caller

To load ``.env`` automatically, install ``python-dotenv`` and call
``load_dotenv()`` at the top of ``app.py`` (before any imports that need
the env var).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Resolve path: settings.yaml lives at <project_root>/configs/settings.yaml
# This file lives at <project_root>/utils/config.py  → parent = project_root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_SETTINGS_PATH: Path = _PROJECT_ROOT / "configs" / "settings.yaml"


def load_config(path: Path = _SETTINGS_PATH) -> dict[str, Any]:
    """Load and return the YAML configuration file as a dictionary.

    Args:
        path: Absolute path to the YAML settings file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the settings file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
        ValueError: If required configuration keys are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Ensure configs/settings.yaml exists in the project root."
        )

    with open(path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    _validate_config_schema(config)
    return config


# ---------------------------------------------------------------------------
# Configuration validation helpers
# ---------------------------------------------------------------------------

def _require_keys(section: dict[str, Any], keys: list[str], where: str) -> None:
    """Raise ValueError if any key in *keys* is absent from *section*.

    Args:
        section: The configuration sub-dictionary to inspect.
        keys: Required key names.
        where: Dotted path label used in the error message (e.g. ``"models.yolo"``).

    Raises:
        ValueError: If *section* is not a mapping or any key is missing.
    """
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{where}' must be a mapping.")
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(f"Missing required configuration keys in '{where}': {missing}")


def _require_number(
    section: dict[str, Any],
    key: str,
    where: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> None:
    """Validate that ``section[key]`` is a number within an optional range.

    Args:
        section: The configuration sub-dictionary containing *key*.
        key: The key to validate.
        where: Dotted path label used in error messages.
        minimum: Inclusive lower bound, if any.
        maximum: Inclusive upper bound, if any.

    Raises:
        ValueError: If the value is missing, not numeric, or out of range.
    """
    if key not in section:
        raise ValueError(f"Missing required configuration key '{where}.{key}'.")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Configuration value '{where}.{key}' must be numeric, "
            f"got {type(value).__name__}."
        )
    if minimum is not None and value < minimum:
        raise ValueError(
            f"Configuration value '{where}.{key}' must be >= {minimum}, got {value}."
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"Configuration value '{where}.{key}' must be <= {maximum}, got {value}."
        )


def _validate_config_schema(config: dict[str, Any]) -> None:
    """Validate configuration structure, required keys, and numeric ranges.

    This checks, in order:

    1. All required top-level sections exist.
    2. Each section contains its required keys.
    3. Numeric values fall within reasonable ranges.
    4. The feature-engineering configuration is complete (non-empty column list).
    5. The model configuration is complete (weights + params for every model).

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If any structural, key, or range check fails.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping (parsed YAML dict).")

    # 1. Required top-level sections
    required_sections = [
        "weather", "models", "classification", "physics",
        "feature_engineering", "recommendations", "logging",
    ]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")

    # 2. Weather
    weather = config["weather"]
    _require_keys(
        weather,
        ["base_url", "timeout_seconds", "default_city", "units", "defaults"],
        "weather",
    )
    _require_number(weather, "timeout_seconds", "weather", minimum=1, maximum=300)
    if not str(weather["base_url"]).strip():
        raise ValueError("Configuration value 'weather.base_url' must be non-empty.")
    if not str(weather["default_city"]).strip():
        raise ValueError("Configuration value 'weather.default_city' must be non-empty.")
    _require_keys(
        weather["defaults"],
        ["ambient_temp_c", "humidity_pct", "wind_speed_ms",
         "cloud_cover_pct", "pressure_hpa", "latitude", "longitude"],
        "weather.defaults",
    )
    defaults = weather["defaults"]
    _require_number(defaults, "humidity_pct", "weather.defaults", minimum=0, maximum=100)
    _require_number(defaults, "cloud_cover_pct", "weather.defaults", minimum=0, maximum=100)
    _require_number(defaults, "wind_speed_ms", "weather.defaults", minimum=0)
    _require_number(defaults, "latitude", "weather.defaults", minimum=-90, maximum=90)
    _require_number(defaults, "longitude", "weather.defaults", minimum=-180, maximum=180)

    # 3. Models — completeness (weights + params for every model)
    models = config["models"]
    _require_keys(models, ["yolo", "mobilenet", "xgboost"], "models")
    _require_keys(
        models["yolo"],
        ["weights", "confidence_threshold", "iou_threshold", "image_size"],
        "models.yolo",
    )
    _require_number(models["yolo"], "confidence_threshold", "models.yolo",
                    minimum=0.0, maximum=1.0)
    _require_number(models["yolo"], "iou_threshold", "models.yolo",
                    minimum=0.0, maximum=1.0)
    _require_number(models["yolo"], "image_size", "models.yolo", minimum=1)
    _require_keys(
        models["mobilenet"], ["weights", "num_classes", "input_size"], "models.mobilenet",
    )
    _require_number(models["mobilenet"], "num_classes", "models.mobilenet", minimum=1)
    _require_number(models["mobilenet"], "input_size", "models.mobilenet", minimum=1)
    _require_keys(models["xgboost"], ["weights"], "models.xgboost")

    # 4. Classification labels
    labels = config["classification"].get("labels") if isinstance(
        config["classification"], dict) else None
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            "Configuration 'classification.labels' must be a non-empty list."
        )

    # 5. Physics — required constants used by services/physics.py
    physics = config["physics"]
    _require_keys(
        physics,
        ["max_irradiance_wm2", "noct_celsius",
         "noct_irradiance_ref", "noct_ambient_ref", "wind_cooling_coefficient",
         "temp_coefficient_pmax", "stc_temperature", "soiling_ratios",
         "panel_rated_power_wp"],
        "physics",
    )
    _require_number(physics, "max_irradiance_wm2", "physics", minimum=0)
    _require_number(physics, "noct_irradiance_ref", "physics", minimum=1)
    _require_number(physics, "panel_rated_power_wp", "physics", minimum=0)
    if not isinstance(physics["soiling_ratios"], dict) or not physics["soiling_ratios"]:
        raise ValueError(
            "Configuration 'physics.soiling_ratios' must be a non-empty mapping."
        )

    # 6. Feature engineering — completeness
    fe = config["feature_engineering"]
    _require_keys(fe, ["feature_columns"], "feature_engineering")
    columns = fe["feature_columns"]
    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "Configuration 'feature_engineering.feature_columns' must be a "
            "non-empty list."
        )
    if not all(isinstance(c, str) and c.strip() for c in columns):
        raise ValueError(
            "Configuration 'feature_engineering.feature_columns' must contain "
            "only non-empty strings."
        )

    # 7. Recommendations — threshold percentages
    rec = config["recommendations"]
    _require_keys(
        rec,
        ["efficiency_loss_critical_pct", "efficiency_loss_warning_pct",
         "hotspot_max_temp_c", "cleaning_humidity_threshold_pct"],
        "recommendations",
    )
    _require_number(rec, "efficiency_loss_critical_pct", "recommendations",
                    minimum=0, maximum=100)
    _require_number(rec, "efficiency_loss_warning_pct", "recommendations",
                    minimum=0, maximum=100)
    _require_number(rec, "cleaning_humidity_threshold_pct", "recommendations",
                    minimum=0, maximum=100)

    # 8. Logging
    _require_keys(config["logging"], ["level", "format"], "logging")


def get_secret(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve a secret value from Streamlit secrets or environment variables.

    Resolution order
    ----------------
    1. ``st.secrets[key]``  — available when running under Streamlit with a
       configured ``.streamlit/secrets.toml``.
    2. ``os.environ[key]``  — set by a ``.env`` file (loaded via
       ``python-dotenv``) or a shell ``export`` statement.
    3. *fallback* — returned as-is (may be ``None``).

    Args:
        key: Name of the secret (e.g. ``"OPENWEATHER_API_KEY"``).
        fallback: Value to return when the secret is not found anywhere.

    Returns:
        The resolved secret string, or *fallback* if not found.

    Example::

        from utils.config import get_secret
        api_key = get_secret("OPENWEATHER_API_KEY")
        if not api_key:
            raise WeatherAPIError("city", "API key not configured")
    """
    # 1. Try Streamlit secrets (only available inside a Streamlit process).
    #    A missing secrets file / not-in-Streamlit context is expected and
    #    handled by falling through to the environment lookup below.  We use
    #    the stdlib logger here (not utils.logger) to avoid a circular import:
    #    utils.logger imports CFG from this module.
    try:
        import streamlit as st  # type: ignore
        value = st.secrets.get(key)
        if value:
            return str(value)
    except ImportError:
        # Streamlit not installed / not running under Streamlit — expected.
        pass
    except Exception as exc:  # noqa: BLE001 — unexpected secrets-backend failure
        import logging
        logging.getLogger(__name__).debug(
            "Streamlit secrets lookup for '%s' failed (%s: %s); "
            "falling back to environment variable.",
            key, type(exc).__name__, exc,
        )

    # 2. Try environment variable (covers .env via python-dotenv)
    value = os.environ.get(key)
    if value:
        return value

    # 3. Fallback
    return fallback


# ---------------------------------------------------------------------------
# Module-level singleton — import CFG wherever config values are needed.
# ---------------------------------------------------------------------------
CFG: dict[str, Any] = load_config()
