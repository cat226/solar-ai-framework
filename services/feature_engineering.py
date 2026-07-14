"""services/feature_engineering.py — ML feature DataFrame construction and validation.

Responsibility
--------------
- Accept structured outputs from the detector, classifier, weather service,
  and physics engine.
- Assemble them into a single-row :class:`pandas.DataFrame` whose columns
  match the feature schema expected by the XGBoost predictor.
- Validate the assembled DataFrame before it reaches the predictor.

Public API
----------
- :func:`build_features` — assemble the feature vector.
- :func:`validate_features` — verify schema, ranges, and missing values.
- :func:`build_feature_dataframe` — convenience wrapper: build then validate.

No prediction logic lives here — only data assembly and validation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from models.classifier import ClassificationResult
from models.detector import DetectionResult
from services.physics import PhysicsResult
from services.weather import WeatherData
from utils.config import CFG
from utils.exceptions import FeatureValidationError
from utils.logger import get_logger

logger = get_logger(__name__)

# Expected feature column order from config
_FEATURE_COLUMNS: list[str] = CFG["feature_engineering"]["feature_columns"]

# Map label string → integer id for the ML model
_LABELS: list[str] = CFG["classification"]["labels"]
_LABEL_TO_ID: dict[str, int] = {lbl: idx for idx, lbl in enumerate(_LABELS)}

# Valid numeric ranges per feature (inclusive) — used by validate_features()
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "irradiance_wm2":       (0.0,   1500.0),
    "module_temp_c":        (-40.0,  120.0),
    "ambient_temp_c":       (-40.0,   60.0),
    "humidity_pct":         (0.0,   100.0),
    "wind_speed_ms":        (0.0,    60.0),
    "cloud_cover_pct":      (0.0,   100.0),
    "soiling_ratio":        (0.0,     1.0),
    "fault_class_id":       (0.0,    10.0),
    "detection_confidence": (0.0,     1.0),
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_features(
    weather: WeatherData,
    physics: PhysicsResult,
    classification: ClassificationResult,
    detection: DetectionResult,
) -> pd.DataFrame:
    """Assemble the raw feature vector DataFrame for XGBoost inference.

    The resulting DataFrame has **one row** and column names matching the
    ``feature_engineering.feature_columns`` list in ``configs/settings.yaml``.
    Missing columns are filled with ``0.0`` and a warning is logged.

    Args:
        weather: Weather observation from :func:`services.weather.fetch_weather`.
        physics: Physics calculations from :func:`services.physics.compute_physics`.
        classification: Fault classification from the MobileNet classifier.
        detection: Panel detection result from the YOLO detector.

    Returns:
        Single-row :class:`pandas.DataFrame` with the feature columns in the
        order expected by the trained XGBoost pipeline.
    """
    fault_class_id: int = _LABEL_TO_ID.get(classification.label, 0)

    row: dict[str, float] = {
        "irradiance_wm2":       physics.irradiance_wm2,
        "module_temp_c":        physics.module_temp_c,
        "ambient_temp_c":       weather.ambient_temp_c,
        "humidity_pct":         weather.humidity_pct,
        "wind_speed_ms":        weather.wind_speed_ms,
        "cloud_cover_pct":      weather.cloud_cover_pct,
        "soiling_ratio":        physics.soiling_ratio,
        "fault_class_id":       float(fault_class_id),
        "detection_confidence": detection.best_confidence,
    }

    # Fill any schema gaps introduced by config changes
    missing = set(_FEATURE_COLUMNS) - set(row.keys())
    if missing:
        logger.warning(
            "Feature engineering: missing columns %s — filling with 0.0", missing
        )
        for col in missing:
            row[col] = 0.0

    df = pd.DataFrame([row])[_FEATURE_COLUMNS]

    logger.info("Feature vector assembled (%d features).", len(df.columns))
    logger.debug("Feature vector:\n%s", df.to_string())

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_features(df: pd.DataFrame) -> None:
    """Validate a feature DataFrame before it is passed to the predictor.

    Checks performed
    ----------------
    1. **Schema** — all expected columns are present.
    2. **Missing values** — no NaN values in any column.
    3. **Numeric ranges** — each feature value falls within its valid range.

    Args:
        df: Single-row feature DataFrame produced by :func:`build_features`.

    Raises:
        FeatureValidationError: If any check fails.  The error message
            describes the specific column and violation.
    """
    # 1. Schema check
    missing_cols = set(_FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise FeatureValidationError(
            f"DataFrame is missing required columns: {sorted(missing_cols)}"
        )

    # 2. Missing value check
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise FeatureValidationError(
            f"DataFrame contains NaN values in columns: {null_cols}"
        )

    # 3. Range checks
    row: dict[str, Any] = df.iloc[0].to_dict()
    violations: list[str] = []
    for col, (lo, hi) in _FEATURE_RANGES.items():
        if col not in row:
            continue
        val = float(row[col])
        if not (lo <= val <= hi):
            violations.append(
                f"  '{col}': {val:.4g} is outside [{lo}, {hi}]"
            )

    if violations:
        raise FeatureValidationError(
            "One or more feature values are out of valid range:\n"
            + "\n".join(violations)
        )

    logger.debug("Feature validation passed (%d features).", len(df.columns))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    weather: WeatherData,
    physics: PhysicsResult,
    classification: ClassificationResult,
    detection: DetectionResult,
) -> pd.DataFrame:
    """Build and validate the feature DataFrame in a single call.

    Combines :func:`build_features` and :func:`validate_features`.
    This is the function called by :mod:`services.pipeline`.

    Args:
        weather: Weather observation.
        physics: Physics calculation results.
        classification: Fault classification result.
        detection: Panel detection result.

    Returns:
        Validated single-row :class:`pandas.DataFrame`.

    Raises:
        FeatureValidationError: If validation fails.
    """
    df = build_features(weather, physics, classification, detection)
    validate_features(df)
    return df
