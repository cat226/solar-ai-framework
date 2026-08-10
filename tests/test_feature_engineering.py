"""tests/test_feature_engineering.py - Deterministic unit tests for
``services/feature_engineering.py`` (Sprint 3.3.5).

Scope
-----
Purely a characterization / regression suite for the CURRENT implementation:

* basic feature-vector construction (row count, columns, ordering, values)
* the strict XGBoost feature schema (``feature_engineering.feature_columns``)
* propagation of environmental / physics / vision inputs
* derived features (temperature difference, cloud factor, wind cooling)
* label -> ``fault_class_id`` mapping (incl. unknown-label fallback)
* schema-gap fill behaviour (missing columns filled with 0.0)
* ``validate_features`` happy paths, inclusive boundaries, ranges, NaN/inf
  and numeric-consistency checks
* ``build_feature_dataframe`` model-boundary stripping
* determinism and edge cases

Design rules honoured:
* deterministic, isolated, fast, model-free, weight-free, network-free.
  No Streamlit / YOLO / MobileNet / XGBoost / OpenWeatherMap is invoked.
* expected values are derived from the CURRENT implementation and live
  configuration (``configs/settings.yaml``) - never invented.
* the feature schema and ranges are read from the module under test
  (``_FEATURE_COLUMNS`` / ``_FEATURE_RANGES``) so the tests follow the
  implementation rather than guessing.

Heavy-import note:
* importing ``services.feature_engineering`` pulls in ``models.classifier``
  (torch) and ``models.detector`` (numpy/PIL) at collection time. This mirrors
  the production import chain and is not redesigned for test speed.

Coverage target: 100% statement/branch of ``services/feature_engineering.py``.
"""

from __future__ import annotations

import inspect
import logging

import pandas as pd
import pytest

from models.classifier import ClassificationResult
from models.detector import DetectionResult
from services.feature_engineering import (
    _FEATURE_COLUMNS,
    _FEATURE_RANGES,
    _LABEL_TO_ID,
    build_feature_dataframe,
    build_features,
    validate_features,
)
from services.physics import PhysicsResult
from services.weather import WeatherData
from utils.config import CFG
from utils.exceptions import FeatureValidationError, SolarAIError

# ---------------------------------------------------------------------------
# Contract constants - read from the module under test / live config.
# ---------------------------------------------------------------------------

# The strict model schema (9 columns) expected by the XGBoost predictor.
FEATURE_COLUMNS = _FEATURE_COLUMNS

# The 12 validated feature ranges (inclusive).
FEATURE_RANGES = _FEATURE_RANGES

# Derived features assembled by build_features but stripped for the model.
DERIVED_COLUMNS = [
    "temperature_difference_c",
    "cloud_factor",
    "wind_cooling_factor",
]

# Exact assembly order returned by build_features (row-dict insertion order).
FULL_COLUMN_ORDER = [
    "irradiance_wm2",
    "module_temp_c",
    "ambient_temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "cloud_cover_pct",
    "soiling_ratio",
    "fault_class_id",
    "detection_confidence",
    "temperature_difference_c",
    "cloud_factor",
    "wind_cooling_factor",
]


# ---------------------------------------------------------------------------
# Deterministic domain-object factories (no network, no randomness).
# ---------------------------------------------------------------------------

def _make_weather(**overrides):
    base = {
        "city": "Chennai", "ambient_temp_c": 25.0, "humidity_pct": 50.0,
        "wind_speed_ms": 2.0, "cloud_cover_pct": 30.0, "pressure_hpa": 1013.25,
        "latitude": 13.08, "longitude": 80.27, "description": "clear sky",
        "fetch_successful": True,
    }
    base.update(overrides)
    return WeatherData(**base)


def _make_physics(**overrides):
    base = {
        "irradiance_wm2": 888.0, "module_temp_c": 49.75, "soiling_ratio": 1.0,
        "temp_loss_pct": 9.9, "effective_efficiency": 0.901,
        "cloud_factor": 0.925, "wind_cooling_factor": 3.0,
    }
    base.update(overrides)
    return PhysicsResult(**base)


def _make_classification(**overrides):
    base = {
        "label": "Clean", "class_id": 0, "confidence": 0.95,
        "classification_successful": True,
    }
    base.update(overrides)
    return ClassificationResult(**base)


def _make_detection(**overrides):
    base = {
        "boxes": [[10.0, 10.0, 210.0, 190.0]], "confidences": [0.92],
        "class_ids": [0], "panel_count": 1, "best_confidence": 0.92,
        "detection_successful": True,
    }
    base.update(overrides)
    return DetectionResult(**base)


def make_inputs(weather=None, physics=None, classification=None, detection=None):
    """Four typed domain objects with deterministic, validation-safe values."""
    return (
        weather if weather is not None else _make_weather(),
        physics if physics is not None else _make_physics(),
        classification if classification is not None else _make_classification(),
        detection if detection is not None else _make_detection(),
    )


def make_valid_row(**overrides):
    """A full 12-feature row that passes ``validate_features``."""
    row = {
        "irradiance_wm2": 888.0,
        "module_temp_c": 49.75,
        "ambient_temp_c": 25.0,
        "humidity_pct": 50.0,
        "wind_speed_ms": 2.0,
        "cloud_cover_pct": 30.0,
        "soiling_ratio": 1.0,
        "fault_class_id": 0.0,
        "detection_confidence": 0.92,
        "temperature_difference_c": 24.75,
        "cloud_factor": 0.925,
        "wind_cooling_factor": 3.0,
    }
    row.update(overrides)
    return row


def make_valid_df(**overrides):
    """Single-row DataFrame mirroring the output of ``build_features``."""
    return pd.DataFrame([make_valid_row(**overrides)])


# ---------------------------------------------------------------------------
# 1. Basic feature construction
# ---------------------------------------------------------------------------

class TestBasicConstruction:
    """``build_features`` produces a single-row 12-column feature vector."""

    def test_returns_single_row_dataframe(self):
        df = build_features(*make_inputs())
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 12)

    def test_all_assembled_columns_present(self):
        df = build_features(*make_inputs())
        assert set(df.columns) == set(FULL_COLUMN_ORDER)

    def test_column_order_matches_assembly_order(self):
        df = build_features(*make_inputs())
        assert list(df.columns) == FULL_COLUMN_ORDER

    def test_environmental_values_transferred(self):
        df = build_features(*make_inputs())
        row = df.iloc[0]
        assert row["ambient_temp_c"] == pytest.approx(25.0)
        assert row["humidity_pct"] == pytest.approx(50.0)
        assert row["wind_speed_ms"] == pytest.approx(2.0)
        assert row["cloud_cover_pct"] == pytest.approx(30.0)

    def test_physics_values_transferred(self):
        df = build_features(*make_inputs())
        row = df.iloc[0]
        assert row["irradiance_wm2"] == pytest.approx(888.0)
        assert row["module_temp_c"] == pytest.approx(49.75)
        assert row["soiling_ratio"] == pytest.approx(1.0)
        assert row["cloud_factor"] == pytest.approx(0.925)
        assert row["wind_cooling_factor"] == pytest.approx(3.0)

    def test_detection_confidence_transferred(self):
        df = build_features(*make_inputs())
        assert df.loc[0, "detection_confidence"] == pytest.approx(0.92)

    def test_clean_label_maps_to_zero(self):
        df = build_features(*make_inputs())
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)

    def test_temperature_difference_derived(self):
        # module 49.75 - ambient 25.0 = 24.75
        df = build_features(*make_inputs())
        assert df.loc[0, "temperature_difference_c"] == pytest.approx(24.75)

    def test_pressure_is_not_consumed(self):
        df = build_features(*make_inputs())
        assert "pressure_hpa" not in df.columns

    def test_temp_loss_and_effective_efficiency_not_consumed(self):
        df = build_features(*make_inputs())
        assert "temp_loss_pct" not in df.columns
        assert "effective_efficiency" not in df.columns


# ---------------------------------------------------------------------------
# 2. Feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    """The strict model schema mirrors config; derived columns stay out."""

    def test_strict_schema_matches_config(self, project_config):
        assert FEATURE_COLUMNS == project_config["feature_engineering"][
            "feature_columns"
        ]

    def test_strict_schema_has_nine_features(self):
        assert len(FEATURE_COLUMNS) == 9

    def test_derived_columns_not_in_strict_schema(self):
        for col in DERIVED_COLUMNS:
            assert col not in FEATURE_COLUMNS

    def test_strict_schema_is_subset_of_assembled_columns(self):
        assert set(FEATURE_COLUMNS) <= set(FULL_COLUMN_ORDER)

    def test_final_dataframe_schema_exactly_matches_strict(self):
        final = build_feature_dataframe(*make_inputs())
        assert list(final.columns) == FEATURE_COLUMNS
        assert len(final.columns) == len(FEATURE_COLUMNS)

    def test_no_duplicate_columns(self):
        df = build_features(*make_inputs())
        assert len(df.columns) == len(set(df.columns))


# ---------------------------------------------------------------------------
# 3. Environmental features
# ---------------------------------------------------------------------------

class TestEnvironmentalFeatures:
    """Weather values propagate into the feature vector."""

    def test_custom_weather_values_propagate(self):
        weather = WeatherData(ambient_temp_c=31.5, humidity_pct=62.0,
                              wind_speed_ms=7.5, cloud_cover_pct=85.0)
        df = build_features(*make_inputs(weather=weather))
        row = df.iloc[0]
        assert row["ambient_temp_c"] == pytest.approx(31.5)
        assert row["humidity_pct"] == pytest.approx(62.0)
        assert row["wind_speed_ms"] == pytest.approx(7.5)
        assert row["cloud_cover_pct"] == pytest.approx(85.0)

    def test_weather_defaults_used_when_unset(self):
        # WeatherData() applies the config ``weather.defaults`` section.
        df = build_features(*make_inputs(weather=WeatherData()))
        row = df.iloc[0]
        assert row["ambient_temp_c"] == pytest.approx(25.0)
        assert row["humidity_pct"] == pytest.approx(50.0)
        assert row["wind_speed_ms"] == pytest.approx(2.0)
        assert row["cloud_cover_pct"] == pytest.approx(0.0)

    def test_negative_ambient_temperature_propagates(self):
        weather = WeatherData(ambient_temp_c=-35.0)
        df = build_features(*make_inputs(weather=weather))
        assert df.loc[0, "ambient_temp_c"] == pytest.approx(-35.0)


# ---------------------------------------------------------------------------
# 4. Physics features
# ---------------------------------------------------------------------------

class TestPhysicsFeatures:
    """Physics outputs propagate into the feature vector."""

    def test_custom_physics_values_propagate(self):
        physics = PhysicsResult(irradiance_wm2=700.5, module_temp_c=55.2,
                                soiling_ratio=0.85, cloud_factor=0.8,
                                wind_cooling_factor=9.0)
        df = build_features(*make_inputs(physics=physics))
        row = df.iloc[0]
        assert row["irradiance_wm2"] == pytest.approx(700.5)
        assert row["module_temp_c"] == pytest.approx(55.2)
        assert row["soiling_ratio"] == pytest.approx(0.85)
        assert row["cloud_factor"] == pytest.approx(0.8)
        assert row["wind_cooling_factor"] == pytest.approx(9.0)

    def test_physics_defaults_used_when_unset(self):
        df = build_features(*make_inputs(physics=PhysicsResult()))
        row = df.iloc[0]
        assert row["irradiance_wm2"] == pytest.approx(0.0)
        assert row["module_temp_c"] == pytest.approx(25.0)
        assert row["soiling_ratio"] == pytest.approx(1.0)
        assert row["cloud_factor"] == pytest.approx(1.0)
        assert row["wind_cooling_factor"] == pytest.approx(0.0)

    def test_temp_loss_and_effective_efficiency_not_consumed(self):
        physics = PhysicsResult(temp_loss_pct=42.0, effective_efficiency=0.123)
        df = build_features(*make_inputs(physics=physics))
        assert "temp_loss_pct" not in df.columns
        assert "effective_efficiency" not in df.columns


# ---------------------------------------------------------------------------
# 5. Asset / user features
# ---------------------------------------------------------------------------

class TestAssetUserFeatures:
    """No panel-age / maintenance / electrical fields are consumed today."""

    def test_no_asset_or_user_fields_in_schema(self):
        for col in ("panel_age_years", "maintenance_count", "installation_type",
                    "voltage_v", "current_a"):
            assert col not in FEATURE_COLUMNS
            assert col not in FULL_COLUMN_ORDER

    def test_build_features_accepts_only_typed_domain_inputs(self):
        sig = inspect.signature(build_features)
        assert list(sig.parameters) == [
            "weather", "physics", "classification", "detection",
        ]


# ---------------------------------------------------------------------------
# 6. Vision features
# ---------------------------------------------------------------------------

class TestVisionFeatures:
    """Only ``best_confidence`` is consumed from the detection result."""

    def test_best_confidence_used(self):
        detection = DetectionResult(best_confidence=0.55)
        df = build_features(*make_inputs(detection=detection))
        assert df.loc[0, "detection_confidence"] == pytest.approx(0.55)

    def test_detection_defaults_confidence_zero(self):
        df = build_features(*make_inputs(detection=DetectionResult()))
        assert df.loc[0, "detection_confidence"] == pytest.approx(0.0)

    def test_detection_geometry_fields_not_consumed(self):
        detection = DetectionResult(boxes=[[1.0, 2.0, 3.0, 4.0]],
                                    confidences=[0.5], class_ids=[7],
                                    panel_count=3)
        df = build_features(*make_inputs(detection=detection))
        for col in ("boxes", "confidences", "class_ids", "panel_count",
                    "detection_successful"):
            assert col not in df.columns

    def test_classification_confidence_not_consumed(self):
        classification = ClassificationResult(label="Dusty", class_id=1,
                                              confidence=0.99)
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(1.0)
        assert "classification_confidence" not in df.columns
        assert "class_id" not in df.columns
        assert "probabilities" not in df.columns


# ---------------------------------------------------------------------------
# 7. Derived features
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    """Temperature difference is computed; cloud/wind factors pass through."""

    @pytest.mark.parametrize(
        ("module_temp", "ambient", "expected"),
        [
            (49.75, 25.0, 24.75),
            (30.0, 30.0, 0.0),
            (-10.0, 30.0, -40.0),
            (100.0, 40.0, 60.0),
            (-40.0, 0.0, -40.0),
        ],
    )
    def test_temperature_difference_is_module_minus_ambient(
        self, module_temp, ambient, expected
    ):
        weather = WeatherData(ambient_temp_c=ambient)
        physics = PhysicsResult(module_temp_c=module_temp)
        df = build_features(*make_inputs(weather=weather, physics=physics))
        assert df.loc[0, "temperature_difference_c"] == pytest.approx(expected)

    def test_cloud_factor_passthrough(self):
        physics = PhysicsResult(cloud_factor=0.925)
        df = build_features(*make_inputs(physics=physics))
        assert df.loc[0, "cloud_factor"] == pytest.approx(0.925)

    def test_wind_cooling_factor_passthrough(self):
        physics = PhysicsResult(wind_cooling_factor=3.0)
        df = build_features(*make_inputs(physics=physics))
        assert df.loc[0, "wind_cooling_factor"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 8. Missing-column (schema-gap) fill
# ---------------------------------------------------------------------------

class TestMissingColumnFill:
    """Columns added to config but missing from the row are filled with 0.0."""

    def test_missing_schema_column_filled_with_zero(self, monkeypatch):
        patched = list(FEATURE_COLUMNS) + ["pressure_hpa"]
        monkeypatch.setattr(
            "services.feature_engineering._FEATURE_COLUMNS", patched
        )
        df = build_features(*make_inputs())
        assert "pressure_hpa" in df.columns
        assert df.loc[0, "pressure_hpa"] == pytest.approx(0.0)

    def test_multiple_missing_columns_all_filled(self, monkeypatch):
        patched = list(FEATURE_COLUMNS) + ["pressure_hpa", "dew_point_c"]
        monkeypatch.setattr(
            "services.feature_engineering._FEATURE_COLUMNS", patched
        )
        df = build_features(*make_inputs())
        assert df.loc[0, "pressure_hpa"] == pytest.approx(0.0)
        assert df.loc[0, "dew_point_c"] == pytest.approx(0.0)

    def test_missing_column_fill_logs_warning(self, monkeypatch, caplog):
        patched = list(FEATURE_COLUMNS) + ["pressure_hpa"]
        monkeypatch.setattr(
            "services.feature_engineering._FEATURE_COLUMNS", patched
        )
        with caplog.at_level(logging.WARNING,
                             logger="services.feature_engineering"):
            build_features(*make_inputs())
        messages = [r.message for r in caplog.records]
        assert any(
            "pressure_hpa" in m and "filling with 0.0" in m for m in messages
        )

    def test_no_missing_column_warning_for_normal_path(self, caplog):
        with caplog.at_level(logging.WARNING,
                             logger="services.feature_engineering"):
            build_features(*make_inputs())
        assert all("filling with 0.0" not in r.message
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# 9. Label -> fault_class_id mapping
# ---------------------------------------------------------------------------

class TestFaultClassMapping:
    """``fault_class_id`` follows config label order; unknown labels -> 0."""

    def test_label_to_id_mapping_matches_config_order(self):
        labels = CFG["classification"]["labels"]
        assert _LABEL_TO_ID == {lbl: idx for idx, lbl in enumerate(labels)}

    @pytest.mark.parametrize(
        ("label", "expected_id"),
        [(lbl, idx) for idx, lbl in enumerate(CFG["classification"]["labels"])],
    )
    def test_each_configured_label_maps_to_its_index(self, label, expected_id):
        classification = ClassificationResult(label=label, class_id=-1)
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(float(expected_id))

    def test_unknown_label_maps_to_zero(self):
        classification = ClassificationResult(label="Mystery-Fault")
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)

    def test_default_classification_label_maps_to_zero(self):
        df = build_features(*make_inputs(classification=ClassificationResult()))
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)

    def test_empty_string_label_maps_to_zero(self):
        classification = ClassificationResult(label="")
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)

    def test_class_id_field_is_ignored(self):
        # fault_class_id is derived from label only, never classification.class_id.
        classification = ClassificationResult(label="Clean", class_id=5)
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. validate_features - happy path & inclusive boundaries
# ---------------------------------------------------------------------------

class TestValidationHappyPath:
    """Valid DataFrames (full or strict schema) pass silently."""

    def test_valid_full_dataframe_passes(self):
        validate_features(make_valid_df())

    def test_valid_strict_schema_dataframe_passes(self):
        # Derived columns are optional: the range loop skips absent columns.
        validate_features(make_valid_df()[FEATURE_COLUMNS])

    def test_extra_columns_are_permitted(self):
        validate_features(make_valid_df(extra_thing=5.0))

    def test_integer_values_pass_validation(self):
        df = pd.DataFrame([{
            "irradiance_wm2": 888, "module_temp_c": 49, "ambient_temp_c": 25,
            "humidity_pct": 50, "wind_speed_ms": 2, "cloud_cover_pct": 30,
            "soiling_ratio": 1, "fault_class_id": 0, "detection_confidence": 0,
            "temperature_difference_c": 24, "cloud_factor": 0,
            "wind_cooling_factor": 3,
        }])
        validate_features(df)

    def test_built_dataframe_passes_validation(self):
        validate_features(build_features(*make_inputs()))

    def test_all_features_at_minimum_pass(self):
        validate_features(make_valid_df(
            irradiance_wm2=0.0, module_temp_c=-40.0, ambient_temp_c=-40.0,
            humidity_pct=0.0, wind_speed_ms=0.0, cloud_cover_pct=0.0,
            soiling_ratio=0.0, fault_class_id=0.0, detection_confidence=0.0,
            temperature_difference_c=-40.0, cloud_factor=0.0,
            wind_cooling_factor=0.0,
        ))

    def test_all_features_at_maximum_pass(self):
        validate_features(make_valid_df(
            irradiance_wm2=1500.0, module_temp_c=120.0, ambient_temp_c=60.0,
            humidity_pct=100.0, wind_speed_ms=60.0, cloud_cover_pct=100.0,
            soiling_ratio=1.0, fault_class_id=10.0, detection_confidence=1.0,
            temperature_difference_c=100.0, cloud_factor=1.0,
            wind_cooling_factor=50.0,
        ))

    @pytest.mark.parametrize("col", list(FEATURE_RANGES.keys()))
    def test_minimum_boundary_inclusive(self, col):
        lo, _ = FEATURE_RANGES[col]
        overrides = {col: float(lo)}
        if col == "irradiance_wm2":
            # At 0 W/m2 the numeric-consistency check must stay quiet.
            overrides.update({"module_temp_c": 25.0, "ambient_temp_c": 25.0,
                              "temperature_difference_c": 0.0})
        validate_features(make_valid_df(**overrides))

    @pytest.mark.parametrize("col", list(FEATURE_RANGES.keys()))
    def test_maximum_boundary_inclusive(self, col):
        _, hi = FEATURE_RANGES[col]
        validate_features(make_valid_df(**{col: float(hi)}))


# ---------------------------------------------------------------------------
# 11. validate_features - schema
# ---------------------------------------------------------------------------

class TestValidationSchema:
    """Required columns must be present; derived columns are optional."""

    def test_missing_required_column_rejected(self):
        df = make_valid_df().drop(columns=["irradiance_wm2"])
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert "missing required columns" in msg
        assert "irradiance_wm2" in msg

    def test_multiple_missing_columns_reported_sorted(self):
        df = make_valid_df().drop(columns=["soiling_ratio", "humidity_pct"])
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert "humidity_pct" in msg and "soiling_ratio" in msg
        assert msg.index("humidity_pct") < msg.index("soiling_ratio")

    def test_missing_derived_column_is_permitted(self):
        # cloud_factor is not part of the strict model schema.
        df = make_valid_df().drop(columns=["cloud_factor"])
        validate_features(df)

    def test_missing_column_raises_solar_ai_error_hierarchy(self):
        assert issubclass(FeatureValidationError, SolarAIError)
        df = make_valid_df().drop(columns=["wind_speed_ms"])
        with pytest.raises(FeatureValidationError):
            validate_features(df)

    def test_missing_column_error_message_uses_standard_prefix(self):
        df = make_valid_df().drop(columns=["humidity_pct"])
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert str(exc.value).startswith("Feature validation failed:")


# ---------------------------------------------------------------------------
# 12. validate_features - missing values (NaN / infinity)
# ---------------------------------------------------------------------------

class TestValidationMissingValues:
    """NaN in any column is rejected; infinity is caught by range checks."""

    def test_nan_value_rejected(self):
        df = make_valid_df(humidity_pct=float("nan"))
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert "NaN" in msg
        assert "humidity_pct" in msg

    def test_nan_in_derived_column_rejected(self):
        df = make_valid_df(cloud_factor=float("nan"))
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert "cloud_factor" in str(exc.value)

    def test_nan_in_extra_column_rejected(self):
        df = make_valid_df(extra_thing=float("nan"))
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert "extra_thing" in str(exc.value)

    def test_positive_infinity_rejected_by_range_check(self):
        df = make_valid_df(irradiance_wm2=float("inf"))
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert "outside" in msg
        assert "irradiance_wm2" in msg

    def test_negative_infinity_rejected_by_range_check(self):
        df = make_valid_df(module_temp_c=float("-inf"))
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert "outside" in str(exc.value)


# ---------------------------------------------------------------------------
# 13. validate_features - numeric ranges
# ---------------------------------------------------------------------------

class TestValidationRanges:
    """Values outside the inclusive ranges are rejected per column."""

    @pytest.mark.parametrize("col", list(FEATURE_RANGES.keys()))
    def test_value_below_minimum_rejected(self, col):
        lo, _ = FEATURE_RANGES[col]
        df = make_valid_df(**{col: lo - 1.0})
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert col in msg
        assert "outside" in msg

    @pytest.mark.parametrize("col", list(FEATURE_RANGES.keys()))
    def test_value_above_maximum_rejected(self, col):
        _, hi = FEATURE_RANGES[col]
        df = make_valid_df(**{col: hi + 1.0})
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert col in msg
        assert "outside" in msg

    def test_multiple_violations_reported_together(self):
        df = make_valid_df(irradiance_wm2=2000.0, humidity_pct=150.0)
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        msg = str(exc.value)
        assert "One or more feature values failed validation" in msg
        assert "irradiance_wm2" in msg
        assert "humidity_pct" in msg


# ---------------------------------------------------------------------------
# 14. validate_features - numeric consistency
# ---------------------------------------------------------------------------

class TestValidationNumericConsistency:
    """Module temp anomalously high given near-zero irradiance is rejected."""

    def test_low_irradiance_with_hot_module_rejected(self):
        df = make_valid_df(irradiance_wm2=0.0, module_temp_c=40.01,
                           ambient_temp_c=25.0, temperature_difference_c=15.01)
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert "Module temp anomalously high" in str(exc.value)

    def test_irradiance_just_below_10_with_hot_module_rejected(self):
        df = make_valid_df(irradiance_wm2=9.99, module_temp_c=40.01,
                           ambient_temp_c=25.0, temperature_difference_c=15.01)
        with pytest.raises(FeatureValidationError) as exc:
            validate_features(df)
        assert "Module temp anomalously high" in str(exc.value)

    def test_exactly_10_irradiance_skips_consistency_check(self):
        df = make_valid_df(irradiance_wm2=10.0, module_temp_c=120.0,
                           ambient_temp_c=25.0, temperature_difference_c=95.0)
        validate_features(df)

    def test_module_temp_at_exactly_ambient_plus_15_passes(self):
        df = make_valid_df(irradiance_wm2=0.0, module_temp_c=40.0,
                           ambient_temp_c=25.0, temperature_difference_c=15.0)
        validate_features(df)

    def test_cool_module_with_low_irradiance_passes(self):
        df = make_valid_df(irradiance_wm2=0.0, module_temp_c=25.0,
                           ambient_temp_c=25.0, temperature_difference_c=0.0)
        validate_features(df)


# ---------------------------------------------------------------------------
# 15. build_feature_dataframe - model-boundary wrapper
# ---------------------------------------------------------------------------

class TestBuildFeatureDataframe:
    """Wrapper builds, validates and strips to the strict model schema."""

    def test_returns_strict_model_schema(self):
        final = build_feature_dataframe(*make_inputs())
        assert list(final.columns) == FEATURE_COLUMNS

    def test_derived_features_stripped(self):
        final = build_feature_dataframe(*make_inputs())
        for col in DERIVED_COLUMNS:
            assert col not in final.columns

    def test_single_row_returned(self):
        final = build_feature_dataframe(*make_inputs())
        assert final.shape[0] == 1

    def test_matches_build_features_stripped_to_schema(self):
        w, p, c, d = make_inputs()
        full = build_features(w, p, c, d)
        final = build_feature_dataframe(w, p, c, d)
        pd.testing.assert_frame_equal(final, full[FEATURE_COLUMNS])

    def test_value_round_trip(self):
        final = build_feature_dataframe(*make_inputs())
        assert final.loc[0, "irradiance_wm2"] == pytest.approx(888.0)
        assert final.loc[0, "fault_class_id"] == pytest.approx(0.0)
        assert final.loc[0, "detection_confidence"] == pytest.approx(0.92)

    def test_out_of_range_physics_raises(self):
        physics = PhysicsResult(irradiance_wm2=2000.0)
        with pytest.raises(FeatureValidationError):
            build_feature_dataframe(*make_inputs(physics=physics))

    def test_nan_input_raises(self):
        weather = WeatherData(humidity_pct=float("nan"))
        with pytest.raises(FeatureValidationError):
            build_feature_dataframe(*make_inputs(weather=weather))

    def test_default_objects_flow_through(self):
        final = build_feature_dataframe(
            WeatherData(), PhysicsResult(),
            ClassificationResult(), DetectionResult(),
        )
        assert list(final.columns) == FEATURE_COLUMNS
        assert final.loc[0, "fault_class_id"] == pytest.approx(0.0)
        assert final.loc[0, "detection_confidence"] == pytest.approx(0.0)

    def test_patched_missing_column_survives_to_final(self, monkeypatch):
        patched = list(FEATURE_COLUMNS) + ["pressure_hpa"]
        monkeypatch.setattr(
            "services.feature_engineering._FEATURE_COLUMNS", patched
        )
        final = build_feature_dataframe(*make_inputs())
        assert "pressure_hpa" in final.columns
        assert final.loc[0, "pressure_hpa"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 16. Model boundary
# ---------------------------------------------------------------------------

class TestModelBoundary:
    """The DataFrame reaching prediction matches the configured XGBoost schema."""

    def test_final_schema_matches_xgboost_contract(self, project_config):
        final = build_feature_dataframe(*make_inputs())
        assert list(final.columns) == project_config["feature_engineering"][
            "feature_columns"
        ]

    def test_all_columns_are_float64(self):
        final = build_feature_dataframe(*make_inputs())
        for col in FEATURE_COLUMNS:
            assert final[col].dtype == "float64"

    def test_boundary_dataframe_shape_is_one_row_nine_features(self):
        final = build_feature_dataframe(*make_inputs())
        assert final.shape == (1, 9)


# ---------------------------------------------------------------------------
# 17. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Identical inputs produce byte-identical feature DataFrames."""

    def test_build_features_repeatable(self):
        df_a = build_features(*make_inputs())
        df_b = build_features(*make_inputs())
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_build_feature_dataframe_repeatable(self):
        final_a = build_feature_dataframe(*make_inputs())
        final_b = build_feature_dataframe(*make_inputs())
        pd.testing.assert_frame_equal(final_a, final_b)

    def test_repeated_calls_do_not_mutate_inputs(self):
        w, p, c, d = make_inputs()
        snapshot = (w.ambient_temp_c, p.module_temp_c, c.label, d.best_confidence)
        for _ in range(3):
            build_features(w, p, c, d)
        assert (w.ambient_temp_c, p.module_temp_c, c.label,
                d.best_confidence) == snapshot


# ---------------------------------------------------------------------------
# 18. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Meaningful boundary cases supported by the current implementation."""

    def test_zero_values_flow_through(self):
        weather = WeatherData(ambient_temp_c=0.0, humidity_pct=0.0,
                              wind_speed_ms=0.0, cloud_cover_pct=0.0)
        physics = PhysicsResult(irradiance_wm2=0.0, module_temp_c=0.0,
                                soiling_ratio=0.0, cloud_factor=0.0,
                                wind_cooling_factor=0.0)
        df = build_features(weather, physics,
                            ClassificationResult(label="Clean"),
                            DetectionResult())
        validate_features(df)
        assert df.loc[0, "irradiance_wm2"] == pytest.approx(0.0)
        assert df.loc[0, "soiling_ratio"] == pytest.approx(0.0)

    def test_healthy_panel_features(self):
        df = build_features(*make_inputs())
        assert df.loc[0, "fault_class_id"] == pytest.approx(0.0)
        assert df.loc[0, "soiling_ratio"] == pytest.approx(1.0)

    def test_faulted_panel_features(self):
        classification = ClassificationResult(label="Hotspot", class_id=5)
        df = build_features(*make_inputs(classification=classification))
        assert df.loc[0, "fault_class_id"] == pytest.approx(5.0)

    def test_floating_point_values_preserved(self):
        physics = PhysicsResult(irradiance_wm2=123.456789,
                                module_temp_c=48.123456)
        weather = WeatherData(ambient_temp_c=22.345678)
        df = build_features(*make_inputs(weather=weather, physics=physics))
        assert df.loc[0, "irradiance_wm2"] == pytest.approx(123.456789)
        assert df.loc[0, "module_temp_c"] == pytest.approx(48.123456)
        assert df.loc[0, "temperature_difference_c"] == pytest.approx(
            48.123456 - 22.345678
        )

    def test_extreme_valid_values_within_range(self):
        df = build_features(*make_inputs(
            weather=WeatherData(ambient_temp_c=60.0, humidity_pct=100.0,
                                wind_speed_ms=60.0, cloud_cover_pct=100.0),
            physics=PhysicsResult(irradiance_wm2=1500.0, module_temp_c=120.0,
                                  soiling_ratio=1.0, cloud_factor=1.0,
                                  wind_cooling_factor=50.0),
        ))
        validate_features(df)








