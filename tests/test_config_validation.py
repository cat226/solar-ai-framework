"""tests/test_config_validation.py - Deterministic unit tests for utils/config.py.

Covers configuration validation for ``configs/settings.yaml``:

1. the real settings file validates through the existing schema validator
2. required top-level and nested sections / keys are enforced
3. numeric range bounds (confidence, IoU, coordinates, percentages, timeouts,
   positive-only counts)
4. type checks (bool/str where numbers are expected, non-mapping sections,
   empty strings where non-empty values are required)
5. feature-engineering configuration (column list shape and contents)
6. physics configuration (soiling ratios, required constants, value ranges)
7. secret fallback resolution order (Streamlit -> env var -> fallback)
8. regression protection: the current settings.yaml stays accepted

Design rules honoured:
- deterministic, isolated, fast: no network, no model weights, no Streamlit
  runtime, no weather/YOLO/MobileNet/XGBoost execution
- expected behaviour derived from the CURRENT implementation in
  ``utils/config.py`` and the CURRENT ``configs/settings.yaml`` - no invented
  schema expectations
"""

from __future__ import annotations

import copy
import sys
import types

import pytest
import yaml

from utils.config import (
    _require_keys,
    _require_number,
    _validate_config_schema,
    get_secret,
    load_config,
)

# ---------------------------------------------------------------------------
# Reference pins - mirror the CURRENT schema in utils/config.py and the modules
# that consume the configuration (services/physics.py,
# services/feature_engineering.py).  These are pins, not invented rules.
# ---------------------------------------------------------------------------

# Top-level sections required by _validate_config_schema().
REQUIRED_TOP_LEVEL_SECTIONS = [
    "weather", "models", "classification", "physics",
    "feature_engineering", "recommendations", "logging",
]

# Physics constants consumed by services/physics.py at import time and required
# by _validate_config_schema().
PHYSICS_CONSUMER_KEYS = [
    "max_irradiance_wm2",
    "noct_celsius", "noct_irradiance_ref", "noct_ambient_ref",
    "wind_cooling_coefficient", "temp_coefficient_pmax",
    "stc_temperature", "soiling_ratios", "panel_rated_power_wp",
]

# Feature column order consumed by services/feature_engineering.py
# (_FEATURE_COLUMNS == CFG["feature_engineering"]["feature_columns"]).
EXPECTED_FEATURE_COLUMNS = [
    "irradiance_wm2", "module_temp_c", "ambient_temp_c",
    "humidity_pct", "wind_speed_ms", "cloud_cover_pct",
    "soiling_ratio", "fault_class_id", "detection_confidence",
]

# ---------------------------------------------------------------------------
# Mutating helpers - every test works on a deep copy of the real config so the
# shared CFG singleton is never modified.
# ---------------------------------------------------------------------------


def _drop(config, *path):
    """Return a deep copy of *config* with the nested key at *path* removed."""
    cfg = copy.deepcopy(config)
    section = cfg
    for part in path[:-1]:
        section = section[part]
    del section[path[-1]]
    return cfg


def _set(config, path, value):
    """Return a deep copy of *config* with the nested key at *path* set."""
    cfg = copy.deepcopy(config)
    section = cfg
    for part in path[:-1]:
        section = section[part]
    section[path[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# 1. Real configuration validation
# ---------------------------------------------------------------------------


class TestRealConfigurationValidates:
    """The CURRENT configs/settings.yaml must pass the schema validator."""

    def test_load_config_revalidates_real_settings(self, project_config):
        cfg = load_config()
        assert cfg == project_config
        assert set(REQUIRED_TOP_LEVEL_SECTIONS) <= set(cfg)

    def test_real_settings_pass_schema_validator(self, project_config):
        # Must not raise - a regression guard against accidental schema
        # tightening beyond what the implementation actually requires.
        _validate_config_schema(copy.deepcopy(project_config))

    def test_section_and_value_pins(self, project_config):
        """Pin representative values that downstream modules depend on."""
        assert project_config["weather"]["timeout_seconds"] == 10
        assert project_config["weather"]["base_url"].startswith("https://")
        assert project_config["models"]["yolo"]["confidence_threshold"] == 0.45
        assert project_config["models"]["yolo"]["iou_threshold"] == 0.50
        assert project_config["models"]["yolo"]["image_size"] == 640
        assert project_config["models"]["mobilenet"]["num_classes"] == 6
        assert project_config["physics"]["noct_celsius"] == 45.0
        assert project_config["physics"]["soiling_ratios"]["Dusty"] == 0.92
        assert project_config["recommendations"]["efficiency_loss_critical_pct"] == 20.0
        assert project_config["logging"]["level"] == "INFO"


# ---------------------------------------------------------------------------
# 2. Required sections
# ---------------------------------------------------------------------------


class TestRequiredSections:
    """Missing or non-mapping top-level configuration sections must fail."""

    def test_root_must_be_a_mapping(self):
        for bad_root in ([], "settings", 42, None):
            with pytest.raises(ValueError, match="root must be a mapping"):
                _validate_config_schema(bad_root)

    def test_empty_config_reports_all_sections(self):
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema({})
        message = str(exc_info.value)
        assert "Missing required configuration sections" in message
        for section in REQUIRED_TOP_LEVEL_SECTIONS:
            assert section in message

    @pytest.mark.parametrize("section", REQUIRED_TOP_LEVEL_SECTIONS)
    def test_missing_single_section_fails(self, project_config, section):
        cfg = _drop(project_config, section)
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert "Missing required configuration sections" in str(exc_info.value)
        assert section in str(exc_info.value)

    def test_missing_multiple_sections_are_all_reported(self, project_config):
        cfg = _drop(project_config, "weather")
        cfg = _drop(cfg, "logging")
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        message = str(exc_info.value)
        assert "weather" in message
        assert "logging" in message

# ---------------------------------------------------------------------------
# 3. Required keys
# ---------------------------------------------------------------------------

MISSING_KEY_CASES = [
    ("weather.base_url", ("weather", "base_url"), "weather"),
    ("weather.timeout_seconds", ("weather", "timeout_seconds"), "weather"),
    ("weather.defaults.latitude", ("weather", "defaults", "latitude"), "weather.defaults"),
    ("weather.defaults.ambient_temp_c", ("weather", "defaults", "ambient_temp_c"), "weather.defaults"),
    ("models.yolo.weights", ("models", "yolo", "weights"), "models.yolo"),
    ("models.yolo.iou_threshold", ("models", "yolo", "iou_threshold"), "models.yolo"),
    ("models.mobilenet.weights", ("models", "mobilenet", "weights"), "models.mobilenet"),
    ("models.mobilenet.input_size", ("models", "mobilenet", "input_size"), "models.mobilenet"),
    ("models.xgboost.weights", ("models", "xgboost", "weights"), "models.xgboost"),
    ("physics.noct_celsius", ("physics", "noct_celsius"), "physics"),
    ("physics.soiling_ratios", ("physics", "soiling_ratios"), "physics"),
    ("physics.panel_rated_power_wp", ("physics", "panel_rated_power_wp"), "physics"),
    ("feature_engineering.feature_columns", ("feature_engineering", "feature_columns"), "feature_engineering"),
    ("recommendations.hotspot_max_temp_c", ("recommendations", "hotspot_max_temp_c"), "recommendations"),
    ("recommendations.cleaning_humidity_threshold_pct", ("recommendations", "cleaning_humidity_threshold_pct"), "recommendations"),
    ("logging.format", ("logging", "format"), "logging"),
    ("logging.level", ("logging", "level"), "logging"),
]


class TestRequiredKeys:
    """Missing nested keys inside present sections must fail."""

    @pytest.mark.parametrize(
        "case", MISSING_KEY_CASES, ids=[c[0] for c in MISSING_KEY_CASES]
    )
    def test_missing_nested_key_fails(self, project_config, case):
        _, path, section = case
        cfg = _drop(project_config, *path)
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert f"Missing required configuration keys in '{section}'" in str(exc_info.value)

    def test_missing_key_error_is_actionable(self, project_config):
        cfg = _drop(project_config, "weather", "timeout_seconds")
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert ("Missing required configuration keys in 'weather': "
                "['timeout_seconds']") in str(exc_info.value)

    def test_missing_classification_labels_reported(self, project_config):
        cfg = _drop(project_config, "classification", "labels")
        with pytest.raises(
            ValueError, match="'classification.labels' must be a non-empty list"
        ):
            _validate_config_schema(cfg)

    def test_require_keys_raises_for_missing_key(self):
        with pytest.raises(ValueError) as exc_info:
            _require_keys({"a": 1}, ["a", "b"], "demo")
        assert "Missing required configuration keys in 'demo': ['b']" in str(exc_info.value)

    def test_require_number_raises_for_missing_key(self):
        with pytest.raises(ValueError) as exc_info:
            _require_number({"a": 1}, "b", "demo")
        assert "Missing required configuration key 'demo.b'." in str(exc_info.value)

# ---------------------------------------------------------------------------
# 4. Non-mapping sections
# ---------------------------------------------------------------------------

NON_MAPPING_CASES = [
    ("weather as list", ("weather",), []),
    ("weather.defaults as list", ("weather", "defaults"), []),
    ("models as list", ("models",), []),
    ("models.yolo as string", ("models", "yolo"), "not-a-mapping"),
    ("physics as list", ("physics",), []),
    ("feature_engineering as string", ("feature_engineering",), "not-a-mapping"),
    ("recommendations as list", ("recommendations",), []),
    ("logging as list", ("logging",), []),
]


class TestNonMappingSections:
    """Sections that are not mappings must be rejected with a clear message."""

    @pytest.mark.parametrize(
        "case", NON_MAPPING_CASES, ids=[c[0] for c in NON_MAPPING_CASES]
    )
    def test_non_mapping_section_rejected(self, project_config, case):
        _, path, bad_value = case
        cfg = _set(project_config, path, bad_value)
        with pytest.raises(ValueError, match="must be a mapping"):
            _validate_config_schema(cfg)

    def test_non_mapping_classification_reports_labels(self, project_config):
        cfg = _set(project_config, ("classification",), ["Clean"])
        with pytest.raises(
            ValueError, match="'classification.labels' must be a non-empty list"
        ):
            _validate_config_schema(cfg)

    def test_require_keys_rejects_non_mapping_section(self):
        with pytest.raises(ValueError, match="'demo' must be a mapping"):
            _require_keys(["not", "a", "dict"], ["k"], "demo")

# ---------------------------------------------------------------------------
# 5. Numeric range validation
# ---------------------------------------------------------------------------

RANGE_CASES = [
    ("timeout below minimum", ("weather", "timeout_seconds"), 0, ">= 1"),
    ("timeout above maximum", ("weather", "timeout_seconds"), 301, "<= 300"),
    ("humidity below zero", ("weather", "defaults", "humidity_pct"), -0.01, ">= 0"),
    ("humidity above 100", ("weather", "defaults", "humidity_pct"), 100.01, "<= 100"),
    ("cloud cover below zero", ("weather", "defaults", "cloud_cover_pct"), -1, ">= 0"),
    ("cloud cover above 100", ("weather", "defaults", "cloud_cover_pct"), 101, "<= 100"),
    ("wind speed negative", ("weather", "defaults", "wind_speed_ms"), -0.1, ">= 0"),
    ("latitude above 90", ("weather", "defaults", "latitude"), 90.1, "<= 90"),
    ("latitude below -90", ("weather", "defaults", "latitude"), -90.1, ">= -90"),
    ("longitude above 180", ("weather", "defaults", "longitude"), 180.1, "<= 180"),
    ("longitude below -180", ("weather", "defaults", "longitude"), -180.1, ">= -180"),
    ("yolo confidence above 1", ("models", "yolo", "confidence_threshold"), 1.01, "<= 1"),
    ("yolo iou below 0", ("models", "yolo", "iou_threshold"), -0.01, ">= 0"),
    ("yolo image size zero", ("models", "yolo", "image_size"), 0, ">= 1"),
    ("mobilenet num_classes zero", ("models", "mobilenet", "num_classes"), 0, ">= 1"),
    ("mobilenet input_size zero", ("models", "mobilenet", "input_size"), 0, ">= 1"),
    ("physics max irradiance negative", ("physics", "max_irradiance_wm2"), -1, ">= 0"),
    ("physics noct irradiance ref zero", ("physics", "noct_irradiance_ref"), 0, ">= 1"),
    ("physics panel rating negative", ("physics", "panel_rated_power_wp"), -400, ">= 0"),
    ("critical loss pct above 100", ("recommendations", "efficiency_loss_critical_pct"), 120, "<= 100"),
    ("warning loss pct below 0", ("recommendations", "efficiency_loss_warning_pct"), -1, ">= 0"),
    ("cleaning humidity pct above 100", ("recommendations", "cleaning_humidity_threshold_pct"), 101, "<= 100"),
]


class TestNumericRangeValidation:
    """Out-of-range numeric values must fail with the offending bound."""

    @pytest.mark.parametrize("case", RANGE_CASES, ids=[c[0] for c in RANGE_CASES])
    def test_out_of_range_value_rejected(self, project_config, case):
        _, path, bad_value, bound = case
        cfg = _set(project_config, path, bad_value)
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert bound in str(exc_info.value)

    def test_max_range_boundaries_are_inclusive(self, project_config):
        cfg = _set(project_config, ("weather", "timeout_seconds"), 300)
        cfg = _set(cfg, ("weather", "defaults", "humidity_pct"), 100.0)
        cfg = _set(cfg, ("weather", "defaults", "cloud_cover_pct"), 0.0)
        cfg = _set(cfg, ("weather", "defaults", "latitude"), 90.0)
        cfg = _set(cfg, ("weather", "defaults", "longitude"), -180.0)
        cfg = _set(cfg, ("models", "yolo", "confidence_threshold"), 1.0)
        cfg = _set(cfg, ("models", "yolo", "iou_threshold"), 0.0)
        cfg = _set(cfg, ("physics", "max_irradiance_wm2"), 0)
        cfg = _set(cfg, ("physics", "noct_irradiance_ref"), 1)
        cfg = _set(cfg, ("physics", "panel_rated_power_wp"), 0)
        cfg = _set(cfg, ("recommendations", "efficiency_loss_critical_pct"), 0.0)
        cfg = _set(cfg, ("recommendations", "cleaning_humidity_threshold_pct"), 100.0)
        _validate_config_schema(cfg)  # must not raise

    def test_min_range_boundaries_are_inclusive(self, project_config):
        cfg = _set(project_config, ("weather", "timeout_seconds"), 1)
        cfg = _set(cfg, ("weather", "defaults", "humidity_pct"), 0.0)
        cfg = _set(cfg, ("weather", "defaults", "cloud_cover_pct"), 0.0)
        cfg = _set(cfg, ("weather", "defaults", "latitude"), -90.0)
        cfg = _set(cfg, ("weather", "defaults", "longitude"), 180.0)
        cfg = _set(cfg, ("weather", "defaults", "wind_speed_ms"), 0.0)
        cfg = _set(cfg, ("models", "yolo", "confidence_threshold"), 0.0)
        cfg = _set(cfg, ("models", "yolo", "iou_threshold"), 0.0)
        cfg = _set(cfg, ("models", "yolo", "image_size"), 1)
        cfg = _set(cfg, ("models", "mobilenet", "num_classes"), 1)
        cfg = _set(cfg, ("models", "mobilenet", "input_size"), 1)
        cfg = _set(cfg, ("physics", "max_irradiance_wm2"), 0)
        cfg = _set(cfg, ("physics", "noct_irradiance_ref"), 1)
        cfg = _set(cfg, ("physics", "panel_rated_power_wp"), 0)
        cfg = _set(cfg, ("recommendations", "efficiency_loss_critical_pct"), 0.0)
        cfg = _set(cfg, ("recommendations", "efficiency_loss_warning_pct"), 0.0)
        cfg = _set(cfg, ("recommendations", "cleaning_humidity_threshold_pct"), 0.0)
        _validate_config_schema(cfg)  # must not raise

# ---------------------------------------------------------------------------
# 6. Type validation
# ---------------------------------------------------------------------------

TYPE_CASES = [
    ("bool confidence rejected", ("models", "yolo", "confidence_threshold"), True, "must be numeric, got bool"),
    ("bool timeout rejected", ("weather", "timeout_seconds"), False, "must be numeric, got bool"),
    ("str humidity rejected", ("weather", "defaults", "humidity_pct"), "50", "must be numeric, got str"),
    ("str image size rejected", ("models", "yolo", "image_size"), "640", "must be numeric, got str"),
    ("float latitude accepted", ("weather", "defaults", "latitude"), 13.08, None),
]


class TestTypeValidation:
    """Booleans/strings where numbers are expected must be rejected."""

    @pytest.mark.parametrize("case", TYPE_CASES, ids=[c[0] for c in TYPE_CASES])
    def test_type_rules(self, project_config, case):
        _, path, value, expected = case
        cfg = _set(project_config, path, value)
        if expected is None:
            _validate_config_schema(cfg)  # must not raise
        else:
            with pytest.raises(ValueError, match=expected):
                _validate_config_schema(cfg)

    def test_empty_base_url_rejected(self, project_config):
        cfg = _set(project_config, ("weather", "base_url"), "")
        with pytest.raises(ValueError, match="'weather.base_url' must be non-empty"):
            _validate_config_schema(cfg)

    def test_blank_base_url_rejected(self, project_config):
        cfg = _set(project_config, ("weather", "base_url"), "   ")
        with pytest.raises(ValueError, match="'weather.base_url' must be non-empty"):
            _validate_config_schema(cfg)

    def test_empty_default_city_rejected(self, project_config):
        cfg = _set(project_config, ("weather", "default_city"), "")
        with pytest.raises(ValueError, match="'weather.default_city' must be non-empty"):
            _validate_config_schema(cfg)

    def test_require_number_rejects_bool(self):
        with pytest.raises(ValueError, match="must be numeric, got bool"):
            _require_number({"n": True}, "n", "demo")


# ---------------------------------------------------------------------------
# 7. Feature engineering configuration
# ---------------------------------------------------------------------------


class TestFeatureConfigurationValidation:
    """feature_engineering.feature_columns shape and content rules."""

    def test_feature_columns_must_be_non_empty_list(self, project_config):
        for bad in ([], "irradiance_wm2", None):
            cfg = _set(project_config, ("feature_engineering", "feature_columns"), bad)
            with pytest.raises(
                ValueError,
                match="'feature_engineering.feature_columns' must be a non-empty list",
            ):
                _validate_config_schema(cfg)

    def test_feature_column_entries_must_be_strings(self, project_config):
        cfg = _set(
            project_config,
            ("feature_engineering", "feature_columns"),
            ["irradiance_wm2", 7, "cloud_cover_pct"],
        )
        with pytest.raises(ValueError, match="must contain only non-empty strings"):
            _validate_config_schema(cfg)

    def test_feature_column_entries_must_not_be_empty_strings(self, project_config):
        cfg = _set(
            project_config,
            ("feature_engineering", "feature_columns"),
            ["irradiance_wm2", ""],
        )
        with pytest.raises(ValueError, match="must contain only non-empty strings"):
            _validate_config_schema(cfg)

    def test_whitespace_only_feature_column_entry_rejected(self, project_config):
        cfg = _set(
            project_config,
            ("feature_engineering", "feature_columns"),
            ["irradiance_wm2", "   "],
        )
        with pytest.raises(ValueError, match="must contain only non-empty strings"):
            _validate_config_schema(cfg)

    def test_feature_column_shape_matches_predictor_schema(self, project_config):
        columns = project_config["feature_engineering"]["feature_columns"]
        assert columns == EXPECTED_FEATURE_COLUMNS  # order matters
        assert len(columns) == len(set(columns))  # no duplicates
        assert all(isinstance(c, str) and c for c in columns)

# ---------------------------------------------------------------------------
# 8. Physics configuration
# ---------------------------------------------------------------------------


class TestPhysicsConfigurationValidation:
    """physics section - constants the implementation needs and value ranges."""

    def test_required_physics_constants_exist_in_real_config(self, project_config):
        physics = project_config["physics"]
        for key in PHYSICS_CONSUMER_KEYS:
            assert key in physics, f"missing physics key: {key}"

    @pytest.mark.parametrize("key", PHYSICS_CONSUMER_KEYS)
    def test_missing_physics_constant_rejected(self, project_config, key):
        cfg = _drop(project_config, "physics", key)
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert "Missing required configuration keys in 'physics'" in str(exc_info.value)
        assert key in str(exc_info.value)

    def test_soiling_ratios_must_be_non_empty_mapping(self, project_config):
        for bad in ({}, []):
            cfg = _set(project_config, ("physics", "soiling_ratios"), bad)
            with pytest.raises(
                ValueError,
                match="'physics.soiling_ratios' must be a non-empty mapping",
            ):
                _validate_config_schema(cfg)

    def test_every_classification_label_has_a_soiling_ratio(self, project_config):
        labels = project_config["classification"]["labels"]
        ratios = project_config["physics"]["soiling_ratios"]
        assert set(ratios) == set(labels)
        for label in labels:
            assert 0.0 < ratios[label] <= 1.0

    def test_real_physics_values_in_expected_ranges(self, project_config):
        physics = project_config["physics"]
        assert physics["max_irradiance_wm2"] > 0
        assert physics["noct_irradiance_ref"] >= 1
        assert physics["panel_rated_power_wp"] > 0

# ---------------------------------------------------------------------------
# 9. Secret fallback behaviour
# ---------------------------------------------------------------------------


class _RaisingSecretsBackend:
    """Fake ``streamlit`` module whose secrets backend blows up on access."""

    @property
    def secrets(self):
        raise RuntimeError("not running inside a Streamlit session")


class TestSecretFallback:
    """get_secret resolves Streamlit secrets -> env var -> fallback."""

    def test_environment_variable_is_returned(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY", "default") == "from-env"

    def test_fallback_returned_when_env_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.delenv("SOLAR_AI_TEST_KEY", raising=False)
        assert get_secret("SOLAR_AI_TEST_KEY", "fallback") == "fallback"

    def test_none_returned_when_nowhere_defined(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.delenv("SOLAR_AI_TEST_KEY", raising=False)
        assert get_secret("SOLAR_AI_TEST_KEY") is None

    def test_streamlit_secrets_take_priority_over_env(self, monkeypatch):
        fake = types.SimpleNamespace(secrets={"SOLAR_AI_TEST_KEY": "from-streamlit"})
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY") == "from-streamlit"

    def test_empty_streamlit_value_falls_through_to_env(self, monkeypatch):
        fake = types.SimpleNamespace(secrets={"SOLAR_AI_TEST_KEY": ""})
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY") == "from-env"

    def test_empty_env_value_falls_through_to_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "")
        assert get_secret("SOLAR_AI_TEST_KEY", "fallback") == "fallback"

    def test_streamlit_backend_error_falls_back_to_env(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", _RaisingSecretsBackend())
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY") == "from-env"

    def test_missing_streamlit_module_falls_back_to_env(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY") == "from-env"


# ---------------------------------------------------------------------------
# 10. load_config error handling
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    """File-level loading errors surface as typed, actionable exceptions."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does-not-exist.yaml"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(missing)

    def test_invalid_yaml_raises_yaml_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("weather: [unclosed", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_config(bad)

    def test_empty_yaml_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="root must be a mapping"):
            load_config(empty)

    def test_valid_yaml_round_trips_through_load_config(
        self, tmp_path, project_root, project_config
    ):
        copy_path = tmp_path / "settings.yaml"
        copy_path.write_text(
            (project_root / "configs" / "settings.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        assert load_config(copy_path) == project_config

