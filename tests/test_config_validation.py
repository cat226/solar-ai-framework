"""tests/test_config_validation.py - Deterministic unit tests for utils/config.py.

Covers configuration validation for configs/settings.yaml and ensures the
physics configuration remains internally consistent with the implementation.
"""

from __future__ import annotations

import copy
import sys

import pytest
import yaml

from utils.config import (
    _require_keys,
    _require_number,
    _validate_config_schema,
    get_secret,
    load_config,
)

REQUIRED_TOP_LEVEL_SECTIONS = [
    "weather", "models", "classification", "physics",
    "feature_engineering", "recommendations", "logging",
]

PHYSICS_CONSUMER_KEYS = [
    "max_irradiance_wm2", "irradiance_cloud_factor",
    "noct_celsius", "noct_irradiance_ref", "noct_ambient_ref",
    "wind_cooling_coefficient", "temp_coefficient_pmax",
    "stc_temperature", "soiling_ratios", "panel_rated_power_wp",
]

EXPECTED_FEATURE_COLUMNS = [
    "irradiance_wm2", "module_temp_c", "ambient_temp_c",
    "humidity_pct", "wind_speed_ms", "cloud_cover_pct",
    "soiling_ratio", "fault_class_id", "detection_confidence",
]


def _drop(config, *path):
    cfg = copy.deepcopy(config)
    section = cfg
    for part in path[:-1]:
        section = section[part]
    del section[path[-1]]
    return cfg


def _set(config, path, value):
    cfg = copy.deepcopy(config)
    section = cfg
    for part in path[:-1]:
        section = section[part]
    section[path[-1]] = value
    return cfg


class TestRealConfigurationValidates:
    def test_load_config_revalidates_real_settings(self, project_config):
        cfg = load_config()
        assert cfg == project_config
        assert set(REQUIRED_TOP_LEVEL_SECTIONS) <= set(cfg)

    def test_real_settings_pass_schema_validator(self, project_config):
        _validate_config_schema(copy.deepcopy(project_config))

    def test_section_and_value_pins(self, project_config):
        assert project_config["weather"]["timeout_seconds"] == 10
        assert project_config["weather"]["base_url"].startswith("https://")
        # Recalibrated 2026-09-05 (Phase 6B) from 0.45 to 0.30 via a
        # validation-split-only confidence-threshold sweep - see
        # docs/ML_HARDENING_PHASE6B.md for the full methodology and evidence.
        assert project_config["models"]["yolo"]["confidence_threshold"] == 0.30
        assert project_config["models"]["yolo"]["iou_threshold"] == 0.50
        assert project_config["models"]["yolo"]["image_size"] == 640
        assert project_config["models"]["mobilenet"]["num_classes"] == 6
        assert project_config["physics"]["noct_celsius"] == 45.0
        assert project_config["physics"]["soiling_ratios"]["Dusty"] == 0.92
        assert project_config["recommendations"]["efficiency_loss_critical_pct"] == 20.0
        assert project_config["logging"]["level"] == "INFO"


class TestRequiredSections:
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


MISSING_KEY_CASES = [
    ("weather.base_url", ("weather", "base_url"), "weather"),
    ("weather.timeout_seconds", ("weather", "timeout_seconds"), "weather"),
    ("weather.defaults.latitude", ("weather", "defaults", "latitude"), "weather.defaults"),
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
    @pytest.mark.parametrize("case", MISSING_KEY_CASES, ids=[c[0] for c in MISSING_KEY_CASES])
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
        assert "Missing required configuration keys in 'weather': ['timeout_seconds']" in str(exc_info.value)

    def test_missing_classification_labels_reported(self, project_config):
        cfg = _drop(project_config, "classification", "labels")
        with pytest.raises(ValueError, match="'classification.labels' must be a non-empty list"):
            _validate_config_schema(cfg)

    def test_require_keys_raises_for_missing_key(self):
        with pytest.raises(ValueError) as exc_info:
            _require_keys({"a": 1}, ["a", "b"], "demo")
        assert "Missing required configuration keys in 'demo': ['b']" in str(exc_info.value)

    def test_require_number_raises_for_missing_key(self):
        with pytest.raises(ValueError) as exc_info:
            _require_number({"a": 1}, "b", "demo")
        assert "Missing required configuration key 'demo.b'." in str(exc_info.value)


class TestNonMappingSections:
    @pytest.mark.parametrize("path,bad", [
        (("weather",), []), (("weather", "defaults"), []), (("models",), []),
        (("models", "yolo"), "not-a-mapping"), (("physics",), []),
        (("feature_engineering",), "not-a-mapping"), (("recommendations",), []),
        (("logging",), []),
    ])
    def test_non_mapping_section_rejected(self, project_config, path, bad):
        cfg = _set(project_config, path, bad)
        with pytest.raises(ValueError, match="must be a mapping"):
            _validate_config_schema(cfg)

    def test_non_mapping_classification_reports_labels(self, project_config):
        cfg = _set(project_config, ("classification",), ["Clean"])
        with pytest.raises(ValueError, match="'classification.labels' must be a non-empty list"):
            _validate_config_schema(cfg)


RANGE_CASES = [
    (("weather", "timeout_seconds"), 0, ">= 1"),
    (("weather", "timeout_seconds"), 301, "<= 300"),
    (("weather", "defaults", "humidity_pct"), -0.01, ">= 0"),
    (("weather", "defaults", "humidity_pct"), 100.01, "<= 100"),
    (("weather", "defaults", "cloud_cover_pct"), -1, ">= 0"),
    (("weather", "defaults", "cloud_cover_pct"), 101, "<= 100"),
    (("weather", "defaults", "wind_speed_ms"), -0.1, ">= 0"),
    (("weather", "defaults", "latitude"), 90.1, "<= 90"),
    (("weather", "defaults", "latitude"), -90.1, ">= -90"),
    (("weather", "defaults", "longitude"), 180.1, "<= 180"),
    (("weather", "defaults", "longitude"), -180.1, ">= -180"),
    (("models", "yolo", "confidence_threshold"), 1.01, "<= 1"),
    (("models", "yolo", "iou_threshold"), -0.01, ">= 0"),
    (("models", "yolo", "image_size"), 0, ">= 1"),
    (("models", "mobilenet", "num_classes"), 0, ">= 1"),
    (("models", "mobilenet", "input_size"), 0, ">= 1"),
    (("physics", "max_irradiance_wm2"), -1, ">= 0"),
    (("physics", "irradiance_cloud_factor"), 1.5, "<= 1"),
    (("physics", "irradiance_cloud_factor"), -0.5, ">= 0"),
    (("physics", "noct_irradiance_ref"), 0, ">= 1"),
    (("physics", "panel_rated_power_wp"), -400, ">= 0"),
    (("recommendations", "efficiency_loss_critical_pct"), 120, "<= 100"),
    (("recommendations", "efficiency_loss_warning_pct"), -1, ">= 0"),
    (("recommendations", "cleaning_humidity_threshold_pct"), 101, "<= 100"),
]


class TestNumericRangeValidation:
    @pytest.mark.parametrize("path,bad,bound", RANGE_CASES)
    def test_out_of_range_value_rejected(self, project_config, path, bad, bound):
        cfg = _set(project_config, path, bad)
        with pytest.raises(ValueError) as exc_info:
            _validate_config_schema(cfg)
        assert bound in str(exc_info.value)


class TestFeatureEngineering:
    def test_feature_columns_match_expected_order(self, project_config):
        assert project_config["feature_engineering"]["feature_columns"] == EXPECTED_FEATURE_COLUMNS

    def test_feature_columns_must_be_non_empty_list(self, project_config):
        for bad in ({}, [], "columns"):
            cfg = _set(project_config, ("feature_engineering", "feature_columns"), bad)
            with pytest.raises(ValueError, match="'feature_engineering.feature_columns' must be a non-empty list"):
                _validate_config_schema(cfg)


class TestPhysicsConfigurationValidation:
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
            with pytest.raises(ValueError, match="'physics.soiling_ratios' must be a non-empty mapping"):
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
        # Zero is valid here: the runtime model now derives the cloud
        # transmission factor directly from cloud cover rather than using this
        # legacy minimum as an interpolation endpoint.
        assert 0.0 <= physics["irradiance_cloud_factor"] <= 1.0
        assert physics["noct_irradiance_ref"] >= 1
        assert physics["panel_rated_power_wp"] > 0


class TestSecretFallback:
    def test_environment_variable_is_returned(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "from-env")
        assert get_secret("SOLAR_AI_TEST_KEY", "default") == "from-env"

    def test_fallback_returned_when_env_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.delenv("SOLAR_AI_TEST_KEY", raising=False)
        assert get_secret("SOLAR_AI_TEST_KEY", "default") == "default"

    def test_empty_environment_variable_falls_back(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setenv("SOLAR_AI_TEST_KEY", "")
        assert get_secret("SOLAR_AI_TEST_KEY", "default") == "default"

    def test_yaml_loader_reads_mapping(self, project_config):
        assert isinstance(project_config, dict)
        assert yaml.safe_load(open("configs/settings.yaml", encoding="utf-8")) == project_config
