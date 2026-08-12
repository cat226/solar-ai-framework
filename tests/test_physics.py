"""Deterministic unit tests for services.physics."""

from datetime import datetime, timezone

import pytest

from services.physics import (
    PhysicsResult,
    calculate_cloud_factor,
    calculate_irradiance,
    calculate_module_temperature,
    calculate_soiling_ratio,
    calculate_temperature_loss,
    calculate_wind_cooling,
    compute_physics,
)

NOON = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
MORNING = datetime(2026, 7, 14, 9, 0, 0, tzinfo=timezone.utc)
AFTERNOON = datetime(2026, 7, 14, 15, 0, 0, tzinfo=timezone.utc)
EARLY_MORNING = datetime(2026, 7, 14, 7, 0, 0, tzinfo=timezone.utc)


class TestCloudFactor:
    def test_zero_cloud_cover_is_clear_sky(self):
        assert calculate_cloud_factor(0.0) == 1.0

    def test_full_cloud_cover_is_zero_transmission(self):
        assert calculate_cloud_factor(100.0) == 0.0

    def test_partial_cloud_cover_is_linear(self):
        assert calculate_cloud_factor(50.0) == pytest.approx(0.5)
        assert calculate_cloud_factor(90.0) == pytest.approx(0.1)
        assert calculate_cloud_factor(33.33) == pytest.approx(0.6667)

    def test_cloud_cover_is_clamped(self):
        assert calculate_cloud_factor(-100.0) == 1.0
        assert calculate_cloud_factor(200.0) == 0.0

    def test_monotonic_decreasing(self):
        assert calculate_cloud_factor(20.0) > calculate_cloud_factor(60.0)


class TestIrradiance:
    def test_nighttime_is_zero(self):
        midnight = datetime(2026, 7, 14, 0, 0, 0, tzinfo=timezone.utc)
        assert calculate_irradiance(1.0, midnight, 0.0, 0.0) == 0.0

    def test_solar_noon_peak_clear_sky(self):
        assert calculate_irradiance(1.0, NOON, 0.0, 0.0) == pytest.approx(1000.0)

    def test_morning_and_afternoon_are_symmetric(self):
        assert calculate_irradiance(1.0, MORNING, 0.0, 0.0) == pytest.approx(707.11)
        assert calculate_irradiance(1.0, AFTERNOON, 0.0, 0.0) == pytest.approx(707.11)

    def test_cloud_factor_is_multiplicative(self):
        clear = calculate_irradiance(1.0, NOON, 0.0, 0.0)
        cloudy = calculate_irradiance(0.1, NOON, 0.0, 0.0)
        assert cloudy == pytest.approx(clear * 0.1)
        assert cloudy < 300.0

    def test_latitude_factor(self):
        assert calculate_irradiance(1.0, NOON, 60.0, 0.0) == pytest.approx(500.0)
        assert calculate_irradiance(1.0, NOON, 90.0, 0.0) == pytest.approx(100.0)

    def test_longitude_shifts_solar_time(self):
        assert calculate_irradiance(1.0, NOON, 0.0, 15.0) == pytest.approx(965.93)
        assert calculate_irradiance(1.0, NOON, 0.0, -15.0) == pytest.approx(965.93)

    def test_deterministic_for_fixed_inputs(self):
        a = calculate_irradiance(0.9, MORNING, 13.08, 80.27)
        b = calculate_irradiance(0.9, MORNING, 13.08, 80.27)
        assert a == b

    def test_default_time_can_be_stubbed(self, monkeypatch):
        class FixedNow:
            @staticmethod
            def now(tz=None):
                return NOON

        import services.physics as physics
        monkeypatch.setattr(physics, "datetime", FixedNow)
        assert calculate_irradiance(1.0) == pytest.approx(1000.0)


class TestModuleTemperature:
    def test_no_irradiance_returns_ambient_minus_wind(self):
        assert calculate_module_temperature(25.0, 0.0, 0.0) == 25.0
        assert calculate_module_temperature(25.0, 0.0, 3.0) == 22.0

    def test_irradiance_contribution(self, project_config):
        p = project_config["physics"]
        noct = float(p["noct_celsius"])
        ref_g = float(p["noct_irradiance_ref"])
        ref_ta = float(p["noct_ambient_ref"])
        expected = 25.0 + (noct - ref_ta) / ref_g * 800.0
        assert calculate_module_temperature(25.0, 800.0, 0.0) == pytest.approx(expected)

    def test_noct_reference_condition(self, project_config):
        p = project_config["physics"]
        assert calculate_module_temperature(
            float(p["noct_ambient_ref"]), float(p["noct_irradiance_ref"]), 0.0
        ) == pytest.approx(float(p["noct_celsius"]))


class TestWindCooling:
    def test_zero_wind(self):
        assert calculate_wind_cooling(0.0) == 0.0

    def test_configured_coefficient(self, project_config):
        coeff = float(project_config["physics"]["wind_cooling_coefficient"])
        assert calculate_wind_cooling(2.0) == pytest.approx(round(coeff * 2.0, 2))

    def test_increasing_wind_increases_cooling(self):
        assert calculate_wind_cooling(1.0) < calculate_wind_cooling(2.0) < calculate_wind_cooling(5.0)


class TestSoiling:
    def test_all_configured_labels(self, project_config):
        for label, expected in project_config["physics"]["soiling_ratios"].items():
            assert calculate_soiling_ratio(label) == pytest.approx(float(expected))

    def test_unknown_labels_default_to_clean(self):
        for label in ("Unknown", "Dust", "", "dusty"):
            assert calculate_soiling_ratio(label) == 1.0


class TestTemperatureLoss:
    def test_at_stc_is_zero(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        assert calculate_temperature_loss(stc) == 0.0

    def test_above_stc_is_positive(self, project_config):
        p = project_config["physics"]
        expected = abs(float(p["temp_coefficient_pmax"])) * (50.0 - float(p["stc_temperature"])) * 100.0
        assert calculate_temperature_loss(50.0) == pytest.approx(round(expected, 2))

    def test_below_stc_is_clamped_to_zero(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        assert calculate_temperature_loss(stc - 10.0) == 0.0

    def test_loss_increases_with_temperature(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        assert calculate_temperature_loss(stc + 40.0) > calculate_temperature_loss(stc + 10.0)


_CLEAR = dict(
    ambient_temp_c=25.0,
    wind_speed_ms=2.0,
    cloud_cover_pct=0.0,
    fault_label="Clean",
    latitude=0.0,
    longitude=0.0,
    observation_time=NOON,
)


class TestComputePhysics:
    def test_result_type_and_fields(self):
        result = compute_physics(**_CLEAR)
        assert isinstance(result, PhysicsResult)
        for field in (
            "irradiance_wm2", "module_temp_c", "soiling_ratio",
            "temp_loss_pct", "effective_efficiency", "cloud_factor",
            "wind_cooling_factor",
        ):
            assert hasattr(result, field)

    def test_clear_noon_values(self):
        result = compute_physics(**_CLEAR)
        assert result.cloud_factor == pytest.approx(1.0)
        assert result.wind_cooling_factor == pytest.approx(3.0)
        assert result.irradiance_wm2 == pytest.approx(1000.0)
        assert result.module_temp_c == pytest.approx(53.25)
        assert result.temp_loss_pct == pytest.approx(11.3)
        assert result.effective_efficiency == pytest.approx(0.887)

    def test_dusty_label_propagates(self):
        result = compute_physics(**{**_CLEAR, "fault_label": "Dusty"})
        assert result.soiling_ratio == pytest.approx(0.92)
        assert result.effective_efficiency == pytest.approx(round(0.92 * 0.887, 4))

    def test_ninety_percent_cloud_cover_is_low_irradiance(self):
        result = compute_physics(**{**_CLEAR, "ambient_temp_c": 18.0, "cloud_cover_pct": 90.0})
        assert result.cloud_factor == pytest.approx(0.1)
        assert result.irradiance_wm2 == pytest.approx(100.0)
        assert result.irradiance_wm2 < 300.0
        assert result.module_temp_c < 53.25
        assert result.temp_loss_pct < 11.3

    def test_full_cloud_cover_is_zero_at_noon(self):
        result = compute_physics(**{**_CLEAR, "cloud_cover_pct": 100.0})
        assert result.cloud_factor == 0.0
        assert result.irradiance_wm2 == 0.0

    def test_nighttime_is_zero(self):
        midnight = datetime(2026, 7, 14, 0, 0, 0, tzinfo=timezone.utc)
        result = compute_physics(**{**_CLEAR, "observation_time": midnight})
        assert result.irradiance_wm2 == 0.0
        assert result.module_temp_c == pytest.approx(22.0)
        assert result.temp_loss_pct == 0.0
        assert result.effective_efficiency == pytest.approx(1.0)

    def test_geographic_inputs_are_accepted(self):
        result = compute_physics(**{**_CLEAR, "latitude": 13.08, "longitude": 80.27})
        assert isinstance(result.irradiance_wm2, float)
        assert result.irradiance_wm2 >= 0.0

    def test_repeated_execution_is_deterministic(self):
        assert compute_physics(**_CLEAR) == compute_physics(**_CLEAR)


class TestBoundaryConditions:
    def test_cloud_cover_bounds(self):
        assert calculate_cloud_factor(0.0) == 1.0
        assert calculate_cloud_factor(100.0) == 0.0

    def test_longitude_wraps(self):
        assert calculate_irradiance(1.0, NOON, 0.0, 180.0) == 0.0
        assert calculate_irradiance(1.0, NOON, 0.0, -180.0) == 0.0

    def test_night_to_day_transition(self):
        sunrise = datetime(2026, 7, 14, 6, 0, 0, tzinfo=timezone.utc)
        assert calculate_irradiance(1.0, sunrise, 0.0, 0.0) == 0.0
        assert calculate_irradiance(1.0, EARLY_MORNING, 0.0, 0.0) == pytest.approx(258.82)
