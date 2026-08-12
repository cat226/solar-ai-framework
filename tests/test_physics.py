"""tests/test_physics.py - Deterministic unit tests for services/physics.py.

Covers cloud factor, irradiance (solar geometry), NOCT module temperature,
wind cooling, soiling ratios, temperature loss, the consolidated
``compute_physics`` entry point and sensible boundary conditions.

Design rules honoured:
- deterministic: fixed timezone-aware datetimes, fixed coordinates
- isolated and fast: no network, no model weights, no external APIs
- expected values derived from the CURRENT implementation and configuration
  (configs/settings.yaml) - no invented formulas
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Fixed deterministic instants (UTC) - never datetime.now()
# ---------------------------------------------------------------------------

NOON = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)      # solar noon at lon=0
MORNING = datetime(2026, 7, 14, 9, 0, 0, tzinfo=timezone.utc)     # 3 h before noon
AFTERNOON = datetime(2026, 7, 14, 15, 0, 0, tzinfo=timezone.utc)  # 3 h after noon
EARLY_MORNING = datetime(2026, 7, 14, 7, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Cloud factor
# ---------------------------------------------------------------------------


class TestCloudFactor:
    """``calculate_cloud_factor`` linear attenuation model."""

    def test_zero_cloud_cover_is_clear_sky(self):
        assert calculate_cloud_factor(0.0) == 1.0

    def test_full_cloud_cover_returns_zero(self):
        assert calculate_cloud_factor(100.0) == 0.0

    def test_partial_cloud_cover_linear_interpolation(self):
        assert calculate_cloud_factor(50.0) == pytest.approx(0.5)

    def test_example_value_matches_formula(self):
        assert calculate_cloud_factor(30.0) == pytest.approx(0.7)

    def test_values_above_100_are_clamped_to_zero(self):
        assert calculate_cloud_factor(200.0) == 0.0

    def test_negative_cloud_cover_extrapolates_above_one(self):
        assert calculate_cloud_factor(-100.0) == pytest.approx(2.0)

    def test_monotonic_decreasing_with_cloud_cover(self):
        assert calculate_cloud_factor(20.0) > calculate_cloud_factor(60.0)

    def test_rounding_to_four_decimals(self):
        assert calculate_cloud_factor(33.33) == pytest.approx(0.6667)

    def test_deterministic(self):
        assert calculate_cloud_factor(45.5) == calculate_cloud_factor(45.5)


# ---------------------------------------------------------------------------
# 2. Irradiance (solar geometry)
# ---------------------------------------------------------------------------


class TestIrradiance:
    """``calculate_irradiance`` cosine solar-elevation model."""

    def test_nighttime_is_zero(self, fixed_utc_midnight):
        assert calculate_irradiance(1.0, fixed_utc_midnight, 0.0, 0.0) == 0.0

    def test_solar_noon_peak_clear_sky(self):
        # lat=0, lon=0 at 12:00 UTC -> local solar noon -> full max irradiance
        assert calculate_irradiance(1.0, NOON, 0.0, 0.0) == pytest.approx(1000.0)

    def test_morning_positive(self):
        # 09:00 UTC -> sin(45 deg) * 1000 = 707.11 W/m2
        assert calculate_irradiance(1.0, MORNING, 0.0, 0.0) == pytest.approx(707.11)

    def test_afternoon_symmetric_with_morning(self):
        assert calculate_irradiance(1.0, AFTERNOON, 0.0, 0.0) == pytest.approx(707.11)

    def test_sunrise_transition_boundary(self, fixed_utc_datetime):
        # 06:00 UTC at equator -> sin(0) -> zero irradiance
        assert calculate_irradiance(1.0, fixed_utc_datetime, 0.0, 0.0) == 0.0

    def test_cloud_cover_reduces_irradiance(self):
        clear = calculate_irradiance(1.0, NOON, 0.0, 0.0)
        cloudy = calculate_irradiance(0.25, NOON, 0.0, 0.0)
        assert cloudy < clear
        assert cloudy == pytest.approx(250.0)

    def test_latitude_is_accepted(self):
        # At noon, cos(60 deg) -> 0.5 factor
        assert calculate_irradiance(1.0, NOON, 60.0, 0.0) == pytest.approx(500.0)

    def test_latitude_floored_at_extremes(self):
        # cos(90 deg) ~ 0 -> floored at 0.1 by the implementation
        assert calculate_irradiance(1.0, NOON, 90.0, 0.0) == pytest.approx(100.0)
        assert calculate_irradiance(1.0, NOON, -90.0, 0.0) == pytest.approx(100.0)

    def test_longitude_shifts_local_solar_time(self):
        # lon=15 deg -> local solar hour 13 -> sin(105 deg) = sin(75 deg)
        assert calculate_irradiance(1.0, NOON, 0.0, 15.0) == pytest.approx(965.93)
        # lon=-15 deg -> local solar hour 11 -> symmetric value
        assert calculate_irradiance(1.0, NOON, 0.0, -15.0) == pytest.approx(965.93)

    def test_deterministic_for_fixed_time_and_place(self):
        a = calculate_irradiance(0.9, MORNING, 13.08, 80.27)
        b = calculate_irradiance(0.9, MORNING, 13.08, 80.27)
        assert a == b

    def test_default_time_uses_utc_now_but_is_stubbed(self, monkeypatch):
        # Stub datetime.now so no wall clock is ever consulted.
        class _FixedNow:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)

        import services.physics as physics

        monkeypatch.setattr(physics, "datetime", _FixedNow)
        assert calculate_irradiance(1.0) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 3. Module temperature (NOCT model)
# ---------------------------------------------------------------------------


class TestModuleTemperature:
    """``calculate_module_temperature`` NOCT thermal model."""

    def test_ambient_only_when_no_irradiance_and_no_wind(self):
        assert calculate_module_temperature(25.0, 0.0, 0.0) == 25.0

    def test_irradiance_contribution(self, project_config):
        noct = float(project_config["physics"]["noct_celsius"])
        ref_g = float(project_config["physics"]["noct_irradiance_ref"])
        ref_ta = float(project_config["physics"]["noct_ambient_ref"])
        # T_cell = Ta + (NOCT - 20)/800 * G at zero wind
        expected_800 = 25.0 + (noct - ref_ta) / ref_g * 800.0
        expected_1000 = 25.0 + (noct - ref_ta) / ref_g * 1000.0
        assert calculate_module_temperature(25.0, 800.0, 0.0) == pytest.approx(expected_800)
        assert calculate_module_temperature(25.0, 1000.0, 0.0) == pytest.approx(expected_1000)

    def test_noct_reference_conditions_yield_noct(self, project_config):
        # Ambient = NOCT ambient ref, G = NOCT irradiance ref, no wind -> NOCT
        noct = float(project_config["physics"]["noct_celsius"])
        ref_g = float(project_config["physics"]["noct_irradiance_ref"])
        ref_ta = float(project_config["physics"]["noct_ambient_ref"])
        assert calculate_module_temperature(ref_ta, ref_g, 0.0) == pytest.approx(noct)

    def test_wind_cooling_reduces_temperature(self):
        still = calculate_module_temperature(25.0, 800.0, 0.0)
        breezy = calculate_module_temperature(25.0, 800.0, 3.0)
        assert breezy == pytest.approx(still - 3.0)

    def test_deterministic(self):
        a = calculate_module_temperature(20.5, 812.0, 2.5)
        b = calculate_module_temperature(20.5, 812.0, 2.5)
        assert a == b


# ---------------------------------------------------------------------------
# 4. Wind cooling
# ---------------------------------------------------------------------------


class TestWindCooling:
    """``calculate_wind_cooling`` linear cooling model."""

    def test_zero_wind_no_cooling(self):
        assert calculate_wind_cooling(0.0) == 0.0

    def test_normal_wind_uses_configured_coefficient(self, project_config):
        coeff = float(project_config["physics"]["wind_cooling_coefficient"])
        assert coeff == 1.5  # pin the configured coefficient
        assert calculate_wind_cooling(2.0) == pytest.approx(3.0)

    def test_increasing_wind_increases_cooling(self):
        assert calculate_wind_cooling(1.0) < calculate_wind_cooling(2.0) < calculate_wind_cooling(5.0)

    def test_configured_coefficient_is_respected(self, project_config):
        coeff = float(project_config["physics"]["wind_cooling_coefficient"])
        for wind in (0.0, 1.0, 2.5, 10.0):
            assert calculate_wind_cooling(wind) == pytest.approx(round(coeff * wind, 2))

    def test_deterministic(self):
        assert calculate_wind_cooling(3.7) == calculate_wind_cooling(3.7)


# ---------------------------------------------------------------------------
# 5. Soiling ratios
# ---------------------------------------------------------------------------


class TestSoiling:
    """``calculate_soiling_ratio`` config-driven mapping."""

    def test_all_configured_labels_map_to_config(self, project_config):
        ratios = project_config["physics"]["soiling_ratios"]
        assert ratios  # config sanity: mapping must not be empty
        for label, expected in ratios.items():
            assert calculate_soiling_ratio(label) == pytest.approx(float(expected))

    def test_known_label_regression_pins(self):
        # Pins matching configs/settings.yaml exactly
        assert calculate_soiling_ratio("Clean") == 1.0
        assert calculate_soiling_ratio("Dusty") == 0.92
        assert calculate_soiling_ratio("Bird-Drop") == 0.88
        assert calculate_soiling_ratio("Electrical-Damage") == 0.80
        assert calculate_soiling_ratio("Physical-Damage") == 0.75
        assert calculate_soiling_ratio("Hotspot") == 0.70

    def test_unconfigured_labels_fall_back_to_one(self):
        # The implementation falls back to 1.0 (no soiling) for any label
        # that is NOT a configured mapping key.
        assert calculate_soiling_ratio("Unknown") == 1.0
        assert calculate_soiling_ratio("Snow Coverage") == 1.0
        assert calculate_soiling_ratio("Dust") == 1.0
        assert calculate_soiling_ratio("Bird Droppings") == 1.0
        assert calculate_soiling_ratio("") == 1.0

    def test_lookup_is_case_sensitive(self):
        # Exact-match dict lookup: lowercase variant is not a configured key
        assert calculate_soiling_ratio("dusty") == 1.0

    def test_deterministic(self):
        assert calculate_soiling_ratio("Dusty") == calculate_soiling_ratio("Dusty")


# ---------------------------------------------------------------------------
# 6. Temperature loss / temperature coefficient
# ---------------------------------------------------------------------------


class TestTemperatureLoss:
    """``calculate_temperature_loss`` linear Pmax coefficient model."""

    def test_at_stc_temperature_zero_loss(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        assert calculate_temperature_loss(stc) == 0.0

    def test_above_stc_positive_loss(self, project_config):
        coeff = float(project_config["physics"]["temp_coefficient_pmax"])
        stc = float(project_config["physics"]["stc_temperature"])
        # loss = |coeff| * (T - T_STC) * 100
        expected = abs(coeff) * (50.0 - stc) * 100.0
        assert calculate_temperature_loss(50.0) == pytest.approx(round(expected, 2))

    def test_below_stc_clamped_to_zero(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        # Cooler-than-STC panels produce no *negative* loss (no gain)
        assert calculate_temperature_loss(stc - 10.0) == 0.0

    def test_sign_and_direction(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        assert calculate_temperature_loss(stc + 10.0) > 0.0
        assert calculate_temperature_loss(stc + 40.0) > calculate_temperature_loss(stc + 10.0)

    def test_deterministic(self):
        assert calculate_temperature_loss(47.25) == calculate_temperature_loss(47.25)


# ---------------------------------------------------------------------------
# 7. compute_physics() - consolidated calculation
# ---------------------------------------------------------------------------

_CLEAR_ARGS = {
    "ambient_temp_c": 25.0,
    "wind_speed_ms": 2.0,
    "cloud_cover_pct": 0.0,
    "fault_label": "Clean",
    "latitude": 0.0,
    "longitude": 0.0,
}


class TestComputePhysics:
    """``compute_physics`` end-to-end deterministic calculation."""

    def test_returns_physics_result_object(self, fixed_utc_noon):
        result = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_noon)
        assert isinstance(result, PhysicsResult)

    def test_all_required_fields_present(self, fixed_utc_noon):
        result = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_noon)
        for field in (
            "irradiance_wm2",
            "module_temp_c",
            "soiling_ratio",
            "temp_loss_pct",
            "effective_efficiency",
            "cloud_factor",
            "wind_cooling_factor",
        ):
            assert hasattr(result, field)

    def test_deterministic_repeated_execution(self, fixed_utc_noon):
        r1 = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_noon)
        r2 = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_noon)
        assert r1 == r2

    def test_clear_noon_expected_values(self, fixed_utc_noon):
        result = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_noon)
        # cloud 0% -> factor 1.0; wind 2 m/s -> cooling 3.0 C
        assert result.cloud_factor == pytest.approx(1.0)
        assert result.wind_cooling_factor == pytest.approx(3.0)
        # noon at equator, clear sky -> 1000 W/m2
        assert result.irradiance_wm2 == pytest.approx(1000.0)
        # T = 25 + (45-20)/800*1000 - 3 = 53.25 C
        assert result.module_temp_c == pytest.approx(53.25)
        # Clean -> soiling 1.0; loss = 0.004*28.25*100 = 11.3 %
        assert result.soiling_ratio == pytest.approx(1.0)
        assert result.temp_loss_pct == pytest.approx(11.3)
        # eff = 1.0 * (1 - 0.113) = 0.887
        assert result.effective_efficiency == pytest.approx(0.887)

    def test_soiling_label_propagates_to_effective_efficiency(self, fixed_utc_noon):
        args = {**_CLEAR_ARGS, "fault_label": "Dusty"}
        result = compute_physics(**args, observation_time=fixed_utc_noon)
        assert result.soiling_ratio == pytest.approx(0.92)
        # same temperature factor as the Clean run (0.887), reduced by soiling
        assert result.effective_efficiency == pytest.approx(round(0.92 * 0.887, 4))

    def test_cloud_cover_reduces_irradiance_and_temperature(self, fixed_utc_noon):
        args = {**_CLEAR_ARGS, "cloud_cover_pct": 100.0}
        result = compute_physics(**args, observation_time=fixed_utc_noon)
        assert result.cloud_factor == 0.0
        assert result.irradiance_wm2 == 0.0
        assert result.module_temp_c == pytest.approx(22.0)
        assert result.temp_loss_pct == 0.0
        assert result.effective_efficiency == pytest.approx(1.0)

    def test_nighttime_zero_irradiance(self, fixed_utc_midnight):
        result = compute_physics(**_CLEAR_ARGS, observation_time=fixed_utc_midnight)
        assert result.irradiance_wm2 == 0.0
        # T = 25 + 0 - 3 = 22 C; no temperature loss; efficiency unaffected
        assert result.module_temp_c == pytest.approx(22.0)
        assert result.temp_loss_pct == 0.0
        assert result.effective_efficiency == pytest.approx(1.0)

    def test_geographic_inputs_accepted(self, fixed_utc_noon):
        result = compute_physics(
            ambient_temp_c=25.0,
            wind_speed_ms=2.0,
            cloud_cover_pct=0.0,
            fault_label="Clean",
            latitude=13.08,
            longitude=80.27,
            observation_time=fixed_utc_noon,
        )
        assert isinstance(result.irradiance_wm2, float)
        assert result.irradiance_wm2 >= 0.0


# ---------------------------------------------------------------------------
# 8. Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """Sensible boundaries based on the actual implementation/config."""

    def test_cloud_cover_zero_and_full(self):
        assert calculate_cloud_factor(0.0) == 1.0
        assert calculate_cloud_factor(100.0) == 0.0

    def test_wind_speed_zero(self):
        assert calculate_wind_cooling(0.0) == 0.0

    def test_ambient_at_stc_yields_zero_loss(self, project_config):
        stc = float(project_config["physics"]["stc_temperature"])
        # No irradiance and no wind -> module temperature == ambient == STC
        assert calculate_module_temperature(stc, 0.0, 0.0) == pytest.approx(stc)
        assert calculate_temperature_loss(stc) == 0.0

    def test_extreme_latitudes_use_floor_factor(self):
        # cos(+-90 deg) ~ 0 -> floored at 0.1 -> 100 W/m2 at clear noon
        assert calculate_irradiance(1.0, NOON, 90.0, 0.0) == pytest.approx(100.0)
        assert calculate_irradiance(1.0, NOON, -90.0, 0.0) == pytest.approx(100.0)

    def test_longitude_wraps_around_the_globe(self):
        # lon=180 at 12:00 UTC -> local solar hour 0 -> night at equator model
        assert calculate_irradiance(1.0, NOON, 0.0, 180.0) == 0.0
        # lon=-180 at 12:00 UTC -> local solar hour 0 -> same
        assert calculate_irradiance(1.0, NOON, 0.0, -180.0) == 0.0

    def test_nighttime_to_daytime_transition(self, fixed_utc_datetime):
        # 06:00 UTC boundary -> 0; 07:00 UTC -> positive irradiance
        assert calculate_irradiance(1.0, fixed_utc_datetime, 0.0, 0.0) == 0.0
        assert calculate_irradiance(1.0, EARLY_MORNING, 0.0, 0.0) == pytest.approx(258.82)





