"""services/physics.py — Solar physics calculations.

Responsibility
--------------
- Compute plane-of-array irradiance from cloud cover and time of day.
- Compute solar cell module temperature using the NOCT thermal model.
- Derive the soiling ratio from the fault classification label.
- Compute temperature-corrected power loss.

All constants originate from ``configs/settings.yaml`` (``physics`` section).
No model inference or UI logic lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils.config import CFG
from utils.logger import get_logger

logger = get_logger(__name__)

# Pull physics constants once
_P: dict = CFG["physics"]
_MAX_IRR: float = float(_P["max_irradiance_wm2"])
_CLOUD_FACTOR: float = float(_P["irradiance_cloud_factor"])
_NOCT: float = float(_P["noct_celsius"])
_NOCT_IRR_REF: float = float(_P["noct_irradiance_ref"])
_NOCT_AMB_REF: float = float(_P["noct_ambient_ref"])
_WIND_COOL: float = float(_P["wind_cooling_coefficient"])
_TEMP_COEFF: float = float(_P["temp_coefficient_pmax"])
_STC_TEMP: float = float(_P["stc_temperature"])
_SOILING_RATIOS: dict[str, float] = _P["soiling_ratios"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PhysicsResult:
    """Structured output of all solar physics calculations.

    Attributes:
        irradiance_wm2: Plane-of-array irradiance in W/m².
        module_temp_c: Estimated solar cell module temperature in °C.
        soiling_ratio: Fraction of clean-panel output (0.0–1.0).
        temp_loss_pct: Percentage power loss due to elevated module temperature.
        effective_efficiency: Combined efficiency factor (soiling × temperature).
    """

    irradiance_wm2: float = 0.0
    module_temp_c: float = 25.0
    soiling_ratio: float = 1.0
    temp_loss_pct: float = 0.0
    effective_efficiency: float = 1.0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def calculate_irradiance(
    cloud_cover_pct: float,
    observation_time: Optional[datetime] = None,
) -> float:
    """Estimate plane-of-array irradiance from cloud cover and solar angle.

    Uses a cosine solar-elevation model to approximate the diurnal cycle.
    Cloud cover linearly attenuates irradiance between the clear-sky peak and
    a minimum fraction defined by ``physics.irradiance_cloud_factor``.

    Args:
        cloud_cover_pct: Fractional cloud cover (0–100 %).
        observation_time: UTC datetime of observation.  Defaults to now (UTC).

    Returns:
        Estimated irradiance in W/m².
    """
    if observation_time is None:
        observation_time = datetime.now(tz=timezone.utc)

    # Solar elevation proxy: peak at solar noon (12:00 UTC ≈ 12:00 local)
    hour = observation_time.hour + observation_time.minute / 60.0
    # Map [6h, 18h] to [0, π] — cosine peak at noon
    solar_angle = math.pi * (hour - 6.0) / 12.0
    solar_factor = max(0.0, math.sin(solar_angle))

    # Clear-sky irradiance at this solar angle
    clear_sky = _MAX_IRR * solar_factor

    # Cloud attenuation: linearly interpolate between clear and cloudy floor
    cloud_fraction = cloud_cover_pct / 100.0
    min_irr = clear_sky * _CLOUD_FACTOR
    irradiance = clear_sky - (clear_sky - min_irr) * cloud_fraction

    logger.debug(
        "Irradiance: hour=%.1f, solar_factor=%.3f, cloud=%.0f%% → %.1f W/m²",
        hour, solar_factor, cloud_cover_pct, irradiance,
    )
    return round(irradiance, 2)


def calculate_module_temperature(
    ambient_temp_c: float,
    irradiance_wm2: float,
    wind_speed_ms: float,
) -> float:
    """Estimate solar cell module temperature using the NOCT thermal model.

    Formula: T_cell = T_ambient + (NOCT - T_ref) / G_ref × G - k × v_wind

    Args:
        ambient_temp_c: Ambient (air) temperature in °C.
        irradiance_wm2: Incident irradiance in W/m².
        wind_speed_ms: Wind speed in m/s (cooling effect).

    Returns:
        Estimated module temperature in °C.
    """
    noct_rise = (_NOCT - _NOCT_AMB_REF) / _NOCT_IRR_REF * irradiance_wm2
    wind_cooling = _WIND_COOL * wind_speed_ms
    module_temp = ambient_temp_c + noct_rise - wind_cooling

    logger.debug(
        "Module temp: ambient=%.1f°C, irr=%.1f W/m², wind=%.1f m/s → %.2f°C",
        ambient_temp_c, irradiance_wm2, wind_speed_ms, module_temp,
    )
    return round(module_temp, 2)


def get_soiling_ratio(fault_label: str) -> float:
    """Return the soiling ratio for a given fault class label.

    Falls back to ``1.0`` (no soiling) for unknown labels.

    Args:
        fault_label: Fault class label from the classifier
                     (e.g. ``"Dusty"``, ``"Clean"``).

    Returns:
        Soiling ratio in [0.0, 1.0].
    """
    ratio = _SOILING_RATIOS.get(fault_label, 1.0)
    logger.debug("Soiling ratio for '%s': %.2f", fault_label, ratio)
    return ratio


def calculate_temperature_loss(module_temp_c: float) -> float:
    """Compute percentage power loss due to elevated module temperature.

    Uses the linear temperature coefficient of maximum power (Pmax):
        loss = temp_coefficient_pmax × (T_cell − T_STC) × 100

    Args:
        module_temp_c: Module temperature in °C.

    Returns:
        Power loss as a non-negative percentage (0.0 if below STC temp).
    """
    delta_t = module_temp_c - _STC_TEMP
    loss_pct = _TEMP_COEFF * delta_t * 100.0  # coeff is negative → loss positive
    loss_pct = max(0.0, -loss_pct)            # ensure non-negative

    logger.debug(
        "Temp loss: module=%.1f°C, ΔT=%.1f°C → %.2f%% loss",
        module_temp_c, delta_t, loss_pct,
    )
    return round(loss_pct, 2)


def compute_physics(
    ambient_temp_c: float,
    wind_speed_ms: float,
    cloud_cover_pct: float,
    fault_label: str,
    observation_time: Optional[datetime] = None,
) -> PhysicsResult:
    """Run all physics calculations and return a consolidated :class:`PhysicsResult`.

    This is the single entry point used by :mod:`services.pipeline`.

    Args:
        ambient_temp_c: Ambient temperature in °C.
        wind_speed_ms: Wind speed in m/s.
        cloud_cover_pct: Cloud cover percentage (0–100).
        fault_label: Fault class label from the MobileNet classifier.
        observation_time: UTC datetime of observation; defaults to now (UTC).

    Returns:
        :class:`PhysicsResult` with all computed values.
    """
    irradiance = calculate_irradiance(cloud_cover_pct, observation_time)
    module_temp = calculate_module_temperature(ambient_temp_c, irradiance, wind_speed_ms)
    soiling_ratio = get_soiling_ratio(fault_label)
    temp_loss_pct = calculate_temperature_loss(module_temp)

    # Effective efficiency combines both soiling and temperature degradation
    temp_factor = 1.0 - temp_loss_pct / 100.0
    effective_efficiency = round(soiling_ratio * temp_factor, 4)

    result = PhysicsResult(
        irradiance_wm2=irradiance,
        module_temp_c=module_temp,
        soiling_ratio=soiling_ratio,
        temp_loss_pct=temp_loss_pct,
        effective_efficiency=effective_efficiency,
    )

    logger.info(
        "Physics: irr=%.1f W/m2, T_mod=%.1f C, soiling=%.2f, temp_loss=%.2f%%, eff=%.4f",
        result.irradiance_wm2, result.module_temp_c, result.soiling_ratio,
        result.temp_loss_pct, result.effective_efficiency,
    )
    return result
