"""services/physics.py — Solar physics calculations.

Responsibility
--------------
- Compute plane-of-array irradiance using solar geometry.
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
_PHYSICS_CFG: dict = CFG["physics"]
_MAX_IRRADIANCE_WM2: float = float(_PHYSICS_CFG["max_irradiance_wm2"])
_NOCT_CELSIUS: float = float(_PHYSICS_CFG["noct_celsius"])
_NOCT_IRRADIANCE_REF: float = float(_PHYSICS_CFG["noct_irradiance_ref"])
_NOCT_AMBIENT_REF: float = float(_PHYSICS_CFG["noct_ambient_ref"])
_WIND_COOLING_COEFF: float = float(_PHYSICS_CFG["wind_cooling_coefficient"])
_TEMP_COEFF_PMAX: float = float(_PHYSICS_CFG["temp_coefficient_pmax"])
_STC_TEMPERATURE: float = float(_PHYSICS_CFG["stc_temperature"])
_SOILING_RATIOS: dict[str, float] = _PHYSICS_CFG["soiling_ratios"]


@dataclass
class PhysicsResult:
    """Structured output of all solar physics calculations."""

    irradiance_wm2: float = 0.0
    module_temp_c: float = 25.0
    soiling_ratio: float = 1.0
    temp_loss_pct: float = 0.0
    effective_efficiency: float = 1.0
    cloud_factor: float = 1.0
    wind_cooling_factor: float = 0.0


def calculate_cloud_factor(cloud_cover_pct: float) -> float:
    """Return cloud transmission in [0, 1] for 0–100% cloud cover.

    The model intentionally treats the reported cloud-cover percentage as a
    direct attenuation signal: clear sky transmits 100%, while full cloud
    cover transmits 0%. Inputs outside the physical range are clamped so the
    returned factor can never become negative or exceed one.
    """
    cloud_fraction = min(max(float(cloud_cover_pct), 0.0), 100.0) / 100.0
    factor = 1.0 - cloud_fraction
    return round(factor, 4)


def calculate_wind_cooling(wind_speed_ms: float) -> float:
    """Calculate the wind cooling factor in °C."""
    return round(_WIND_COOLING_COEFF * wind_speed_ms, 2)


def calculate_irradiance(
    cloud_factor: float,
    observation_time: Optional[datetime] = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> float:
    """Estimate plane-of-array irradiance from cloud factor and solar angle."""
    if observation_time is None:
        observation_time = datetime.now(tz=timezone.utc)

    utc_hour = observation_time.hour + observation_time.minute / 60.0
    local_solar_hour = (utc_hour + longitude / 15.0) % 24.0

    solar_angle = math.pi * (local_solar_hour - 6.0) / 12.0
    lat_rad = math.radians(latitude)
    lat_factor = math.cos(lat_rad)
    solar_factor = max(0.0, math.sin(solar_angle)) * max(0.1, lat_factor)

    clear_sky = _MAX_IRRADIANCE_WM2 * solar_factor
    irradiance = clear_sky * min(max(cloud_factor, 0.0), 1.0)

    logger.debug(
        "Irradiance calc: local_hour=%.1f, solar_factor=%.3f, cloud_factor=%.3f → %.1f W/m²",
        local_solar_hour, solar_factor, cloud_factor, irradiance,
    )
    return round(irradiance, 2)


def calculate_module_temperature(
    ambient_temp_c: float,
    irradiance_wm2: float,
    wind_cooling_factor: float,
) -> float:
    """Estimate solar cell module temperature using the NOCT thermal model."""
    noct_rise = (_NOCT_CELSIUS - _NOCT_AMBIENT_REF) / _NOCT_IRRADIANCE_REF * irradiance_wm2
    module_temp = ambient_temp_c + noct_rise - wind_cooling_factor
    logger.debug(
        "Module temp calc: ambient=%.1f°C, irr=%.1f W/m², wind_cooling=%.1f°C → %.2f°C",
        ambient_temp_c, irradiance_wm2, wind_cooling_factor, module_temp,
    )
    return round(module_temp, 2)


def calculate_soiling_ratio(fault_label: str) -> float:
    """Return the configured soiling ratio for a fault class label."""
    ratio = _SOILING_RATIOS.get(fault_label, 1.0)
    logger.debug("Soiling ratio for '%s': %.2f", fault_label, ratio)
    return ratio


def calculate_temperature_loss(module_temp_c: float) -> float:
    """Compute percentage power loss due to elevated module temperature."""
    delta_t = module_temp_c - _STC_TEMPERATURE
    loss_pct = _TEMP_COEFF_PMAX * delta_t * 100.0
    loss_pct = max(0.0, -loss_pct)
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
    latitude: float = 0.0,
    longitude: float = 0.0,
    observation_time: Optional[datetime] = None,
) -> PhysicsResult:
    """Run all physics calculations and return a consolidated result."""
    logger.info("Physics Calculations: Starting...")

    cloud_factor = calculate_cloud_factor(cloud_cover_pct)
    wind_cooling_factor = calculate_wind_cooling(wind_speed_ms)
    irradiance = calculate_irradiance(cloud_factor, observation_time, latitude, longitude)
    module_temp = calculate_module_temperature(ambient_temp_c, irradiance, wind_cooling_factor)
    soiling_ratio = calculate_soiling_ratio(fault_label)
    temp_loss_pct = calculate_temperature_loss(module_temp)

    temp_factor = 1.0 - temp_loss_pct / 100.0
    effective_efficiency = round(soiling_ratio * temp_factor, 4)

    result = PhysicsResult(
        irradiance_wm2=irradiance,
        module_temp_c=module_temp,
        soiling_ratio=soiling_ratio,
        temp_loss_pct=temp_loss_pct,
        effective_efficiency=effective_efficiency,
        cloud_factor=cloud_factor,
        wind_cooling_factor=wind_cooling_factor,
    )

    logger.info(
        "Physics Calculations complete: irr=%.1f W/m2, T_mod=%.1f C, soiling=%.2f, temp_loss=%.2f%%, eff=%.4f",
        result.irradiance_wm2, result.module_temp_c, result.soiling_ratio,
        result.temp_loss_pct, result.effective_efficiency,
    )
    return result
