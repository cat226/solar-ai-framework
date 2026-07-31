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
_CLOUD_FACTOR_MIN: float = float(_PHYSICS_CFG["irradiance_cloud_factor"])
_NOCT_CELSIUS: float = float(_PHYSICS_CFG["noct_celsius"])
_NOCT_IRRADIANCE_REF: float = float(_PHYSICS_CFG["noct_irradiance_ref"])
_NOCT_AMBIENT_REF: float = float(_PHYSICS_CFG["noct_ambient_ref"])
_WIND_COOLING_COEFF: float = float(_PHYSICS_CFG["wind_cooling_coefficient"])
_TEMP_COEFF_PMAX: float = float(_PHYSICS_CFG["temp_coefficient_pmax"])
_STC_TEMPERATURE: float = float(_PHYSICS_CFG["stc_temperature"])
_SOILING_RATIOS: dict[str, float] = _PHYSICS_CFG["soiling_ratios"]


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
        cloud_factor: Derived factor for cloud cover impact.
        wind_cooling_factor: Derived factor for wind cooling effect in °C.
    """

    irradiance_wm2: float = 0.0
    module_temp_c: float = 25.0
    soiling_ratio: float = 1.0
    temp_loss_pct: float = 0.0
    effective_efficiency: float = 1.0
    cloud_factor: float = 1.0
    wind_cooling_factor: float = 0.0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def calculate_cloud_factor(cloud_cover_pct: float) -> float:
    """Calculate the cloud transmission factor.
    
    Args:
        cloud_cover_pct: Fractional cloud cover (0–100 %).
        
    Returns:
        Fractional transmission factor in [0.0, 1.0].
    """
    cloud_fraction = cloud_cover_pct / 100.0
    # Linearly interpolate between 1.0 (clear) and _CLOUD_FACTOR_MIN (fully cloudy)
    factor = 1.0 - (1.0 - _CLOUD_FACTOR_MIN) * cloud_fraction
    return round(factor, 4)


def calculate_wind_cooling(wind_speed_ms: float) -> float:
    """Calculate the wind cooling factor.
    
    Args:
        wind_speed_ms: Wind speed in m/s.
        
    Returns:
        Temperature reduction in °C.
    """
    return round(_WIND_COOLING_COEFF * wind_speed_ms, 2)


def calculate_irradiance(
    cloud_factor: float,
    observation_time: Optional[datetime] = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> float:
    """Estimate plane-of-array irradiance from cloud factor and solar angle.

    Uses a cosine solar-elevation model to approximate the diurnal cycle,
    enhanced by simple geographical coordinates.

    Args:
        cloud_factor: Computed cloud transmission factor.
        observation_time: UTC datetime of observation.  Defaults to now (UTC).
        latitude: Observer latitude.
        longitude: Observer longitude.

    Returns:
        Estimated irradiance in W/m².
    """
    if observation_time is None:
        observation_time = datetime.now(tz=timezone.utc)

    # Calculate local solar time approximation using longitude
    utc_hour = observation_time.hour + observation_time.minute / 60.0
    local_solar_hour = (utc_hour + longitude / 15.0) % 24.0

    # Solar elevation proxy: map [6h, 18h] to [0, π] — cosine peak at solar noon
    solar_angle = math.pi * (local_solar_hour - 6.0) / 12.0
    
    # Adjust for latitude (simplistic cosine approximation for solar zenith)
    lat_rad = math.radians(latitude)
    lat_factor = math.cos(lat_rad)
    
    # Combined solar factor
    solar_factor = max(0.0, math.sin(solar_angle)) * max(0.1, lat_factor)

    # Clear-sky irradiance at this solar angle
    clear_sky = _MAX_IRRADIANCE_WM2 * solar_factor

    # Apply cloud attenuation
    irradiance = clear_sky * cloud_factor

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
    """Estimate solar cell module temperature using the NOCT thermal model.

    Formula: T_cell = T_ambient + (NOCT - T_ref) / G_ref × G - wind_cooling

    Args:
        ambient_temp_c: Ambient (air) temperature in °C.
        irradiance_wm2: Incident irradiance in W/m².
        wind_cooling_factor: Wind cooling effect in °C.

    Returns:
        Estimated module temperature in °C.
    """
    noct_rise = (_NOCT_CELSIUS - _NOCT_AMBIENT_REF) / _NOCT_IRRADIANCE_REF * irradiance_wm2
    module_temp = ambient_temp_c + noct_rise - wind_cooling_factor

    logger.debug(
        "Module temp calc: ambient=%.1f°C, irr=%.1f W/m², wind_cooling=%.1f°C → %.2f°C",
        ambient_temp_c, irradiance_wm2, wind_cooling_factor, module_temp,
    )
    return round(module_temp, 2)


def calculate_soiling_ratio(fault_label: str) -> float:
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
    delta_t = module_temp_c - _STC_TEMPERATURE
    loss_pct = _TEMP_COEFF_PMAX * delta_t * 100.0  # coeff is negative → loss positive
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
    latitude: float = 0.0,
    longitude: float = 0.0,
    observation_time: Optional[datetime] = None,
) -> PhysicsResult:
    """Run all physics calculations and return a consolidated :class:`PhysicsResult`.

    This is the single entry point used by :mod:`services.pipeline`.

    Args:
        ambient_temp_c: Ambient temperature in °C.
        wind_speed_ms: Wind speed in m/s.
        cloud_cover_pct: Cloud cover percentage (0–100).
        fault_label: Fault class label from the MobileNet classifier.
        latitude: Observer latitude.
        longitude: Observer longitude.
        observation_time: UTC datetime of observation; defaults to now (UTC).

    Returns:
        :class:`PhysicsResult` with all computed values.
    """
    logger.info("Physics Calculations: Starting...")
    
    cloud_factor = calculate_cloud_factor(cloud_cover_pct)
    wind_cooling_factor = calculate_wind_cooling(wind_speed_ms)
    
    irradiance = calculate_irradiance(cloud_factor, observation_time, latitude, longitude)
    module_temp = calculate_module_temperature(ambient_temp_c, irradiance, wind_cooling_factor)
    soiling_ratio = calculate_soiling_ratio(fault_label)
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
        cloud_factor=cloud_factor,
        wind_cooling_factor=wind_cooling_factor,
    )

    logger.info(
        "Physics Calculations complete: irr=%.1f W/m2, T_mod=%.1f C, soiling=%.2f, temp_loss=%.2f%%, eff=%.4f",
        result.irradiance_wm2, result.module_temp_c, result.soiling_ratio,
        result.temp_loss_pct, result.effective_efficiency,
    )
    return result
