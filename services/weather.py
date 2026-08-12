"""services/weather.py — OpenWeatherMap API client.

Responsibility
--------------
- Make a single HTTP GET request to the OpenWeatherMap ``/weather`` endpoint.
- Parse the JSON response into a typed :class:`WeatherData` dataclass.
- Expose a single public function :func:`fetch_weather` used by the pipeline.

The API key is resolved at call time via :func:`utils.config.get_secret`
(``OPENWEATHER_API_KEY``).  It is never read from ``configs/settings.yaml``.

This module has no knowledge of images, models, or the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from utils.config import CFG, get_secret
from utils.logger import get_logger

logger = get_logger(__name__)

# Pull non-secret API config once from YAML
_W_CFG: dict = CFG["weather"]
_BASE_URL: str = _W_CFG["base_url"]
_TIMEOUT: int = int(_W_CFG["timeout_seconds"])
_UNITS: str = _W_CFG["units"]
_DEFAULTS: dict = _W_CFG["defaults"]

_CITY_MAX_LENGTH = 100
_CITY_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def _validate_city(city: str) -> str:
    """Validate and sanitize a city name for safe external API use.

    Rules:
    - Reject non-string values.
    - Reject control characters (C0 controls and DEL).
    - Strip leading/trailing whitespace.
    - Reject empty/whitespace-only strings.
    - Truncate to a safe maximum length.

    Args:
        city: Raw city name from caller.

    Returns:
        Sanitized city string.

    Raises:
        ValueError: If the city is invalid.
    """
    if not isinstance(city, str):
        raise ValueError("City must be a string.")

    if _CITY_CONTROL_CHAR_RE.search(city):
        raise ValueError("City contains invalid control characters.")

    city = city.strip()
    if not city:
        raise ValueError("City must not be empty.")

    if len(city) > _CITY_MAX_LENGTH:
        city = city[:_CITY_MAX_LENGTH]
        logger.warning("City name truncated to %d characters.", _CITY_MAX_LENGTH)

    return city


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class WeatherData:
    """Structured weather observation returned by the API.

    Attributes:
        city: Location name returned by the API.
        ambient_temp_c: Dry-bulb air temperature in °C.
        humidity_pct: Relative humidity percentage (0–100).
        wind_speed_ms: Wind speed in m/s.
        cloud_cover_pct: Cloud cover percentage (0–100).
        pressure_hpa: Atmospheric pressure in hPa.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        timestamp: UTC datetime of the observation.
        description: Human-readable weather description (e.g. "light rain").
        fetch_successful: True if the API call and parsing succeeded.
    """

    city: str = ""
    ambient_temp_c: float = _DEFAULTS["ambient_temp_c"]
    humidity_pct: float = _DEFAULTS["humidity_pct"]
    wind_speed_ms: float = _DEFAULTS["wind_speed_ms"]
    cloud_cover_pct: float = _DEFAULTS["cloud_cover_pct"]
    pressure_hpa: float = _DEFAULTS["pressure_hpa"]
    latitude: float = _DEFAULTS["latitude"]
    longitude: float = _DEFAULTS["longitude"]
    timestamp: Optional[datetime] = None
    description: str = ""
    fetch_successful: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_weather(city: str) -> WeatherData:
    """Fetch current weather conditions for *city* from OpenWeatherMap.

    Args:
        city: City name to query (e.g. ``"Chennai"``).

    Returns:
        :class:`WeatherData` populated from the API response.
        On any network or parsing error the dataclass is returned with
        ``fetch_successful=False`` and safe default values so the pipeline
        can continue degraded rather than crash.
    """
    try:
        sanitized_city = _validate_city(city)
    except ValueError as exc:
        logger.warning("Invalid city name '%s': %s. Using defaults.", city, exc)
        return WeatherData(city=city, fetch_successful=False)

    api_key = get_secret("OPENWEATHER_API_KEY")
    if not api_key:
        logger.warning(
            "OPENWEATHER_API_KEY is not set. Using weather defaults for '%s'.", sanitized_city
        )
        return WeatherData(city=sanitized_city, fetch_successful=False)

    params = {
        "q": sanitized_city,
        "appid": api_key,
        "units": _UNITS,
    }

    logger.info("Weather Request: Fetching data for city '%s'", sanitized_city)

    try:
        response = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data: dict = response.json()

        weather = WeatherData(
            city=data.get("name", sanitized_city),
            ambient_temp_c=float(data["main"]["temp"]),
            humidity_pct=float(data["main"]["humidity"]),
            wind_speed_ms=float(data["wind"]["speed"]),
            cloud_cover_pct=float(data["clouds"]["all"]),
            pressure_hpa=float(data["main"]["pressure"]),
            latitude=float(data["coord"]["lat"]),
            longitude=float(data["coord"]["lon"]),
            timestamp=datetime.fromtimestamp(data["dt"], tz=timezone.utc),
            description=data["weather"][0]["description"],
            fetch_successful=True,
        )

        logger.info(
            "Weather Response: %s | %.1f°C | humidity=%d%% | wind=%.1f m/s | "
            "clouds=%d%% | lat=%.2f | lon=%.2f",
            weather.description,
            weather.ambient_temp_c,
            weather.humidity_pct,
            weather.wind_speed_ms,
            weather.cloud_cover_pct,
            weather.latitude,
            weather.longitude,
        )
        return weather

    except requests.exceptions.Timeout:
        logger.warning(
            "Weather API timed out after %d s for city '%s'. Using defaults.",
            _TIMEOUT, sanitized_city,
        )
    except requests.exceptions.HTTPError as exc:
        logger.warning(
            "Weather API HTTP error for city '%s': %s. Using defaults.", sanitized_city, exc,
        )
    except (KeyError, TypeError, ValueError, requests.exceptions.RequestException) as exc:
        logger.warning(
            "Weather API error for city '%s' (%s): %s. Using defaults.",
            sanitized_city, type(exc).__name__, exc,
        )

    # Return safe defaults so the pipeline can continue
    return WeatherData(city=sanitized_city, fetch_successful=False)
