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

from dataclasses import dataclass

import requests

from utils.config import CFG, get_secret
from utils.exceptions import WeatherAPIError
from utils.logger import get_logger

logger = get_logger(__name__)

# Pull non-secret API config once from YAML
_W_CFG: dict = CFG["weather"]
_BASE_URL: str = _W_CFG["base_url"]
_TIMEOUT: int = int(_W_CFG["timeout_seconds"])
_UNITS: str = _W_CFG["units"]


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
        description: Human-readable weather description (e.g. "light rain").
        fetch_successful: True if the API call and parsing succeeded.
    """

    city: str = ""
    ambient_temp_c: float = 25.0
    humidity_pct: float = 50.0
    wind_speed_ms: float = 2.0
    cloud_cover_pct: float = 0.0
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
    # Resolve API key at call time so .env / secrets.toml changes take effect
    api_key = get_secret("OPENWEATHER_API_KEY")
    if not api_key:
        logger.warning(
            "OPENWEATHER_API_KEY is not set. Using weather defaults for '%s'.", city
        )
        return WeatherData(city=city, fetch_successful=False)

    params = {
        "q": city,
        "appid": api_key,
        "units": _UNITS,
    }

    logger.info("Fetching weather data for city: %s", city)

    try:
        response = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data: dict = response.json()

        weather = WeatherData(
            city=data.get("name", city),
            ambient_temp_c=float(data["main"]["temp"]),
            humidity_pct=float(data["main"]["humidity"]),
            wind_speed_ms=float(data["wind"]["speed"]),
            cloud_cover_pct=float(data["clouds"]["all"]),
            description=data["weather"][0]["description"],
            fetch_successful=True,
        )

        logger.info(
            "Weather fetched: %s | %.1f°C | humidity=%d%% | wind=%.1f m/s | "
            "clouds=%d%%",
            weather.description,
            weather.ambient_temp_c,
            weather.humidity_pct,
            weather.wind_speed_ms,
            weather.cloud_cover_pct,
        )
        return weather

    except requests.exceptions.Timeout:
        logger.warning("Weather API timed out after %d s. Using defaults.", _TIMEOUT)
    except requests.exceptions.HTTPError as exc:
        logger.warning("Weather API HTTP error: %s. Using defaults.", exc)
    except (KeyError, ValueError, requests.exceptions.RequestException) as exc:
        logger.warning("Weather API error: %s. Using defaults.", exc)

    # Return safe defaults so the pipeline can continue
    return WeatherData(city=city, fetch_successful=False)
