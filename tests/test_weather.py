"""tests/test_weather.py - Deterministic unit tests for services/weather.py.

Covers the OpenWeatherMap client behaviour actually implemented by
``fetch_weather``:

A. Successful response parsing
   - all JSON fields mapped to WeatherData
   - numeric conversions
   - timestamp parsing
   - fallback to input city when response name missing
   - fetch_successful=True state

B. Fallback / graceful-degradation behaviour
   - missing API key
   - request timeout
   - HTTP error
   - generic requests exception
   - malformed JSON / missing keys
   - invalid numeric values

C. Configuration behaviour
   - endpoint, timeout, units, defaults sourced from config
   - fallback WeatherData uses config defaults

D. Logging behaviour
   - warning on missing API key
   - info before request
   - info on successful response
   - warning on failures with city and exception details

E. Determinism
   - repeated calls with the same mock response produce equivalent results
   - no real network access

Design rules honoured:
- deterministic, isolated, fast: no real network, no API keys, no model weights
- expected behaviour derived strictly from the CURRENT implementation
- no production code modified
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

from services.weather import WeatherData, _DEFAULTS, _BASE_URL, _TIMEOUT, _UNITS, fetch_weather


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(json_data, status_code=200):
    """Create a mock requests.Response with the given JSON body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_data)
    return resp


def _success_payload(city="Chennai", description="clear sky", dt=1697904000):
    """Return a minimal but complete OpenWeatherMap-style JSON payload."""
    return {
        "name": city,
        "main": {
            "temp": 28.5,
            "humidity": 65,
            "pressure": 1008.5,
        },
        "wind": {
            "speed": 3.2,
        },
        "clouds": {
            "all": 40,
        },
        "coord": {
            "lat": 13.08,
            "lon": 80.27,
        },
        "dt": dt,
        "weather": [
            {"description": description},
        ],
    }


# ---------------------------------------------------------------------------
# A. Successful response parsing
# ---------------------------------------------------------------------------

class TestSuccessfulResponseParsing:
    """fetch_weather returns a fully populated WeatherData on success."""

    def test_returns_fetch_successful_true(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is True

    def test_city_from_response_name(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload(city="Mumbai")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.city == "Mumbai"

    def test_city_falls_back_to_input_when_name_missing(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload(city="Chennai")
        del payload["name"]
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Delhi")
        assert result.city == "Delhi"

    def test_temperature_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.ambient_temp_c == pytest.approx(28.5)

    def test_humidity_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.humidity_pct == pytest.approx(65.0)

    def test_wind_speed_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.wind_speed_ms == pytest.approx(3.2)

    def test_cloud_cover_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.cloud_cover_pct == pytest.approx(40.0)

    def test_pressure_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.pressure_hpa == pytest.approx(1008.5)

    def test_coordinates_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.latitude == pytest.approx(13.08)
        assert result.longitude == pytest.approx(80.27)

    def test_timestamp_parsed_as_utc(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        dt_epoch = 1697904000  # 2023-10-21 00:00:00 UTC
        payload = _success_payload(dt=dt_epoch)
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        expected = datetime.fromtimestamp(dt_epoch, tz=timezone.utc)
        assert result.timestamp == expected
        assert result.timestamp.tzinfo is timezone.utc

    def test_description_parsed(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload(description="light rain")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.description == "light rain"

    def test_all_fields_populated_on_success(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        result = fetch_weather("Chennai")
        assert result.city == "Chennai"
        assert result.fetch_successful is True
        assert result.ambient_temp_c > 0
        assert 0 <= result.humidity_pct <= 100
        assert result.wind_speed_ms >= 0
        assert 0 <= result.cloud_cover_pct <= 100
        assert result.pressure_hpa > 0
        assert result.timestamp is not None


# ---------------------------------------------------------------------------
# B. Fallback / graceful-degradation behaviour
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    """fetch_weather returns safe defaults on any handled failure."""

    def test_missing_api_key_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: None)
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"
        assert result.ambient_temp_c == _DEFAULTS["ambient_temp_c"]
        assert result.humidity_pct == _DEFAULTS["humidity_pct"]
        assert result.wind_speed_ms == _DEFAULTS["wind_speed_ms"]
        assert result.cloud_cover_pct == _DEFAULTS["cloud_cover_pct"]
        assert result.pressure_hpa == _DEFAULTS["pressure_hpa"]
        assert result.latitude == _DEFAULTS["latitude"]
        assert result.longitude == _DEFAULTS["longitude"]
        assert result.timestamp is None
        assert result.description == ""
        assert result.fetch_successful is False

    def test_request_timeout_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"
        assert result.ambient_temp_c == _DEFAULTS["ambient_temp_c"]

    def test_http_error_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        resp = _make_response({}, status_code=404)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found", response=resp)
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: resp)
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"

    def test_generic_request_exception_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.ConnectionError("no internet")))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"

    def test_missing_json_key_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload()
        del payload["main"]["temp"]  # remove required key
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"

    def test_malformed_numeric_value_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload()
        payload["main"]["temp"] = "not-a-number"
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"

    def test_fallback_city_preserved_when_response_name_missing(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload()
        del payload["name"]
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Pune")
        assert result.city == "Pune"
        assert result.fetch_successful is True  # .get() fallback means parsing continues


# ---------------------------------------------------------------------------
# C. Configuration behaviour
# ---------------------------------------------------------------------------

class TestConfigurationBehaviour:
    """fetch_weather reads endpoint, timeout, units, and defaults from config."""

    def test_uses_configured_base_url(self, project_config, monkeypatch):
        expected_url = project_config["weather"]["base_url"]
        captured = {}

        def fake_get(url, *a, **kw):
            captured["url"] = url
            return _make_response(_success_payload())

        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", fake_get)
        fetch_weather("Chennai")
        assert captured["url"] == expected_url

    def test_passes_configured_timeout(self, project_config, monkeypatch):
        expected_timeout = project_config["weather"]["timeout_seconds"]
        captured = {}

        def fake_get(url, *a, **kw):
            captured["timeout"] = kw.get("timeout")
            return _make_response(_success_payload())

        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", fake_get)
        fetch_weather("Chennai")
        assert captured["timeout"] == expected_timeout

    def test_passes_configured_units(self, project_config, monkeypatch):
        expected_units = project_config["weather"]["units"]
        captured = {}

        def fake_get(url, *a, **kw):
            captured["params"] = kw.get("params", {})
            return _make_response(_success_payload())

        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", fake_get)
        fetch_weather("Chennai")
        assert captured["params"]["units"] == expected_units

    def test_api_key_passed_as_appid(self, monkeypatch):
        captured = {}

        def fake_get(url, *a, **kw):
            captured["params"] = kw.get("params", {})
            return _make_response(_success_payload())

        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "test-api-key")
        monkeypatch.setattr("services.weather.requests.get", fake_get)
        fetch_weather("Chennai")
        assert captured["params"]["appid"] == "test-api-key"
        assert captured["params"]["q"] == "Chennai"

    def test_fallback_uses_config_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: None)
        result = fetch_weather("Chennai")
        assert result.ambient_temp_c == _DEFAULTS["ambient_temp_c"]
        assert result.humidity_pct == _DEFAULTS["humidity_pct"]
        assert result.wind_speed_ms == _DEFAULTS["wind_speed_ms"]
        assert result.cloud_cover_pct == _DEFAULTS["cloud_cover_pct"]
        assert result.pressure_hpa == _DEFAULTS["pressure_hpa"]
        assert result.latitude == _DEFAULTS["latitude"]
        assert result.longitude == _DEFAULTS["longitude"]


# ---------------------------------------------------------------------------
# D. Logging behaviour
# ---------------------------------------------------------------------------

class TestLoggingBehaviour:
    """fetch_weather emits the expected log messages."""

    def test_warning_on_missing_api_key(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: None)
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_weather("Chennai")
        assert "OPENWEATHER_API_KEY is not set" in caplog.text
        assert "Chennai" in caplog.text

    def test_info_before_request(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        import logging
        with caplog.at_level(logging.INFO):
            fetch_weather("Chennai")
        assert "Weather Request" in caplog.text
        assert "Chennai" in caplog.text

    def test_info_on_successful_response(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        import logging
        with caplog.at_level(logging.INFO):
            fetch_weather("Chennai")
        assert "Weather Response" in caplog.text
        assert "clear sky" in caplog.text

    def test_warning_on_timeout_includes_city_and_timeout(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")))
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_weather("Chennai")
        assert "timed out" in caplog.text
        assert "Chennai" in caplog.text

    def test_warning_on_http_error_includes_city(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        resp = _make_response({}, status_code=500)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error", response=resp)
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: resp)
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_weather("Chennai")
        assert "Chennai" in caplog.text

    def test_warning_on_generic_error_includes_exception_type(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad json")))
        import logging
        with caplog.at_level(logging.WARNING):
            fetch_weather("Chennai")
        assert "ValueError" in caplog.text
        assert "Chennai" in caplog.text

    def test_no_credentials_exposed_in_logs(self, monkeypatch, caplog):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "super-secret-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        import logging
        with caplog.at_level(logging.INFO):
            fetch_weather("Chennai")
        assert "super-secret-key" not in caplog.text


# ---------------------------------------------------------------------------
# E. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated mocked calls produce equivalent results."""

    def test_repeated_calls_are_equivalent(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        r1 = fetch_weather("Chennai")
        r2 = fetch_weather("Chennai")
        assert r1 == r2

    def test_no_real_network_access(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        # requests.get should never be called if we mock it - if it were, the test would fail
        # because real network access is not possible in this environment.
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(_success_payload()))
        fetch_weather("Chennai")  # must not raise


# ---------------------------------------------------------------------------
# F. WeatherData defaults
# ---------------------------------------------------------------------------

class TestWeatherDataDefaults:
    """WeatherData fields use config-derived defaults."""

    def test_default_city_is_empty(self):
        w = WeatherData()
        assert w.city == ""

    def test_defaults_match_config(self, project_config):
        defaults = project_config["weather"]["defaults"]
        w = WeatherData()
        assert w.ambient_temp_c == defaults["ambient_temp_c"]
        assert w.humidity_pct == defaults["humidity_pct"]
        assert w.wind_speed_ms == defaults["wind_speed_ms"]
        assert w.cloud_cover_pct == defaults["cloud_cover_pct"]
        assert w.pressure_hpa == defaults["pressure_hpa"]
        assert w.latitude == defaults["latitude"]
        assert w.longitude == defaults["longitude"]

    def test_default_timestamp_is_none(self):
        w = WeatherData()
        assert w.timestamp is None

    def test_default_description_is_empty(self):
        w = WeatherData()
        assert w.description == ""

    def test_default_fetch_successful_is_false(self):
        w = WeatherData()
        assert w.fetch_successful is False


# ---------------------------------------------------------------------------
# G. Boundary / malformed cases
# ---------------------------------------------------------------------------

class TestBoundaryAndMalformedCases:
    """Only test cases the implementation actually handles."""

    def test_empty_response_body_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response({}))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False

    @pytest.mark.parametrize("field", [
        ("main", "temp"),
        ("main", "humidity"),
        ("main", "pressure"),
        ("wind", "speed"),
        ("clouds", "all"),
    ])
    def test_null_numeric_field_returns_defaults(self, monkeypatch, field):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload()
        payload[field[0]][field[1]] = None
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"
        assert result.ambient_temp_c == _DEFAULTS["ambient_temp_c"]

    def test_multiple_null_fields_returns_defaults(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: "fake-key")
        payload = _success_payload()
        payload["main"]["temp"] = None
        payload["main"]["humidity"] = None
        payload["wind"]["speed"] = None
        monkeypatch.setattr("services.weather.requests.get", lambda *a, **kw: _make_response(payload))
        result = fetch_weather("Chennai")
        assert result.fetch_successful is False
        assert result.city == "Chennai"

    def test_fallback_preserves_query_city(self, monkeypatch):
        monkeypatch.setattr("services.weather.get_secret", lambda key, fallback=None: None)
        result = fetch_weather("Kolkata")
        assert result.city == "Kolkata"