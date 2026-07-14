"""utils/config.py — YAML configuration loader and secret resolver.

Loads ``configs/settings.yaml`` once at import time and exposes the parsed
dictionary as :data:`CFG`.  All other modules should import ``CFG`` from
here instead of reading the YAML themselves.

Secret resolution
-----------------
Sensitive values (API keys, credentials) must **not** live in YAML.
Use :func:`get_secret` to read them in priority order:

1. ``st.secrets["OPENWEATHER_API_KEY"]``  — if running inside Streamlit
2. ``os.environ["OPENWEATHER_API_KEY"]``  — from a ``.env`` or shell export
3. *fallback* — optional default value supplied by the caller

To load ``.env`` automatically, install ``python-dotenv`` and call
``load_dotenv()`` at the top of ``app.py`` (before any imports that need
the env var).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Resolve path: settings.yaml lives at <project_root>/configs/settings.yaml
# This file lives at <project_root>/utils/config.py  → parent = project_root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_SETTINGS_PATH: Path = _PROJECT_ROOT / "configs" / "settings.yaml"


def load_config(path: Path = _SETTINGS_PATH) -> dict[str, Any]:
    """Load and return the YAML configuration file as a dictionary.

    Args:
        path: Absolute path to the YAML settings file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the settings file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Ensure configs/settings.yaml exists in the project root."
        )

    with open(path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    return config


def get_secret(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve a secret value from Streamlit secrets or environment variables.

    Resolution order
    ----------------
    1. ``st.secrets[key]``  — available when running under Streamlit with a
       configured ``.streamlit/secrets.toml``.
    2. ``os.environ[key]``  — set by a ``.env`` file (loaded via
       ``python-dotenv``) or a shell ``export`` statement.
    3. *fallback* — returned as-is (may be ``None``).

    Args:
        key: Name of the secret (e.g. ``"OPENWEATHER_API_KEY"``).
        fallback: Value to return when the secret is not found anywhere.

    Returns:
        The resolved secret string, or *fallback* if not found.

    Example::

        from utils.config import get_secret
        api_key = get_secret("OPENWEATHER_API_KEY")
        if not api_key:
            raise WeatherAPIError("city", "API key not configured")
    """
    # 1. Try Streamlit secrets (only available inside a Streamlit process)
    try:
        import streamlit as st  # type: ignore
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 — not in Streamlit context, or key missing
        pass

    # 2. Try environment variable (covers .env via python-dotenv)
    value = os.environ.get(key)
    if value:
        return value

    # 3. Fallback
    return fallback


# ---------------------------------------------------------------------------
# Module-level singleton — import CFG wherever config values are needed.
# ---------------------------------------------------------------------------
CFG: dict[str, Any] = load_config()
