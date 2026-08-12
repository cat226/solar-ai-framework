"""utils/security.py — Input sanitization and safe-string utilities.

Provides reusable helpers for cleaning user-controlled strings before
they are logged, displayed, or sent to external services.
"""

from __future__ import annotations

import re

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_for_log(value: str, max_length: int = 200) -> str:
    """Strip control characters and truncate a string for safe log emission.

    Args:
        value: Raw user-controlled string.
        max_length: Maximum length of the returned string.

    Returns:
        Sanitized string safe for inclusion in log messages.
    """
    if not isinstance(value, str):
        return str(value)

    cleaned = _CONTROL_CHAR_RE.sub("", value)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return cleaned
