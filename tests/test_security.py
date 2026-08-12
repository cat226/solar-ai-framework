"""tests/test_security.py - Input sanitization and safe-string utilities.

Covers utils/security.py sanitize_for_log behavior.
"""

from __future__ import annotations

import pytest

from utils.security import sanitize_for_log


class TestSanitizeForLog:
    """sanitize_for_log strips control characters and truncates long strings."""

    def test_plain_string_unchanged(self):
        assert sanitize_for_log("Chennai") == "Chennai"

    def test_leading_trailing_whitespace_preserved(self):
        assert sanitize_for_log("  Chennai  ") == "  Chennai  "

    def test_control_characters_stripped(self):
        assert "\x00" not in sanitize_for_log("Chennai\x00\x01\x02")
        assert "\x07" not in sanitize_for_log("tab\x07here")

    def test_newline_stripped(self):
        assert "\n" not in sanitize_for_log("line1\nline2")
        assert "\r" not in sanitize_for_log("line1\r\nline2")

    def test_del_character_stripped(self):
        assert "\x7f" not in sanitize_for_log("hello\x7fworld")

    def test_long_string_truncated(self):
        long_str = "a" * 300
        result = sanitize_for_log(long_str, max_length=50)
        assert len(result) == 53
        assert result.endswith("...")

    def test_default_max_length(self):
        long_str = "a" * 300
        result = sanitize_for_log(long_str)
        assert len(result) == 203
        assert result.endswith("...")

    def test_non_string_coerced(self):
        assert sanitize_for_log(123) == "123"

    def test_none_returns_string_none(self):
        assert sanitize_for_log(None) == "None"

    def test_empty_string_unchanged(self):
        assert sanitize_for_log("") == ""
