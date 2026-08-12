"""Runtime security regression tests."""

from app import _sanitize_city


def test_sanitize_city_removes_control_characters():
    assert _sanitize_city("Chennai\nIN\t") == "Chennai IN"


def test_sanitize_city_collapses_whitespace_and_trims():
    assert _sanitize_city("  Chennai   Tamil Nadu  ") == "Chennai Tamil Nadu"


def test_sanitize_city_limits_length():
    assert len(_sanitize_city("x" * 500)) == 100


def test_sanitize_city_accepts_normal_unicode_text():
    assert _sanitize_city("São Paulo") == "São Paulo"
