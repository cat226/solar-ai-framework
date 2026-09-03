"""Tests for utils/auth.py — the single shared-password access gate.

Driven through Streamlit's AppTest framework against
tests/_auth_gate_script.py, since require_access() depends on
st.session_state / st.stop(), which need a real script context.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_SCRIPT = str(Path(__file__).resolve().parent / "_auth_gate_script.py")


def _run(monkeypatch, password_env: str | None):
    if password_env is None:
        monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("APP_ACCESS_PASSWORD", password_env)
    at = AppTest.from_file(_SCRIPT)
    at.run(timeout=30)
    return at


def test_noop_when_no_password_configured(monkeypatch):
    at = _run(monkeypatch, None)
    assert not at.exception
    assert any("GATE_PASSED_MARKER" in t.value for t in at.text)


def test_blocks_access_until_password_entered(monkeypatch):
    at = _run(monkeypatch, "correct-horse")
    assert not at.exception
    # Gated: the marker past the gate must not have rendered.
    assert not any("GATE_PASSED_MARKER" in t.value for t in at.text)
    # A password field should be present.
    assert len(at.text_input) == 1


def test_wrong_password_shows_error_and_stays_blocked(monkeypatch):
    at = _run(monkeypatch, "correct-horse")
    at.text_input[0].input("wrong-password").run()
    at.button[0].click().run()
    assert not at.exception
    assert not any("GATE_PASSED_MARKER" in t.value for t in at.text)
    assert len(at.error) == 1


def test_correct_password_grants_access(monkeypatch):
    at = _run(monkeypatch, "correct-horse")
    at.text_input[0].input("correct-horse").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("GATE_PASSED_MARKER" in t.value for t in at.text)


def test_session_state_persists_across_reruns(monkeypatch):
    """Once authenticated, a fresh run() within the same session must not
    re-prompt for the password."""
    at = _run(monkeypatch, "correct-horse")
    at.text_input[0].input("correct-horse").run()
    at.button[0].click().run()
    assert any("GATE_PASSED_MARKER" in t.value for t in at.text)

    at.run()  # simulate another script rerun in the same session
    assert not at.exception
    assert any("GATE_PASSED_MARKER" in t.value for t in at.text)
