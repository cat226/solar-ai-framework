"""tests/test_pages_xss_escaping.py — defense-in-depth check that
pages/09_🚨_Alerts.py and pages/12_📷_Cameras.py HTML-escape stored
`worst_label`/`camera_name` values before interpolating them into an
`unsafe_allow_html=True` markdown block.

api/main.py's ingestion validation (`_validate_report`) is the primary
fix — it rejects a non-taxonomy label before it's ever stored — but
`services/storage.py` itself does not validate labels (nothing stops some
other future write path from inserting one), so these pages must not
trust stored strings either. These tests bypass the API entirely and
write directly via `services.storage`, the same way a bug in a future
caller could, to prove the page's own escaping holds regardless.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALERTS_PAGE = str(_REPO_ROOT / "pages" / "09_🚨_Alerts.py")
_CAMERAS_PAGE = str(_REPO_ROOT / "pages" / "12_📷_Cameras.py")
_PAYLOAD = "<img src=x onerror=alert(1)>"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


def test_alerts_page_escapes_malicious_worst_label():
    flagged_panel = {"condition": {"status": "predicted", "label": _PAYLOAD, "confidence": 0.9}}
    storage.record_ground_inspection(
        [{"label": "a.jpg", "report": {}}], flagged=[flagged_panel], unread=[], confirmed_clean=[],
    )
    at = AppTest.from_file(_ALERTS_PAGE)
    at.run(timeout=30)
    assert not at.exception
    rendered = "\n".join(m.value for m in at.markdown)
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered


def test_alerts_page_escapes_malicious_camera_name():
    cam = storage.register_camera(_PAYLOAD)
    flagged_panel = {"condition": {"status": "predicted", "label": "DUSTY", "confidence": 0.9}}
    storage.record_ground_inspection(
        [{"label": "a.jpg", "report": {}}], flagged=[flagged_panel], unread=[], confirmed_clean=[],
        camera_id=cam["id"],
    )
    at = AppTest.from_file(_ALERTS_PAGE)
    at.run(timeout=30)
    assert not at.exception
    rendered = "\n".join(m.value for m in at.markdown)
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered


def test_cameras_page_escapes_malicious_worst_label():
    cam = storage.register_camera("Cam K")
    flagged_panel = {"condition": {"status": "predicted", "label": _PAYLOAD, "confidence": 0.9}}
    storage.record_ground_inspection(
        [{"label": "a.jpg", "report": {}}], flagged=[flagged_panel], unread=[], confirmed_clean=[],
        camera_id=cam["id"],
    )
    at = AppTest.from_file(_CAMERAS_PAGE)
    at.run(timeout=30)
    assert not at.exception
    rendered = "\n".join(m.value for m in at.markdown)
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
