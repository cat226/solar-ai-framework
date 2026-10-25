"""Smoke tests for the multi-page Streamlit UI (pages/) using Streamlit's
AppTest framework. These verify each page renders without raising, in both
the empty-history state and after a real Ground Inspection run has been
recorded - they do not assert on fabricated data, only that real/empty
states render.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_FILES = sorted(str(p) for p in (_REPO_ROOT / "pages").glob("*.py"))
_APP_PY = str(_REPO_ROOT / "app.py")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


@pytest.fixture
def recorded_ground_inspection():
    """Record one real Ground Inspection run via services.storage's real
    write path (not mocked) — a minimal but schema-accurate report, so
    every dashboard page's "has history" branch gets genuine exercise."""
    flagged_panel = {"display_id": "P1", "condition": {"status": "predicted", "label": "DUSTY", "confidence": 0.9}}
    clean_panel = {"display_id": "P2", "condition": {"status": "predicted", "label": "CLEAN", "confidence": 0.95}}
    fake_report = {
        "schema_version": "1.0",
        "image": "smoke_test.jpg",
        "panel_count": 2,
        "analysis_status": "ok",
        "panels": [
            {
                "panel_id": "P1",
                "detection": {"confidence": 0.9, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "status": "valid"},
                "condition": {"status": "predicted", "label": "DUSTY", "confidence": 0.9},
                "classifier_condition": {"status": "predicted", "label": "DUSTY", "confidence": 0.9},
                "crop": {"bbox": [0, 0, 10, 10], "warnings": []},
            },
            {
                "panel_id": "P2",
                "detection": {"confidence": 0.95, "bbox": {"x1": 10, "y1": 0, "x2": 20, "y2": 10}, "status": "valid"},
                "condition": {"status": "predicted", "label": "CLEAN", "confidence": 0.95},
                "classifier_condition": {"status": "predicted", "label": "CLEAN", "confidence": 0.95},
                "crop": {"bbox": [10, 0, 20, 10], "warnings": []},
            },
        ],
    }
    captures = [{"label": "smoke_test.jpg", "report": fake_report}]
    storage.record_ground_inspection(captures, flagged=[flagged_panel], unread=[], confirmed_clean=[clean_panel])


@pytest.mark.parametrize("page_path", PAGE_FILES)
class TestPageSmoke:
    def test_renders_without_exception_empty_history(self, page_path):
        at = AppTest.from_file(page_path)
        at.run(timeout=30)
        assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"

    def test_renders_without_exception_with_history(self, page_path, recorded_ground_inspection):
        at = AppTest.from_file(page_path)
        at.run(timeout=30)
        assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"


def test_app_entrypoint_renders_without_exception():
    at = AppTest.from_file(_APP_PY)
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
