"""Smoke tests for the multi-page Streamlit UI (pages/) using Streamlit's
AppTest framework. These verify each page renders without raising, in both
the empty-history state and after a real inspection has been recorded -
they do not assert on fabricated data, only that real/empty states render.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import storage
from services.pipeline import PipelineResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_FILES = sorted(str(p) for p in (_REPO_ROOT / "pages").glob("*.py"))
_APP_PY = str(_REPO_ROOT / "app.py")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


@pytest.fixture
def recorded_inspection(detection_single_panel, classification_clean, default_physics, prediction_normal):
    from services.recommendation import generate_recommendations

    recs = generate_recommendations(
        classification=classification_clean, physics=default_physics, prediction=prediction_normal,
    )
    result = PipelineResult(
        detection_result=detection_single_panel,
        classification_result=classification_clean,
        physics_data=default_physics,
        efficiency_prediction=prediction_normal,
        recommendations=recs,
        processing_time=1.0,
        status="SUCCESS",
        city="Testville",
    )
    storage.record_inspection(result, city="Testville")


@pytest.mark.parametrize("page_path", PAGE_FILES)
class TestPageSmoke:
    def test_renders_without_exception_empty_history(self, page_path):
        at = AppTest.from_file(page_path)
        at.run(timeout=30)
        assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"

    def test_renders_without_exception_with_history(self, page_path, recorded_inspection):
        at = AppTest.from_file(page_path)
        at.run(timeout=30)
        assert not at.exception, f"{page_path} raised: {[str(e) for e in at.exception]}"


def test_app_entrypoint_renders_without_exception():
    at = AppTest.from_file(_APP_PY)
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
