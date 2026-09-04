"""tests/test_app_analyze_gating.py — app.py must not re-run the pipeline
(or write a duplicate history row) on every sidebar rerun.

Streamlit reruns the whole script on every widget interaction. Before this
was fixed, an uploaded image sitting in the sidebar meant nudging *any*
other sidebar input (panel age, voltage, city, ...) re-triggered a full
detection + classification + prediction pass and inserted another history
row for the same photo. These tests exercise the real app.py end-to-end
via Streamlit's AppTest (real pipeline, real locally-available artifacts —
no mocking of services.pipeline), asserting only on the *count* of stored
inspections, which is robust whether the real model artifacts happen to be
present in this environment or not.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from services import storage

_APP_PY = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(at: AppTest, *, content: bytes | None = None, filename: str = "panel.png") -> AppTest:
    at.sidebar.file_uploader[0].clear().upload(filename, content or _png_bytes(), "image/png")
    at.run(timeout=60)
    return at


class TestAnalyzeGating:
    def test_uploading_alone_does_not_run_the_pipeline(self):
        """An uploaded image with no button click yet must not produce a
        result or a history row - only an explicit prompt to click Analyze."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _upload(at)

        assert not at.exception
        assert any("Click" in i.value and "Analyze" in i.value for i in at.info)
        assert storage.get_recent_inspections(limit=10) == []

    def test_analyze_click_runs_pipeline_and_records_once(self):
        """A click must produce exactly one real outcome. Whether that's a
        recorded inspection (SUCCESS) or none at all (a real ModelLoadError
        when this environment has no model artifacts, e.g. CI) depends on
        which artifacts happen to be present - both are legitimate; a
        silent no-op or more than one row is not."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _upload(at)

        at.button[0].click().run(timeout=60)
        assert not at.exception
        succeeded = any("Pipeline completed" in s.value for s in at.success)
        errored = len(at.error) > 0
        assert succeeded or errored, "Analyze click produced neither a success nor an error message."
        assert len(storage.get_recent_inspections(limit=10)) == (1 if succeeded else 0)

    def test_widget_only_rerun_after_analyze_does_not_duplicate_history(self):
        """The core regression this suite guards: after a real Analyze
        click, touching an unrelated sidebar widget (no second click) must
        rerun the script without invoking the pipeline again."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _upload(at)

        at.button[0].click().run(timeout=60)
        after_click_count = len(storage.get_recent_inspections(limit=10))

        # Nudge a sidebar number_input (panel age) - a plain rerun, no click.
        at.sidebar.number_input[0].set_value(5.0).run(timeout=60)

        assert not at.exception
        assert len(storage.get_recent_inspections(limit=10)) == after_click_count

    def test_repeated_reruns_with_same_image_still_do_not_duplicate(self):
        """Multiple plain reruns in a row (simulating several widget nudges)
        must never accumulate more than the one recorded inspection."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _upload(at)
        at.button[0].click().run(timeout=60)
        after_click_count = len(storage.get_recent_inspections(limit=10))

        for value in (1.0, 2.0, 3.0):
            at.sidebar.number_input[0].set_value(value).run(timeout=60)

        assert not at.exception
        assert len(storage.get_recent_inspections(limit=10)) == after_click_count

    def test_uploading_a_different_image_clears_the_previous_result(self):
        """A newly uploaded image must not keep showing the previous
        photo's result until the user explicitly re-analyzes."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _upload(at)
        at.button[0].click().run(timeout=60)
        # The click must have produced a real outcome - not still be sitting
        # on the pre-click "click Analyze" prompt.
        assert not any("Click" in i.value and "Analyze" in i.value for i in at.info)

        # Re-upload a different image without clicking Analyze again.
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), color=(200, 50, 5)).save(buf, format="PNG")
        _upload(at, content=buf.getvalue(), filename="panel2.png")

        assert not at.exception
        assert any("Click" in i.value and "Analyze" in i.value for i in at.info)
