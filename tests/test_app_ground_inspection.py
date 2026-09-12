"""tests/test_app_ground_inspection.py — behavioral tests for app.py's
Ground Inspection landing page: the parts that are genuinely app-specific
logic (capture-count truncation, invalid-upload rejection) rather than
just "does it render," which tests/test_pages_smoke.py already covers.

Uses Streamlit's real AppTest harness against the real app.py — no mocking
of the pipeline. Real model artifacts must be present locally for the
run-flow assertions; if they are not, those assertions are skipped rather
than failing, matching this project's existing convention for tests that
depend on gitignored, non-committed weight files.
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


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30), size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _switch_to_upload_mode(at: AppTest) -> AppTest:
    at.radio[0].set_value("Upload your own photo(s)")
    at.run(timeout=30)
    return at


class TestCaptureCountTruncation:
    def test_uploading_more_than_the_cap_truncates_with_a_warning(self):
        """More than _MAX_CAPTURES_PER_RUN (20) valid uploads must be
        truncated to the cap, with a clear on-screen warning — never
        silently processed in full (which could make one run take a very
        long time with no way to see partial progress) and never a crash."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _switch_to_upload_mode(at)

        files = [(f"panel{i}.png", _png_bytes(color=(i, i, i)), "image/png") for i in range(25)]
        at.file_uploader[0].set_value(files)
        at.run(timeout=60)

        assert not at.exception
        assert any("25 photos selected" in w.value and "first 20" in w.value for w in at.warning)

    def test_uploading_at_or_below_the_cap_shows_no_truncation_warning(self):
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _switch_to_upload_mode(at)

        files = [(f"panel{i}.png", _png_bytes(color=(i, i, i)), "image/png") for i in range(5)]
        at.file_uploader[0].set_value(files)
        at.run(timeout=60)

        assert not at.exception
        assert not any("photos selected" in w.value for w in at.warning)


class TestInvalidUploadRejection:
    def test_corrupt_file_is_rejected_not_crashed(self):
        """A non-image file with an image-like name must be rejected with a
        clear per-file error, never crash the page and never silently
        become a phantom capture."""
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _switch_to_upload_mode(at)

        at.file_uploader[0].set_value(("not_really_an_image.png", b"this is not image data", "image/png"))
        at.run(timeout=30)

        assert not at.exception
        assert any("isn't a valid/safe image" in e.value for e in at.error)
        # The Run button must stay disabled — no valid capture exists to run.
        assert at.button[0].disabled

    def test_mix_of_valid_and_invalid_uploads_keeps_only_the_valid_ones(self):
        at = AppTest.from_file(_APP_PY)
        at.run(timeout=30)
        _switch_to_upload_mode(at)

        at.file_uploader[0].set_value([
            ("good.png", _png_bytes(), "image/png"),
            ("bad.png", b"not an image", "image/png"),
        ])
        at.run(timeout=30)

        assert not at.exception
        assert any("isn't a valid/safe image" in e.value for e in at.error)
        # One real capture survived, so the Run button must be enabled.
        assert not at.button[0].disabled
