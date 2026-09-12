"""tests/test_pages_camera_admin.py — real AppTest interaction with the
Phase 18 camera admin pages (registration, one-time token display,
enable/disable, delete-with-confirmation), and the Cameras overview page's
problem-camera banner. Goes beyond tests/test_pages_smoke.py's
render-without-exception check by actually driving the widgets.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PAGE = str(_REPO_ROOT / "pages" / "13_🛠️_Camera_Configuration.py")
_CAMERAS_PAGE = str(_REPO_ROOT / "pages" / "12_📷_Cameras.py")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


class TestCameraConfigurationPage:
    def test_registering_a_camera_shows_its_one_time_token(self):
        at = AppTest.from_file(_CONFIG_PAGE)
        at.run(timeout=30)
        at.text_input[0].set_value("Rooftop North").run()
        at.button[0].click().run()  # "➕ Register camera"
        assert not at.exception

        cam = storage.list_cameras()[0]
        assert cam["name"] == "Rooftop North"
        # the real token is shown somewhere on the rendered page
        codes = [c.value for c in at.code]
        assert any(len(v) > 20 for v in codes)

    def test_empty_name_rejected(self):
        at = AppTest.from_file(_CONFIG_PAGE)
        at.run(timeout=30)
        at.button[0].click().run()
        assert not at.exception
        assert storage.list_cameras() == []
        assert any("required" in e.value.lower() for e in at.error)

    def test_disable_then_enable_round_trips(self):
        cam = storage.register_camera("Cam A")
        at = AppTest.from_file(_CONFIG_PAGE)
        at.run(timeout=30)
        disable_btn = next(b for b in at.button if b.label == "Disable")
        disable_btn.click().run()
        assert storage.get_camera(cam["id"])["status"] == "disabled"

        at2 = AppTest.from_file(_CONFIG_PAGE)
        at2.run(timeout=30)
        enable_btn = next(b for b in at2.button if b.label == "Enable")
        enable_btn.click().run()
        assert storage.get_camera(cam["id"])["status"] == "active"

    def test_remove_requires_confirmation_click(self):
        cam = storage.register_camera("Cam B")
        at = AppTest.from_file(_CONFIG_PAGE)
        at.run(timeout=30)
        remove_btn = next(b for b in at.button if b.label == "Remove")
        remove_btn.click().run()
        # first click only arms confirmation - camera still exists
        assert storage.get_camera(cam["id"]) is not None
        confirm_btn = next(b for b in at.button if "Confirm delete" in b.label)
        confirm_btn.click().run()
        assert storage.get_camera(cam["id"]) is None


class TestCamerasOverviewPage:
    def test_no_cameras_shows_empty_state(self):
        at = AppTest.from_file(_CAMERAS_PAGE)
        at.run(timeout=30)
        assert not at.exception
        assert any("No cameras registered" in i.value for i in at.info)

    def test_camera_with_no_runs_shows_never_reported(self):
        storage.register_camera("Cam C")
        at = AppTest.from_file(_CAMERAS_PAGE)
        at.run(timeout=30)
        assert not at.exception
        assert any("No connected camera" in s.value for s in at.success)

    def test_flagged_run_surfaces_problem_banner_with_camera_number(self):
        cam = storage.register_camera("Cam D")
        flagged_panel = {"condition": {"status": "predicted", "label": "DUSTY", "confidence": 0.9}}
        storage.record_ground_inspection(
            [{"label": "a.jpg", "report": {}}],
            flagged=[flagged_panel], unread=[], confirmed_clean=[], camera_id=cam["id"],
        )
        at = AppTest.from_file(_CAMERAS_PAGE)
        at.run(timeout=30)
        assert not at.exception
        assert any(f"#{cam['camera_number']}" in e.value for e in at.error)
