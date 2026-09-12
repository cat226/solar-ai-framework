"""tests/test_storage_cameras.py — tests for services/storage.py's Phase 18
camera registry functions, and for camera-scoped panel identity matching.
"""
from __future__ import annotations

import pytest

from services import storage


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


class TestRegisterCamera:
    def test_first_camera_gets_number_one(self):
        cam = storage.register_camera("Front roof", "North pole")
        assert cam["camera_number"] == 1
        assert cam["name"] == "Front roof"
        assert cam["status"] == "active"
        assert cam["api_token"]  # a real token was issued

    def test_camera_numbers_increment_and_never_reuse_after_delete(self):
        cam1 = storage.register_camera("A")
        cam2 = storage.register_camera("B")
        assert cam2["camera_number"] == cam1["camera_number"] + 1
        storage.delete_camera(cam1["id"])
        cam3 = storage.register_camera("C")
        assert cam3["camera_number"] == cam2["camera_number"] + 1  # not reused


class TestTokenRedaction:
    def test_list_cameras_redacts_token_by_default(self):
        storage.register_camera("A")
        [cam] = storage.list_cameras()
        assert cam["api_token"] is None

    def test_list_cameras_can_include_token_explicitly(self):
        registered = storage.register_camera("A")
        [cam] = storage.list_cameras(include_token=True)
        assert cam["api_token"] == registered["api_token"]

    def test_get_camera_redacts_token(self):
        registered = storage.register_camera("A")
        cam = storage.get_camera(registered["id"])
        assert cam["api_token"] is None

    def test_get_cameras_with_latest_status_redacts_token(self):
        storage.register_camera("A")
        [cam] = storage.get_cameras_with_latest_status()
        assert cam["api_token"] is None


class TestAuthenticateCamera:
    def test_correct_token_authenticates(self):
        registered = storage.register_camera("A")
        result = storage.authenticate_camera(registered["api_token"])
        assert result is not None
        assert result["id"] == registered["id"]

    def test_wrong_token_fails(self):
        storage.register_camera("A")
        assert storage.authenticate_camera("not-the-real-token") is None

    def test_empty_token_fails(self):
        assert storage.authenticate_camera("") is None

    def test_disabled_camera_token_no_longer_authenticates(self):
        registered = storage.register_camera("A")
        storage.set_camera_status(registered["id"], "disabled")
        assert storage.authenticate_camera(registered["api_token"]) is None

    def test_deleted_camera_token_no_longer_authenticates(self):
        registered = storage.register_camera("A")
        storage.delete_camera(registered["id"])
        assert storage.authenticate_camera(registered["api_token"]) is None


class TestCameraStatus:
    def test_invalid_status_rejected(self):
        registered = storage.register_camera("A")
        with pytest.raises(ValueError, match="active.*disabled"):
            storage.set_camera_status(registered["id"], "banana")

    def test_touch_last_seen_updates_timestamp(self):
        registered = storage.register_camera("A")
        assert storage.get_camera(registered["id"])["last_seen_at"] is None
        storage.touch_camera_last_seen(registered["id"])
        assert storage.get_camera(registered["id"])["last_seen_at"] is not None


class TestCamerasWithLatestStatus:
    def test_camera_with_no_runs_has_none_latest_run(self):
        storage.register_camera("A")
        [cam] = storage.get_cameras_with_latest_status()
        assert cam["latest_run"] is None

    def test_camera_with_a_run_shows_its_summary(self):
        cam = storage.register_camera("A")
        storage.record_ground_inspection(
            [{"label": "x.jpg", "report": {}}],
            flagged=[], unread=[], confirmed_clean=[{"condition": {"label": "CLEAN", "status": "predicted"}}],
            camera_id=cam["id"],
        )
        [row] = storage.get_cameras_with_latest_status()
        assert row["latest_run"] is not None
        assert row["latest_run"]["panel_count"] == 1


class TestGroundInspectionsFilteredByCamera:
    def test_no_filter_returns_all(self):
        cam = storage.register_camera("A")
        storage.record_ground_inspection([{"label": "a", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=cam["id"])
        storage.record_ground_inspection([{"label": "b", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=None)
        assert len(storage.get_recent_ground_inspections()) == 2

    def test_filter_by_camera_id_excludes_manual_uploads(self):
        cam = storage.register_camera("A")
        storage.record_ground_inspection([{"label": "a", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=cam["id"])
        storage.record_ground_inspection([{"label": "b", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=None)
        rows = storage.get_recent_ground_inspections(camera_id=cam["id"])
        assert len(rows) == 1
        assert rows[0]["capture_count"] == 1

    def test_filter_by_none_returns_only_manual_uploads(self):
        cam = storage.register_camera("A")
        storage.record_ground_inspection([{"label": "a", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=cam["id"])
        storage.record_ground_inspection([{"label": "b", "report": {}}], flagged=[], unread=[], confirmed_clean=[], camera_id=None)
        rows = storage.get_recent_ground_inspections(camera_id=None)
        assert len(rows) == 1


class TestPanelIdentityScopedByCamera:
    def _capture(self, label, panel_id="P1", bbox=(0, 0, 400, 1000), status="predicted", cond_label="CLEAN"):
        return {
            "label": label,
            "report": {
                "image_width": 1000, "image_height": 1000,
                "panels": [{
                    "panel_id": panel_id,
                    "detection": {"bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox))},
                    "classifier_condition": {"status": status, "label": cond_label, "confidence": 0.9},
                }],
            },
        }

    def _record(self, camera_id, captures):
        flagged = [{"condition": p["classifier_condition"]} for c in captures for p in c["report"]["panels"]
                   if p["classifier_condition"]["label"] != "CLEAN"]
        clean = [{"condition": p["classifier_condition"]} for c in captures for p in c["report"]["panels"]
                 if p["classifier_condition"]["label"] == "CLEAN"]
        run_id = storage.record_ground_inspection(captures, flagged=flagged, unread=[], confirmed_clean=clean, camera_id=camera_id)
        return storage.record_panel_identities(run_id, captures, camera_id=camera_id)

    def test_two_cameras_at_identical_position_never_match_each_other(self):
        """The exact same normalized bbox, reported by two DIFFERENT
        cameras, must never be treated as the same physical panel — they
        are, or may be, entirely different arrays."""
        cam1 = storage.register_camera("Camera 1")
        cam2 = storage.register_camera("Camera 2")

        result1 = self._record(cam1["id"], [self._capture("d1.jpg")])
        assert result1["new_panel_ids"] == ["PANEL-001"]

        result2 = self._record(cam2["id"], [self._capture("d2.jpg")])
        assert result2["new_panel_ids"] == ["PANEL-002"]  # NOT matched to cam1's panel
        assert result2["matched"] == []

    def test_same_camera_second_run_matches_correctly(self):
        cam = storage.register_camera("Camera 1")
        self._record(cam["id"], [self._capture("d1.jpg")])
        result2 = self._record(cam["id"], [self._capture("d2.jpg")])
        assert result2["matched"][0]["persistent_id"] == "PANEL-001"

    def test_manual_upload_and_camera_never_cross_match(self):
        cam = storage.register_camera("Camera 1")
        self._record(None, [self._capture("manual.jpg")])  # manual upload, camera_id=None
        result = self._record(cam["id"], [self._capture("d1.jpg")])
        assert result["new_panel_ids"] == ["PANEL-002"]
        assert result["matched"] == []
