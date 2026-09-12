"""tests/test_api_ingest.py — Phase 18 camera ingestion API
(api/main.py). Verifies auth, persistence via services/storage.py, and the
"notify with camera number" contract the Cameras/Alerts pages depend on —
without ever running real model inference (captures carry pre-built
`report` dicts, exactly the shape a Pi's own `ground_pipeline.run_capture()`
would produce).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from services import storage


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


@pytest.fixture
def client():
    return TestClient(app)


def _report(panel_id="P1", label="CLEAN", status="predicted", confidence=0.9, bbox=(0, 0, 400, 1000)):
    return {
        "image_width": 1000, "image_height": 1000, "panel_count": 1, "analysis_status": "ok",
        "panels": [{
            "panel_id": panel_id,
            "detection": {"bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox)), "status": "valid", "confidence": 0.95},
            "condition": {"status": status, "label": label, "confidence": confidence},
            "classifier_condition": {"status": status, "label": label, "confidence": confidence,
                                      "probabilities": {"CLEAN": 0.9, "DUSTY": 0.05, "BIRD_DROP": 0.03, "PHYSICAL_DAMAGE": 0.02}},
        }],
    }


class TestAuth:
    def test_missing_authorization_header_rejected(self, client):
        resp = client.post("/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]})
        assert resp.status_code == 401

    def test_malformed_authorization_header_rejected(self, client):
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": "not-bearer-scheme"},
        )
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, client):
        storage.register_camera("Cam A")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": "Bearer not-the-real-token"},
        )
        assert resp.status_code == 401

    def test_disabled_camera_rejected(self, client):
        cam = storage.register_camera("Cam A")
        storage.set_camera_status(cam["id"], "disabled")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 401


class TestIngest:
    def test_clean_capture_persists_and_reports_no_problem(self, client):
        cam = storage.register_camera("Cam A", "Roof north")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["problem_detected"] is False
        assert body["camera_number"] == cam["camera_number"]
        assert body["clean_count"] == 1

        rows = storage.get_recent_ground_inspections(camera_id=cam["id"])
        assert len(rows) == 1

    def test_flagged_capture_reports_problem_with_camera_number(self, client):
        cam = storage.register_camera("Cam B")
        resp = client.post(
            "/v1/ingest",
            json={"captures": [{"label": "a.jpg", "report": _report(label="DUSTY")}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        body = resp.json()
        assert body["problem_detected"] is True
        assert body["flagged_count"] == 1
        assert body["camera_number"] == cam["camera_number"]

    def test_successful_ingest_touches_last_seen(self, client):
        cam = storage.register_camera("Cam C")
        assert storage.get_camera(cam["id"])["last_seen_at"] is None
        client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert storage.get_camera(cam["id"])["last_seen_at"] is not None

    def test_second_run_same_camera_matches_panel_identity(self, client):
        cam = storage.register_camera("Cam D")
        headers = {"Authorization": f"Bearer {cam['api_token']}"}
        client.post("/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]}, headers=headers)
        resp2 = client.post("/v1/ingest", json={"captures": [{"label": "b.jpg", "report": _report()}]}, headers=headers)
        body2 = resp2.json()
        assert body2["new_panel_ids"] == []  # matched, not new

    def test_condition_change_surfaced_in_response(self, client):
        cam = storage.register_camera("Cam E")
        headers = {"Authorization": f"Bearer {cam['api_token']}"}
        client.post("/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report(label="CLEAN")}]}, headers=headers)
        resp2 = client.post("/v1/ingest", json={"captures": [{"label": "b.jpg", "report": _report(label="DUSTY")}]}, headers=headers)
        body2 = resp2.json()
        assert len(body2["changes"]) == 1
        assert body2["changes"][0]["previous_label"] == "CLEAN"
        assert body2["changes"][0]["new_label"] == "DUSTY"

    def test_malformed_report_returns_422_not_500(self, client):
        cam = storage.register_camera("Cam F")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": {"panel_count": 1}}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 422

    def test_invalid_condition_label_rejected(self, client):
        """A camera device is authenticated but not fully trusted content-
        wise — the ingestion API never re-runs inference, so it must reject
        any classifier_condition.label outside the fixed 4-class taxonomy
        rather than storing it verbatim (see api/main.py::_validate_report;
        this is what closes off the stored-XSS path into worst_label)."""
        cam = storage.register_camera("Cam I")
        malicious_report = _report(label="<img src=x onerror=alert(1)>")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": malicious_report}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 422
        assert storage.get_recent_ground_inspections(camera_id=cam["id"]) == []

    def test_invalid_condition_status_rejected(self, client):
        cam = storage.register_camera("Cam J")
        bad_report = _report(status="not_a_real_status")
        resp = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": bad_report}]},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 422

    def test_empty_captures_list_rejected(self, client):
        cam = storage.register_camera("Cam G")
        resp = client.post(
            "/v1/ingest", json={"captures": []},
            headers={"Authorization": f"Bearer {cam['api_token']}"},
        )
        assert resp.status_code == 422

    def test_two_cameras_never_cross_match_via_api(self, client):
        cam1 = storage.register_camera("Cam H1")
        cam2 = storage.register_camera("Cam H2")
        client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": f"Bearer {cam1['api_token']}"},
        )
        resp2 = client.post(
            "/v1/ingest", json={"captures": [{"label": "a.jpg", "report": _report()}]},
            headers={"Authorization": f"Bearer {cam2['api_token']}"},
        )
        assert resp2.json()["new_panel_ids"] != []  # NOT matched to cam1's panel


class TestHealth:
    def test_health_endpoint_ok(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
