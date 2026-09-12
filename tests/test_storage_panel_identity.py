"""tests/test_storage_panel_identity.py — tests for services/storage.py's
Phase 17 persistent-panel-identity functions
(record_panel_identities/get_panel_identities_summary/
get_panel_observation_history/get_recent_condition_changes).

Builds minimal-but-schema-accurate capture/report dicts directly (the same
shape app.py's real pipeline produces) rather than running real model
inference — this module is pure orchestration/persistence logic, not
inference, so that's the right thing to test in isolation.
"""
from __future__ import annotations

import pytest

from services import storage


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test_inspections.db")


def _capture(label: str, *, width: int, height: int, panels: list[dict]) -> dict:
    """panels: [{"panel_id": "P1", "bbox": [x1,y1,x2,y2], "label": "CLEAN",
    "confidence": 0.9, "status": "predicted"}, ...]"""
    return {
        "label": label,
        "report": {
            "image_width": width,
            "image_height": height,
            "panels": [
                {
                    "panel_id": p["panel_id"],
                    "detection": {"bbox": dict(zip(("x1", "y1", "x2", "y2"), p["bbox"]))},
                    "classifier_condition": {
                        "status": p.get("status", "predicted"),
                        "label": p["label"],
                        "confidence": p.get("confidence", 0.9),
                    },
                }
                for p in panels
            ],
        },
    }


def _record_run(captures: list[dict], *, flagged_labels=("DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE")) -> tuple[int, dict]:
    """Record a ground_inspections row (real record_ground_inspection call,
    not a stub) then run identity matching against it — the same two-step
    sequence app.py performs."""
    flagged, unread, confirmed_clean = [], [], []
    for cap in captures:
        for p in cap["report"]["panels"]:
            cond = p["classifier_condition"]
            entry = {"condition": cond}
            if cond["status"] == "predicted" and cond["label"] == "CLEAN":
                confirmed_clean.append(entry)
            elif cond["status"] == "predicted":
                flagged.append(entry)
            else:
                unread.append(entry)
    run_id = storage.record_ground_inspection(captures, flagged=flagged, unread=unread, confirmed_clean=confirmed_clean)
    result = storage.record_panel_identities(run_id, captures)
    return run_id, result


class TestFirstRunAllNew:
    def test_first_run_assigns_new_ids_to_every_panel(self):
        captures = [_capture("photo1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
            {"panel_id": "P2", "bbox": [600, 0, 1000, 1000], "label": "DUSTY"},
        ])]
        _run_id, result = _record_run(captures)
        assert result["new_panel_ids"] == ["PANEL-001", "PANEL-002"]
        assert result["matched"] == []
        assert result["changes"] == []

    def test_summary_reflects_first_run_state(self):
        captures = [_capture("photo1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN", "confidence": 0.91},
        ])]
        _record_run(captures)
        summary = storage.get_panel_identities_summary()
        assert len(summary) == 1
        assert summary[0]["persistent_id"] == "PANEL-001"
        assert summary[0]["last_condition_label"] == "CLEAN"
        assert summary[0]["observation_count"] == 1


class TestSecondRunMatching:
    def test_same_position_reuses_the_same_persistent_id(self):
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
        ])]
        _record_run(captures1)

        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [5, 5, 405, 995], "label": "CLEAN"},  # near-identical position
        ])]
        _run_id2, result2 = _record_run(captures2)

        assert result2["new_panel_ids"] == []
        assert len(result2["matched"]) == 1
        assert result2["matched"][0]["persistent_id"] == "PANEL-001"

        summary = storage.get_panel_identities_summary()
        assert len(summary) == 1  # still only one tracked panel, not two
        assert summary[0]["observation_count"] == 2

    def test_condition_change_is_detected_between_confident_reads(self):
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN", "confidence": 0.9},
        ])]
        _record_run(captures1)

        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "DUSTY", "confidence": 0.8},
        ])]
        _run_id2, result2 = _record_run(captures2)

        assert len(result2["changes"]) == 1
        change = result2["changes"][0]
        assert change["persistent_id"] == "PANEL-001"
        assert change["previous_label"] == "CLEAN"
        assert change["new_label"] == "DUSTY"

        changes = storage.get_recent_condition_changes()
        assert len(changes) == 1
        assert changes[0]["previous_label"] == "CLEAN"
        assert changes[0]["new_label"] == "DUSTY"

    def test_uncertain_read_never_counted_as_a_condition_change(self):
        """An 'uncertain' read must never be compared as if it were a
        confirmed condition — matches the page's existing discipline that
        uncertain is never treated as a fact."""
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN", "confidence": 0.9},
        ])]
        _record_run(captures1)

        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "DUSTY", "confidence": 0.3, "status": "uncertain"},
        ])]
        _run_id2, result2 = _record_run(captures2)

        assert result2["changes"] == []
        assert storage.get_recent_condition_changes() == []

    def test_far_away_panel_is_treated_as_new_not_matched(self):
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
        ])]
        _record_run(captures1)

        # A panel on the opposite side of the frame — beyond the match cutoff.
        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [900, 900, 1000, 1000], "label": "CLEAN"},
        ])]
        _run_id2, result2 = _record_run(captures2)

        assert result2["new_panel_ids"] == ["PANEL-002"]
        assert result2["matched"] == []
        assert len(storage.get_panel_identities_summary()) == 2

    def test_new_panel_in_second_run_gets_its_own_id_alongside_matched_one(self):
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
        ])]
        _record_run(captures1)

        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
            {"panel_id": "P2", "bbox": [600, 0, 1000, 1000], "label": "DUSTY"},
        ])]
        _run_id2, result2 = _record_run(captures2)

        assert result2["new_panel_ids"] == ["PANEL-002"]
        assert len(result2["matched"]) == 1
        assert result2["matched"][0]["persistent_id"] == "PANEL-001"


class TestObservationHistory:
    def test_history_is_chronological_and_complete(self):
        captures1 = [_capture("d1.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
        ])]
        _record_run(captures1)
        captures2 = [_capture("d2.jpg", width=1000, height=1000, panels=[
            {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "DUSTY"},
        ])]
        _record_run(captures2)

        history = storage.get_panel_observation_history("PANEL-001")
        assert len(history) == 2
        assert [h["condition_label"] for h in history] == ["CLEAN", "DUSTY"]

    def test_unknown_persistent_id_returns_empty_history(self):
        assert storage.get_panel_observation_history("PANEL-999") == []


class TestCrossSlotIsolation:
    def test_panels_in_different_capture_slots_never_match_each_other(self):
        """Two DIFFERENT captures in the SAME run, each contributing one
        panel at the identical normalized position — they must get
        DIFFERENT persistent ids (slot 0's panel and slot 1's panel are
        never compared), matching panel_identity.py's documented no-cross-
        slot-matching guarantee."""
        captures = [
            _capture("left.jpg", width=1000, height=1000, panels=[
                {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
            ]),
            _capture("right.jpg", width=1000, height=1000, panels=[
                {"panel_id": "P1", "bbox": [0, 0, 400, 1000], "label": "CLEAN"},
            ]),
        ]
        _run_id, result = _record_run(captures)
        assert result["new_panel_ids"] == ["PANEL-001", "PANEL-002"]
