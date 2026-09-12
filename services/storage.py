"""services/storage.py — Local Ground Inspection history persistence.

Responsibility
--------------
- Record each real, completed Ground Inspection run to a local SQLite
  database so the UI's Overview, Panel Results, Site Health, History,
  Analytics, and Alerts pages have genuine data to show instead of
  fabricated placeholders.
- Provide read helpers for those pages.

This module intentionally does NOT store the uploaded image itself — only
the structured per-panel results already computed by the Ground Inspection
pipeline (app.py). This keeps the database small and avoids an unbounded
local image archive.

The legacy aerial-pipeline `inspections` table (and record_inspection /
get_summary_stats / get_recent_inspections / get_inspection / get_alerts)
was removed alongside the rest of the aerial pipeline — see
docs/PHASE_16_ARCHITECTURE_REVIEW.md. Ground Inspection has a genuinely
different result shape (a dynamic number of panels per run, each
independently read by two condition engines, no weather/physics/XGBoost
involved at all), so it always had its own separate table below.

This is a single-user local SQLite file, not a multi-tenant production
database — appropriate for this Streamlit application's actual deployment
shape. If genuine multi-user/production persistence is needed later, this
module is the seam to replace, not app.py or the individual pages.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Local runtime data — never committed (see .gitignore), created lazily on first use.
DB_PATH = Path("data") / "inspections.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ground_inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 0,
    panel_count INTEGER NOT NULL DEFAULT 0,
    clean_count INTEGER NOT NULL DEFAULT 0,
    dusty_count INTEGER NOT NULL DEFAULT 0,
    bird_drop_count INTEGER NOT NULL DEFAULT 0,
    physical_damage_count INTEGER NOT NULL DEFAULT 0,
    unread_count INTEGER NOT NULL DEFAULT 0,
    flagged_count INTEGER NOT NULL DEFAULT 0,
    worst_label TEXT NOT NULL DEFAULT '',
    worst_confidence REAL NOT NULL DEFAULT 0.0,
    result_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ground_inspections_created_at ON ground_inspections (created_at);
CREATE INDEX IF NOT EXISTS idx_ground_inspections_flagged ON ground_inspections (flagged_count);

-- Phase 17: persistent panel identity across runs (geometry-matched, best
-- effort — see training/detection/ground_pv/panel_identity.py). Current
-- known state of every tracked panel — one row per persistent_id, always
-- reflecting its MOST RECENT observation.
CREATE TABLE IF NOT EXISTS panel_identities (
    persistent_id TEXT PRIMARY KEY,
    capture_slot INTEGER NOT NULL,
    first_seen_run_id INTEGER NOT NULL,
    last_seen_run_id INTEGER NOT NULL,
    last_center_x REAL NOT NULL,
    last_center_y REAL NOT NULL,
    last_condition_status TEXT NOT NULL DEFAULT '',
    last_condition_label TEXT,
    last_condition_confidence REAL,
    observation_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_panel_identities_capture_slot ON panel_identities (capture_slot);

-- Full observation history — one row per (persistent panel, run) it was
-- actually seen in. This is what a per-panel history view reads.
CREATE TABLE IF NOT EXISTS panel_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persistent_id TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    capture_slot INTEGER NOT NULL,
    display_id TEXT NOT NULL,
    condition_status TEXT NOT NULL,
    condition_label TEXT,
    condition_confidence REAL,
    center_x REAL NOT NULL,
    center_y REAL NOT NULL,
    match_distance REAL
);
CREATE INDEX IF NOT EXISTS idx_panel_observations_persistent_id ON panel_observations (persistent_id);
CREATE INDEX IF NOT EXISTS idx_panel_observations_run_id ON panel_observations (run_id);

-- Detected condition transitions (e.g. CLEAN -> DUSTY) for a matched panel
-- between two consecutive CONFIDENT ("predicted") reads. Computed once at
-- write time (same pattern as ground_inspections' precomputed counts)
-- rather than re-derived from panel_observations on every page load.
CREATE TABLE IF NOT EXISTS panel_condition_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persistent_id TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    previous_label TEXT NOT NULL,
    previous_confidence REAL,
    new_label TEXT NOT NULL,
    new_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_panel_condition_changes_created_at ON panel_condition_changes (created_at);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ground Inspection history
# ---------------------------------------------------------------------------

_GROUND_SEVERITY_ORDER = {"PHYSICAL_DAMAGE": 3, "BIRD_DROP": 2, "DUSTY": 1, "CLEAN": 0}


def record_ground_inspection(
    captures: list[dict],
    *, flagged: list[dict], unread: list[dict], confirmed_clean: list[dict],
) -> int:
    """Persist one Ground Inspection run to local history.

    Args:
        captures: The same ``[{"label", "image", "report"}, ...]`` list
            app.py holds in ``st.session_state["ground_inspection_captures"]``
            — only ``label`` and ``report`` are serialized (never the raw
            image).
        flagged: Panels whose CLASSIFIER read (primary signal) is a
            confidently-predicted non-CLEAN condition.
        unread: Panels whose classifier read is uncertain.
        confirmed_clean: Panels whose classifier read is confidently CLEAN.

    Returns:
        The new row's integer id.
    """
    clean_count = len(confirmed_clean)
    dusty_count = sum(1 for p in flagged if p["condition"]["label"] == "DUSTY")
    bird_drop_count = sum(1 for p in flagged if p["condition"]["label"] == "BIRD_DROP")
    physical_damage_count = sum(1 for p in flagged if p["condition"]["label"] == "PHYSICAL_DAMAGE")
    panel_count = clean_count + len(flagged) + len(unread)

    if flagged:
        worst = max(
            flagged,
            key=lambda p: (_GROUND_SEVERITY_ORDER.get(p["condition"]["label"], 0), p["condition"]["confidence"]),
        )
        worst_label, worst_confidence = worst["condition"]["label"], worst["condition"]["confidence"]
    elif unread:
        worst_label, worst_confidence = "UNCERTAIN", 0.0
    elif confirmed_clean:
        worst_label, worst_confidence = "CLEAN", 1.0
    else:
        worst_label, worst_confidence = "", 0.0

    result_dict = {"captures": [{"label": c["label"], "report": c["report"]} for c in captures]}

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO ground_inspections (
                created_at, capture_count, panel_count, clean_count, dusty_count,
                bird_drop_count, physical_damage_count, unread_count, flagged_count,
                worst_label, worst_confidence, result_json
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                len(captures), panel_count, clean_count, dusty_count,
                bird_drop_count, physical_damage_count, len(unread), len(flagged),
                worst_label, worst_confidence, json.dumps(result_dict),
            ),
        )
        row_id = int(cur.lastrowid)
    logger.info(
        "Recorded ground inspection id=%d panels=%d flagged=%d unread=%d",
        row_id, panel_count, len(flagged), len(unread),
    )
    return row_id


def get_recent_ground_inspections(limit: int = 50) -> list[dict]:
    """Return the most recent Ground Inspection runs, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ground_inspections ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_ground_inspection(inspection_id: int) -> Optional[dict]:
    """Return one Ground Inspection run by id, or None if it doesn't exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ground_inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
    return dict(row) if row else None


def get_ground_alerts(limit: int = 50) -> list[dict]:
    """Return recent Ground Inspection runs that flagged at least one
    confidently-non-CLEAN panel, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM ground_inspections
            WHERE flagged_count > 0
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_ground_summary_stats() -> dict:
    """Return aggregate KPIs for Ground Inspection history. All-zero/None
    defaults when no run has been recorded yet — callers should render an
    explicit empty state rather than treat zeros as a real measurement."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM ground_inspections").fetchone()["n"]
        if total == 0:
            return {
                "total_runs": 0, "total_panels_analyzed": 0, "avg_panels_per_run": 0.0,
                "runs_with_flagged_panels": 0, "total_flagged_panels": 0,
                "total_unread_panels": 0, "total_clean_panels": 0,
                "condition_distribution": {},
            }
        total_panels = conn.execute("SELECT SUM(panel_count) AS v FROM ground_inspections").fetchone()["v"]
        avg_panels = conn.execute("SELECT AVG(panel_count) AS v FROM ground_inspections").fetchone()["v"]
        runs_flagged = conn.execute(
            "SELECT COUNT(*) AS n FROM ground_inspections WHERE flagged_count > 0"
        ).fetchone()["n"]
        totals = conn.execute(
            """
            SELECT SUM(clean_count) AS clean, SUM(dusty_count) AS dusty,
                   SUM(bird_drop_count) AS bird_drop, SUM(physical_damage_count) AS physical_damage,
                   SUM(unread_count) AS unread
            FROM ground_inspections
            """
        ).fetchone()

    return {
        "total_runs": total,
        "total_panels_analyzed": int(total_panels or 0),
        "avg_panels_per_run": float(avg_panels or 0.0),
        "runs_with_flagged_panels": runs_flagged,
        "total_flagged_panels": int((totals["dusty"] or 0) + (totals["bird_drop"] or 0) + (totals["physical_damage"] or 0)),
        "total_unread_panels": int(totals["unread"] or 0),
        "total_clean_panels": int(totals["clean"] or 0),
        "condition_distribution": {
            "CLEAN": int(totals["clean"] or 0),
            "DUSTY": int(totals["dusty"] or 0),
            "BIRD_DROP": int(totals["bird_drop"] or 0),
            "PHYSICAL_DAMAGE": int(totals["physical_damage"] or 0),
        },
    }


# ---------------------------------------------------------------------------
# Phase 17 — persistent panel identity across runs (geometry-matched, best
# effort). See training/detection/ground_pv/panel_identity.py for the
# matching algorithm and its documented assumptions/limitations. Matching
# always uses the CLASSIFIER's condition read (primary signal), consistent
# with every other display/history/alert already built on this page.
# ---------------------------------------------------------------------------

_PERSISTENT_ID_PREFIX = "PANEL-"


def _next_persistent_id(conn: sqlite3.Connection) -> str:
    n = conn.execute("SELECT COUNT(*) AS n FROM panel_identities").fetchone()["n"]
    return f"{_PERSISTENT_ID_PREFIX}{n + 1:03d}"


def record_panel_identities(run_id: int, captures: list[dict]) -> dict:
    """Match this run's panels against the most recently known position of
    every tracked panel (per capture slot — see
    training.detection.ground_pv.panel_identity.match_run_to_previous for
    exactly what that assumes), assign/reuse persistent ids, and record
    each observation plus any detected condition change.

    Must be called AFTER record_ground_inspection() for the same run_id —
    it only attributes observations, it does not create the run row.

    Args:
        run_id: The ground_inspections row id this batch of panels belongs to.
        captures: The same [{"label", "image", "report"}, ...] list passed
            to record_ground_inspection (only "label" and "report" are read).

    Returns:
        A dict with new_panel_ids/matched/changes lists — enough detail for
        a caller to show a live summary without a second query.
    """
    from training.detection.ground_pv.panel_identity import (
        CaptureSlotPanels,
        PanelRef,
        match_run_to_previous,
        normalized_center,
    )

    current_slots: list[CaptureSlotPanels] = []
    panel_lookup: dict[str, dict] = {}
    for slot_idx, cap in enumerate(captures):
        report = cap["report"]
        w, h = report.get("image_width"), report.get("image_height")
        panels: list[PanelRef] = []
        for p in report.get("panels", []):
            temp_id = f"slot{slot_idx}-{p['panel_id']}"
            bbox = p["detection"]["bbox"]
            cx, cy = normalized_center([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], w, h)
            panels.append(PanelRef(temp_id, cx, cy))
            panel_lookup[temp_id] = {
                "display_id": p["panel_id"],
                "center_x": cx, "center_y": cy,
                "condition": p["classifier_condition"],
            }
        current_slots.append(CaptureSlotPanels(slot_idx, panels))

    new_panel_ids: list[str] = []
    matched: list[dict] = []
    changes: list[dict] = []

    with _connect() as conn:
        prev_rows = conn.execute("SELECT * FROM panel_identities").fetchall()
        previous_by_slot: dict[int, list] = {}
        prev_state: dict[str, sqlite3.Row] = {}
        for row in prev_rows:
            previous_by_slot.setdefault(row["capture_slot"], []).append(
                PanelRef(row["persistent_id"], row["last_center_x"], row["last_center_y"])
            )
            prev_state[row["persistent_id"]] = row
        previous_slots = [CaptureSlotPanels(slot, panels) for slot, panels in previous_by_slot.items()]

        match_results = match_run_to_previous(current_slots, previous_slots)

        for slot_idx, results in match_results.items():
            for r in results:
                info = panel_lookup[r.current_source_id]
                cond = info["condition"]

                if r.matched_previous_source_id is not None:
                    persistent_id = r.matched_previous_source_id
                    prev_row = prev_state.get(persistent_id)
                    matched.append({
                        "persistent_id": persistent_id, "display_id": info["display_id"],
                        "distance": r.distance,
                    })
                else:
                    persistent_id = _next_persistent_id(conn)
                    prev_row = None
                    new_panel_ids.append(persistent_id)
                    # Reserve the id immediately so two new panels assigned in
                    # the same call never collide on the same COUNT(*)-derived id.
                    conn.execute(
                        "INSERT INTO panel_identities (persistent_id, capture_slot, first_seen_run_id, "
                        "last_seen_run_id, last_center_x, last_center_y, observation_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (persistent_id, slot_idx, run_id, run_id, info["center_x"], info["center_y"]),
                    )

                # A real condition change requires BOTH the previous and the
                # new read to be a CONFIDENT ("predicted") read with a
                # different label - an "uncertain" read is never compared as
                # if it were a confirmed condition (matches this page's
                # existing no_prediction/uncertain discipline).
                if (
                    prev_row is not None
                    and prev_row["last_condition_status"] == "predicted"
                    and cond["status"] == "predicted"
                    and prev_row["last_condition_label"] != cond["label"]
                ):
                    changes.append({
                        "persistent_id": persistent_id, "display_id": info["display_id"],
                        "previous_label": prev_row["last_condition_label"],
                        "previous_confidence": prev_row["last_condition_confidence"],
                        "new_label": cond["label"], "new_confidence": cond["confidence"],
                    })
                    conn.execute(
                        "INSERT INTO panel_condition_changes (persistent_id, run_id, created_at, "
                        "previous_label, previous_confidence, new_label, new_confidence) "
                        "VALUES (?, ?, datetime('now'), ?, ?, ?, ?)",
                        (persistent_id, run_id, prev_row["last_condition_label"],
                         prev_row["last_condition_confidence"], cond["label"], cond["confidence"]),
                    )

                conn.execute(
                    "INSERT INTO panel_observations (persistent_id, run_id, created_at, capture_slot, "
                    "display_id, condition_status, condition_label, condition_confidence, "
                    "center_x, center_y, match_distance) "
                    "VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)",
                    (persistent_id, run_id, slot_idx, info["display_id"], cond["status"],
                     cond["label"], cond["confidence"], info["center_x"], info["center_y"], r.distance),
                )
                conn.execute(
                    "UPDATE panel_identities SET last_seen_run_id = ?, last_center_x = ?, last_center_y = ?, "
                    "last_condition_status = ?, last_condition_label = ?, last_condition_confidence = ?, "
                    "observation_count = observation_count + 1 WHERE persistent_id = ?",
                    (run_id, info["center_x"], info["center_y"], cond["status"], cond["label"],
                     cond["confidence"], persistent_id),
                )

    logger.info(
        "Panel identity matching for run %d: %d new, %d matched, %d condition change(s)",
        run_id, len(new_panel_ids), len(matched), len(changes),
    )
    return {"new_panel_ids": new_panel_ids, "matched": matched, "changes": changes}


def get_panel_identities_summary() -> list[dict]:
    """Every currently-tracked panel's latest known state, most-recently-seen first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM panel_identities ORDER BY last_seen_run_id DESC, persistent_id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_panel_observation_history(persistent_id: str) -> list[dict]:
    """Every observation of one persistent panel, oldest first - a
    chronological condition timeline for that physical panel."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM panel_observations WHERE persistent_id = ? ORDER BY run_id ASC",
            (persistent_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_condition_changes(limit: int = 50) -> list[dict]:
    """Recently detected condition transitions (e.g. CLEAN -> DUSTY) for
    matched panels, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM panel_condition_changes ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
