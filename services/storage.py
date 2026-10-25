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
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

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

-- Phase 18: registered camera/Pi devices. A run/panel with camera_id NULL
-- means it came from a manual upload in the Streamlit UI, not a registered
-- device — that path is deliberately kept working without requiring a
-- camera registration.
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_number INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    api_token TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cameras_camera_number ON cameras (camera_number);
CREATE INDEX IF NOT EXISTS idx_cameras_api_token ON cameras (api_token);
"""

# Columns added to already-existing tables after their original CREATE TABLE
# above (idempotent ALTER TABLE, not a migration framework — appropriate for
# this module's own stated single-user-local-SQLite scope; duplicate-column
# errors are swallowed, any other error is not).
_SCHEMA_MIGRATIONS: list[str] = [
    "ALTER TABLE ground_inspections ADD COLUMN camera_id INTEGER",
    "ALTER TABLE panel_identities ADD COLUMN camera_id INTEGER",
    "ALTER TABLE panel_observations ADD COLUMN camera_id INTEGER",
    "ALTER TABLE panel_condition_changes ADD COLUMN camera_id INTEGER",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        _apply_migrations(conn)
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
    camera_id: int | None = None,
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
        camera_id: The registered camera (services.storage.cameras) this run
            came from, or None for a manual upload in the Streamlit UI —
            manual uploads are never required to belong to a camera.

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
                worst_label, worst_confidence, result_json, camera_id
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                len(captures), panel_count, clean_count, dusty_count,
                bird_drop_count, physical_damage_count, len(unread), len(flagged),
                worst_label, worst_confidence, json.dumps(result_dict), camera_id,
            ),
        )
        row_id = int(cur.lastrowid)
    logger.info(
        "Recorded ground inspection id=%d camera_id=%s panels=%d flagged=%d unread=%d",
        row_id, camera_id, panel_count, len(flagged), len(unread),
    )
    return row_id


_UNSET: Any = object()


def get_recent_ground_inspections(limit: int = 50, camera_id: Any = _UNSET) -> list[dict]:
    """Return the most recent Ground Inspection runs, newest first.

    camera_id: omit for all runs (any camera or manual upload). Pass an int
    to filter to one camera, or explicitly pass None to see only manual
    (non-camera) uploads.
    """
    with _connect() as conn:
        if camera_id is _UNSET:
            rows = conn.execute(
                "SELECT * FROM ground_inspections ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ground_inspections WHERE camera_id IS ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (camera_id, limit),
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
    confidently-non-CLEAN panel, newest first. Each row carries
    `camera_number`/`camera_name` (NULL for a manual upload, or for a run
    whose camera has since been deleted — see delete_camera()'s orphaned-
    reference note) so an alert can always say WHICH camera saw the
    problem, per the admin panel's "notify with camera number" requirement."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT gi.*, c.camera_number AS camera_number, c.name AS camera_name
            FROM ground_inspections gi
            LEFT JOIN cameras c ON c.id = gi.camera_id
            WHERE gi.flagged_count > 0
            ORDER BY gi.created_at DESC, gi.id DESC LIMIT ?
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


def record_panel_identities(run_id: int, captures: list[dict], *, camera_id: int | None = None) -> dict:
    """Match this run's panels against the most recently known position of
    every tracked panel FROM THE SAME CAMERA (per capture slot — see
    training.detection.ground_pv.panel_identity.match_run_to_previous for
    exactly what that assumes), assign/reuse persistent ids, and record
    each observation plus any detected condition change.

    Must be called AFTER record_ground_inspection() for the same run_id —
    it only attributes observations, it does not create the run row.

    Args:
        run_id: The ground_inspections row id this batch of panels belongs to.
        captures: The same [{"label", "image", "report"}, ...] list passed
            to record_ground_inspection (only "label" and "report" are read).
        camera_id: Must match the camera_id passed to record_ground_inspection
            for this run (or None for both, for a manual upload). Matching is
            STRICTLY scoped to this camera_id — a panel from one camera can
            never match a panel from a different camera (or from a manual
            upload), since they are, or may be, different physical arrays.

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
        prev_rows = conn.execute(
            "SELECT * FROM panel_identities WHERE camera_id IS ?", (camera_id,)
        ).fetchall()
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
                        "last_seen_run_id, last_center_x, last_center_y, observation_count, camera_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                        (persistent_id, slot_idx, run_id, run_id, info["center_x"], info["center_y"], camera_id),
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
                        "previous_label, previous_confidence, new_label, new_confidence, camera_id) "
                        "VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)",
                        (persistent_id, run_id, prev_row["last_condition_label"],
                         prev_row["last_condition_confidence"], cond["label"], cond["confidence"], camera_id),
                    )

                conn.execute(
                    "INSERT INTO panel_observations (persistent_id, run_id, created_at, capture_slot, "
                    "display_id, condition_status, condition_label, condition_confidence, "
                    "center_x, center_y, match_distance, camera_id) "
                    "VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (persistent_id, run_id, slot_idx, info["display_id"], cond["status"],
                     cond["label"], cond["confidence"], info["center_x"], info["center_y"], r.distance, camera_id),
                )
                conn.execute(
                    "UPDATE panel_identities SET last_seen_run_id = ?, last_center_x = ?, last_center_y = ?, "
                    "last_condition_status = ?, last_condition_label = ?, last_condition_confidence = ?, "
                    "observation_count = observation_count + 1 WHERE persistent_id = ?",
                    (run_id, info["center_x"], info["center_y"], cond["status"], cond["label"],
                     cond["confidence"], persistent_id),
                )

    logger.info(
        "Panel identity matching for run %d (camera_id=%s): %d new, %d matched, %d condition change(s)",
        run_id, camera_id, len(new_panel_ids), len(matched), len(changes),
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


# ---------------------------------------------------------------------------
# Phase 18 — registered camera/Pi devices. A camera reports captures to the
# ingestion API (see api/main.py), authenticated with its own api_token.
# Deleting a camera does NOT cascade-delete its historical runs/panels —
# their camera_id becomes an orphaned reference on purpose (never silently
# destroy real recorded history); callers should handle a camera_id with no
# matching row gracefully (e.g. "deleted camera").
# ---------------------------------------------------------------------------


def register_camera(name: str, location: str = "") -> dict:
    """Register a new camera and issue it an API token.

    The token is only ever returned HERE, at creation time — list_cameras()
    deliberately never re-exposes it (show/copy it to configure the Pi
    immediately, matching how e.g. GitHub only shows a new access token once).

    Returns the full new row, including the one-time api_token.
    """
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        next_number = conn.execute(
            "SELECT COALESCE(MAX(camera_number), 0) + 1 AS n FROM cameras"
        ).fetchone()["n"]
        cur = conn.execute(
            "INSERT INTO cameras (camera_number, name, location, api_token, status, created_at) "
            "VALUES (?, ?, ?, ?, 'active', datetime('now'))",
            (next_number, name.strip(), location.strip(), token),
        )
        camera_id = int(cur.lastrowid)
        row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    logger.info("Registered camera id=%d camera_number=%d name=%r", camera_id, next_number, name)
    return dict(row)


def list_cameras(*, include_token: bool = False) -> list[dict]:
    """All registered cameras, ordered by camera_number. api_token is
    redacted by default — pass include_token=True only for the
    configuration page's own "show token again for re-setup" action, never
    for a general listing/overview page."""
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM cameras ORDER BY camera_number ASC").fetchall()]
    if not include_token:
        for r in rows:
            r["api_token"] = None
    return rows


def get_camera(camera_id: int) -> Optional[dict]:
    """One camera by internal id (api_token redacted)."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["api_token"] = None
    return d


def authenticate_camera(api_token: str) -> Optional[dict]:
    """Used by the ingestion API to identify+authenticate a request. Returns
    the FULL row (including the token, since the caller already has it and
    this is the one legitimate internal use) only for an 'active' camera —
    a disabled camera's token stops working without deleting its history."""
    if not api_token:
        return None
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM cameras WHERE status = 'active'").fetchall()
    for row in rows:
        if secrets.compare_digest(row["api_token"], api_token):
            return dict(row)
    return None


def touch_camera_last_seen(camera_id: int) -> None:
    """Record that a camera successfully reported in — called on every
    successful ingestion so the Cameras page can show real last-seen times."""
    with _connect() as conn:
        conn.execute(
            "UPDATE cameras SET last_seen_at = datetime('now') WHERE id = ?", (camera_id,)
        )


def set_camera_status(camera_id: int, status: str) -> None:
    """status: 'active' or 'disabled'. Disabling stops the camera's token
    from authenticating without deleting any of its history — the
    reversible option, preferred over delete_camera() for "pause this
    camera" rather than "remove it permanently"."""
    if status not in ("active", "disabled"):
        raise ValueError(f"status must be 'active' or 'disabled', got {status!r}")
    with _connect() as conn:
        conn.execute("UPDATE cameras SET status = ? WHERE id = ?", (status, camera_id))
    logger.info("Camera id=%d status set to %s", camera_id, status)


def delete_camera(camera_id: int) -> None:
    """Permanently remove a camera registration (its token stops working
    immediately). Historical ground_inspections/panel rows are NOT deleted
    or reassigned — they keep their camera_id as an orphaned reference,
    never silently destroying real recorded history."""
    with _connect() as conn:
        conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    logger.info("Deleted camera id=%d", camera_id)


def get_cameras_with_latest_status() -> list[dict]:
    """Every registered camera plus a summary of its most recent run (or
    None if it has never reported in) — the single query the Cameras
    overview page needs."""
    with _connect() as conn:
        cameras = conn.execute("SELECT * FROM cameras ORDER BY camera_number ASC").fetchall()
        result = []
        for cam in cameras:
            latest = conn.execute(
                "SELECT id, created_at, panel_count, flagged_count, unread_count, worst_label "
                "FROM ground_inspections WHERE camera_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (cam["id"],),
            ).fetchone()
            row = dict(cam)
            row["api_token"] = None
            row["latest_run"] = dict(latest) if latest else None
            result.append(row)
    return result
