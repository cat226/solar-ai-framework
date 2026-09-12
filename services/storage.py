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
