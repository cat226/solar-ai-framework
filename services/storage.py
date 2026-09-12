"""services/storage.py — Local inspection history persistence.

Responsibility
--------------
- Record each real, successfully-completed pipeline run to a local SQLite
  database so the UI's Dashboard, History, Analytics, and Alerts pages have
  genuine data to show instead of fabricated placeholders.
- Provide read helpers for those pages.

This module intentionally does NOT store the uploaded image itself (only a
SHA-256 hash of it, for identity/dedup purposes) — this keeps the database
small and avoids an unbounded local image archive. It stores only the
structured results already computed by :func:`services.pipeline.run_pipeline`.

Failed pipeline runs (status == "ERROR") are not recorded; there is nothing
genuine to show for them beyond what the UI already displays live.

This is a single-user local SQLite file, not a multi-tenant production
database — appropriate for this Streamlit application's actual deployment
shape. If genuine multi-user/production persistence is needed later, this
module is the seam to replace, not app.py or the individual pages.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Local runtime data — never committed (see .gitignore), created lazily on first use.
DB_PATH = Path("data") / "inspections.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    city TEXT NOT NULL DEFAULT '',
    image_sha256 TEXT NOT NULL DEFAULT '',
    panel_count INTEGER NOT NULL DEFAULT 0,
    detection_confidence REAL NOT NULL DEFAULT 0.0,
    fault_label TEXT NOT NULL DEFAULT '',
    fault_confidence REAL NOT NULL DEFAULT 0.0,
    efficiency_loss_pct REAL NOT NULL DEFAULT 0.0,
    estimated_output_w REAL NOT NULL DEFAULT 0.0,
    severity TEXT NOT NULL DEFAULT 'OK',
    processing_time_s REAL NOT NULL DEFAULT 0.0,
    result_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_inspections_created_at ON inspections (created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_severity ON inspections (severity);

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

# Columns added after the original schema above. Added via idempotent
# ALTER TABLE rather than a migration framework - appropriate for this
# module's own stated scope (a single-user local SQLite file, not a
# multi-tenant production database). Existing rows predate the interim-
# classifier/optional-XGBoost work and are backfilled with honest values:
# every historical row was only ever written by a SUCCESS pipeline run
# under the code that existed when it ran, and prior to this change
# run_pipeline could not reach status=SUCCESS at all unless XGBoost had
# genuinely run - so xgboost_available=1 is a true fact about those rows,
# not a guess. classifier_source is genuinely unknown for rows written
# before this column existed (the pipeline didn't track it yet).
_SCHEMA_MIGRATIONS: list[str] = [
    "ALTER TABLE inspections ADD COLUMN classifier_source TEXT NOT NULL DEFAULT 'unknown'",
    "ALTER TABLE inspections ADD COLUMN xgboost_available INTEGER NOT NULL DEFAULT 1",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


def image_sha256(raw_bytes: bytes) -> str:
    """Compute a SHA-256 hex digest for uploaded image bytes.

    Used only as a stable identifier for dedup/history display — the raw
    image bytes themselves are never persisted.
    """
    return hashlib.sha256(raw_bytes).hexdigest()


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


def record_inspection(result, *, city: str, image_bytes: Optional[bytes] = None) -> int:
    """Persist one successfully-completed pipeline run.

    Args:
        result: A :class:`services.pipeline.PipelineResult` with
                ``status == "SUCCESS"``.
        city: The city string used for this run (kept alongside the result
              since ``PipelineResult.city`` may be empty on some paths).
        image_bytes: Optional raw uploaded image bytes, hashed but not stored.

    Returns:
        The new row's integer id.

    Raises:
        ValueError: If ``result.status`` is not ``"SUCCESS"`` — recording a
                    failed run would misrepresent it as a real inspection.
    """
    if result.status != "SUCCESS":
        raise ValueError("record_inspection() requires a SUCCESS PipelineResult")

    det = result.detection_result
    clf = result.classification_result
    pred = result.efficiency_prediction
    rec = result.recommendations

    result_dict = {
        "detection": asdict(det),
        "classification": asdict(clf),
        "weather": asdict(result.weather_data),
        "physics": asdict(result.physics_data),
        "prediction": asdict(pred),
        "panels": [asdict(p) for p in getattr(result, "panels", [])],
        "site_summary": asdict(getattr(result, "site_summary", None)) if getattr(result, "site_summary", None) is not None else {},
        "classifier_source": getattr(result, "classifier_source", "unknown"),
        "xgboost_available": getattr(result, "xgboost_available", True),
        "recommendations": rec.to_dict(),
        "processing_time": result.processing_time,
        "city": city,
    }

    def _json_default(obj):
        # Weather/physics results carry datetime fields; store them as ISO
        # strings rather than fail the whole record on a non-serializable type.
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO inspections (
                created_at, city, image_sha256, panel_count, detection_confidence,
                fault_label, fault_confidence, efficiency_loss_pct, estimated_output_w,
                severity, processing_time_s, result_json, classifier_source, xgboost_available
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                city,
                image_sha256(image_bytes) if image_bytes else "",
                det.panel_count,
                det.best_confidence,
                clf.label,
                clf.confidence,
                pred.efficiency_loss_pct,
                pred.estimated_output_w,
                rec.overall_severity.value,
                result.processing_time,
                json.dumps(result_dict, default=_json_default),
                getattr(result, "classifier_source", "unknown"),
                1 if getattr(result, "xgboost_available", True) else 0,
            ),
        )
        row_id = int(cur.lastrowid)
    logger.info("Recorded inspection id=%d severity=%s fault=%s", row_id, rec.overall_severity.value, clf.label)
    return row_id


def get_recent_inspections(limit: int = 50) -> list[dict]:
    """Return the most recent inspections, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM inspections ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_inspection(inspection_id: int) -> Optional[dict]:
    """Return one inspection by id, or None if it doesn't exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
    return dict(row) if row else None


def get_alerts(limit: int = 50) -> list[dict]:
    """Return recent inspections with CRITICAL or WARNING severity, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM inspections
            WHERE severity IN ('CRITICAL', 'WARNING')
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    """Return aggregate KPIs for the dashboard, computed from real stored history.

    Returns a dict with total_inspections=0 and all other fields at neutral
    defaults when no history exists yet — callers should render an explicit
    empty state rather than treat zeros as a real measurement.
    """
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM inspections").fetchone()["n"]
        if total == 0:
            return {
                "total_inspections": 0,
                "avg_efficiency_loss_pct": None,
                "avg_panel_count": 0.0,
                "total_panels_analyzed": 0,
                "critical_count": 0,
                "warning_count": 0,
                "fault_distribution": {},
                "clean_count": 0,
                "dusty_count": 0,
                "hotspot_count": 0,
                "xgboost_unavailable_count": 0,
            }
        # Efficiency loss is only meaningful over inspections that actually
        # got a real prediction - averaging in the honest 0.0 placeholder
        # from an unavailable-XGBoost row would understate real losses.
        avg_eff = conn.execute(
            "SELECT AVG(efficiency_loss_pct) AS v FROM inspections WHERE xgboost_available = 1"
        ).fetchone()["v"]
        avg_panels = conn.execute(
            "SELECT AVG(panel_count) AS v FROM inspections"
        ).fetchone()["v"]
        total_panels = conn.execute(
            "SELECT SUM(panel_count) AS v FROM inspections"
        ).fetchone()["v"]
        critical = conn.execute(
            "SELECT COUNT(*) AS n FROM inspections WHERE severity = 'CRITICAL'"
        ).fetchone()["n"]
        warning = conn.execute(
            "SELECT COUNT(*) AS n FROM inspections WHERE severity = 'WARNING'"
        ).fetchone()["n"]
        fault_rows = conn.execute(
            "SELECT fault_label, COUNT(*) AS n FROM inspections GROUP BY fault_label"
        ).fetchall()
        xgb_unavailable = conn.execute(
            "SELECT COUNT(*) AS n FROM inspections WHERE xgboost_available = 0"
        ).fetchone()["n"]

    fault_distribution = {r["fault_label"]: r["n"] for r in fault_rows}
    return {
        "total_inspections": total,
        # None (not 0.0) when no stored inspection ever produced a real
        # prediction (e.g. every row has xgboost_available=0) - a computed
        # 0.0 here would be indistinguishable from a genuine zero-loss
        # measurement. Callers must render this as "unavailable", not "0%".
        "avg_efficiency_loss_pct": float(avg_eff) if avg_eff is not None else None,
        "avg_panel_count": float(avg_panels or 0.0),
        "total_panels_analyzed": int(total_panels or 0),
        "critical_count": critical,
        "warning_count": warning,
        "fault_distribution": fault_distribution,
        "clean_count": fault_distribution.get("Clean", 0),
        "dusty_count": fault_distribution.get("Dusty", 0),
        "hotspot_count": fault_distribution.get("Hotspot", 0),
        "xgboost_unavailable_count": xgb_unavailable,
    }


# ---------------------------------------------------------------------------
# Ground Inspection history — SEPARATE table from `inspections` above, which
# is aerial-pipeline-specific (whole-image fault_label, XGBoost efficiency/
# output prediction). Ground Inspection has a genuinely different result
# shape (a dynamic number of panels per run, each independently read by two
# condition engines, no weather/physics/XGBoost involved at all) — forcing
# it into the aerial schema would mean inventing meaningless values for
# columns like efficiency_loss_pct. One row is recorded per "Run Ground
# Inspection" click (which may bundle multiple photo captures of the same
# array), not per photo — matching how a user thinks about "one inspection."
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
            image, matching ``record_inspection``'s no-image-storage policy).
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
