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
"""


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
                severity, processing_time_s, result_json
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "avg_efficiency_loss_pct": 0.0,
                "avg_panel_count": 0.0,
                "critical_count": 0,
                "warning_count": 0,
                "fault_distribution": {},
            }
        avg_eff = conn.execute(
            "SELECT AVG(efficiency_loss_pct) AS v FROM inspections"
        ).fetchone()["v"]
        avg_panels = conn.execute(
            "SELECT AVG(panel_count) AS v FROM inspections"
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

    return {
        "total_inspections": total,
        "avg_efficiency_loss_pct": float(avg_eff or 0.0),
        "avg_panel_count": float(avg_panels or 0.0),
        "critical_count": critical,
        "warning_count": warning,
        "fault_distribution": {r["fault_label"]: r["n"] for r in fault_rows},
    }
