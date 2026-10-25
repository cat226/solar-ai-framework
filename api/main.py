"""api/main.py — Phase 18 ingestion API for Raspberry Pi camera devices.

Role in the pipeline (per the confirmed design — "Just capture + analyze +
log"): each Pi runs the SAME `services/ground_pipeline.py` inference code
locally (loading the frozen research checkpoints itself) and only POSTs the
already-computed structured result here. This service does NOT run model
inference and never receives a raw image — it authenticates the camera,
persists the result via `services/storage.py` (tagged with that camera's
id), and reports back whether a problem was detected so the Pi's own log
line (and the admin panel) can show that immediately.

Why not send raw images to a central inference server instead: a Pi doing
its own capture+analyze+log keeps this a real edge pipeline (works even if
the admin panel's host is briefly unreachable — the Pi can retry the POST),
and avoids needing to design/secure a second, heavier image-upload endpoint
for a student project's actual deployment shape (see
docs/PHASE_16_ARCHITECTURE_REVIEW.md's "single-user local SQLite, not a
multi-tenant production database" framing — this API is the same scale of
ambition, just reachable over the local network from Pi devices).

Auth: a per-camera bearer token issued once by `storage.register_camera()`
(shown only at creation time in the Cameras Configuration admin page).
Sent as `Authorization: Bearer <token>`. Compared with
`secrets.compare_digest` inside `storage.authenticate_camera()` — never a
plain `==`.

Run locally with: `uvicorn api.main:app --host 0.0.0.0 --port 8008`
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from services import ground_pipeline, notifications, storage
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_STATUSES = frozenset({"predicted", "uncertain", "no_prediction"})
_VALID_LABELS = frozenset(CONDITION_4CLASS_ORDER)


def _validate_condition_dict(cond: Any, where: str) -> None:
    """A camera's `report` is otherwise-unvalidated JSON from an
    authenticated-but-not-fully-trusted device — accepted because this
    service intentionally does not re-run inference (see module docstring),
    but its `label`/`status` values still end up stored as plain SQLite TEXT
    (`ground_inspections.worst_label`) and later rendered by Streamlit pages
    that use `unsafe_allow_html=True` for a status pill on the same line.
    Fails closed here (422) rather than let an arbitrary string reach that
    render path — the fixed 4-class taxonomy is the only legitimate value
    coming out of `ground_pipeline`'s own classifier/detector."""
    if not isinstance(cond, dict):
        raise ValueError(f"{where} must be an object")
    status_val = cond.get("status")
    if status_val not in _VALID_STATUSES:
        raise ValueError(f"{where}.status must be one of {sorted(_VALID_STATUSES)}, got {status_val!r}")
    label = cond.get("label")
    if status_val == "no_prediction":
        if label is not None:
            raise ValueError(f"{where}.label must be null when status is no_prediction, got {label!r}")
    elif label not in _VALID_LABELS:
        raise ValueError(f"{where}.label must be one of {sorted(_VALID_LABELS)}, got {label!r}")
    confidence = cond.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ValueError(f"{where}.confidence must be a number or null, got {confidence!r}")


def _validate_report(report: dict[str, Any]) -> None:
    """Validate the parts of a capture `report` this service actually
    trusts and stores (panel condition reads) against the project's fixed
    4-class taxonomy. Everything else in `report` is opaque/passed through
    unchanged, same as before — this only closes the specific fields that
    reach a rendered page."""
    for i, panel in enumerate(report.get("panels", []) or []):
        if "classifier_condition" in panel:
            _validate_condition_dict(panel["classifier_condition"], f"panels[{i}].classifier_condition")
        if "condition" in panel:
            _validate_condition_dict(panel["condition"], f"panels[{i}].condition")

app = FastAPI(
    title="Solar AI — Camera Ingestion API",
    description="Receives already-analyzed Ground Inspection results from registered Pi camera devices.",
    version="1.0.0",
)


class CaptureIn(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)
    report: dict[str, Any]


class IngestRequest(BaseModel):
    captures: list[CaptureIn] = Field(..., min_length=1, max_length=ground_pipeline.MAX_CAPTURES_PER_RUN)


def _authenticate(authorization: str | None = Header(default=None)) -> dict:
    """Resolve the calling camera from a `Authorization: Bearer <token>`
    header. Fails closed with 401 on anything missing/malformed/wrong —
    never guesses which camera a request is from."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    camera = storage.authenticate_camera(token)
    if camera is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid, disabled, or unknown camera token.")
    return camera


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/ingest")
def ingest(payload: IngestRequest, camera: dict = Depends(_authenticate)) -> dict:
    """Persist one already-analyzed capture batch from a registered camera.

    Each capture's `report` must be the exact JSON `run_panel_condition_pipeline`
    + `ground_pipeline.classify_crop` output shape (i.e. what
    `ground_pipeline.run_capture()` returns on the Pi) — each panel entry
    needs a `classifier_condition` block, since that is the primary signal
    every summary/alert/identity-match on this project is built on.
    """
    captures = [{"label": c.label, "report": c.report} for c in payload.captures]

    try:
        for c in captures:
            _validate_report(c["report"])
        summary = ground_pipeline.summarize_panels(captures)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            422,
            f"Malformed capture report (expected the ground_pipeline.run_capture() output shape): {exc}",
        ) from exc

    run_id = storage.record_ground_inspection(
        captures,
        flagged=summary["flagged"], unread=summary["unread"], confirmed_clean=summary["confirmed_clean"],
        camera_id=camera["id"],
    )

    identity_result: dict[str, Any] | None = None
    try:
        identity_result = storage.record_panel_identities(run_id, captures, camera_id=camera["id"])
    except Exception:  # noqa: BLE001 — identity tracking must never fail the already-saved run
        logger.exception("Failed to match panel identities for camera_id=%s run_id=%d", camera["id"], run_id)

    storage.touch_camera_last_seen(camera["id"])

    # Off by default; never raises (see services/notifications.py). Uses the same flagged list as the stored run.
    notification = notifications.notify_ingest(summary["flagged"], camera, run_id)
    logger.info("Notification for run_id=%d: sent=%s %s", run_id, notification["sent"], notification["reason"])

    problem_detected = len(summary["flagged"]) > 0
    logger.info(
        "Ingested run_id=%d from camera_number=%s (id=%s): %d flagged, %d unread, %d clean%s",
        run_id, camera["camera_number"], camera["id"],
        len(summary["flagged"]), len(summary["unread"]), len(summary["confirmed_clean"]),
        " — PROBLEM DETECTED" if problem_detected else "",
    )

    return {
        "run_id": run_id,
        "camera_id": camera["id"],
        "camera_number": camera["camera_number"],
        "problem_detected": problem_detected,
        "flagged_count": len(summary["flagged"]),
        "unread_count": len(summary["unread"]),
        "clean_count": len(summary["confirmed_clean"]),
        "new_panel_ids": identity_result["new_panel_ids"] if identity_result else [],
        "changes": identity_result["changes"] if identity_result else [],
    }
