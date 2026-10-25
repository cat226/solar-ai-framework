"""services/notifications.py — outbound alert notifications for the Ground Inspection ingestion path
(Raspberry Pi camera -> api/main.py /v1/ingest).

OFF BY DEFAULT. Enabled only when SOLAR_AI_NOTIFY is on AND at least one channel is configured.
Channels: a generic JSON webhook (works with ntfy, Slack/Discord incoming webhooks, n8n, ...) and SMTP email.

Safety properties (each covered by tests/test_notifications.py):
  * never raises into the caller - a failing channel is logged and ingestion still succeeds;
  * only confident non-CLEAN reads alert (status == "predicted" AND confidence >= SOLAR_AI_NOTIFY_MIN_CONF);
  * per-(camera, label) cooldown so a persistent mark is not re-sent on every capture; the cooldown is only
    consumed when at least one channel actually delivered;
  * every message states that the classifier is NOT validated on the target camera, because a CLEAN result is
    not evidence of a clean panel (see datasets/ground_pv_research/web_condition_v2/README.md);
  * secrets (webhook URL, SMTP password) come from the environment only and are never logged or put in messages.

Environment variables:
  SOLAR_AI_NOTIFY              on/1/true/yes to enable (default off)
  SOLAR_AI_NOTIFY_WEBHOOK_URL  http(s) URL that receives a JSON POST
  SOLAR_AI_NOTIFY_SMTP_HOST / _PORT (587) / _USER / _PASSWORD / _FROM / _TO (comma separated) / _STARTTLS (on)
  SOLAR_AI_NOTIFY_COOLDOWN_HOURS (12)     SOLAR_AI_NOTIFY_MIN_CONF (0.60)
  SOLAR_AI_NOTIFY_STATE        path of the cooldown state file (data/notify_state.json)
"""
from __future__ import annotations

import json
import os
import smtplib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from utils.logger import get_logger

logger = get_logger(__name__)

UNVALIDATED_BANNER = (
    "UNVALIDATED RESEARCH MODEL: this classifier has not been verified on this camera. An alert may be a false "
    "alarm, and the ABSENCE of an alert does not mean the panels are clean. Inspect the panel before acting."
)
_TRUE = {"1", "on", "true", "yes"}


@dataclass
class SmtpConfig:
    host: str
    port: int = 587
    user: str = ""
    password: str = field(default="", repr=False)
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    starttls: bool = True


@dataclass
class NotifyConfig:
    enabled: bool = False
    webhook_url: str = field(default="", repr=False)
    smtp: SmtpConfig | None = None
    cooldown_hours: float = 12.0
    min_confidence: float = 0.60
    state_path: Path = Path("data/notify_state.json")
    reason: str = ""  # why disabled, for logs

    def channels(self) -> list[str]:
        return (["webhook"] if self.webhook_url else []) + (["email"] if self.smtp else [])


def _valid_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and bool(u.netloc)


def config_from_env(env: dict[str, str] | None = None) -> NotifyConfig:
    env = dict(os.environ) if env is None else env
    cfg = NotifyConfig(
        cooldown_hours=float(env.get("SOLAR_AI_NOTIFY_COOLDOWN_HOURS", "12")),
        min_confidence=float(env.get("SOLAR_AI_NOTIFY_MIN_CONF", "0.60")),
        state_path=Path(env.get("SOLAR_AI_NOTIFY_STATE", "data/notify_state.json")),
    )
    if env.get("SOLAR_AI_NOTIFY", "").strip().lower() not in _TRUE:
        cfg.reason = "SOLAR_AI_NOTIFY is not on"
        return cfg
    url = env.get("SOLAR_AI_NOTIFY_WEBHOOK_URL", "").strip()
    if url:
        if _valid_http_url(url):
            cfg.webhook_url = url
        else:
            logger.warning("SOLAR_AI_NOTIFY_WEBHOOK_URL is not a valid http(s) URL - webhook channel disabled")
    host = env.get("SOLAR_AI_NOTIFY_SMTP_HOST", "").strip()
    to = [a.strip() for a in env.get("SOLAR_AI_NOTIFY_SMTP_TO", "").split(",") if a.strip()]
    if host and to:
        cfg.smtp = SmtpConfig(
            host=host, port=int(env.get("SOLAR_AI_NOTIFY_SMTP_PORT", "587")),
            user=env.get("SOLAR_AI_NOTIFY_SMTP_USER", ""), password=env.get("SOLAR_AI_NOTIFY_SMTP_PASSWORD", ""),
            sender=env.get("SOLAR_AI_NOTIFY_SMTP_FROM", "") or env.get("SOLAR_AI_NOTIFY_SMTP_USER", ""),
            recipients=to, starttls=env.get("SOLAR_AI_NOTIFY_SMTP_STARTTLS", "on").strip().lower() in _TRUE)
    cfg.enabled = bool(cfg.channels())
    if not cfg.enabled:
        cfg.reason = "SOLAR_AI_NOTIFY is on but no channel is configured"
    return cfg


# ---------------------------------------------------------------- cooldown state

def _load_state(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def select_alerts(flagged: list[dict[str, Any]], camera: dict[str, Any], state: dict[str, str],
                  now: datetime, cfg: NotifyConfig) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick the flagged panels that should alert now. Returns (alerts, cooldown_keys_to_consume).
    One alert per (camera, label): the most confident panel of that label."""
    best: dict[str, dict[str, Any]] = {}
    for p in flagged:
        c = p["condition"]
        if c.get("status") != "predicted" or c.get("label") in (None, "CLEAN") or c.get("confidence", 0) < cfg.min_confidence:
            continue
        if c["label"] not in best or c["confidence"] > best[c["label"]]["confidence"]:
            best[c["label"]] = {"panel": p["display_id"], "label": c["label"], "confidence": round(float(c["confidence"]), 3)}
    alerts, keys = [], []
    for label, a in sorted(best.items()):
        key = f"{camera.get('id')}:{label}"
        last = state.get(key)
        if last:
            try:
                if now - datetime.fromisoformat(last) < timedelta(hours=cfg.cooldown_hours):
                    continue
            except ValueError:
                pass
        alerts.append(a)
        keys.append(key)
    return alerts, keys


def build_message(camera: dict[str, Any], run_id: int, alerts: list[dict[str, Any]]) -> tuple[str, str]:
    cam = f"camera {camera.get('camera_number', '?')}" + (f" ({camera['name']})" if camera.get("name") else "")
    labels = ", ".join(sorted({a["label"] for a in alerts}))
    subject = f"[Solar AI] possible {labels} on {cam}"
    lines = [f"Possible condition(s) flagged on {cam}, ingestion run #{run_id}:", ""]
    lines += [f"  - panel {a['panel']}: {a['label']} (confidence {a['confidence']:.2f})" for a in alerts]
    lines += ["", UNVALIDATED_BANNER]
    return subject, "\n".join(lines)


# ---------------------------------------------------------------- transports

def send_webhook(url: str, subject: str, body: str, camera: dict[str, Any], run_id: int,
                 alerts: list[dict[str, Any]], post: Callable[..., Any] | None = None) -> None:
    if post is None:
        import requests  # noqa: PLC0415 - already a project dependency (pi_agent uses it)

        post = requests.post
    payload = {"title": subject, "text": body, "content": body, "validated": False, "run_id": run_id,
               "camera_number": camera.get("camera_number"), "alerts": alerts}
    resp = post(url, json=payload, timeout=10)
    resp.raise_for_status()


def send_email(smtp: SmtpConfig, subject: str, body: str,
               smtp_factory: Callable[..., Any] | None = None) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, smtp.sender, ", ".join(smtp.recipients)
    msg.set_content(body)
    with (smtp_factory or smtplib.SMTP)(smtp.host, smtp.port, timeout=15) as s:
        if smtp.starttls:
            s.starttls()
        if smtp.user:
            s.login(smtp.user, smtp.password)
        s.send_message(msg)


def notify_ingest(flagged: list[dict[str, Any]], camera: dict[str, Any], run_id: int,
                  cfg: NotifyConfig | None = None, *, now: datetime | None = None,
                  post: Callable[..., Any] | None = None,
                  smtp_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Alert on one ingested run. Never raises. Returns {"sent": bool, "channels": [...], "alerts": n, "reason": str}."""
    try:
        cfg = cfg or config_from_env()
        if not cfg.enabled:
            return {"sent": False, "channels": [], "alerts": 0, "reason": cfg.reason or "disabled"}
        now = now or datetime.now(timezone.utc)
        state = _load_state(cfg.state_path)
        alerts, keys = select_alerts(flagged, camera, state, now, cfg)
        if not alerts:
            return {"sent": False, "channels": [], "alerts": 0, "reason": "nothing new above threshold / in cooldown"}
        subject, body = build_message(camera, run_id, alerts)
        delivered: list[str] = []
        if cfg.webhook_url:
            try:
                send_webhook(cfg.webhook_url, subject, body, camera, run_id, alerts, post=post)
                delivered.append("webhook")
            except Exception as exc:  # noqa: BLE001 - never leak the URL; log only the exception type
                logger.warning("Notification webhook failed: %s", type(exc).__name__)
        if cfg.smtp:
            try:
                send_email(cfg.smtp, subject, body, smtp_factory=smtp_factory)
                delivered.append("email")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Notification email failed: %s", type(exc).__name__)
        if delivered:
            for k in keys:
                state[k] = now.isoformat()
            _save_state(cfg.state_path, state)
        return {"sent": bool(delivered), "channels": delivered, "alerts": len(alerts),
                "reason": "" if delivered else "all channels failed"}
    except Exception as exc:  # noqa: BLE001 - notification problems must never fail ingestion
        logger.warning("Notification step failed: %s", type(exc).__name__)
        return {"sent": False, "channels": [], "alerts": 0, "reason": f"error: {type(exc).__name__}"}
