"""services/notifications.py - off by default, thresholds, cooldown, failure isolation, secrecy."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services import notifications as n

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
CAM = {"id": 3, "camera_number": 2, "name": "Roof east"}


def _p(label="BIRD_DROP", conf=0.9, status="predicted", pid="P1"):
    return {"display_id": pid, "condition": {"label": label, "confidence": conf, "status": status}}


def _cfg(tmp_path, **kw):
    base = dict(enabled=True, webhook_url="https://hooks.example/abc", state_path=tmp_path / "s.json")
    base.update(kw)
    return n.NotifyConfig(**base)


class _Resp:
    def raise_for_status(self):
        pass


def test_disabled_by_default_and_without_a_channel():
    assert config_ok({}) is False
    assert n.config_from_env({"SOLAR_AI_NOTIFY": "on"}).reason.endswith("no channel is configured")
    assert n.notify_ingest([_p()], CAM, 1, n.config_from_env({}))["sent"] is False


def config_ok(env):
    return n.config_from_env(env).enabled


def test_env_config_parses_both_channels_and_rejects_bad_webhook():
    env = {"SOLAR_AI_NOTIFY": "yes", "SOLAR_AI_NOTIFY_WEBHOOK_URL": "https://ntfy.sh/mine",
           "SOLAR_AI_NOTIFY_SMTP_HOST": "smtp.x", "SOLAR_AI_NOTIFY_SMTP_TO": "a@x.com, b@x.com"}
    c = n.config_from_env(env)
    assert c.enabled and c.channels() == ["webhook", "email"] and c.smtp.recipients == ["a@x.com", "b@x.com"]
    bad = n.config_from_env({"SOLAR_AI_NOTIFY": "on", "SOLAR_AI_NOTIFY_WEBHOOK_URL": "file:///etc/passwd"})
    assert bad.enabled is False and bad.webhook_url == ""


def test_secrets_are_not_in_repr():
    c = n.config_from_env({"SOLAR_AI_NOTIFY": "on", "SOLAR_AI_NOTIFY_WEBHOOK_URL": "https://h/SECRETTOKEN",
                           "SOLAR_AI_NOTIFY_SMTP_HOST": "h", "SOLAR_AI_NOTIFY_SMTP_TO": "a@x.com",
                           "SOLAR_AI_NOTIFY_SMTP_PASSWORD": "hunter2"})
    assert "SECRETTOKEN" not in repr(c) and "hunter2" not in repr(c)


def test_only_confident_non_clean_predicted_reads_alert(tmp_path):
    cfg = _cfg(tmp_path)
    flagged = [_p("CLEAN", 0.99), _p("DUSTY", 0.5), _p("PHYSICAL_DAMAGE", 0.9, status="uncertain"), _p("BIRD_DROP", 0.8)]
    alerts, _ = n.select_alerts(flagged, CAM, {}, NOW, cfg)
    assert [a["label"] for a in alerts] == ["BIRD_DROP"]


def test_one_alert_per_label_most_confident_panel(tmp_path):
    alerts, keys = n.select_alerts([_p(conf=0.7, pid="P1"), _p(conf=0.95, pid="P2")], CAM, {}, NOW, _cfg(tmp_path))
    assert len(alerts) == 1 and alerts[0]["panel"] == "P2" and keys == ["3:BIRD_DROP"]


def test_send_then_cooldown_then_resend_after_expiry(tmp_path):
    cfg, sent = _cfg(tmp_path), []
    post = lambda url, json, timeout: (sent.append(json), _Resp())[1]  # noqa: E731
    assert n.notify_ingest([_p()], CAM, 7, cfg, now=NOW, post=post)["sent"] is True
    again = n.notify_ingest([_p()], CAM, 8, cfg, now=NOW + timedelta(hours=1), post=post)
    assert again["sent"] is False and "cooldown" in again["reason"]
    assert n.notify_ingest([_p()], CAM, 9, cfg, now=NOW + timedelta(hours=13), post=post)["sent"] is True
    assert len(sent) == 2
    # a different label is not blocked by the BIRD_DROP cooldown
    assert n.notify_ingest([_p("PHYSICAL_DAMAGE")], CAM, 10, cfg, now=NOW + timedelta(hours=14), post=post)["sent"] is True


def test_message_carries_the_unvalidated_banner_and_camera(tmp_path):
    got = {}
    post = lambda url, json, timeout: (got.update(json), _Resp())[1]  # noqa: E731
    n.notify_ingest([_p()], CAM, 42, _cfg(tmp_path), now=NOW, post=post)
    assert "UNVALIDATED" in got["text"] and "camera 2" in got["title"] and got["validated"] is False and "#42" in got["text"]


def test_failed_channel_never_raises_and_does_not_consume_cooldown(tmp_path):
    cfg = _cfg(tmp_path)

    def boom(url, json, timeout):
        raise ConnectionError("https://hooks.example/abc is down")

    r = n.notify_ingest([_p()], CAM, 1, cfg, now=NOW, post=boom)
    assert r["sent"] is False and r["reason"] == "all channels failed"
    assert not cfg.state_path.exists()
    ok = n.notify_ingest([_p()], CAM, 2, cfg, now=NOW, post=lambda url, json, timeout: _Resp())
    assert ok["sent"] is True  # retried because the failed attempt did not start a cooldown


def test_one_channel_failing_still_delivers_on_the_other(tmp_path):
    cfg = _cfg(tmp_path, smtp=n.SmtpConfig(host="h", sender="s@x.com", recipients=["a@x.com"]))

    class _S:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, m): self.__class__.last = m

    r = n.notify_ingest([_p()], CAM, 1, cfg, now=NOW, post=lambda *a, **k: (_ for _ in ()).throw(OSError("x")),
                        smtp_factory=_S)
    assert r["channels"] == ["email"] and "UNVALIDATED" in _S.last.get_content()


def test_corrupt_state_file_is_tolerated(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_path.write_text("{not json")
    assert n.notify_ingest([_p()], CAM, 1, cfg, now=NOW, post=lambda *a, **k: _Resp())["sent"] is True
    assert json.loads(cfg.state_path.read_text())["3:BIRD_DROP"]
