"""pi_agent/capture_and_send.py — runs on the Raspberry Pi itself.

Role: capture + analyze + log (the confirmed Phase 18 design) — this script
captures a photo from the Pi's camera, runs the SAME inference pipeline as
app.py (`services/ground_pipeline.py`, no divergent copy), and POSTs the
already-computed structured result to the central ingestion API
(`api/main.py::POST /v1/ingest`). It never sends the raw image — only the
JSON `report` per capture, exactly like `services/ground_pipeline.run_capture()`
returns.

Requires this repo (or a trimmed checkout containing at least `services/`,
`training/`, `utils/`, and the three frozen `weights/*` checkpoints) present
on the Pi, plus `picamera2` for real hardware capture (Raspberry Pi OS only
— cannot be installed or exercised off-Pi, so this module keeps capture
behind a small seam and is only integration-tested here via `--test-image`,
never against real hardware). See docs/RASPBERRY_PI_DEPLOYMENT.md.

Usage:
    python -m pi_agent.capture_and_send --config pi_agent/config.yaml --once
    python -m pi_agent.capture_and_send --config pi_agent/config.yaml        # loop forever

Dry-run without a physical camera (e.g. to test the analyze+send half on a
dev machine that already has the frozen weights):
    python -m pi_agent.capture_and_send --config pi_agent/config.yaml --once --test-image path/to/photo.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image

from services import ground_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_CONFIG_KEYS = ("api_base_url", "api_token", "camera_label")
_INGEST_TIMEOUT_SECONDS = 30


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the Pi agent's local config. Fails closed (raises)
    on a missing file or a missing required key rather than silently
    running with an unset token or endpoint."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Pi agent config not found at {config_path}. Copy "
            "pi_agent/config.example.yaml there and fill in real values."
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    missing = [k for k in _REQUIRED_CONFIG_KEYS if not config.get(k)]
    if missing:
        raise ValueError(f"Pi agent config at {config_path} is missing required key(s): {', '.join(missing)}")

    config.setdefault("capture_interval_seconds", 300)
    config.setdefault("detection_confidence", 0.25)
    return config


def capture_image() -> Image.Image:
    """Capture one still frame from the Pi's camera via picamera2. Only
    importable/runnable on actual Raspberry Pi OS — raises a clear,
    actionable error everywhere else instead of a confusing ImportError,
    so --test-image is the documented way to exercise the rest of this
    script off-Pi."""
    try:
        from picamera2 import Picamera2  # noqa: PLC0415 — Pi-only dependency, deferred
    except ImportError as exc:
        raise RuntimeError(
            "picamera2 is not installed — this only runs on Raspberry Pi OS with the "
            "camera stack set up (see docs/RASPBERRY_PI_DEPLOYMENT.md). For a dry run on "
            "a dev machine, pass --test-image <path> instead of relying on real capture."
        ) from exc

    camera = Picamera2()
    try:
        camera.configure(camera.create_still_configuration())
        camera.start()
        time.sleep(1.0)  # let auto-exposure/auto-focus settle before capturing
        array = camera.capture_array()
    finally:
        camera.close()
    return Image.fromarray(array).convert("RGB")


def analyze(image: Image.Image, label: str, detection_conf: float, models: tuple) -> dict[str, Any]:
    """Run the shared ground pipeline on ONE captured image — identical
    code path to app.py's manual-upload run, just called directly instead
    of through Streamlit."""
    panel_model, condition_model, classifier_model, classifier_device = models
    return ground_pipeline.run_capture(
        panel_model, condition_model, classifier_model, classifier_device,
        image, label, detection_conf,
    )


def send_result(config: dict[str, Any], label: str, report: dict[str, Any]) -> dict[str, Any]:
    """POST one analyzed capture to the ingestion API. Raises on a network
    or non-2xx failure — callers decide whether to retry or just log and
    move on to the next scheduled capture."""
    url = config["api_base_url"].rstrip("/") + "/v1/ingest"
    resp = requests.post(
        url,
        json={"captures": [{"label": label, "report": report}]},
        headers={"Authorization": f"Bearer {config['api_token']}"},
        timeout=_INGEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def run_once(config: dict[str, Any], models: tuple, *, test_image: str | None = None) -> dict[str, Any]:
    label = f"{config['camera_label']}_{time.strftime('%Y%m%dT%H%M%S')}.jpg"
    if test_image:
        image = Image.open(test_image).convert("RGB")
        logger.info("Using --test-image %s instead of real camera capture", test_image)
    else:
        image = capture_image()

    report = analyze(image, label, config["detection_confidence"], models)
    result = send_result(config, label, report)

    if result.get("problem_detected"):
        logger.warning(
            "[%s] camera #%s: PROBLEM DETECTED — %d flagged panel(s) (run_id=%s)",
            config["camera_label"], result.get("camera_number"), result.get("flagged_count"), result.get("run_id"),
        )
    else:
        logger.info(
            "[%s] camera #%s: OK — %d clean, %d unread (run_id=%s)",
            config["camera_label"], result.get("camera_number"),
            result.get("clean_count"), result.get("unread_count"), result.get("run_id"),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi_agent/config.yaml", help="Path to this Pi's config.yaml")
    parser.add_argument("--once", action="store_true", help="Capture+analyze+send once, then exit (cron style)")
    parser.add_argument("--test-image", default=None, help="Use this image file instead of real camera capture")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    try:
        panel_model, condition_model = ground_pipeline.load_ground_models()
        classifier_model, classifier_device = ground_pipeline.load_condition_classifier()
    except Exception:  # noqa: BLE001 — surfaced as a clear failed-to-start log, not a stack trace to a cron mailbox
        logger.exception("Failed to load models — cannot start")
        return 1
    models = (panel_model, condition_model, classifier_model, classifier_device)

    if args.once:
        try:
            run_once(config, models, test_image=args.test_image)
        except Exception:  # noqa: BLE001 — one bad capture must not look like a silent success to cron
            logger.exception("Capture+analyze+send failed")
            return 1
        return 0

    logger.info(
        "Starting loop mode: capturing every %ds (Ctrl+C to stop)", config["capture_interval_seconds"]
    )
    while True:
        try:
            run_once(config, models, test_image=args.test_image)
        except Exception:  # noqa: BLE001 — one bad capture must never kill the whole loop
            logger.exception("Capture+analyze+send failed — will retry next interval")
        time.sleep(config["capture_interval_seconds"])


if __name__ == "__main__":
    sys.exit(main())
