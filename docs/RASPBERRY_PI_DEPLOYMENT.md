# Phase 18 — Raspberry Pi camera deployment

Research/demo deployment guide, not a hardened production rollout. Every
claim below about the software (endpoints, config keys, test coverage) is
verified in this repo (`api/main.py`, `pi_agent/capture_and_send.py`,
`tests/test_api_ingest.py`, `tests/test_pi_agent.py`). Everything about
real Raspberry Pi hardware behavior (capture latency, outdoor image
quality, thermal behavior) is **not benchmarked** — no physical Pi or
camera was available in this development environment. Where this doc gives
a hardware recommendation, it says so explicitly and gives the reasoning,
rather than presenting it as a measured result.

## Architecture (confirmed design)

> "Just capture + analyze + log."

Each Pi runs the **same inference code as the Streamlit app** — it is a
full edge node, not a dumb camera:

```
Pi camera  ──▶  pi_agent/capture_and_send.py
                   │
                   ├── capture (picamera2)
                   ├── analyze (services/ground_pipeline.py — the SAME
                   │            module app.py imports; identical detector
                   │            + classifier + box-detector pipeline)
                   └── log (POST the JSON result — never the raw image —
                             to the ingestion API)
                             │
                             ▼
                    api/main.py  (FastAPI, api/main.py::POST /v1/ingest)
                       │
                       ├── authenticate the camera's token
                       │   (services/storage.py::authenticate_camera)
                       ├── persist the run (record_ground_inspection +
                       │   record_panel_identities, tagged camera_id)
                       └── respond with problem_detected + camera_number
                             │
                             ▼
                    Streamlit admin pages read the same SQLite DB:
                      pages/12_📷_Cameras.py       (live status by camera #)
                      pages/13_🛠️_Camera_Configuration.py (add/remove/disable)
                      pages/09_🚨_Alerts.py         (alerts, camera # attached)
                      pages/11_🕰️_Panel_History.py  (cross-run panel tracking)
```

Why inference runs ON the Pi rather than centrally: it keeps this a real
edge pipeline (a Pi keeps working, and can retry its POST, even if the
admin host is briefly down), and it avoids building/securing a second,
heavier raw-image-upload endpoint for what is still a single-user,
local-network deployment (same scale of ambition as
`services/storage.py`'s own "single-user local SQLite, not a multi-tenant
production database" framing — see `docs/PHASE_16_ARCHITECTURE_REVIEW.md`).
The cost is real: every Pi needs the frozen model weights and enough
compute to run them, which the next section sizes honestly.

## Hardware

### Raspberry Pi board

**Recommended: Raspberry Pi 4 Model B (4GB+) or Raspberry Pi 5.** Reasoning,
not a benchmark:
- The pipeline runs two YOLO-nano-scale models plus one MobileNetV2 pass
  per detected panel, entirely on CPU (`torch.cuda.is_available()` is
  `False` on a Pi, so `ground_pipeline.load_condition_classifier()` already
  falls back to CPU with no code change needed).
- At a 5-minute default capture interval (`capture_interval_seconds` in
  `pi_agent/config.yaml`), even a conservatively slow multi-second
  CPU inference pass per image is well within budget — this is a
  low-frequency polling workload, not real-time video.
- 4GB+ RAM gives headroom for loading three model checkpoints plus
  Raspberry Pi OS itself without swapping.
- **Not verified on real hardware in this environment** — if actual
  per-image inference time on a Pi 4 turns out too slow for a shorter
  interval you want, increase `capture_interval_seconds` rather than
  assuming it will keep up.

### Camera

**Recommended: Raspberry Pi Camera Module 3 (Sony IMX708, 12MP,
autofocus).** Reasoning:
- Autofocus matters here specifically because the panel detector's real
  working range (per `docs/PANEL_IDENTITY_TRACKING.md` /
  `docs/GROUND_PV_RESEARCH.md`) is close-up shots of ~1-3 panels — a fixed-
  focus camera risks a soft image at whatever distance the pole ends up
  mounted at, and this pipeline has no separate focus-quality check.
- Choose the **Wide FOV variant** if the pole mount will be close to the
  array (covers more panels per single capture, consistent with this
  project's own "multiple closer photos beat one distant wide shot"
  finding); choose the **standard/telephoto variant** if the mount will be
  farther back.
- Outdoor daylight/glare on a reflective panel surface is a genuine open
  risk this project has not tested against real Pi hardware — mount the
  camera with a lens hood/shade and avoid direct backlighting if possible.
  This is a stated risk, not a solved one.

## Software setup — on the Pi

1. Raspberry Pi OS Bookworm (64-bit), camera enabled via
   `raspi-config` (Interface Options → Camera). `picamera2` ships
   preinstalled on current Raspberry Pi OS images.
2. Copy (or `git clone`) this repository, or a trimmed copy containing at
   minimum: `services/`, `training/`, `utils/`, `pi_agent/`, and these
   three frozen checkpoints under `weights/` (the exact files
   `services/ground_pipeline.py` loads):
   - `weights/yolo_solar_ground_research_v1.pt`
   - `weights/yolo_solar_condition_4class_research_v2.pt`
   - `weights/mobilenet_condition_4class_research_v1.pth`
3. Create a venv and install only what the Pi agent needs (skip
   `streamlit`/`fastapi`/`uvicorn` — those run on the admin host, not the
   Pi):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch torchvision ultralytics Pillow numpy requests PyYAML
   ```
   PyTorch publishes official Linux aarch64 wheels — no source build should
   be needed on a 64-bit Raspberry Pi OS. Install time and inference speed
   on real Pi hardware are not benchmarked here.
4. Copy `pi_agent/config.example.yaml` to `pi_agent/config.yaml` and fill
   in `api_base_url` (the admin host's address, e.g.
   `http://192.168.1.50:8008`) and `api_token` (see next section for where
   the token comes from). `pi_agent/config.yaml` is git-ignored — it holds
   a real secret and must never be committed.

## Software setup — on the admin host

The admin host runs both the existing Streamlit app and the new ingestion
API side by side (they share the same SQLite file via `services/storage.py`
— Streamlit's own writes and the API's writes are both short-lived SQLite
transactions, not a long-held connection, so this is safe for this
project's single-file-SQLite, local-network scale).

1. `pip install -r requirements.txt` (already includes `fastapi`,
   `uvicorn`, `python-multipart`).
2. Register each camera from the **Camera Configuration** admin page
   (`pages/13_🛠️_Camera_Configuration.py`) — this issues a fresh API
   token, shown **once**, to paste into that Pi's `config.yaml`.
3. Run the ingestion API:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8008
   ```
   (Bind to `0.0.0.0` only on a trusted local network — see "Security
   scope" below.)
4. Run Streamlit as usual (`streamlit run app.py`).

## Running the Pi agent

**Cron style** (simple, one process per capture, easiest to reason about):
```
*/5 * * * * cd /path/to/repo && .venv/bin/python -m pi_agent.capture_and_send --config pi_agent/config.yaml --once >> /var/log/solarai_pi_agent.log 2>&1
```

**systemd service style** (loop mode, keeps models loaded in memory between
captures — avoids reloading three checkpoints every 5 minutes):
```ini
# /etc/systemd/system/solarai-pi-agent.service
[Unit]
Description=Solar AI Pi camera agent
After=network-online.target

[Service]
WorkingDirectory=/path/to/repo
ExecStart=/path/to/repo/.venv/bin/python -m pi_agent.capture_and_send --config pi_agent/config.yaml
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now solarai-pi-agent
```

**Dry run without hardware** (useful for testing the analyze+send half on
any machine that already has the three frozen checkpoints — this is how
`tests/test_pi_agent.py::TestRunOnceRealAnalysis` verifies the real model
stack works end to end):
```bash
python -m pi_agent.capture_and_send --config pi_agent/config.yaml --once \
    --test-image assets/demo/ground_pv_real/some_photo.jpeg
```

## Classifier choice on the Pi, and alerts

**Which classifier.** `SOLAR_AI_CLASSIFIER` selects the condition classifier used by the shared pipeline (default
`rooftop_mix`, the MobileNetV2). Opt-in web-trained alternatives need `weights/web_condition_ensemble_v1/`:
`web_ensemble` (5 models) or `web_ensemble_lite` (2 models: ConvNeXt-Tiny 288 + EfficientNetV2-S 384).

| backend | out-of-fold macro-F1 (web photos) | cost per panel crop on a desktop CPU, 4 threads |
|---|---|---|
| `rooftop_mix` | not comparable (different test set) | ~0.02 s (single pass, no flip TTA) |
| `web_ensemble_lite` | 0.875 | ~1.2 s (with flip TTA) |
| `web_ensemble` | 0.878 | ~3.8 s |

These timings are **measured on a desktop CPU, not on a Pi**. A Pi 4/5 is several times slower, so budget on the order
of 5-10x as an unmeasured estimate: a capture with 6 panels would take tens of seconds to a few minutes with the lite
ensemble. The default 300 s capture interval still fits that, but measure on your board before relying on it.
Neither classifier is validated on a rooftop camera (see `docs/GROUND_INSPECTION_STATUS_AND_LIMITS.md`).

**Alerts.** `api/main.py` calls `services/notifications.py` after every ingest. It is **off by default**. To enable,
set these in the environment of the API process (never commit them):

```
SOLAR_AI_NOTIFY=on
SOLAR_AI_NOTIFY_WEBHOOK_URL=https://ntfy.sh/<your-private-topic>      # any JSON webhook; and/or SMTP below
SOLAR_AI_NOTIFY_SMTP_HOST=smtp.example.com  SOLAR_AI_NOTIFY_SMTP_PORT=587
SOLAR_AI_NOTIFY_SMTP_USER=...  SOLAR_AI_NOTIFY_SMTP_PASSWORD=...  SOLAR_AI_NOTIFY_SMTP_TO=you@example.com
SOLAR_AI_NOTIFY_COOLDOWN_HOURS=12   SOLAR_AI_NOTIFY_MIN_CONF=0.60
```

Behaviour: it alerts only on a *predicted* non-CLEAN read at or above `MIN_CONF`, at most once per camera and label per
cooldown window, and the cooldown is only consumed when a channel really delivered. A failing channel is logged
(exception type only, never the URL) and never fails the ingest. Every message says the model is unvalidated and that
the absence of an alert does not mean the panels are clean. Tests: `tests/test_notifications.py`.

## Security scope (stated honestly, not glossed over)

- Per-camera bearer tokens (`secrets.token_urlsafe(32)`), compared with
  `secrets.compare_digest` — real, not a placeholder.
- **No TLS, no rate limiting, no public-internet exposure is provided or
  recommended.** Run the ingestion API only on a trusted local network
  (the same network the Pi devices are on). Exposing it publicly would
  need TLS termination and stricter hardening this phase does not add —
  matching this project's existing "single-user local database" scope, not
  a claim this is enterprise-ready.
- Disabling a camera (Camera Configuration page) immediately stops its
  token from authenticating without deleting its history — reversible.
  Deleting a camera also stops its token immediately, but does NOT recycle
  its camera_number (verified in
  `tests/test_storage_cameras.py::test_camera_numbers_increment_and_never_reuse_after_delete`)
  and does NOT delete its historical runs/panels — see
  `services/storage.py::delete_camera`'s "orphaned reference" note.

## Testing coverage that exists today

- `tests/test_storage_cameras.py` (21 tests) — camera registry, token
  redaction, auth, camera-scoped panel identity isolation.
- `tests/test_api_ingest.py` (13 tests) — the ingestion API's auth,
  persistence, problem-detection response, and cross-camera isolation,
  via FastAPI's real `TestClient`.
- `tests/test_pi_agent.py` (9 tests) — config validation, the real
  off-Pi `picamera2` error path, and one test that runs the **actual**
  frozen model stack on a real demo photo end-to-end (skipped
  automatically if the weight files aren't present on the machine running
  the tests).
- `tests/test_pages_camera_admin.py` (7 tests) — real Streamlit `AppTest`
  interaction with registration/disable/enable/delete-with-confirmation
  and the Cameras overview's problem banner.

**Not tested**: real Raspberry Pi hardware (capture timing, outdoor image
quality, thermal throttling under sustained use), multi-Pi concurrent
writes at real-world request rates, and behavior over an unreliable Wi-Fi
link (the agent does not currently retry a failed POST within the same
cycle — a failed send is logged and picked up again at the next scheduled
capture).
