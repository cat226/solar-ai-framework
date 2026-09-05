# Deployment Runbook

The reference production runtime is a Dockerized Streamlit application.

> **Solar AI v1 is technically functional but has not been validated as a
> high-accuracy production inspection system. The current YOLO detector
> has a known domain gap for close-up/ground-level imagery.** See
> `docs/ML_EVALUATION_v1.0.0.md`, `docs/ML_HARDENING_PHASE6B.md`, and
> `docs/ML_HARDENING_PHASE6C.md` for the full, independently-measured
> evidence behind this statement. This deployment guide covers running
> the application safely and observably — it does not change that
> conclusion.

## 1. Prerequisites

- Docker (or an OCI-compatible runtime) able to build/run a
  `python:3.12-slim-bookworm`-based image.
- The real, trained model artifacts (see §3) — **not included in this
  repository or the built image**.
- Network egress to `api.openweathermap.org` if live weather data is
  wanted (the app degrades safely without it — see §9 in
  `README.md`'s Known Limitations, and `docs/ML_EVALUATION_v1.0.0.md`).
- A writable location on the host for persistent inspection history
  (§7).

## 2. Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENWEATHER_API_KEY` | No | Live weather lookups. Absent → the app uses configured defaults and discloses this on the Environment page and in `Limitations` — never fabricated live data. |
| `APP_ACCESS_PASSWORD` | No (strongly recommended for any non-local deployment) | Shared access password (see §8). Unset → the app is open to anyone who can reach it — matches local development, not a hardened deployment default. |
| `SOLAR_AI_DATA_ROOT` | No | Overrides where local training/evaluation tooling writes large generated data. Irrelevant to the deployed application itself (only used by `training/`). |

Set secrets via the deployment platform's own secret store, `.env` (never
committed — see `.gitignore` and `.env.example`), or
`.streamlit/secrets.toml` (never committed — see
`.streamlit/secrets.toml.example`). Do not pass secrets as Dockerfile
`ARG`/`ENV` at build time — that bakes them into the image layer history
even if later removed from the running container.

## 3. Model artifact requirements

Model artifacts are intentionally excluded from Git and the image. Mount a
trusted directory containing the trained artifacts at `/app/weights`:

```bash
docker run --rm -p 8501:8501 \
  -e OPENWEATHER_API_KEY="$OPENWEATHER_API_KEY" \
  -e APP_ACCESS_PASSWORD="$APP_ACCESS_PASSWORD" \
  -v "$(pwd)/weights:/app/weights:ro" \
  -v solar-ai-data:/app/data \
  solar-ai-framework:latest
```

Required files for Solar AI v1 (solar-panel detection + Clean/Dusty/Hotspot
classification):

- `yolo_solar.pt`
- `mobilenet_solar_v1.pth`

Optional, not part of v1:

- `mobilenet_solar.pth` — the future six-class classifier. Not yet trained;
  not required for v1. If present, `ModelManager` prefers it automatically
  over the v1 artifact, superseding v1's 3-class scope with no code change.
- `xgboost_solar.joblib` — efficiency-loss prediction. Not yet trained (no
  legitimate training dataset was found - see
  `training/prediction/DATASET_SOURCES.md`); not required for v1. Detection
  and classification run normally without it, and efficiency/output fields
  are reported as genuinely unavailable rather than fabricated
  (`xgboost_available=false`, never a silent `0%`).

Do not substitute generated, placeholder, or unverified model files.

## 4. Artifact verification

Trusted deployment artifacts should be accompanied by a reviewed JSON
manifest containing the SHA-256 digest of every artifact. The repository
ships one such reviewed manifest for the real v1.0.0 artifacts at
`weights/manifest.json` (see `docs/RELEASE_v1.0.0.md` for the full
reproducibility record) - use it as-is if deploying the exact v1.0.0
artifacts, or replace it with a manifest for your own reviewed bundle.
Verify the mounted artifacts before starting an inference-capable
deployment:

```bash
python scripts/verify_model_artifacts.py --manifest /app/weights/manifest.json
```

The verifier reads local files only. It never downloads or creates model
artifacts and fails on missing files or digest mismatches.

## 5. Startup

```bash
docker build -t solar-ai-framework:latest .
docker run --rm -p 8501:8501 \
  -e OPENWEATHER_API_KEY="$OPENWEATHER_API_KEY" \
  -e APP_ACCESS_PASSWORD="$APP_ACCESS_PASSWORD" \
  -v "$(pwd)/weights:/app/weights:ro" \
  -v solar-ai-data:/app/data \
  --name solar-ai \
  solar-ai-framework:latest
```

Model loading is lazy — the process starts and accepts connections even
with no model artifacts present; the first inference attempt then fails
with a typed `ModelLoadError` rather than the process itself failing to
start. This lets liveness and inference-readiness be checked and reported
independently (§6).

## 6. Health / readiness verification

Two independent signals, never conflated:

- **Liveness** ("is the process up"): Docker's own `HEALTHCHECK` polls
  Streamlit's `/_stcore/health` endpoint. `docker ps` shows `healthy` once
  the web process is accepting requests — this does **not** mean model
  artifacts are present.
- **Readiness** ("can real inference run"):
  ```bash
  docker exec solar-ai python scripts/check_runtime_readiness.py
  ```
  Prints e.g. `{"inference_readiness": "ready", "liveness": "ok",
  "missing_artifacts": ["XGBoost"]}` and exits non-zero whenever a
  *required* artifact (YOLO or MobileNet) is missing. **XGBoost missing
  alone is expected for v1 and does not indicate a broken deployment** —
  see `docs/RELEASE_v1.0.0.md`'s readiness section for why. This check
  never runs real inference and never fabricates a missing artifact as
  present.

Neither check requires processing an image — both are safe to run
frequently (an orchestrator's liveness probe) or on demand (an operator's
readiness check before routing traffic).

## 7. Persistent storage

`services/storage.py` writes a single-file SQLite database at
`data/inspections.db` (resolved relative to the container's `/app`
working directory, i.e. `/app/data/inspections.db`). Mount this path as a
named volume (`-v solar-ai-data:/app/data`, as shown above) so inspection
history survives a container restart/upgrade. Without a mounted volume,
history is lost whenever the container is removed — the *live* analysis
result on screen is never affected, only its persistence to History/
Analytics/Alerts (and, since Phase 7, a failed save is now surfaced to the
user on-screen, not just logged — see `docs/PRODUCT_HARDENING_PHASE7.md`).

SQLite is appropriate for this single-instance research/demo deployment
shape (one Streamlit process, one writer at a time in practice). It is
**not** appropriate the moment any of the following becomes true: multiple
application instances writing concurrently (e.g. horizontal scaling behind
a load balancer), a need for network-attached/shared storage, or
multi-user account semantics beyond the current single shared-password
gate. At that point, `services/storage.py` is the intended seam to replace
with a real client/server database (e.g. PostgreSQL) — not `app.py` or the
individual pages, which only call its public functions.

## 8. Authentication

`utils/auth.py` provides a single shared-password gate (`APP_ACCESS_PASSWORD`),
compared via `hmac.compare_digest` (timing-safe). This is explicitly **not**
a multi-user account system — no per-user identity, no password reset, no
SSO, and no brute-force lockout (acceptable for a low-traffic research/demo
deployment behind a password most visitors will never see; not a
substitute for real authentication if genuinely sensitive data or a
public, high-traffic audience is expected). Leaving `APP_ACCESS_PASSWORD`
unset makes the deployment openly accessible — this matches local
development and must be set explicitly for any non-local deployment. If
genuine multi-user authentication is required, treat `utils/auth.py` as
the seam to replace with a real identity provider, not a base to layer
ad-hoc logic onto.

## 9. Logs

Application modules log through the centralized `utils.logger`
(stdout/stderr, captured by `docker logs`). Logs record the inference
lifecycle, model-loading state, failures (with typed exception chaining),
and processing duration. Logs never contain: uploaded image contents,
the access password, API keys, or unsanitized user-controlled strings
(city input is sanitized before logging — see `utils/security.py` and
`services/weather.py`). Default log level is `INFO`; `DEBUG` is available
but not enabled by default and should not be enabled in a
publicly-reachable deployment without reviewing what it additionally
emits.

## 10. Backup

Back up the `data/inspections.db` volume using your platform's normal
volume-snapshot mechanism (it is a single SQLite file — a file-level copy
taken while the container is stopped, or via SQLite's own `.backup`
command, is sufficient; there is no separate secrets/config state to back
up beyond the environment variables in §2, which should already live in
your deployment platform's own secret store). Do not attempt to back up
`weights/` — those are externally-supplied, reproducible from
`docs/RELEASE_v1.0.0.md`'s manifest, not generated state.

## 11. Recovery

If `data/inspections.db` is lost or corrupted: stop the container, replace
the volume contents with the most recent backup (or an empty volume — the
app recreates the schema automatically via `CREATE TABLE IF NOT EXISTS` on
first write), and restart. No inspection history is recoverable beyond the
last backup; this is an accepted characteristic of the single-file SQLite
design (§7), not a defect introduced by this recovery procedure.

## 12. Upgrade procedure

1. Build the new image from the desired commit (`docker build`).
2. Verify the new image's `verify_imports.py` / test suite / readiness
   check pass (already gated by CI on every push to this branch).
3. Stop the running container (`docker stop solar-ai`).
4. Start the new container with the **same** `data/inspections.db` volume
   mounted (§7) and the **same** `weights/` directory (§3) — model
   artifacts and history are independent of the application code version.
5. Verify readiness (§6) before considering the upgrade complete.

## 13. Rollback procedure

Re-run §12 with the previous image tag/commit instead of the new one,
using the same persistent volume. Because the SQLite schema only ever
grows via additive, idempotent `ALTER TABLE ... ADD COLUMN` migrations
(never a destructive schema change), an older application version can
read a database written by a newer one (extra columns it doesn't know
about are simply not selected by its own queries) — a rollback is safe
with respect to history data.

## 14. Known ML limitations (do not omit when describing this deployment)

- **Not validated as a high-accuracy production inspection system** — see
  the notice at the top of this document and `docs/ML_EVALUATION_v1.0.0.md`.
- **YOLO has a known domain-shift limitation**: trained and validated on
  aerial/satellite imagery (BDAPPV); real close-up/ground-level photos
  (the kind a user is likely to upload) see a measured detection rate of
  only ~2.6% (`docs/ML_HARDENING_PHASE6B.md`). Whole-image classification
  still runs and remains accurate on such photos independent of detection
  success (§15).
- **No legitimate remediation dataset was found** as of Phase 6C
  (`docs/ML_HARDENING_PHASE6C.md`) — this is not expected to change until
  new, properly licensed target-domain data is identified.
- **XGBoost efficiency-loss prediction is unavailable** and not planned
  until a genuine training dataset is found (§16).
- **Three of six original fault classes remain unclassifiable**
  (Bird-Drop, Electrical-Damage, Physical-Damage) — v1 is frozen to
  Clean/Dusty/Hotspot only.

## 15. Supported v1 capabilities

- Solar-panel detection (YOLOv8n, real trained artifact).
- 3-class fault classification: **Clean, Dusty, Hotspot** — this is v1's
  complete, frozen scope, not a partial rollout.
- Weather lookup, physics-based feature engineering, maintenance
  recommendations (all real computation on real inputs — no fabricated
  values at any stage).
- Whole-image classification runs and is reported **independently** of
  whether a panel is successfully detected in the image — a real user
  photo that YOLO fails to localize (see §14) still receives a real,
  attempted classification, clearly labeled as such.
- Full dashboard: Overview, Panel Results, Site Health, Environment, Model
  Status, Limitations, History, Analytics, Alerts, Settings.

## 16. Unavailable XGBoost behavior

When `weights/xgboost_solar.joblib` is absent (the expected v1 state):

- Detection and classification run normally and are reported as real
  results.
- Efficiency-loss and estimated-output-power fields are reported as
  genuinely **unavailable** (`prediction_successful=false`,
  `xgboost_available=false`) — **never** a fabricated `0%` or `0 W`.
- Aggregate KPIs (`services/storage.get_summary_stats`) report `None`
  where no real prediction ever ran, rendered as `N/A` in the UI — never
  averaged in as a silent zero.
- `scripts/check_runtime_readiness.py` reports `XGBoost` under
  `missing_artifacts` — this is the correct, expected v1 state, not an
  error to remediate before considering the deployment healthy (§6).

## Production notes

- Python 3.12 is the supported runtime.
- The image runs as an unprivileged `appuser` (fixed UID 10001). Only
  `/app/data` (the SQLite history file's directory) and the user's own
  home directory are writable by `appuser` — application source is
  root-owned and read-only even to the runtime user (Phase 7 hardening).
- **Read-only root filesystem**: not the default `docker run` invocation
  above, but achievable — run with `--read-only --tmpfs /tmp
  --tmpfs /home/appuser/.streamlit` alongside the `data` volume already
  shown in §3/§5. Not required for a typical demo deployment; documented
  here for operators who want it.
- Uploaded files are capped at 10 MB by Streamlit configuration and are
  validated (format, decompression-bomb protection) before decoding.
- XSRF protection is enabled. Uncaught-exception detail is not exposed to
  the browser (`showErrorDetails = "none"` in `.streamlit/config.toml`,
  Phase 7 hardening) — full detail is always in the server-side logs (§9).
- Model loading remains lazy so the application can start without model
  artifacts; an inference request fails with a typed `ModelLoadError` until
  the required artifacts are supplied.
- Dependency requirements remain minimum-version declarations. A reviewed
  lock/constraints strategy is still a separate reproducibility task.
