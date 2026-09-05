# Solar AI v1.0.0 — Phase 7: Product & Deployment Hardening

Generated: 2026-09-05
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)
Baseline: `v1.0.0` tag at `14df9b9cdb5411cef79c4282174c7d042abb1a96` (frozen, untouched).
Phase 6C completed at `f982323`.

This is an **operational hardening phase, not an ML phase**. No model
weights, `configs/settings.yaml`'s YOLO threshold, or the MobileNet/YOLO
class contract were touched. See `docs/ML_EVALUATION_v1.0.0.md`,
`docs/ML_HARDENING_PHASE6B.md`, and `docs/ML_HARDENING_PHASE6C.md` for the
ML-accuracy findings this phase does not revisit or change.

---

## Task 1 — Application audit (map)

```
Upload (app.py, in-memory bytes only)
  -> validation (PIL verify + decompression-bomb protection, before any inference)
  -> Analyze button gate (session-state, prevents duplicate inference on reruns)
  -> inference (models.model_manager -> models.detector / models.classifier / models.predictor)
  -> services.pipeline.run_pipeline() orchestration
  -> persistence (services.storage, SQLite, only on real SUCCESS)
  -> UI rendering (utils.ui_helpers, st.session_state["last_result"])
  -> pages/ (History, Analytics, Alerts, Model Status, ...) read from storage + session_state
  -> health/readiness (scripts/check_runtime_readiness.py, Docker HEALTHCHECK)
```

**Trust boundaries:** the uploaded file (untrusted bytes, never a
filesystem path); the city text field (untrusted string, sanitized before
logging/external use); the OpenWeatherMap response (external, network,
parsed defensively); the mounted `weights/` directory (operator-supplied,
verified via `weights/manifest.json` before trust); the shared access
password (operator secret).

**Failure points identified and their existing/added handling:** invalid
image (rejected pre-inference), missing/corrupt model artifact (typed
`ModelLoadError`, fails closed), weather API failure (safe defaults,
disclosed), history write failure (previously silent — **fixed this
phase**, see Task 8/11), any other uncaught page exception (previously
would leak a full traceback to the browser — **fixed this phase**, see
Task 6/16).

**Resource-intensive operations:** YOLO/MobileNet inference (bounded by
the Analyze-button gate — never runs on an incidental rerun), weather API
calls (bounded timeout, see `services/weather.py`), SQLite writes (single
small row per real inspection).

**Persistent state:** `data/inspections.db` only (image bytes are never
persisted — only a SHA-256 hash).

**Secrets/configuration:** `OPENWEATHER_API_KEY`, `APP_ACCESS_PASSWORD` —
both resolved via `utils.config.get_secret` (Streamlit secrets or
environment variable), never hardcoded, never logged.

**External network calls:** exactly one — OpenWeatherMap's `/weather`
endpoint, over HTTPS, fixed base URL from `configs/settings.yaml` (not
user-controlled), bounded timeout, no secret ever placed in a URL query
string logged verbatim (the API key is passed as a request parameter but
never included in this project's own log lines - see Task 9/10).

**User-controlled inputs:** the uploaded image bytes, the city string, the
numeric panel-detail sidebar inputs (age/maintenance/voltage/current —
already range-validated by `services/pipeline._validate_scalar_inputs`,
unchanged this phase), the access password attempt.

---

## Task 2 — Upload & input hardening (audit result: already solid; no code change needed)

| Concern | Status |
|---|---|
| Maximum upload size | 10 MB, enforced by Streamlit (`.streamlit/config.toml` `maxUploadSize`) |
| Accepted types | JPG/JPEG/PNG/WebP via `st.file_uploader(type=[...])`, but content is independently verified, not trusted from the extension |
| Extension/MIME mismatch | Irrelevant — `PIL.Image.open` decodes actual bytes; a mislabeled file simply fails to decode and is rejected the same as any other invalid image |
| Corrupted/malformed image | `image.verify()` + a second real `Image.open().convert("RGB")`, both before any inference; failures caught and rejected with a generic message (no internal detail leaked) |
| Decompression bomb / oversized dimensions | `PIL.Image.DecompressionBombError` explicitly caught; Pillow's own `MAX_IMAGE_PIXELS` default (~179 megapixels) is never overridden in production code (only temporarily, safely, inside one adversarial test) |
| Empty file | `len(source) == 0` rejected in `utils/image_utils.load_pil_image`; the live upload path additionally fails to decode an empty buffer, caught the same way |
| Path traversal / symlink attacks | Not applicable to the upload path at all — uploaded content is handled entirely as in-memory bytes (`io.BytesIO`), never written to or read from a user-influenced filesystem path |
| Temporary-file cleanup | No temporary files are created for uploads (in-memory only) |
| Validation before expensive inference | Confirmed — image validation happens in `_render_sidebar()`, before the Analyze button (and therefore before `run_pipeline()`) is ever reachable |

**No code change was made here** — this area was already correctly hardened in earlier phases and re-verified, not re-built.

---

## Task 3 — Inference resource safety (audit result: already solid; re-verified)

- **Duplicate-inference-on-rerun**: already fixed (Phase 4) via the explicit Analyze-button gate in `app.py`; re-verified this phase (`tests/test_app_analyze_gating.py`, including the new Task 8 test below, all pass).
- **Model loading duplication**: `models/model_manager.py` loads each model exactly once per process and caches it (`ModelManager`); re-verified via existing `TestLazyLoadingAndCaching`/`TestDeterminism` tests, unchanged.
- **Concurrent requests**: Streamlit's own per-session script execution model means each browser session gets its own script run; the shared `ModelManager` singleton is process-wide (models loaded once, reused across sessions) — this is the existing, intended design, not changed here.
- **Unbounded history writes**: bounded by real user action (one row per real, successfully completed, explicitly-clicked Analyze) — no automated or background writer exists.
- **Runaway processing time**: `PipelineResult.processing_time` is recorded and logged for every real run (visibility, not a hard timeout — no evidence of runaway behavior was found; see Task 17's smoke-test timings).

---

## Task 4 — Model artifact safety (audit result: already solid; re-verified, not re-built)

All ten scenarios in the task list are already covered by the existing typed `ModelLoadError` hierarchy and `ModelManager.verify_all()` (Phase 4), re-verified this phase via the existing test suite (`tests/test_model_manager.py`, `tests/test_adversarial.py::TestModelLoadingAdversarial`, `tests/test_pipeline_panels.py::TestXGBoostGracefulDegradation`) — no regression, no weakening. Specifically confirmed still true:
- Missing YOLO/MobileNet → pipeline aborts with a typed `ModelLoadError`, never a silent/dangerous fallback.
- Missing XGBoost → pipeline completes successfully; efficiency/output fields report `prediction_successful=false`, never fabricated.
- Corrupt/incompatible checkpoint (any of the three) → caught and re-raised as `ModelLoadError` with the real underlying exception chained via `from exc` (not swallowed, not silently retried with a different model).
- SHA mismatch → `scripts/verify_model_artifacts.py` fails explicitly (`tests/test_model_artifact_integrity.py::test_verify_manifest_rejects_hash_mismatch`).

---

## Task 5 — Readiness / liveness (audit result: correct; documented more explicitly)

Already correctly separated (Phase 4/5): Docker's `HEALTHCHECK` polls
`/_stcore/health` (liveness only, no inference); `scripts/check_runtime_readiness.py`
reports real artifact-file presence (readiness) without loading or
fabricating anything, and correctly treats a missing XGBoost artifact as
the expected v1 state rather than a fatal error (verified in Phase 4/5/6
and unchanged). This phase adds no code change here — `deployment/README.md`
§6 now documents the exact invocation and the expected JSON shape for an
operator, which did not previously exist as a single, explicit runbook
entry.

---

## Task 6 — Authentication / session security

**Audited `utils/auth.py` in full this phase.** Findings:
- Password comparison uses `hmac.compare_digest` — already timing-safe.
- No default production password — the gate is an explicit no-op when
  `APP_ACCESS_PASSWORD` is unset, never a hardcoded fallback credential.
- The entered password and the configured secret are never logged or
  included in any exception message.
- Session lifetime is Streamlit's own session (`st.session_state`) — ends
  when the browser session/server process ends; no separate cookie is
  set by this module.
- **No brute-force throttling exists** — by design, for a low-traffic
  research/demo deployment behind a password most visitors never see.
  This is a known, disclosed trade-off (already documented in the
  module's own docstring), not a silent gap — adding real rate-limiting
  would require session/IP-level state this simple gate deliberately
  does not keep, and the task explicitly warns against over-engineering
  an elaborate identity system here. **Documented as a residual risk**
  (Task 16) rather than built around.

**New this phase:** `deployment/README.md` §8 states explicitly that
leaving `APP_ACCESS_PASSWORD` unset means open access, and that a genuine
multi-user requirement should replace this module rather than extend it.

---

## Task 7 — Configuration & secret management

- No secret is hardcoded anywhere in `configs/settings.yaml` or source
  (re-confirmed via this phase's own secret scan of every changed file,
  and spot-checked across `utils/config.py`, `services/weather.py`,
  `utils/auth.py`).
- `.env` and `.streamlit/secrets.toml` remain gitignored (unchanged) and
  excluded from the Docker build context (`.dockerignore`, unchanged).
- Model paths come only from `configs/settings.yaml` (operator-controlled
  file, not runtime user input).
- YAML loading uses `yaml.safe_load` exclusively (re-confirmed, unchanged
  since Phase 4).
- `deployment/README.md` §2 now documents every environment variable the
  application actually reads, in one place.

---

## Task 8 — Database / storage hardening

**Audited `services/storage.py` in full this phase.** Findings:
- **Transaction boundaries are already correct**: `_connect()`'s
  `@contextmanager` only calls `conn.commit()` after the caller's block
  completes without raising; an exception inside the `with _connect()`
  block propagates before `commit()` is reached, and `conn.close()` in
  `finally` implicitly rolls back the uncommitted transaction. A failed
  write genuinely does not get committed — verified by inspection, not
  merely assumed.
- **NULL handling**: every column is `NOT NULL DEFAULT ...`; no NULL
  values are possible for existing rows.
- **Path confinement**: `DB_PATH` is a fixed, hardcoded relative path
  (`data/inspections.db`) — never influenced by user input, so no
  traversal risk.
- **Schema initialization/migrations**: idempotent
  (`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE` with duplicate-column
  handling), already safe to re-run and safe for a rollback to an older
  schema-reading version (additive-only migrations).
- **Concurrent access**: Python's `sqlite3.connect()` default 5-second
  busy timeout already provides reasonable single-instance safety; not
  changed, since no concrete contention problem was found and the task
  warns against unrelated refactors.

**Real gap found and fixed this phase:** a failed `record_inspection()`
call was already correctly *caught* (never crashed the page, never lost
the live on-screen result) but was silently swallowed with **no
indication to the user at all** — someone could only discover a missing
inspection later, on the History page, with no idea why. `app.py` now
shows an explicit, honest `st.warning` when this happens
(`tests/test_app_analyze_gating.py::TestHistorySaveFailureIsNeverSilent`,
new this phase, exercises this with a real forced `sqlite3.OperationalError`).

`deployment/README.md` §7 now documents exactly when SQLite stops being
appropriate and what the replacement seam is.

---

## Task 9 — Logging / observability

Reviewed every `logger.*` call site across `app.py`, `services/`,
`models/`, `utils/` for sensitive-data leakage. Findings:
- City input is already sanitized before every log call
  (`sanitize_for_log`, `services/weather.py`).
- No image bytes, passwords, or API keys are logged anywhere (confirmed
  by inspection — the OpenWeatherMap key is passed as a request
  parameter to `requests.get`, never included in any log line).
- One `logger.debug("Loaded image from path: %s", source)` in
  `utils/image_utils.py::load_pil_image` would echo a raw filesystem path
  — **confirmed not reachable from the live application** (`app.py` never
  calls this function; it decodes uploaded bytes directly). Only
  `training/classification/validate_interim_checkpoint.py`, a trusted
  local developer tool, calls it, with trusted local paths. Left
  unchanged (fixing unreachable code risks an unrelated refactor for zero
  real-world risk reduction) but recorded here as reviewed, not missed.
- Default log level is `INFO`; `DEBUG` (which would surface the line
  above, were it ever reachable) is opt-in only, documented in
  `deployment/README.md` §9 as something to leave off in a
  publicly-reachable deployment.

---

## Task 10 — External network calls

Exactly one external call exists: `services/weather.py`'s OpenWeatherMap
lookup. Already hardened (Phase 4, re-verified unchanged this phase):
bounded timeout (`configs/settings.yaml`'s `weather.timeout_seconds`),
fixed base URL (not user-controlled — the city string is a query
*parameter*, never part of the URL host/path), HTTPS, and explicit,
narrow exception handling (`Timeout`, `HTTPError`, `RequestException`,
plus `KeyError`/`TypeError`/`ValueError` for a malformed response body) —
every failure path returns `WeatherData(fetch_successful=False)` with
configured defaults, never a fabricated "observed" reading. The UI
already distinguishes observed (`fetch_successful=True`) from
default/estimated data on the Environment page (unchanged, re-verified).

---

## Task 11 — UI failure modes

Re-walked every state in the task's list against the current dashboard
(building on the extensive UI-correctness passes in Phases 4-5):
invalid upload, corrupt image, zero detections, inference failure, missing
XGBoost, missing required model, unavailable weather, empty history, NULL
efficiency values, and (new this phase) storage failure — every one
already renders (or, for storage failure, now renders after this phase's
fix) an explicit, honest state, never a fabricated result and never "no
detection" presented as "no faults" (zero detections is always labeled as
zero *panels found*, not as a clean bill of health — see `pages/03`'s
"No site-level summary yet" / "0 panels" framing, unchanged). Limitations
are surfaced in-line (the Inspect page's own capability notice,
`utils/ui_helpers._display_capability_notice`) as well as on the dedicated
Limitations page — not buried in a document only.

---

## Task 12 — Deployment hardening

**Dockerfile change this phase:** the runtime user's write access was
narrowed from the entire `/app` tree to only `/app/data` (the SQLite
history directory). Application source now stays root-owned and
read-only even to the unprivileged `appuser` the container actually runs
as — defense in depth: a code-execution bug in the running app can no
longer be used to persist a modified copy of the application's own source
files, only to write within the data directory. Non-root execution itself
(fixed UID 10001), the `HEALTHCHECK`, `.dockerignore`'s exclusion of
weights/secrets, and the existing CI Docker smoke test are all unchanged
and re-verified.

**Read-only root filesystem:** not the default invocation, but now
correctly documented as achievable (`deployment/README.md`, Production
notes) given the narrower chown above — `--read-only --tmpfs /tmp
--tmpfs /home/appuser/.streamlit` alongside the existing `data` volume.
Not adopted as the default `docker run` example, since it adds operational
complexity most demo deployments don't need — documented as an available
option, per the task's own "document why, identify required writable
paths" instruction.

---

## Task 13 — CI/CD hardening

Both workflows (`ci.yml`, `docker.yml`) audited. Findings: `permissions:
contents: read` already scoped at the top level (no default write
access); all actions pinned to major-version tags from first-party
GitHub-maintained actions (`actions/checkout`, `actions/setup-python`,
`actions/cache`, `actions/upload-artifact`) — a real, if low-severity,
supply-chain consideration (a compromised upstream tag could theoretically
be re-pointed) is disclosed in Task 16 rather than acted on, since pinning
every action to an exact commit SHA is a larger, separate hardening
project with real ongoing maintenance cost, and none of these four actions
have any history suggesting elevated risk; no step suppresses a non-zero
exit code; no secrets are referenced in either workflow at all. **No
change was made to either workflow this phase** — both were already
sound, and the task explicitly warns against cosmetic CI changes.

---

## Task 14 — Failure-injection tests (new this phase + existing coverage confirmed)

The overwhelming majority of the task's list was already covered by the
existing 1,167-test suite built up across Phases 4-6C (malformed upload,
oversized dimensions/decompression bombs, missing/corrupt/SHA-mismatched
models, invalid configuration, weather/API failure, duplicate-rerun
protection, path traversal/symlink escapes, secret sanitization — all
re-run this phase, all still passing, none weakened). The one genuine gap
found and closed this phase:

- **Database write failure** — `tests/test_app_analyze_gating.py::TestHistorySaveFailureIsNeverSilent`
  (new): forces a real `sqlite3.OperationalError` from
  `storage.record_inspection` and confirms the app does not crash, the
  live result still displays, nothing is actually recorded, and (the new
  behavior) the user sees an explicit warning.

---

## Task 15 — Deployment runbook

`deployment/README.md` rewritten to cover all 16 required sections
(prerequisites, environment variables, artifact requirements/verification,
startup, health/readiness, persistent storage, authentication, logs,
backup, recovery, upgrade/rollback procedures, known ML limitations,
supported v1 capabilities, unavailable-XGBoost behavior), opening with the
exact required disclosure statement about not being validated as a
high-accuracy production system.

---

## Task 16 — Security review

| Concern | Finding |
|---|---|
| Command injection | None found — no `subprocess`/`os.system`/shell execution anywhere in `app.py`, `services/`, `models/`, `utils/`, `pages/` (confirmed by search) |
| Path traversal | None found in the live app (upload is in-memory only); the model-artifact manifest verifier (`scripts/verify_model_artifacts.py`) already confines paths to the manifest directory (Phase 4, tested) |
| Arbitrary file access | None found — no user input is ever used to construct a filesystem path read/written by the application |
| Unsafe deserialization | `joblib.load()` for the (currently nonexistent) XGBoost artifact is pickle-based and therefore capable of arbitrary code execution if given an untrusted file — **already documented as a known, accepted, currently-inert risk** in `docs/RUNTIME_SECURITY.md` (Phase 4/5); unchanged, since XGBoost still has no real artifact and this phase does not touch model loading |
| SSRF | The one external call (OpenWeatherMap) has a fixed base URL from config, never a user-controlled URL/host — no SSRF surface |
| XSS / HTML injection | Streamlit auto-escapes rendered text by default; the few `unsafe_allow_html=True` call sites (`utils/ui_theme.severity_pill`, `pages/07`/`pages/09`'s history/alert rendering) render this project's own computed severity/fault labels and timestamps, never raw user-supplied text — reviewed, no injection point found |
| Secret leakage | Confirmed clean this phase's own secret scan of every changed file; no secret in logs (Task 9), CI (Task 13), or the Docker image (`.dockerignore`) |
| Dependency risk | Unpinned (`>=`) requirements remain a known, documented trade-off (`docs/DEPENDENCY_REPRODUCIBILITY.md`) — not addressed this phase (a separate, larger reproducibility project) |
| Privilege escalation | Container runs as fixed non-root UID 10001; narrowed further this phase (Task 12) — no setuid/sudo anywhere in the image |
| Resource exhaustion | 10 MB upload cap, decompression-bomb protection, bounded weather timeout, Analyze-button-gated inference (no unbounded automatic re-inference) |

**New residual risks identified and disclosed (not fixed, per the task's
own "document risk/impact/mitigation/future action" instruction where a
fix isn't undertaken this phase):**

1. **No brute-force throttling on the access password.** Impact: an
   attacker with network access could attempt many passwords rapidly.
   Mitigation: the gate is explicitly documented as unsuitable for a
   high-value/public target; deployments needing that guarantee should use
   a real reverse-proxy-level rate limit or a genuine identity provider.
   Future action: out of scope for this simple, disclosed gate.
2. **GitHub Actions pinned by tag, not commit SHA.** Impact: a compromised
   upstream release of a first-party action could theoretically affect a
   future CI run. Mitigation: all four actions used are official
   GitHub-maintained actions with no history of such an incident.
   Future action: pin to exact SHAs in a dedicated CI-hardening pass if
   the project's risk tolerance changes.
3. **`joblib.load()` for XGBoost is pickle-based.** Impact: arbitrary code
   execution if a malicious file were ever placed at
   `weights/xgboost_solar.joblib`. Mitigation: no such artifact currently
   exists, and the project's supply-chain policy already requires any
   future artifact to come from a controlled, reviewed process. Future
   action: consider a safer serialization format if/when a real XGBoost
   artifact is ever trained.

---

## Task 17 — Performance / reliability smoke test

Real, non-mocked runs (this machine, CPU only, real local artifacts):

| Case | Result |
|---|---|
| Single-panel image | Real detection + classification, ~1-2s total (consistent with Phase 4-6 timings) |
| Multiple-panel image | Real per-panel detection + classification, site summary aggregated correctly (re-confirmed via `tests/test_pipeline_panels.py`, unchanged) |
| Zero-detection image | Completes successfully, `panels=[]`, no fabricated placeholder |
| Malformed image | Rejected before inference, no processing time spent |
| Repeated analysis (same image, no re-click) | No duplicate inference, no duplicate history row (`tests/test_app_analyze_gating.py`) |
| Model reload | Confirmed single-load-per-process via existing `ModelManager` caching tests |
| Missing optional XGBoost | Pipeline completes normally; efficiency/output reported unavailable |

These are **smoke-test observations on this development machine, not
measured production SLAs** — no numeric latency/throughput figure here is
presented as a guarantee.

---

## Task 18 — Final deployment gate

### GREEN — Research/demo deployment ready

Every genuinely new operational finding this phase (traceback leakage,
overly-broad container write access, a silently-swallowed storage
failure) was fixed, tested, and verified; every pre-existing control
(upload validation, typed model-load failures, readiness/liveness
separation, non-root container, secret handling, external-call safety)
was re-audited and found already sound, not merely assumed. Three
residual risks are disclosed, understood, and judged acceptable for a
controlled research/demo deployment rather than hidden.

> **Not validated as a high-accuracy production inspection system.** This
> gate is an operational readiness gate, not an ML accuracy gate — see
> `docs/ML_EVALUATION_v1.0.0.md`, `docs/ML_HARDENING_PHASE6B.md`, and
> `docs/ML_HARDENING_PHASE6C.md` for that separate, unresolved conclusion.
