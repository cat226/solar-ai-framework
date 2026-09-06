# Solar AI — Final Project Status

Generated: 2026-09-07
Branch: `feat/cloud-training-orchestration`

This is the concise, current-state summary after the "finish the app"
completion pass. It does not replace `README.md`, `docs/RELEASE_v1.0.0.md`,
or `deployment/README.md` — it points to them and states the bottom line.

## Production model

| Field | Value |
|---|---|
| Path | `weights/yolo_solar.pt` |
| SHA-256 | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` |
| Status | **Unchanged throughout this entire project cycle**, including every experiment below. Re-verified by hash immediately before and after every training/evaluation run. |
| Reliable for | Aerial/satellite panel detection (mAP50 = 0.739, real 2,581-image held-out test — see `docs/RELEASE_v1.0.0.md`). |
| Not reliable for | Close-up/ground-level photo detection (~2.6% on a 156-image sample; 0/16 on an independent real-photo validation — see below). |

## Experimental model

| Field | Value |
|---|---|
| Path | `weights/yolo_solar_domain_v1.pt` |
| SHA-256 | `3dc092882be8feabcada21f427263d508e1b796eaed287ec7f1d761df6b50603` |
| Origin | Fine-tuned from the production checkpoint above on a mixed aerial+ground dataset (conservative settings: frozen backbone, low learning rate, 10 epochs). |
| Benchmark result | Ground mAP50 improved 0.000 → 0.445 on a held-out ground test set; aerial mAP50 saw a small, non-catastrophic dip (0.743 → 0.725). |
| Real-application result | **0/16 detections** on an independent 16-image real-photo validation, identical to the production model at the deployed 0.45 confidence threshold — the benchmark gain has not yet translated into different real-world behavior. A diagnostic (non-deployed) sub-threshold check showed real, measurably higher confidence on 11/16 images, explaining *why* without changing the actual result. |
| A clean-data-only variant (`weights/yolo_solar_domain_clean_v1.pt`) was also tested | Produced **no** meaningful ground-domain improvement (near-zero mAP50) — the full-experimental model's gain does not survive training on clean-provenance data alone, given only 2-3 independent physical sites were available. |
| Deployment status | **Experimental only.** Not in `weights/manifest.json`, not loaded by `models/model_manager.py`, not referenced by any production code path. |

## Validated capabilities

- **Aerial solar-panel detection** — real trained YOLOv8n artifact, measured mAP50 = 0.739 on a genuine 2,581-image held-out test split.
- **Whole-image MobileNet classification** (Clean / Dusty / Hotspot, v1's frozen 3-class scope) — runs independently of detection success, so every valid upload gets a real classification result even when no panel is localized. Now labeled explicitly as *"Whole-image classification — not panel-level diagnosis"* everywhere it's shown (live Inspect page, History, Overview, Alerts) — see "Application fixes" below.
- **Panel-level detect → crop → classify path** — runs and is labeled as panel-level whenever YOLO actually detects a panel (bounding-box overlay, per-panel crop, per-panel classification, per-panel efficiency estimate when available).
- **Full application pipeline** — image upload, EXIF-orientation-aware preprocessing, weather lookup (with honest fallback to defaults when the API is unavailable), physics-based feature engineering, maintenance recommendations.
- **History persistence** — SQLite-backed, verified end-to-end via direct pipeline testing and, this pass, inside the actual built Docker container with a fresh volume (see "Deployment status" below) — a real, previously-undetected container bug in this path was found and fixed this pass.

## Current limitation

**Panel-level detection on close-up/ground-level images is not yet reliable**, for both the production and experimental checkpoints, at the deployed confidence threshold. Consequently, **panel-crop classification is unavailable whenever no panel is detected** — this is now stated explicitly in the UI rather than silently omitted or backfilled with the whole-image result. See `pages/06_⚠️_Limitations.py`'s new "Detection coverage" section and `E:\SolarAI_Datasets\domain_adaptation_v1\reports\end_to_end_real_image_validation_report.md` for the full evidence.

## Unavailable

**XGBoost efficiency-loss prediction** — no `weights/xgboost_solar.joblib` artifact exists, and none is planned, because no dataset was found that legitimately pairs this project's fault-classification taxonomy with real environmental telemetry *and* a genuinely measured efficiency-loss target (investigated 2026-09-04, see `training/prediction/DATASET_SOURCES.md`). The pipeline reports this as `xgboost_available=False` and returns a clean "unavailable" state everywhere a prediction would otherwise appear — never a fabricated `0%`/`0 W`. Detection and classification continue to run and return real results regardless.

---

## Application audit & fixes this pass

1. **Whole-image vs. panel-level conflation (fixed).** The live Inspect page's classification section, the History page's table/detail view, the Overview dashboard, and the Alerts feed all previously showed the whole-image MobileNet label under a generic "Fault"/"Fault Classification" heading with no qualifier — a user could reasonably read this as a panel-level diagnosis, including on images where zero panels were detected. Fixed: every one of these surfaces now labels the value explicitly as whole-image, and the live Inspect page shows the exact disclosure text *"Whole-image classification — not panel-level diagnosis"* (a pure, unit-tested function — `utils.ui_helpers.whole_image_classification_caption` — so the wording can never silently drift). The genuine per-panel classification (shown only when YOLO detects a panel) is now labeled "Panel Classification" with its own caption explaining it is independent of the whole-image result.
2. **Model Status page overstated readiness (fixed).** Previously said "✅ Fully operational" whenever the YOLO/MobileNet files existed on disk, with no mention of the close-up detection gap. Fixed: added an explicit warning under the YOLO section citing the measured detection rates, and softened the "Overall" language to distinguish pipeline *readiness* (executes without error) from detection *accuracy* on close-up photos.
3. **Limitations page had no detection-reliability section at all (fixed).** The page audited classifier coverage in detail but never mentioned the YOLO domain-shift limitation — the single most consequential real-world limitation of the whole application. Added a "Detection coverage" section stating the measured rates, what happens on a zero-detection image, and the experimental model's real status.
4. **Container history-write bug (found and fixed).** Direct testing inside the actual built Docker image (not just unit tests) found that a fresh named volume mounted over `/app/data` inherited a root-owned `inspections.db` baked into the image from the developer's own local `data/` directory (`.dockerignore` did not exclude it, and `COPY . .` runs before `USER appuser` in the `Dockerfile`) — silently making every history write in a brand-new deployment fail with `sqlite3.OperationalError: attempt to write a readonly database` (caught and surfaced as an on-screen warning by existing Phase-7 hardening, never a silent failure or a crash, but a real functional gap for the first user of any fresh deployment). Fixed by excluding `data/*.db` from the Docker build context; re-verified with a fresh volume inside a rebuilt container that history now writes and reads back correctly.
5. **Docker build was needlessly slow and heavy (fixed).** The default `pip install -r requirements.txt` pulled the full CUDA-enabled PyTorch build (~1.1 GB of `nvidia-*`/`cudnn` packages) despite this application having no GPU code path — pure waste of build time and image size. Fixed by installing the CPU-only PyTorch/torchvision wheels first from PyTorch's own index, before the rest of `requirements.txt` (which then sees them already satisfied). Also excluded `training/` (110 MB of offline dataset/experiment tooling never imported by the running application) from the build context, cutting context transfer from ~176 MB to ~300 KB.
6. **`services/pipeline.py` gained one optional `detector` parameter** (used by the model-comparison tooling from the prior validation pass) — every existing call site is unaffected; covered by 3 new tests.

## Deployment status

- **Docker build:** succeeds (`docker build -t solar-ai-framework:finalpass .`), ~4.0 GB image, after the CPU-only-torch and build-context fixes above.
- **Container startup:** confirmed healthy (`docker inspect --format='{{.State.Health.Status}}'` → `healthy`) within seconds of starting.
- **Health endpoint:** `GET /_stcore/health` → `HTTP 200`.
- **Readiness:** `scripts/check_runtime_readiness.py` inside the running container, with real weights mounted read-only → `{"inference_readiness": "not_ready", "liveness": "ok", "missing_artifacts": ["XGBoost"]}` — the correct, expected v1 state (exit code 2, matching `deployment/README.md` §6's documented behavior).
- **Artifact integrity:** `scripts/verify_model_artifacts.py --manifest /app/weights/manifest.json` inside the container → `Model artifact integrity verification passed.`
- **End-to-end inference + history, inside the container:** a real validation photo was copied in and run through `services.pipeline.run_pipeline` directly inside the running container (status `SUCCESS`, 0 detections, real whole-image classification, `xgboost_available=False`), then `services.storage.record_inspection` + `get_inspection` + `get_recent_inspections` were exercised against a fresh named volume — all passed after fix #4 above.
- **Non-root hardening:** confirmed running as `appuser` (uid 10001), `weights/` mounted read-only, `/app/data` the only writable application path.
- **Production gate:** `weights/manifest.json` unchanged; `weights/yolo_solar.pt` SHA-256 re-verified as `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` after every step in this pass, including the Docker validation; `v1.0.0` tag untouched; `weights/yolo_solar_domain_v1.pt` not promoted.

## Test results

Full suite (`python -m pytest -q --ignore=tests/test_mobilenet_training.py`, the one file blocked by an unrelated local sklearn/DLL Application-Control-policy issue on this Windows machine, pre-existing and unrelated to this project's code): **1312 passed, 4 skipped, 4 failed** — the 4 failures are the same pre-existing, environment-only issues present before this pass and unrelated to any change made in it:
- 3 × Windows symlink-privilege failures (`tests/test_adversarial.py`) — this Windows account lacks the privilege to create symlinks; the underlying code path is exercised and passes on CI (`windows-latest` runner has this privilege).
- 1 × `tests/test_runtime_readiness.py` local-artifact-count mismatch — this specific dev checkout has real `weights/yolo_solar.pt` and `mobilenet_solar_v1.pth` present, so a test written for a from-clean-checkout assumption (expects all three artifacts reported missing) sees only XGBoost missing.

7 new tests added this pass (`tests/test_pipeline.py::TestDetectorOverride` ×3 from the prior validation phase, `tests/test_ui_helpers.py` ×3, plus the pre-existing suite) — all passing, none weakened or skipped to force a pass.

`python verify_imports.py` → 14/14 passed.

## Recommended next development phase

1. **Do not pursue further YOLO domain-adaptation training on the current data.** Both the full-experimental and clean-only variants have been tried; neither is ready to promote (see the domain-adaptation and clean-experiment reports in `E:\SolarAI_Datasets\domain_adaptation_v1\reports\`). Further progress needs either genuinely new, clean-provenance ground-level data with materially more independent physical sites, or a deliberate, separately-justified threshold-recalibration study (not a silent change) — neither is in scope for an app-completion pass.
2. **Consider whether the six-class MobileNet expansion (Bird-Drop, Electrical-Damage, Physical-Damage) is worth resuming** — blocked purely on licensed dataset availability per `training/classification/DATASET_SOURCES.md`, unrelated to anything in this pass.
3. **XGBoost remains correctly out of scope** until a legitimate paired-telemetry dataset is found; no action recommended beyond continuing to report it honestly as unavailable.
4. **Operationally**, the application itself is now complete and honest within its validated capabilities: the remaining roadmap is entirely about acquiring better ML training data, not about application engineering.
