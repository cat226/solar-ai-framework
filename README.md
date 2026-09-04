# ☀️ Solar AI Framework

> **A Multimodal AI Framework for Intelligent Solar Panel Fault Diagnosis, Efficiency Prediction, and Maintenance Recommendation**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

Solar AI Framework is a research-oriented intelligent solar panel monitoring system designed for **journal publication**.

The framework combines computer vision, weather intelligence, physics-based feature engineering, and machine learning to automatically:

- Detect solar panel regions
- Classify panel faults
- Collect real-time environmental data
- Estimate physical operating conditions
- Predict solar panel efficiency
- Generate maintenance recommendations

Unlike conventional monitoring systems that rely only on sensor measurements or image classification, this framework integrates **visual**, **environmental**, and **physics-inspired** information into a unified prediction pipeline.

---

# Research Objective

The primary objective of this research is to develop an intelligent decision-support framework capable of improving photovoltaic maintenance through multimodal artificial intelligence.

The proposed system aims to:

- Detect visible panel faults automatically
- Estimate environmental operating conditions
- Predict efficiency degradation
- Assist maintenance planning
- Provide an explainable workflow suitable for real-world deployment

---

# Current AI Pipeline

```text
Solar Panel Image
        │
        ▼
YOLOv8 Object Detection
        │
        ▼
MobileNetV2 Classification
        │
        ▼
OpenWeatherMap API
        │
        ▼
Physics-Based Feature Engineering
        │
        ▼
XGBoost Efficiency Prediction
        │
        ▼
Maintenance Recommendation
        │
        ▼
Streamlit Dashboard
```

*Current deployment status (Solar AI v1, frozen release): YOLOv8 detection
uses a real trained artifact; MobileNetV2 classification runs the real,
frozen v1 3-class checkpoint (Clean/Dusty/Hotspot) - this is v1's
intentional, complete scope, not a placeholder; XGBoost has no trained
artifact and none is planned until a genuine dataset is found. See
[Current Project Status](#current-project-status) below.*

---

# Features

### Computer Vision

- YOLOv8 object detection
- MobileNetV2 fault classification
- Image preprocessing
- Confidence scoring

**Solar AI v1 supports solar-panel detection and three fault classes: Clean, Dusty, and
Hotspot.** This is v1's real, frozen classifier scope - not a placeholder pending
completion.

Supported now (v1):

- Clean
- Dusty
- Hotspot

Documented future roadmap - not part of v1, not currently classifiable:

- Bird-Drop
- Electrical-Damage
- Physical-Damage

See [Current Project Status](#current-project-status) and
[Known Limitations](#known-limitations) for exactly why, and
`training/classification/DATASET_SOURCES.md` for per-class dataset provenance.

---

### Weather Intelligence

Current implementation collects:

- Temperature
- Humidity
- Cloud Coverage
- Wind Speed
- Atmospheric Pressure

Future versions will integrate more accurate irradiance estimation.

---

### Physics-Based Feature Engineering

Current features include:

- Irradiance estimation
- Module temperature estimation
- Soiling ratio
- Environmental feature generation

---

### Machine Learning

Regression model predicts:

- Solar panel efficiency
- Estimated power degradation

Current implementation uses:

- XGBoost

---

### Maintenance Recommendation

Automatically evaluates:

- Panel cleanliness
- Weather conditions
- Panel age
- Maintenance history

Generates maintenance suggestions based on the prediction pipeline.

---

# Project Architecture

```text
solar-ai-framework/

├── app.py
├── models/
├── services/
├── utils/
├── configs/
├── tests/
├── docs/
└── deployment/
```

The architecture follows a modular design where each module has a single responsibility.

---

# Folder Structure

## app.py

Contains only the Streamlit user interface.

Responsibilities:

- User interaction
- Input collection
- Result visualization

No AI logic exists inside the UI. This is the app's main entry point and
implements the **Inspect** workflow (upload → detect → classify → estimate).

---

## pages/

The rest of the multi-page dashboard, each page reading only real data
(`services/storage.py`'s recorded history, `models/model_manager.py`'s live
artifact status, or `st.session_state["last_result"]` set by `app.py` after
a completed inspection) — no page fabricates a value it doesn't have:

| # | Page | Shows |
|---|------|-------|
| 01 | Overview | Top-level KPIs, real recorded history, live system status |
| 02 | Panel Results | Per-panel table for the most recent live inspection |
| 03 | Site Health | Site-level rollup (this inspection + aggregate history) |
| 04 | Environment | Weather/physics inputs actually used, and their source |
| 05 | Model Status | Per-model readiness, real artifact SHA-256, v1/six-class state |
| 06 | Limitations | Honest, live-cross-checked capability disclosure |
| 07 | History | Searchable full inspection history |
| 08 | Analytics | Trend charts over real recorded history |
| 09 | Alerts | Real CRITICAL/WARNING inspections + live system-availability alerts |
| 10 | Settings | Access control, privacy, inference configuration |

---

## models/

Contains AI model wrappers.

- YOLO Detector
- MobileNet Classifier
- XGBoost Predictor
- Model Manager

---

## services/

Business logic layer.

Contains:

- Weather Service
- Physics Engine
- Feature Engineering
- Recommendation Engine
- AI Pipeline

---

## utils/

Shared utilities.

Examples:

- Configuration Loader
- Logger
- Image Utilities
- Custom Exceptions

---

## configs/

Stores project configuration.

Contains:

- Default parameters
- Thresholds
- Weather settings

Sensitive credentials are stored separately using environment variables or Streamlit Secrets.

---

## tests/

Contains unit and integration tests.

---

## docs/

Research documentation.

- Architecture
- Experimental Notes
- Roadmap

---

## deployment/

Deployment configuration.

- Docker
- Streamlit configuration

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Detection | YOLOv8 |
| Classification | MobileNetV2 |
| Regression | XGBoost |
| Deep Learning | PyTorch |
| Image Processing | Pillow |
| Weather API | OpenWeatherMap |
| Data Processing | Pandas |
| Configuration | YAML |
| Logging | Python Logging |

---

# Design Principles

The framework follows these principles:

- Modular Architecture
- Single Responsibility Principle
- Separation of Concerns
- Configuration-Driven Development
- Reproducibility
- Maintainability
- Extensibility

---

# Current Workflow

```text
Image
   │
   ▼
YOLO Detection  ──────────────► Panel crops (one per detection box)
   │                                   │
   ▼                                   ▼
Whole-image Classification      Per-panel Classification
   │                                   │
   ▼                                   ▼
Weather Collection ──────► Physics Features (per panel, shared weather)
   │                                   │
   ▼                                   ▼
Feature Engineering ─────► Feature Engineering (per panel)
   │                                   │
   ▼                                   ▼
Regression Prediction*    Regression Prediction* (per panel)
   │                                   │
   ▼                                   ▼
Maintenance Recommendation*    Site-level Summary (aggregated)
```

\* Regression prediction requires the XGBoost artifact (`weights/xgboost_solar.joblib`).
When it's absent, detection and classification results above are still
real and returned — every prediction/recommendation field is reported as
explicitly unavailable, never fabricated. See `services/pipeline.py`'s
`PipelineResult.xgboost_available` / `PanelResult.prediction.prediction_successful`.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd solar-ai-framework
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
streamlit run app.py
```

This is a native Streamlit multi-page app. `app.py` is the **Inspect** page:
upload a solar panel image (JPG/PNG/WebP, max 10 MB) in the sidebar, adjust
the panel inputs if needed, then click **Analyze** to run the full
pipeline — YOLO detection, whole-image and per-panel MobileNet
classification, weather lookup, physics, and (when
`weights/xgboost_solar.joblib` exists) efficiency/output prediction.
Analysis only runs on that explicit click, not on every sidebar rerun —
Streamlit reruns the whole script on any widget interaction, and running
real inference (and writing a new history row) on every unrelated slider
nudge would both waste real compute and silently duplicate history. The
sidebar navigation lists the other ten pages (see the `pages/` table
above) — **Overview**, **Panel Results**, **Site Health**, **Environment**,
**Model Status**, **Limitations**, **History**, **Analytics**, **Alerts**,
and **Settings** — all driven by `services/storage.py` (a local SQLite log
of every real, successfully-completed inspection — never fabricated data;
the uploaded image itself is never stored, only its SHA-256 hash) or by
`st.session_state["last_result"]` for the pages tied to the most recent
live inspection. With no inspections recorded/run yet, each page shows an
explicit empty state rather than invented numbers.

---

# Environment Variables

Create a `.env` file:

```text
OPENWEATHER_API_KEY=YOUR_API_KEY
```

Optional — protect the deployment behind a single shared password (see
`utils/auth.py`; this is **not** a multi-user account system, just a gate
against casual unauthenticated access):

```text
APP_ACCESS_PASSWORD=your-shared-password
```

Do **not** commit this file.

---

# Models Required

Place the trained models inside the `weights/` directory (all gitignored —
never committed):

- `weights/yolo_solar.pt` — real YOLOv8n panel detector, and the
  production artifact for v1. A real checkpoint exists as of this writing,
  trained on the full audited 17,107-image BDAPPV IGN dataset via the
  Kaggle cloud pipeline (see `training/cloud/`); test-split mAP50 ≈ 0.74.
  Record: `training/experiments/registry.jsonl`, experiment
  `solar-yolo-full-v1`.
- `weights/mobilenet_solar_v1.pth` — **the real, frozen v1 fault
  classifier** (Clean, Dusty, Hotspot). Not a placeholder or a fallback -
  this is the production artifact for this release. `models/model_manager.py`
  loads this automatically (`models.mobilenet.v1_weights` in
  `configs/settings.yaml`).
- `weights/mobilenet_solar.pth` — the future six-class fault classifier
  (Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Hotspot).
  **Not yet available and not part of v1** — three of the six classes have
  no genuinely licensed, accessible dataset yet (see
  `training/classification/DATASET_SOURCES.md`). `models/model_manager.py`
  prefers this automatically the moment it exists, superseding v1 without
  any application code change.
- `weights/xgboost_solar.joblib` — efficiency-loss regressor. **Not yet
  trained, and not part of v1** — investigated 2026-09-04 (see
  `training/prediction/DATASET_SOURCES.md`): no dataset was found that
  legitimately pairs this project's own `fault_class_id` taxonomy with
  real environmental telemetry *and* a genuinely measured efficiency-loss
  target, so none was fabricated. When absent, `services/pipeline.py`
  still runs detection and classification and returns real results; every
  efficiency/output field is reported as genuinely unavailable
  (`prediction_successful=False`), never a fabricated `0.0`.

None of these files are included in the repository. See
`training/cloud/README.md` for how each was (or will be) produced.

---

# Adding a Future Dataset or Model

- **A missing MobileNet class** (Bird-Drop, Electrical-Damage,
  Physical-Damage): once a genuinely licensed, discretely-labeled dataset
  is acquired, add it under `E:\Solar AI Training Images\source\<ClassName>\`
  (or the equivalent path on another machine — see `training/cloud/base/storage_paths.py`),
  re-run `training/classification/prepare_dataset.py`, then
  `training/classification/train_mobilenet.py` with **no** `--classes`
  argument (its default is the full six) to produce a real six-class
  checkpoint at `weights/mobilenet_solar.pth`. No application code needs
  to change — `models/model_manager.py` already prefers that artifact
  automatically over the v1 one the moment it exists.
- **The XGBoost predictor**: no training pipeline exists yet because no
  legitimate dataset has been found — see the "re-opening this
  investigation" section of `training/prediction/DATASET_SOURCES.md` for
  exactly what a candidate dataset must independently provide (this
  project's own `fault_class_id` taxonomy, full weather telemetry, and a
  genuinely measured `efficiency_loss_pct`) before it can be used. Once one
  is identified, build `training/prediction/` following the same
  provenance-first pattern as `training/classification/` and
  `training/detection/` — never train on this repository's own
  `services/physics.py` heuristic output restated as a label.
- **A new model entirely**: follow the same three-layer pattern as the
  existing three models — a thin wrapper in `models/` that receives an
  already-loaded object via `set_model()`, a loader in
  `models/model_manager.py`, and orchestration added to
  `services/pipeline.py`. Never duplicate model-loading or inference logic
  inside a UI page.
- **Cloud (Kaggle) training for a new model**: reuse
  `training/cloud/base/job_spec.py` / `registry.py` /
  `artifact_validation.py` and follow the pattern in
  `training/cloud/kaggle/build_mobilenet_package.py` (or
  `build_yolo_full_training_package.py`) — a thin entrypoint that clones
  the repo at an exact pinned commit and invokes the real, unmodified
  training script as a subprocess; never duplicate training logic inside
  the Kaggle wrapper. See `training/cloud/README.md`.

---

# Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the full test suite with coverage:

```bash
py -3.12 -m pytest -q
```

Run import verification:

```bash
py -3.12 verify_imports.py
```

Run a single test file:

```bash
py -3.12 -m pytest tests/test_physics.py
```

Run a single test:

```bash
py -3.12 -m pytest tests/test_pipeline.py::TestEntryPointValidation::test_panel_age_at_boundary_100_is_accepted
```

Coverage is configured in `pytest.ini` and reports term-missing output for `services/`, `models/`, and `utils/`.

Current validated results:

- 1071 tests collected
- ~91% statement coverage (varies by run; `utils/ui_helpers.py` and the
  `pages/` files are exercised via Streamlit's `AppTest` framework in
  `tests/test_pages_smoke.py`, not line-by-line unit tests)
- Core business-logic modules at or near 100% statement coverage

---

# Continuous Integration

GitHub Actions runs on every push to `feature/testing-infrastructure` and on pull requests to `main`.

CI environment:

- Runner: `windows-latest`
- Python: 3.12
- Steps: dependency installation → `verify_imports.py` → `pytest -q`

The CI workflow is defined in `.github/workflows/ci.yml`.

---

# Known Limitations

- **Model weights not included:** The `weights/` directory contains only `.gitkeep`. Trained model files (`yolo_solar.pt`, `mobilenet_solar.pth`, `xgboost_solar.joblib`) must be provided separately.
- **Weather API required:** OpenWeatherMap API key must be configured via `.env` or `.streamlit/secrets.toml`.
- **Python 3.12 required:** The test suite and full dependency stack (torch, torchvision, ultralytics) require Python 3.12. Python 3.14 is not supported.
- **No network in tests:** All tests are designed to run without network access. Model weights and API calls are mocked.
- **Windows CI only:** GitHub Actions currently runs on `windows-latest`.
- **Single-deployment persistence, not multi-tenant:** `services/storage.py` is a local SQLite file appropriate for one deployment's own history — it is not a multi-user production database. See its module docstring for the intended replacement seam if that's ever needed.
- **Access gate is a single shared password, not multi-user auth:** `utils/auth.py` blocks casual unauthenticated access; it has no per-user identity, password reset, or SSO. It is a no-op when `APP_ACCESS_PASSWORD` is unset (matching local development).
- **Sites/Assets management and PDF report export are not implemented.** Building genuine versions would require backend persistence beyond what the current single-user SQLite history honestly supports; they were scoped out rather than built as fabricated placeholders.
- **v1 supports three of six taxonomy classes by design** (Clean, Dusty, Hotspot) - this is v1's frozen, intentional release scope, not an incomplete rollout. Bird-Drop, Electrical-Damage, and Physical-Damage remain a documented future expansion and cannot currently be classified — see `training/classification/DATASET_SOURCES.md` for exactly which datasets are blocked and why (some have no genuinely licensed public source at all; others are access-restricted pending the dataset owner's approval, which requires a human account holder, not something this codebase can obtain on its own). The full future class order (`Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Hotspot`) is unchanged and enforced by `training/classification/_dataset_remap.py` — acquiring the missing classes' data and training on the full set is all that's required to supersede v1; no application code needs to change (`models/model_manager.py` already prefers that six-class artifact automatically whenever it exists).
- **No XGBoost artifact exists yet, and none is planned until a genuine dataset is found.** Investigated 2026-09-04 — see `training/prediction/DATASET_SOURCES.md` for the full per-candidate-dataset rejection analysis. Efficiency-loss/output-power predictions are unavailable, not estimated as zero — every part of the UI that would show a prediction instead shows an explicit "unavailable" state (see `services/pipeline.py`'s `xgboost_available` flag), and aggregate KPIs (`services/storage.get_summary_stats`) report `None`, not `0.0`, when no stored inspection ever produced a real prediction.
- **Large local training data lives outside the repository**, on the development machine's `E:\Solar AI Training Images\` drive — see `training/cloud/README.md`'s "Local storage policy" section. This has no effect on the deployed application, which only reads the small artifacts under `weights/`.

## Health and Readiness

The application distinguishes between **liveness** (process is running) and **readiness** (production inference dependencies are available):

- **Liveness:** Docker healthcheck probes `/_stcore/health`. A healthy container means the Streamlit process is accepting requests.
- **Readiness:** The sidebar displays model artifact readiness. When genuine weights are absent, the UI shows a warning. The CLI tool `scripts/check_runtime_readiness.py` reports `not_ready` with a non-zero exit code.

## Model Output Validation

Model wrappers validate their outputs before downstream consumption:

- **YOLO detector:** Validates that bounding box coordinates, confidences, and class IDs are finite. Confidences must be in `[0, 1]`.
- **MobileNet classifier:** Validates that all probabilities are finite and non-negative. Confidence must be in `[0, 1]`.
- **XGBoost predictor:** Validates that prediction outputs are finite before clamping to `[0, 100]`.

Invalid model outputs raise `PredictionError` and stop the pipeline with a controlled error.

---

# Current Project Status

**Solar AI v1 is frozen and released.** v1 supports solar-panel detection and three fault
classes: **Clean, Dusty, and Hotspot**. Bird-Drop, Electrical-Damage, and Physical-Damage
remain a documented future roadmap item, not part of this release. XGBoost efficiency
prediction is currently unavailable because no legitimate training dataset with the
required telemetry and a genuinely measured efficiency-loss target was found.

| Module | Status |
|---------|--------|
| Architecture | ✅ Complete |
| Streamlit Dashboard (10 pages, see below) | ✅ Complete |
| YOLO Detection | ✅ Real trained artifact (Kaggle P100, full 17,107-image BDAPPV IGN dataset) |
| MobileNet Classification | ✅ **v1 release, 3-class** (Clean/Dusty/Hotspot) — this is v1's frozen, intentional scope; Bird-Drop/Electrical-Damage/Physical-Damage are a documented future expansion, see [Known Limitations](#known-limitations) |
| XGBoost Efficiency Prediction | ❌ No trained artifact, not part of v1 — pipeline reports predictions as honestly unavailable rather than fabricating output, see `services/pipeline.py` |
| Weather Integration | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Maintenance Recommendation | ✅ Complete (skipped, not fabricated, when no real prediction exists) |

This table is deliberately not all-green — see the **Model Status** and **Limitations** pages in the running app for the live, per-deployment version of this same information.

---

# Research Roadmap

### Phase 1
- Architecture Refactoring ✅

### Phase 2
- Physics-informed weather and irradiance enhancement

### Phase 3
- Multimodal feature fusion

### Phase 4
- Explainability (e.g., Grad-CAM and SHAP)

### Phase 5
- Experimental evaluation and ablation studies

---

# Intended Publication

This repository is maintained as part of a research project targeting publication in a peer-reviewed journal in the fields of:

- Artificial Intelligence
- Renewable Energy
- Computer Vision
- Smart Energy Systems
- Intelligent Maintenance

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This work builds upon open-source tools and libraries, including:

- Streamlit
- Ultralytics YOLO
- PyTorch
- XGBoost
- OpenWeatherMap
- Pandas
- Pillow

We thank the maintainers and contributors of these projects for enabling reproducible AI research.

---

# Authors

**Solar AI Framework Research Team**

- Ramana Sree
- Verona Ann

---

> **Version:** Architecture Baseline v1.0  
> **Status:** Research Prototype (Architecture Frozen)  
> **Next Milestone:** TASK-002 – Physics-Informed Weather & Irradiance Enhancement
