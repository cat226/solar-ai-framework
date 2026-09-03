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

---

# Features

### Computer Vision

- YOLOv8 object detection
- MobileNetV2 fault classification
- Image preprocessing
- Confidence scoring

Supported fault classes:

- Clean
- Dust
- Bird Droppings
- Electrical Damage
- Physical Damage
- Snow Coverage

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

No AI logic exists inside the UI.

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
Detection
   │
   ▼
Classification
   │
   ▼
Weather Collection
   │
   ▼
Physics Features
   │
   ▼
Feature Engineering
   │
   ▼
Regression Prediction
   │
   ▼
Maintenance Recommendation
```

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

This is a native Streamlit multi-page app. `app.py` is the **Inspect** page
(upload an image, run the full pipeline); the sidebar navigation also lists
**Dashboard**, **History**, **Analytics**, **Alerts**, and **Settings** —
all driven by `services/storage.py`, a local SQLite log of every real,
successfully-completed inspection (never fabricated data; the uploaded
image itself is never stored, only its SHA-256 hash). With no inspections
recorded yet, each of those pages shows an explicit empty state rather
than invented numbers.

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

Place the trained models inside the `weights/` directory:

- `weights/yolo_solar.pt`
- `weights/mobilenet_solar.pth`
- `weights/xgboost_solar.joblib`

These files are not included in the repository and must be supplied separately.

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

- 676 tests collected
- ~94% statement coverage
- Core business-logic modules at 100% statement coverage

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

| Module | Status |
|---------|--------|
| Architecture | ✅ Complete |
| Streamlit Dashboard | ✅ Complete |
| YOLO Detection | ✅ Complete |
| MobileNet Classification | ✅ Complete |
| Weather Integration | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Regression Prediction | ✅ Complete |
| Maintenance Recommendation | ✅ Complete |

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
