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

---

# Environment Variables

Create a `.env` file:

```text
OPENWEATHER_API_KEY=YOUR_API_KEY
```

Do **not** commit this file.

---

# Models Required

Place the trained models inside the appropriate models directory.

Required models include:

- YOLOv8 detection model
- MobileNetV2 classification model
- XGBoost regression model

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

- **Model weights not included:** The `weights/` directory contains only `.gitkeep`. Trained model files (`yolo_solar.pt`, `mobilenet_solar.pth`, `xgboost_solar.joblib`) must be provided separately. See `weights/README.md` for the artifact contract.
- **Weather API required:** OpenWeatherMap API key must be configured via `.env` or `.streamlit/secrets.toml`.
- **Python 3.12 required:** The test suite and full dependency stack (torch, torchvision, ultralytics) require Python 3.12. Python 3.14 is not supported.
- **No network in tests:** All tests are designed to run without network access. Model weights and API calls are mocked.
- **Windows CI only:** GitHub Actions currently runs on `windows-latest`.

---

# Model Artifacts

Real inference requires three trained model artifacts that are **not** committed to this repository.

| Artifact | Role |
|----------|------|
| `weights/yolo_solar.pt` | YOLOv8 solar panel detection |
| `weights/mobilenet_solar.pth` | MobileNetV2 fault classification |
| `weights/xgboost_solar.joblib` | XGBoost efficiency prediction |

**CI and unit tests do not require these files.** Tests use controlled model boundaries and mocks.

**Real inference requires compatible trained artifacts** fine-tuned on the solar panel fault dataset with the class labels defined in `configs/settings.yaml`. Generic pretrained models are not compatible.

To validate whether artifacts are present, run:

```bash
py -3.12 scripts/check_model_artifacts.py
```

See `weights/README.md` for the complete artifact contract, expected formats, and acquisition guidance.

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
