# Solar AI Framework — Architecture

## Overview

The Solar AI Framework is a **Streamlit** application that analyses uploaded
solar panel images to:

1. Detect panels using a YOLO model
2. Classify fault type using MobileNetV2
3. Fetch live weather data (OpenWeatherMap)
4. Compute solar physics (irradiance, module temperature, soiling)
5. Predict energy efficiency loss using XGBoost regression
6. Generate prioritised maintenance recommendations

---

## Folder Structure

```
solar-ai-framework/
├── app.py                          # Streamlit UI only (<150 lines)
├── ARCHITECTURE.md                 # This document
├── AGENTS.md                       # Git workflow rules
│
├── models/                         # AI model wrappers
│   ├── __init__.py
│   ├── detector.py                 # YOLO panel detection
│   ├── classifier.py               # MobileNetV2 fault classification
│   └── predictor.py                # XGBoost energy output regression
│
├── services/                       # Business logic services
│   ├── __init__.py
│   ├── weather.py                  # OpenWeatherMap API client
│   ├── physics.py                  # Irradiance, module temp, soiling
│   ├── feature_engineering.py      # ML feature DataFrame construction
│   ├── recommendation.py           # Maintenance recommendation engine
│   └── pipeline.py                 # Application orchestrator (called by app.py)
│
├── utils/                          # Foundational utilities
│   ├── __init__.py
│   ├── config.py                   # settings.yaml loader (singleton CFG)
│   ├── logger.py                   # Centralized logging factory
│   └── image_utils.py              # PIL image preprocessing helpers
│
└── configs/
    └── settings.yaml               # All constants, thresholds, paths, API keys
```

---

## Dependency Graph

```
app.py
  └── services/pipeline.py          (orchestrator)
        ├── models/detector.py
        │     └── utils/image_utils.py
        ├── models/classifier.py
        │     └── utils/image_utils.py
        ├── models/predictor.py
        ├── services/weather.py
        ├── services/physics.py
        ├── services/feature_engineering.py
        │     ├── models/detector.py        (type import only)
        │     ├── models/classifier.py      (type import only)
        │     ├── services/weather.py       (type import only)
        │     └── services/physics.py       (type import only)
        └── services/recommendation.py
              ├── models/classifier.py      (type import only)
              ├── models/predictor.py       (type import only)
              └── services/physics.py       (type import only)

All modules:
  └── utils/config.py               (CFG singleton)
  └── utils/logger.py               (get_logger factory)
```

> **No circular imports** — `utils` has no application imports; `models` do not
> import `services`; `services` import `models` only for type annotations.

---

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.py` | Streamlit UI, user inputs, result display |
| `services/pipeline.py` | Orchestrates all steps; sole entry-point for `app.py` |
| `models/detector.py` | YOLO inference → `DetectionResult` |
| `models/classifier.py` | MobileNetV2 inference → `ClassificationResult` |
| `models/predictor.py` | XGBoost regression → `PredictionResult` |
| `services/weather.py` | OpenWeatherMap API → `WeatherData` |
| `services/physics.py` | Physics calculations → `PhysicsResult` |
| `services/feature_engineering.py` | Assemble ML feature DataFrame |
| `services/recommendation.py` | Rule-based recommendations → `RecommendationReport` |
| `utils/config.py` | Load `configs/settings.yaml` once → `CFG` dict |
| `utils/logger.py` | `get_logger(name)` factory |
| `utils/image_utils.py` | YOLO letterbox, MobileNet centre-crop, PIL↔NumPy |

---

## Configuration

All tunable values live in **`configs/settings.yaml`**:

- Weather API key and endpoint
- Model weight file paths
- Classification labels
- Physics constants (NOCT, temperature coefficient, soiling ratios)
- Feature column order
- Recommendation severity thresholds
- Logging level and format

No hardcoded constants appear in business logic modules.

---

## Data Flow

```
User uploads image + selects city
        │
        ▼
  app.py calls run_pipeline(image, city)
        │
        ├─► detector.detect(image)           → DetectionResult
        ├─► classifier.classify(image)       → ClassificationResult
        ├─► fetch_weather(city)              → WeatherData
        ├─► compute_physics(...)             → PhysicsResult
        ├─► build_feature_dataframe(...)     → pd.DataFrame
        ├─► predictor.predict(df)            → PredictionResult
        └─► generate_recommendations(...)   → RecommendationReport
                │
                ▼
          PipelineResult (all results bundled)
                │
                ▼
         app.py renders UI metrics, charts, and recommendations
```

---

## Extension Points

| What to change | Where |
|---------------|-------|
| Add a new fault class | `configs/settings.yaml` → `classification.labels` + `physics.soiling_ratios` |
| Swap YOLO version | `configs/settings.yaml` → `models.yolo.weights` |
| Add a new feature | `services/feature_engineering.py` + `configs/settings.yaml` → `feature_engineering.feature_columns` |
| Add a new recommendation rule | `services/recommendation.py` → add a `_rule_*` function |
| Change physics model | `services/physics.py` only |
| Tune thresholds | `configs/settings.yaml` → `recommendations.*` |

---

## Architecture Rules (frozen)

1. `app.py` must only contain UI logic.
2. `services/pipeline.py` is the **only** file `app.py` imports from the framework.
3. `models/` modules must not import from `services/`.
4. `utils/` modules must not import from `models/` or `services/`.
5. All constants must live in `configs/settings.yaml`.
6. No new top-level directories may be created without architecture review.
