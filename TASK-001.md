# TASK-001
## Title
Application Architecture Refactoring

---

## Sprint

Sprint 0 – Foundation

---

## Objective

Refactor the existing Streamlit application into a modular architecture without changing any functionality.

The UI, predictions, models, and outputs must remain exactly the same.

Only the project structure and code organization should change.

---

## Background

Currently app.py contains:

- UI
- YOLO inference
- MobileNet inference
- Weather API
- Feature Engineering
- Physics calculations
- XGBoost prediction
- Recommendation logic

This violates separation of concerns.

After this task, app.py should only orchestrate the workflow.

---

## Scope

Only perform architectural refactoring.

DO NOT

- Change model weights
- Change prediction logic
- Replace any AI model
- Add new research modules
- Improve algorithms

---

## Target Folder Structure

project/

├── app.py

├── models/

│   ├── detector.py

│   ├── classifier.py

│   └── predictor.py

├── services/

│   ├── weather.py

│   ├── physics.py

│   ├── feature_engineering.py

│   ├── recommendation.py

│   └── pipeline.py

├── utils/

│   ├── config.py

│   ├── logger.py

│   └── image_utils.py

├── configs/

│   └── settings.yaml

---

## Refactoring Requirements

### app.py

Should only contain

- Streamlit UI
- User Inputs
- Display Results
- Call pipeline.py

No AI logic.

---

### detector.py

Move

YOLO loading

YOLO inference

Detection output

---

### classifier.py

Move

MobileNet loading

Transforms

Classification

Softmax

Confidence

---

### predictor.py

Move

Joblib loading

DataFrame prediction

Regression logic

---

### weather.py

Move

OpenWeather API

Request handling

Response parsing

---

### physics.py

Move

Irradiance calculation

Module temperature

Soiling ratio

---

### feature_engineering.py

Construct the ML dataframe.

No prediction logic.

---

### recommendation.py

Move

System Analysis

Issue detection

Maintenance messages

---

### pipeline.py

This becomes the orchestrator.

Workflow

Image

↓

Detection

↓

Classification

↓

Weather

↓

Physics

↓

Feature Engineering

↓

Prediction

↓

Recommendation

↓

Return Results

---

## Configuration

Move

API Key

Magic Numbers

Default Values

into

configs/settings.yaml

---

## Utilities

Create

config.py

logger.py

image_utils.py

---

## Acceptance Criteria

✓ Application behaves exactly the same

✓ Same predictions

✓ Same UI

✓ Same models

✓ app.py < 150 lines

✓ Modular code

✓ No duplicated logic

✓ No hardcoded constants inside business logic

✓ Pipeline orchestrates all processing

---

## Deliverables

1. Refactored project

2. Updated imports

3. settings.yaml

4. logger.py

5. pipeline.py

6. Documentation

7. Testing report

8. Patch summary

---

## Definition of Done

The user should not notice any visual difference.

Only the internal architecture should improve.