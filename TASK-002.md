# TASK-002.md

# TASK-002
## Title
Physics-Informed Weather & Environmental Feature Engineering

---

## Sprint

Sprint 1 – Scientific Feature Enhancement

---

## Objective

Replace the existing heuristic environmental calculations with a modular, physics-informed feature engineering pipeline while preserving the existing application workflow and architecture.

This task strengthens the scientific validity of the efficiency prediction model without modifying the AI models or user interface.

---

## Background

The current implementation estimates solar irradiance using simple cloud-cover thresholds.

Example:

```python
if cloud < 20:
    irradiance = 1000
elif cloud < 50:
    irradiance = 800
```

Although suitable for a prototype, this approach is insufficient for a journal publication because:

- No geographical awareness
- No temporal awareness
- No solar position
- Simplistic irradiance estimation
- Limited environmental representation

The goal of this sprint is to improve environmental feature generation while maintaining complete backward compatibility.

---

# Scope

Only modify:

services/

- weather.py
- physics.py
- feature_engineering.py

configs/

- settings.yaml

utils/

- config.py (if required)

Do NOT modify:

- app.py
- detector.py
- classifier.py
- predictor.py
- ModelManager
- Pipeline interface
- Streamlit UI

---

# Research Objective

Improve environmental feature quality by introducing structured physics-based calculations.

The prediction pipeline must remain unchanged.

---

# Phase 1 — Weather Service Enhancement

Enhance the weather service to return:

- Temperature
- Humidity
- Cloud Cover
- Wind Speed
- Pressure
- Latitude
- Longitude
- Timestamp

Requirements:

- Structured response object
- Proper validation
- Graceful error handling
- Request timeout
- Logging

---

# Phase 2 — Physics Engine

Move all environmental calculations into physics.py.

Implement functions for:

calculate_irradiance()

calculate_module_temperature()

calculate_soiling_ratio()

calculate_cloud_factor()

calculate_wind_cooling()

Each function must perform one responsibility only.

The module must NOT communicate with external APIs.

---

# Phase 3 — Feature Engineering

Refactor feature_engineering.py.

Separate into:

build_features()

validate_features()

New derived features should include:

- Ambient Temperature
- Module Temperature
- Estimated Irradiance
- Cloud Factor
- Wind Cooling Factor
- Temperature Difference
- Soiling Ratio

Validation should verify:

- Missing values
- Invalid ranges
- Numerical consistency
- Required columns

---

# Phase 4 — Configuration

Move all environmental constants into:

configs/settings.yaml

Example:

physics:

  nominal_irradiance: 1000

  cloud_loss_factor: 0.75

  temperature_coefficient: 0.0045

  wind_cooling_factor: 0.02

weather:

  timeout: 10

  units: metric

No hardcoded constants inside Python files.

---

# Error Handling

Weather service must gracefully handle:

- Invalid city
- Missing API response
- Timeout
- Network failure
- Invalid JSON

Physics engine must validate inputs before calculation.

Feature engineering must reject invalid data.

---

# Logging

Log:

- Weather request
- Weather response
- Physics calculations
- Feature generation
- Pipeline execution time

---

# Acceptance Criteria

✓ Existing UI unchanged

✓ Existing prediction workflow unchanged

✓ Architecture unchanged

✓ Physics isolated inside physics.py

✓ Feature generation isolated inside feature_engineering.py

✓ Weather service fully modular

✓ No hardcoded environmental constants

✓ Structured logging

✓ Backward compatibility maintained

---

# Deliverables

Generate:

TASK-002_IMPLEMENTATION.md

PATCH_SUMMARY.md

TEST_REPORT.md

Include:

- Files modified
- Scientific improvements
- Test results
- Performance impact
- Remaining limitations

---

# Definition of Done

The application behaves exactly as before while generating scientifically improved environmental features through the new physics module.

No architectural changes are introduced.