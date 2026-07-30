# PATCH_SUMMARY.md
## TASK-002 — Physics-Informed Weather & Environmental Feature Engineering
**Date:** 2026-07-14  
**Branch:** feature/task-002-physics  
**Base:** TASK-001R

---

## Files Modified

### `configs/settings.yaml`
- Added fallback environmental defaults for weather in case of API failure (`weather.defaults`).
- Centralized all constants (no hardcoded constants remain in Python code).

### `services/weather.py`
- Upgraded `WeatherData` to include `latitude`, `longitude`, `pressure_hpa`, and `timestamp`.
- Added structured logs (`Weather Request`, `Weather Response`).
- Expanded exception handling to use timeout configuration and safe fallbacks strictly mapped from settings.

### `services/physics.py`
- Refactored entire physics engine into 5 clean, single-responsibility functions.
- `calculate_irradiance()`: Added geographical awareness using latitude/longitude and solar zenith approximations.
- `calculate_module_temperature()`: Properly isolates NOCT calculation with wind cooling factor.
- `calculate_cloud_factor()`, `calculate_soiling_ratio()`, `calculate_wind_cooling()` isolated.
- Added `Physics Calculations` logging.

### `services/feature_engineering.py`
- Added new derived features to the raw dataframe: `temperature_difference_c`, `cloud_factor`, `wind_cooling_factor`.
- Added strict numeric consistency validation checks (e.g., verifying module temperature against irradiance/ambient combos).
- Safely slices `_FEATURE_COLUMNS` strictly to length of 9 before returning, ensuring XGBoost predictor compatibility.
- Added `Feature Generation` logging.

### `services/pipeline.py`
- Wired the physics step to the enriched `WeatherData`: `compute_physics()` now
  receives `latitude`, `longitude`, and `observation_time` (from
  `weather_data.timestamp`) so the geo/time-aware irradiance model is actually
  driven by live weather data. Backward compatible — on a failed weather fetch
  these fall back to defaults (`timestamp=None` → `datetime.now(utc)`).
- Added `Pipeline Timing` log at the end of execution.

---

## Backward Compatibility
- **API and UI**: The application interface, entry point, and UI code (`app.py`, `ui_helpers.py`) are untouched.
- **Machine Learning**: Although new scientific features are generated and validated, they are strictly stripped right before inference, meaning the XGBoost prediction pipeline is 100% backwards compatible.

---

## TASK-001R — Architecture Compliance & Refinement Patch
**Date:** 2026-07-14  

*(Previous patch summary retained below)*

## Files Created

| File | Purpose |
|------|---------|
| `utils/exceptions.py` | Domain exception hierarchy (`SolarAIError` + 5 subclasses) |
| `models/model_manager.py` | Centralised AI model lifecycle manager |
| `utils/ui_helpers.py` | Streamlit UI helper functions extracted from `app.py` |
| `.env.example` | Secret configuration template |
| `.streamlit/secrets.toml.example` | Streamlit secrets template |
| `TREE.txt` | Updated project folder tree |
| `ARCHITECTURE_REVIEW_RESPONSE.md` | Architecture review document |
| `PATCH_SUMMARY.md` | This file |
| `TEST_REPORT.md` | Import and runtime verification report |
| `TASK-001R_REPORT.md` | Post-review refinement report |
| `TASK-002_IMPLEMENTATION.md` | Feature enhancement report |
| `verify_imports.py` | Automated import verification script |
