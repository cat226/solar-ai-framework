# PATCH_SUMMARY.md
## TASK-001R — Architecture Compliance & Refinement Patch
**Date:** 2026-07-14  
**Branch:** feature/edith  
**Base:** TASK-001 implementation

---

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
| `verify_imports.py` | Automated import verification script |

---

## Files Modified

### `configs/settings.yaml`
```diff
- weather:
-   api_key: "YOUR_OPENWEATHERMAP_API_KEY"
+ # NOTE: The API key is NOT stored here.
+ # Set OPENWEATHER_API_KEY in .env or .streamlit/secrets.toml.
  weather:
    base_url: ...
```

### `utils/config.py`
- Added `get_secret(key, fallback)` — resolves secrets from Streamlit → env → fallback.

### `.gitignore`
```diff
+ # Secrets — NEVER commit
+ .env
+ .streamlit/secrets.toml
```

### `services/weather.py`
- Removed `_API_KEY` module constant.
- Added `get_secret("OPENWEATHER_API_KEY")` call inside `fetch_weather()`.
- Imports `get_secret` from `utils.config` and `WeatherAPIError` from `utils.exceptions`.

### `services/feature_engineering.py`
- Split into `build_features()` + `validate_features()` + `build_feature_dataframe()`.
- Added `_FEATURE_RANGES` dict with valid bounds for all 9 features.
- `validate_features()` raises `FeatureValidationError` on schema / NaN / range failures.

### `services/recommendation.py`
- Added `Recommendation.to_dict()` → `{severity, message, action}`.
- Added `RecommendationReport.to_dict()` → `{status, summary, issues, recommendation, priority}`.

### `services/physics.py`
- Replaced Unicode characters in log messages (η, °, ²) with ASCII equivalents to fix Windows cp1252 encoding errors.

### `models/detector.py`, `models/classifier.py`, `models/predictor.py`
- Removed self-loading mechanisms.
- Added `set_model(model)` injection methods.
- Refactored exception handling to use custom typed exceptions.

### `services/pipeline.py`
- Uses `model_manager.get_*()` for all model access.
- Expanded `run_pipeline()` signature: `panel_age`, `maintenance_count`, `voltage`, `current`, `installation_type`.
- Standardized `PipelineResult` class fields: `detection_result`, `classification_result`, `weather_data`, `physics_data`, `feature_dataframe`, `efficiency_prediction`, `recommendations`, `processing_time`, `status`, etc.
- Structured exception handling: catches `SolarAIError` subclasses by type.

### `app.py`
- Extracted display logic out into `utils/ui_helpers.py`.
- Final line count reduced to 130 lines.
- Sidebar: added `panel_age`, `maintenance_count`, `voltage`, `current`, `installation_type` inputs.
- `run_pipeline()` called with all new parameters.

---

## Backward Compatibility

- `PipelineResult` fields were renamed/restructured specifically for unified object design as requested by architecture review, but data fidelity is unchanged.
- `RecommendationReport` and `Recommendation` dataclass APIs structurally unchanged.
- `CFG` singleton unchanged.
- `get_logger()` unchanged.
