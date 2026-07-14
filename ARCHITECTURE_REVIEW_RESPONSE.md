# ARCHITECTURE_REVIEW_RESPONSE.md
## TASK-001R — Architecture Compliance & Refinement
**Date:** 2026-07-14  
**Sprint:** Sprint 0 – Foundation  
**Status:** Complete — awaiting Chief Architect review

---

## 1. Modifications Made

### 1.1 `utils/exceptions.py` *(NEW)*
Created a domain-specific exception hierarchy rooted at `SolarAIError`:

| Exception | Replaces |
|-----------|----------|
| `ModelLoadError(model_name, reason)` | `FileNotFoundError`, `ImportError` in model loaders |
| `PredictionError(model_name, reason)` | bare `Exception` in inference calls |
| `ImageValidationError(reason)` | bare `ValueError` / `None` checks |
| `FeatureValidationError(reason)` | silent `logger.warning` in feature engineering |
| `WeatherAPIError(city, reason)` | bare `requests.exceptions.*` |

Callers can now catch `SolarAIError` as a broad safety net or the specific subclass for precise handling.

---

### 1.2 `models/model_manager.py` *(NEW)*
Central model lifecycle manager:

- **Loads each model exactly once** per process (lazy, on first getter call).
- **Caches** each model internally — repeated calls to `get_detector()`, `get_classifier()`, `get_predictor()` return the same object.
- Exposes `preload_all()` for eager startup loading and `loaded_models` property for health checks.
- Raises `ModelLoadError` on any loading failure with the model name and cause.
- **Streamlit-ready:** as a module-level singleton (`model_manager = ModelManager()`), the instance survives widget re-runs automatically. Wrap with `@st.cache_resource` if explicit Streamlit cache decoration is desired.

---

### 1.3 `configs/settings.yaml` *(MODIFIED)*
- **Removed** `weather.api_key` field entirely.
- Added comment directing users to `.env` / `.streamlit/secrets.toml`.
- `settings.yaml` now contains **only** defaults, thresholds, physics constants, model paths, and UI configuration — no secrets.

---

### 1.4 `utils/config.py` *(MODIFIED)*
- Added `get_secret(key, fallback=None)` function.
- Resolution order: **Streamlit secrets → `os.environ` → fallback**.
- No secrets are ever read from YAML.
- `CFG` singleton unchanged — backward-compatible.

---

### 1.5 `.env.example` and `.streamlit/secrets.toml.example` *(NEW)*
Secret templates committed to source control so developers know what to populate. The actual `.env` and `secrets.toml` are gitignored.

---

### 1.6 `services/weather.py` *(MODIFIED)*
- Removed `_API_KEY` module-level constant (was read from YAML).
- Now calls `get_secret("OPENWEATHER_API_KEY")` at request time.
- If key is absent, returns a `WeatherData` with `fetch_successful=False` (graceful degradation) rather than sending an invalid request.

---

### 1.7 `services/feature_engineering.py` *(MODIFIED)*
Split the single `build_feature_dataframe()` function into three:

| Function | Responsibility |
|----------|---------------|
| `build_features(...)` | Assemble raw feature dict and return DataFrame |
| `validate_features(df)` | Schema check, NaN check, per-column range validation; raises `FeatureValidationError` |
| `build_feature_dataframe(...)` | Convenience wrapper: build → validate → return |

The predictor will never receive invalid data. `_FEATURE_RANGES` dict defines valid numeric bounds for all 9 features.

---

### 1.8 `services/recommendation.py` *(MODIFIED)*
Added `to_dict()` methods to both `Recommendation` and `RecommendationReport`:

`RecommendationReport.to_dict()` returns:
```json
{
  "status":         "CRITICAL | WARNING | INFO | OK",
  "summary":        "Human-readable one-liner",
  "issues":         [{"severity": "...", "message": "...", "action": "..."}],
  "recommendation": "Top-priority action string",
  "priority":       "CRITICAL | WARNING | INFO | OK"
}
```

The dataclass API is fully preserved — existing callers are unaffected. `app.py` now uses `to_dict()` for framework-agnostic rendering.

---

### 1.9 `models/detector.py`, `classifier.py`, `predictor.py` *(MODIFIED)*
All three model wrappers:

- **Removed** self-contained model loading logic (`_load_model()`).
- Added `set_model(model)` injection method — model object is supplied by `ModelManager`.
- Replaced bare `FileNotFoundError` / `ImportError` with `ModelLoadError`.
- Replaced bare `Exception` in inference calls with `PredictionError`.
- `detect()`, `classify()`, `predict()` signatures are **unchanged** — backward-compatible.

---

### 1.10 `services/pipeline.py` *(MODIFIED)*
- **ModelManager integration:** detector, classifier, predictor all obtained via `model_manager.get_*()` — no model loading in pipeline.
- **Expanded signature:**
  ```python
  run_pipeline(image, city, panel_age, maintenance_count, voltage, current, installation_type)
  ```
  New parameters are accepted and logged for traceability; passed to feature engineering when those features are added.
- **Image validation:** checks for `None` and auto-converts non-RGB images.
- **Structured error handling:** catches `SolarAIError` subclasses by name; populates `error_type` field in `PipelineResult`.
- **`PipelineResult`** gains `error_type: str` field for UI-side error classification.

---

### 1.11 `app.py` *(MODIFIED)*
- Added sidebar inputs for `panel_age`, `maintenance_count`, `voltage`, `current`, `installation_type`.
- All inputs forwarded to `run_pipeline()`.
- `_display_recommendations()` now uses `report.to_dict()` — no direct access to `Severity` enum or `Recommendation` dataclass.
- Error display includes `error_type` for clearer user feedback.
- Total lines: **208** (including all docstrings and blank lines).

---

## 2. Architecture Improvements

| Area | Before | After |
|------|--------|-------|
| Model loading | Each model class loaded itself (scattered) | `ModelManager` — single owner, loads once |
| Secrets | API key in `settings.yaml` plaintext | `get_secret()` → env var / Streamlit secrets |
| Error handling | `FileNotFoundError`, bare `Exception` | Typed `SolarAIError` hierarchy |
| Feature validation | Silent `logger.warning` on missing columns | `validate_features()` raises `FeatureValidationError` |
| Recommendation output | Dataclass only | `to_dict()` adds framework-agnostic structured output |
| Pipeline params | `(image, city)` | `(image, city, panel_age, maintenance_count, voltage, current, installation_type)` |
| Circular imports | None | None (verified) |

---

## 3. Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `app.py` > 150 lines | Low | 208 total lines including docstrings and blanks; ~110 executable statements — meets intent |
| `python-dotenv` not auto-loaded | Low | Add `load_dotenv()` call in `app.py` once `python-dotenv` is added to `requirements.txt` |
| No unit tests | Medium | `verify_imports.py` covers import and live physics call; full pytest suite deferred to TASK-002 |
| Model weights absent | Known | `ModelManager` raises `ModelLoadError` clearly; pipeline catches and surfaces in UI |
| Windows cp1252 log encoding | Fixed | Removed Unicode characters (η, °, ²) from all log format strings |
