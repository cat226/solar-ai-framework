# Solar AI v1.0.0 — Release Notes

Generated: 2026-09-04

> **Solar AI v1.0.0 supports solar-panel detection and three fault classes:
> Clean, Dusty, and Hotspot.**

See `docs/RELEASE_v1.0.0.md` for the full reproducibility manifest (artifact
hashes, dependency versions, training provenance) and `README.md` for
complete documentation.

## Added

- Solar-panel detection (YOLOv8n, real trained artifact — mAP50 ≈ 0.74 on a
  2,581-image held-out test split)
- 3-class fault classification (MobileNetV2 — Clean, Dusty, Hotspot)
- Full analysis pipeline: detection → per-panel crop → classification →
  weather lookup → physics-based feature engineering → maintenance
  recommendations
- 10-page Streamlit dashboard: Overview, Panel Results, Site Health,
  Environment, Model Status, Limitations, History, Analytics, Alerts,
  Settings
- Inspection history (local SQLite), analytics, and alerting derived from
  real recorded inspections only
- Model status / readiness reporting, including opt-in deep verification
  that actually attempts to load each model
- Docker support: non-root container, health endpoint, fails-closed
  readiness check when model artifacts are absent
- Reviewed artifact-integrity manifest and verifier
  (`weights/manifest.json`, `scripts/verify_model_artifacts.py`)

## Model scope

```text
Clean
Dusty
Hotspot
```

Solar-panel detection is class-agnostic (single "panel" class).

## Known limitations

- **XGBoost efficiency-loss/output-power prediction is unavailable.** No
  legitimate training dataset was found that pairs a real fault
  classification with paired environmental telemetry and a genuinely
  measured efficiency-loss value — see
  `training/prediction/DATASET_SOURCES.md`. Every part of the UI that would
  show a prediction instead shows an explicit unavailable state; aggregate
  KPIs report `N/A`, never a fabricated `0%`.
- **Bird-Drop, Electrical-Damage, and Physical-Damage are not classifiable
  in v1.** Three of the six original taxonomy classes have no genuinely
  licensed, accessible dataset yet — see
  `training/classification/DATASET_SOURCES.md`. This is v1's intentional,
  frozen scope, not an incomplete rollout; acquiring the missing classes'
  data and training on the full six-class set is all that's required to
  supersede v1, with no application code changes (`ModelManager` already
  prefers a six-class artifact automatically the moment one exists).
- **SQLite history is single-deployment oriented, not multi-tenant.** See
  `services/storage.py`'s module docstring for the intended replacement
  seam if multi-user persistence is ever needed.
- **Access gate is a single shared password, not multi-user auth.** No
  per-user identity, password reset, or SSO.
- **Sites/Assets management and PDF report export are not implemented** —
  scoped out rather than built as fabricated placeholders.

## Future six-class expansion

The full original taxonomy (`Clean, Dusty, Bird-Drop, Electrical-Damage,
Physical-Damage, Hotspot`) remains the documented roadmap target. See
`README.md`'s "Adding a Future Dataset or Model" section for the exact,
no-code-change path once each missing class's dataset is legitimately
acquired.

No future capability is presented as shipped in this release.
