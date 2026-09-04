# XGBoost Efficiency-Loss Predictor — Dataset Investigation

Generated: 2026-09-04
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/cloud-training-orchestration

## Required schema

`models/predictor.py` / `services/feature_engineering.py` require a training
dataset providing all nine features, in this exact order, plus a genuine
target:

1. `irradiance_wm2`
2. `module_temp_c`
3. `ambient_temp_c`
4. `humidity_pct`
5. `wind_speed_ms`
6. `cloud_cover_pct`
7. `soiling_ratio`
8. `fault_class_id` — the production/interim fault-class index, i.e. a real
   sample must carry a real label from *this repository's own class order*
   (`Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Hotspot`)
9. `detection_confidence` — a real YOLO detection confidence, not a
   placeholder

**Target**: `efficiency_loss_pct` — a genuinely measured (or otherwise
defensibly-sourced, e.g. derived from a real measured power ratio against a
clean reference panel) efficiency loss, not a value this repository's own
`services/physics.py` computed from a config-defined heuristic table
(`physics.soiling_ratios`). Using that heuristic's own output as a training
label would be circular — the model would only ever learn to re-approximate
a hand-written lookup table, not anything about real panel behaviour, and
presenting the result as a "trained regressor" would overstate what it
actually is.

## Investigation (2026-09-04)

Searched specifically for a dataset that pairs (a) a discrete fault label in
this project's own six-class taxonomy, (b) real environmental readings
across all of irradiance/module temp/ambient temp/humidity/wind/cloud cover,
and (c) a genuinely measured efficiency- or power-loss percentage. None of
the following combine all three:

- **DeepSolarEye / "SPSI"** (`deep-solar-eye.github.io`, previously
  investigated for Bird-Drop — see `training/classification/DATASET_SOURCES.md`
  and [[bird_drop_dataset_correction]]): does have real measured
  power-loss values and real irradiance per image, but **no discrete
  fault-type label at all** (continuous soiling only, confirmed by its own
  README) and no module/ambient temp, humidity, wind, or cloud-cover
  fields. The exact same blocker that ruled it out for MobileNet training
  (no discrete class) also rules it out here for `fault_class_id`.
- **IEEE DataPort "Photovoltaic Module Fault Data"** (DOI
  10.21227/5zs3-a832): a genuinely different fault taxonomy (open-circuit /
  overheating / short-circuit / aging / shading-soiling — electrical fault
  modes, not this project's visual classes), **no efficiency- or
  power-loss percentage column at all**, missing module temp / humidity /
  wind / cloud cover (700 rows, irradiance + ambient temp + electrical
  I-V parameters only), sensor data with no panel images, and gated behind
  a required IEEE DataPort account (an external-account requirement, not
  something obtainable from this environment). Rejected on schema grounds
  alone even before the access question.
- **SolarFCD** (arXiv 2604.23662) and the "Unified Deep Learning Platform
  for Dust and Fault Diagnosis" dataset (arXiv 2511.18514): image
  classification datasets (healthy / surface-obstruction / structural /
  electrical, or dust/fault via thermal+visual imaging) — no efficiency-
  loss regression target, no paired environmental sensor readings.
- **"Dataset of photovoltaic panel performance under different fault
  conditions" (ScienceDirect S2352340925001246)**: potentially closer in
  spirit (fault conditions + performance), but the source page returned
  HTTP 403 (paywalled/access-restricted) — could not independently verify
  its actual schema, fault taxonomy, or license, so it is **not accepted**
  absent that verification (this project's standing rule: never record a
  source as usable without actually inspecting it — see
  [[bird_drop_dataset_correction]]'s account of exactly this mistake for a
  different dataset).
- General "PV performance vs. weather" datasets (e.g. NREL PVDAQ-style
  power-vs-irradiance/temperature logs) exist but carry no fault-type
  dimension whatsoever — they measure generic system performance, not
  performance attributable to a specific visually-classified defect.

## Conclusion

**No dataset was found that legitimately supplies both a real
`fault_class_id` in this project's own taxonomy and a real measured
`efficiency_loss_pct`, under a license this project could use.** This is
not a search-effort gap — the required combination (visual fault
classification + simultaneous full weather telemetry + measured efficiency
loss, all for the same panel/moment) is a genuinely narrow,
installation-specific measurement that does not appear to exist as a
public dataset matching this schema.

**Decision: `weights/xgboost_solar.joblib` is not trained.** Per this
project's no-fabrication policy, a missing artifact — honestly represented
as unavailable everywhere in the application — is preferable to a model
trained on invented labels or on a circular restatement of the existing
physics heuristic. See `services/pipeline.py`'s `xgboost_available` flag
and the **Model Status** / **Limitations** dashboard pages for how this is
disclosed to the user.

## Re-opening this investigation

If a genuine, licensed, image-or-installation-linked dataset combining
fault classification with paired environmental + efficiency-loss
measurements is later identified, it must be independently downloaded and
inspected (not accepted from a paper abstract or listing page alone)
before being used — same standard already applied throughout
`training/classification/DATASET_SOURCES.md` and
`training/detection/PROVENANCE_VERIFICATION.md`.
