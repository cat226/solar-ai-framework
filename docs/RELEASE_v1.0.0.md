# Solar AI v1.0.0 — Release & Reproducibility Manifest

Generated: 2026-09-04
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)

> **Solar AI v1.0.0 supports solar-panel detection and three fault classes:
> Clean, Dusty, and Hotspot.** This is not six-class production software —
> Bird-Drop, Electrical-Damage, and Physical-Damage remain a documented
> future roadmap, not current capability. See `README.md`'s
> [Known Limitations](../README.md#known-limitations) and
> [Current Project Status](../README.md#current-project-status).

This document is the frozen, non-fabricated reproducibility record for the
v1.0.0 release. Every value below was independently computed or read from
this repository's own real files — nothing here is estimated or invented.

---

## Application

| Field | Value |
|---|---|
| Release version | `v1.0.0` |
| Base commit (this manifest was authored on top of) | `9e588181e29dc078886dc508d820743476a6e219` |
| Release commit / tag | `v1.0.0` — resolve the exact commit via `git rev-parse v1.0.0` (recorded here rather than hardcoded, since this file is itself part of the commit being tagged) |
| Python version (validated) | 3.12.10 |
| OS / container base image | `python:3.12-slim-bookworm` (see `Dockerfile`) |

### Major dependency versions (resolved in the environment this release was validated in)

`requirements.txt` pins lower bounds (`>=`) only — see `requirements-constraints.txt` and
`docs/DEPENDENCY_REPRODUCIBILITY.md` for the exact-pin policy. These are the versions
actually installed and exercised by the full test suite / Docker smoke test for this release:

| Package | Version |
|---|---|
| streamlit | 1.61.1 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| ultralytics | 8.4.118 |
| xgboost | 3.4.0 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |
| requests | 2.34.2 |

(GitHub Actions CI additionally captures the complete resolved set via
`pip freeze --all` as the `resolved-dependencies-python312-windows` build artifact
on every run.)

---

## ML artifacts

Neither binary is committed to git (see `.gitignore`: `weights/*.pt`, `weights/*.pth`).
The reviewed hashes below are committed as `weights/manifest.json`, verifiable with:

```bash
python scripts/verify_model_artifacts.py --manifest weights/manifest.json
```

### YOLO — solar-panel detector

| Field | Value |
|---|---|
| Path | `weights/yolo_solar.pt` |
| SHA-256 | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` |
| Size | 6,243,690 bytes |
| Model type | YOLOv8n (ultralytics) |
| Class configuration | single class (solar panel) |
| Inference configuration | `confidence_threshold=0.45`, `iou_threshold=0.50`, `image_size=640` (`configs/settings.yaml`, `models.yolo`) |
| Training dataset | `gabrielkasmi/bdappv` (IGN config), full audited prepared set — 17,107 images (11,347 train / 3,179 val / 2,581 test) |
| Training experiment | `solar-yolo-full-v1` — `training/experiments/registry.jsonl` |
| Real test-split metrics | mAP50 = 0.739, mAP50-95 = 0.476, precision = 0.706, recall = 0.807 (2,581 test images, 1,640 instances) |

### MobileNet — fault classifier (v1 release artifact)

| Field | Value |
|---|---|
| Path | `weights/mobilenet_solar_v1.pth` |
| SHA-256 | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` |
| Size | 9,148,107 bytes |
| Model type | MobileNetV2 (torchvision), 3-class classification head |
| Class configuration | `["Clean", "Dusty", "Hotspot"]` (`configs/settings.yaml`, `models.mobilenet.v1_labels`) — **this is v1's real, frozen, complete scope**, not a subset awaiting completion |
| Inference configuration | `input_size=224` (`configs/settings.yaml`, `models.mobilenet`) |
| Training dataset | SolNET (Clean/Dusty, MDPI *Energies* 2023, CC BY 4.0) + PVMD (Hotspot, Mendeley, CC BY 4.0) |
| Training experiment | `solar-mobilenet-3class-full-v2` — `training/experiments/registry.jsonl` |
| Real test-split metrics | accuracy = 1.0, macro F1 = 1.0 (156 held-out test images: 72 Clean / 64 Dusty / 20 Hotspot) — see the registry entry's own honest caveat: no grouping metadata exists for the source dataset, so near-duplicate leakage across splits cannot be fully ruled out beyond the exact-SHA-256 dedup already verified (zero cross-split collisions) |
| Future six-class artifact | `weights/mobilenet_solar.pth` — **not yet trained**; three of six classes (Bird-Drop, Electrical-Damage, Physical-Damage) have no genuinely licensed, accessible dataset — see `training/classification/DATASET_SOURCES.md` |

### XGBoost — efficiency-loss predictor

| Field | Value |
|---|---|
| Path | `weights/xgboost_solar.joblib` |
| Status | **`unavailable`** — no artifact exists, and none is planned until a genuine dataset is found |
| Reason | Investigated 2026-09-04: no dataset was found that legitimately pairs this project's own `fault_class_id` taxonomy with real environmental telemetry *and* a genuinely measured `efficiency_loss_pct` target — see `training/prediction/DATASET_SOURCES.md` for the full per-candidate rejection analysis |
| Runtime behavior | `services/pipeline.py` runs detection and classification and returns real results regardless; every efficiency/output field is reported as genuinely unavailable (`prediction_successful=False`), never a fabricated `0.0` |

---

## Readiness (v1-correct expectation)

Running `python scripts/check_runtime_readiness.py` against a real v1 deployment
(YOLO + MobileNet v1 artifacts present, XGBoost absent — the expected v1 state)
reports:

```json
{"inference_readiness": "not_ready", "liveness": "ok", "missing_artifacts": ["XGBoost"]}
```

This is **correct and expected**, not a defect: `inference_readiness` covers all
three original model slots, and XGBoost is a genuine, permanent v1 boundary — not
a bug to fix before release. See `scripts/check_runtime_readiness.py`'s own
docstring for this rationale.

---

## Reproducing this manifest

```bash
python -c "
import hashlib
for p in ['weights/yolo_solar.pt', 'weights/mobilenet_solar_v1.pth']:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    print(p, h.hexdigest())
"
python scripts/verify_model_artifacts.py --manifest weights/manifest.json
python scripts/check_runtime_readiness.py
python verify_imports.py
python -m pytest -q
```
