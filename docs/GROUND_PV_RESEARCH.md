# Ground-level PV defect research detector — reproducibility record

**Status: pipeline built; dataset not yet acquired; training not yet run.**
Sections marked _PENDING_ are filled after the blocked steps complete. No
metric in this document is fabricated — a number appears only once it has
actually been produced.

This is a **separate research model** from the production aerial detector
`weights/yolo_solar.pt`, which is not modified, retrained, or replaced by
any of this work. Research / non-commercial use only.

---

## 1. Research objective

Build a research-grade **object detector** for ground-level / close-up RGB
photographs of solar panels that detects:

- individual panel boundaries (`panel`)
- bird-dropping regions (`bird_drop`)
- cracks / physical damage (`cracked`)
- dust / soiling regions (`dusty`)

The application later derives a per-panel **research visual condition
assessment** (Clean / Dust / BirdDrop / Damage — a panel may carry several)
by associating defect detections with panel detections. This is explicitly
_not_ a validated engineering or safety diagnosis.

A panel is **not** forced into one mutually-exclusive class: `panel +
bird_drop` and `panel + dusty` are valid simultaneous detections.

## 2–6. Dataset source, URL, version, license, download date

| field | value |
|---|---|
| name | *panel solar* |
| owner | Susan |
| source | Roboflow Universe |
| URL | <https://universe.roboflow.com/susan-ifblr/panel-solar-bw945> |
| workspace / project / version | `susan-ifblr` / `panel-solar-bw945` / `1` |
| export format | YOLOv8 |
| source license | **CC BY 4.0, as displayed by Roboflow** |
| dataset-level license (this repackaging) | CC BY 4.0, attribution preserved |
| image-level provenance | **not independently verified** |
| download date | _PENDING_ — blocked on `ROBOFLOW_API_KEY` (see §21) |

Source class names (per the Roboflow project page): `panel`, `bird_drop`,
`cracked`, `dusty`.

Published (unverified) figures: ~6,493 images; instance counts cracked
6233 / panel 4477 / dusty 235 / bird_drop 60; ~70/20/10 train/valid/test.
These are audited independently, not trusted.

### Citation

> Susan. "panel solar Dataset." Roboflow Universe.
> https://universe.roboflow.com/susan-ifblr/panel-solar-bw945

### Papers referencing this dataset

_To be recorded during the audit pass. Any metrics reported in third-party
papers are **not** presented here as Solar AI's own results._

## 7. Class definitions

| id | name | definition |
|----|------|------------|
| 0 | panel | boundary of one individual solar panel/module in the frame |
| 1 | bird_drop | a region of bird-dropping contamination on a panel |
| 2 | cracked | a crack or physical-damage region on a panel |
| 3 | dusty | a dust / soiling region on a panel |

Canonical id order is fixed in
`training/detection/ground_pv/class_map.py` (`RESEARCH_CLASS_ORDER`) and
`configs/ground_pv_research.yaml`. The source→research class-id remap is
computed from the real downloaded `data.yaml`, never assumed
(`build_class_id_remap`), and applied only when materializing the
normalized copy at `datasets/ground_pv_research/yolo/` — the raw download
is never edited.

**Hotspot is excluded.** The project's existing Hotspot training data is
thermal IR; RGB photographs cannot validate RGB-hotspot semantics. Hotspot
remains a separate future thermal pipeline. The detector never infers
hotspot from brightness, white spots, reflections, or colour.

## 8. Dataset statistics

_PENDING_ — produced by `scripts/audit_ground_pv_dataset.py` into
`datasets/ground_pv_research/reports/dataset_audit.{md,json}`. Includes:
file structure; per-split image/label counts; class instance counts;
images-containing-each-class; multi-class image counts; image
dimension/aspect-ratio distributions; box-area and tiny-object
distributions; images with only defects and no panel; images with a panel
but no defect.

## 9. Duplicate audit

_PENDING_ — exact duplicates (SHA-256), perceptual near-duplicates (dHash,
Hamming ≤ 5), and likely same-image augmentation families.

## 10. Leakage audit

_PENDING_ — exact and near-duplicate leakage across train/valid/test. If
strong leakage is found, a `leakage_clean_split/` is produced alongside
(never replacing) `official_split/`, preferably grouped by near-duplicate
family.

## 11. Train / val / test split

Policy: **preserve the source's official split initially**
(`configs/ground_pv_research.yaml: splits.policy = preserve_official`). Both
`official_split` and (if needed) `leakage_clean_split` are kept. Splits are
never reshuffled without a documented reason. Rare classes are **never**
balanced by duplicating or synthesizing validation/test examples.

## 12. Model architecture

Ultralytics **YOLO11n** (nano), COCO-pretrained. Falls back to **YOLOv8n**
only if the installed Ultralytics predates YOLO11. Repo Ultralytics version
at build time: `8.4.118` (local) / pinned `8.4.138` on Kaggle. The base
checkpoint is **never** `weights/yolo_solar.pt` — `scripts/train_ground_pv.py`
refuses that explicitly.

## 13. Training hyperparameters

| | baseline (`ground_yolo11n_baseline_v1`) |
|---|---|
| imgsz | 640 |
| epochs | 100 |
| patience | 20 |
| batch | -1 (auto) |
| optimizer | auto |
| seed | 42 |
| cos_lr | true |
| pretrained | true |
| device | Kaggle GPU (`device=0`) |

Second experiment `ground_yolo11n_balanced_v1`: identical except a
**train-only weighted image-sampling** strategy that draws images
containing rare classes (bird_drop, dusty) more often. Validation/test are
untouched. Baseline and balanced are compared on the **same** held-out test
set; neither result is cherry-picked.

## 14. Augmentation

From `configs/ground_pv_research.yaml: augmentation` — conservative, must
not distort defect semantics:

`hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.3,
shear=0, perspective=0, flipud=0.0 (no vertical flip), fliplr=0.5,
mosaic=0.5, mixup=0.0, copy_paste=0.0`.

## 15. Hardware / device

Training: Kaggle GPU (NVIDIA **T4** preferred; **P100** sm_60 also
supported via the `torch==2.7.1` / `torchvision==0.22.1` pin). Detected at
runtime with `torch.cuda.is_available()` / `torch.cuda.get_device_name(0)`;
the notebook fails closed if CUDA is unavailable. Local machine: no usable
GPU (data prep / eval / inference only).

## 16. Random seed

`42` everywhere (`configs/ground_pv_research.yaml`, `train_ground_pv.py`,
notebook).

## 17–18. Evaluation metrics / per-class metrics

_PENDING_ — `scripts/evaluate_ground_pv.py` writes
`datasets/ground_pv_research/reports/ground_research_evaluation.{md,json}`
with overall mAP50 / mAP50-95 / precision / recall **and** per-class
mAP50 / mAP50-95 / precision / recall for `panel`, `bird_drop`, `cracked`,
`dusty`, plus per-split image counts, instances per class, and
images-containing-each-class. BirdDrop and Dust are flagged rare.

## 19. Real-world qualitative evaluation

_PENDING_ — `scripts/infer_ground_pv.py` runs the detector on the owner's
real rooftop photos (`datasets/ground_pv_research/real_world_holdout/`,
qualitative only, never training data) and writes
`datasets/ground_pv_research/reports/real_world_holdout.md`. Each prediction
is classified qualitatively (plausible / likely false positive / missed
defect / localization problem / correct panel localization / uncertain).
No TP/FP/FN is fabricated. Images with zero detections are recorded.

## 20. Known limitations

- `bird_drop` (~60 instances) and `dusty` (~235) are severely
  underrepresented — per-class metrics for them are high-variance.
- Source image-level provenance is unverified.
- Domain gap between the source dataset and the owner's rooftop photos.
- No RGB hotspot validation — out of scope by design.
- A baseline run is not production-ready and is not wired into the app's
  inference path.

## 21. Exact checkpoint SHA-256

_PENDING_ — recorded in
`datasets/ground_pv_research/reports/ground_research_model_manifest.json`
after the selected checkpoint is copied to
`weights/yolo_solar_ground_research_v1.pt`.

## 22. Git commit

Pipeline built on branch `feat/cloud-training-orchestration`. The exact
build commit is recorded in the model manifest and training manifest once
training runs.

---

## Blockers (as of 2026-09-07)

1. **`ROBOFLOW_API_KEY` not set** — the Roboflow Universe API returns
   401/403 without it, so the dataset cannot be downloaded. Nothing
   downstream (audit, Kaggle packaging, training, evaluation) can run.
   `scripts/download_ground_pv_dataset.py` exits with instructions.
2. **Kaggle GPU training is a cost/compute action** — it is not launched
   without explicit owner approval.
3. **`E:` drive not mounted** at build time — holds the real-world holdout
   photos (`E:\Solar Real Images\`) and the project's large-data staging
   area. Reattach before the real-world inference step.

## Production-model safety

`weights/yolo_solar.pt` SHA-256 verified **unchanged** before and after
this work:
`0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`.
`weights/manifest.json` and the `v1.0.0` tag
(`4b0287dc98ad68f4fda8710fb522d6d37b08ffad`) untouched.
