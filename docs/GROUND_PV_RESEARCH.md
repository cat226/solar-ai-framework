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
| download date | 2026-09-07 (UTC), via `scripts/download_ground_pv_dataset.py` (Roboflow package) |
| raw location | `datasets/ground_pv_research/raw/panel_solar_bw945_v1/` — **on D:** (E: drive was unstable; project policy normally puts it on E:); immutable, gitignored |
| raw size | 371 MB (389,015,640 bytes), 6,493 JPEG images @ 640×640 RGB |
| source `names:` order | `bird_drop, cracked, dusty, panel` (alphabetical) → research remap `{0→1, 1→2, 2→3, 3→0}` |
| source preprocessing | auto-orient (EXIF stripped), resize 640×640 **stretch**; no source augmentation |

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

From `scripts/audit_ground_pv_dataset.py` (2026-09-07) →
`datasets/ground_pv_research/reports/dataset_audit.{md,json}` and
`acquisition_manifest.json`. **Audit result: FAILED CLOSED** (see §20 / §Blockers).

- 6,493 images / 6,493 label files (train 4,546 / valid 1,299 / test 648); 0 corrupt; 0 missing labels; 0 truly-empty labels.
- All images 640×640, aspect ratio 1.0 (source stretch-resize).
- 15,791 annotations = **15,177 YOLO bbox lines + 614 YOLO-segmentation polygon lines** (see §Annotation format).
- **True class distribution (bbox + polygon):**

  | class | instances | images containing | published by Roboflow (unverified) |
  |---|---|---|---|
  | panel | 6,319 (6,310 bbox + 9 poly) | 5,227 | 4,477 |
  | cracked | 9,040 (8,780 bbox + 260 poly) | 3,694 | 6,233 |
  | dusty | 351 (6 bbox + **345 poly**) | 156 | 235 |
  | bird_drop | 81 (81 bbox) | **9** | 60 |

  Published counts do not match. `dusty` is almost entirely polygon annotations. **`bird_drop` appears in only 9 images**, all `.rf.`-hash variants of ~3 source photos.
- Multi-class images: 2,593 (`cracked+panel` 2,428, `dusty+panel` 156, `bird_drop+panel` 9).
- Images defect-only (no panel box): 1,266; panel-only (no defect): 2,634.
- Box area: median 5.2% of frame, mean 21%, p95 99.8% (bimodal: small defect boxes + full-frame panel boxes). 731 tiny boxes (< 0.1% area).

### Annotation format (mixed — a blocker)

614 polygon lines across 410 files (244 polygon-only, 166 mixed with
boxes). These are valid YOLO-**segmentation** annotations (`class x1 y1 …
xn yn`), mostly the `IMG_85xx/86xx` soiling photos (`dusty`) and
`pv_crack-*` (`cracked`). A pure-detection run needs them converted to
bounding boxes; `training/detection/ground_pv/normalize.py` does this
(min/max of polygon vertices, clamped to [0,1]) when materializing the
normalized copy — the raw download is never modified.

## 9. Duplicate audit

- **Exact duplicates (SHA-256): 1 group** — `Screenshot-2023-04-02-002436` present in both `train/` and `test/` as two `.rf.`-hash copies.
- **Perceptual near-duplicates (dHash, Hamming ≤ 5): 41,166 pairs.**
- **Likely same-image augmentation/re-export families: 804** — same filename base, multiple `.rf.<hash>` copies, often split across train/val/test. Filenames like `1677069178370` (millisecond timestamps) and `Crack_126/127/…` sequences indicate burst / video-frame captures.

## 10. Leakage audit

- **Cross-split exact leakage: 1 image** (the Screenshot above).
- **Cross-split near-duplicate leakage: 18,204 pairs spanning 3,401 unique images (~52% of the dataset)** — train 1,982 / valid 934 / test 485. Severe: the official split does **not** hold near-duplicate frames together.
- A `leakage_clean_split/` grouped by near-duplicate family is **required** before any metric off this dataset can be trusted. Not yet built (awaiting direction). `official_split/` is preserved regardless.

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

## Status / blockers (as of 2026-09-07)

**Dataset acquired** (D:, immutable) and **audited**. The audit **FAILED
CLOSED** — the dataset is **BLOCKED for training** until remediated:

1. **Mixed annotation format** — 614 YOLO-seg polygon lines / 410 files.
   Fix: build the normalized copy (polygon→bbox conversion is implemented).
2. **Cross-split leakage** — 1 exact + ~18,204 near-duplicate pairs across
   ~52% of images. Fix: build a leakage-clean split grouped by
   near-duplicate family (not yet built; needs direction).
3. **`bird_drop` unusable** — 81 instances but only **9 images** (~3 unique
   source photos). Cannot train or evaluate this class meaningfully;
   options are drop it (3-class model) or keep it flagged unevaluable.
4. **Published counts wrong** — Roboflow page says dusty 235 / bird_drop
   60; actual images-containing are 156 / 9. `dusty` is 98% polygons.

Not blockers but noted:

5. **Kaggle GPU training is a cost/compute action** — not launched without
   explicit owner approval.
6. **`E:` drive intermittently unavailable** — holds the real-world holdout
   photos and normally the large-data staging area; the raw dataset is on
   D: as a result.

## Production-model safety

`weights/yolo_solar.pt` SHA-256 verified **unchanged** before and after
this work:
`0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`.
`weights/manifest.json` and the `v1.0.0` tag
(`4b0287dc98ad68f4fda8710fb522d6d37b08ffad`) untouched.
