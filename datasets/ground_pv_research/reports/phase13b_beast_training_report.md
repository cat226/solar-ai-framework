# Phase 13B — "beast-mode" ground PV PANEL detector training

**Phase 13B status: BLOCKED (compute) on the GPU training step.**
**Model decision: C — keep v1.**

`weights/yolo_solar_ground_research_v1.pt` remains the ground PANEL detector. No
new checkpoint was adopted.

> Research only. No production change, no deployment, no merge, no push. F: is
> hardware-unstable and was **not** touched — all datasets, checkpoints, reports
> and Kaggle packages are on **D:**. The six owner photos were **not** trained on,
> **not** pseudo-labelled, **not** used in validation or threshold tuning.

---

## 1. Data sources

| source | what | immutable |
|---|---|---|
| **Susan 3-class** `datasets/ground_pv_research/yolo/` | audited + remediated leakage-clean panel/cracked/dusty, 6,492 images (train 4,546 / valid 1,297 / test 649), **all 640×640 Roboflow stretch-resized** | yes — not modified |
| **Candidate** `solar-panel-detection-bwpkx/solar-panel-detection-ubwtz` **v2** (native-resolution yolov8 export) | Roboflow Universe → `datasets/ground_pv_research/candidate_acquisition/solar_panel_detection_ubwtz/` | yes — raw download not modified |

Roboflow auth via the project `.env` `ROBOFLOW_API_KEY` (never printed/committed).
Kaggle GPU via `edithstark`.

## 2. Provenance

The candidate stays **`personal-experimental-unverified-provenance`**. The
Roboflow re-uploader displays "CC BY 4.0" but states **no original image source**;
per the Phase 7/12 rule that does not establish image-level rights. **Research
only — NOT publication-cleared, redistribution-cleared, commercial, or
production-approved.** It does **not** resolve the Solar AI ground-PV
data-provenance problem.

## 3. Candidate audit — `candidate_acquisition/solar_panel_detection_ubwtz_audit.json`

| | |
|---|---|
| images | 563 (train 395 / valid 113 / test 55) |
| polygon annotations | 1,537 — **all valid → bbox**, 0 zero-area, 0 scene-spanning (>92 %×92 %), 0 malformed |
| empty-label images | 58 |
| native resolution | **4 distinct sizes**: 605×1056 (×371), 1681×2200 (×155), 1075×594 (×34), 2241×1650 (×3) |
| aspect | **526 portrait / 37 landscape / 0 square** (ratio 0.573–1.81) — the **first non-square ground-level PV panel data Solar AI holds** |
| boxes/image | mean 2.7, median 3, max 6 (module-level) |
| bbox area fraction | median 0.16, p90 0.42, 79 boxes > 0.5 |
| boundary-touching boxes | 79 % |
| **scene analysis** | **ALL 563 frames dated 2024-04-30, timestamps within ~4 minutes** = ONE ground-mounted installation, ONE capture session. 2 video sequences (`133150_mp4` 371 frames, `133008_mp4` 34 frames) + 96 burst-photo events. **Independent physical scenes ≈ 1.** |

Strict 16×16 dHash (Hamming ≤ 12) counts every frame distinct — but the *scene*
diversity is ~1. Frames must not be treated as independent examples.

## 4. Derived dataset — `yolo_beast_v1/beast_dataset_manifest.json`

Built from the two immutable sources (neither modified):

- **candidate selection** (deterministic): drop the 58 empty-label images; keep
  every 12th video frame + 1 image per burst capture-second; cap 130 →
  **125 frames selected** (30 video + 95 photo). 463 polygons → tight bboxes,
  0 rejected. Class `solarpanel` → class 0 `panel` (no `cracked`/`dusty` from
  the candidate).
- **125 candidate frames = 2.7 %** of the B/C training pool (4,546 + 125).
- **all candidate frames → TRAIN split only**, as one family `cand_ubwtz_scene1`
  (one physical scene; valid/test placement would measure memorisation).
- `data_A.yaml` = Susan only; `data_BC.yaml` = Susan train + candidate train,
  **Susan valid/test unchanged** → A / B / C would be compared on the identical
  held-out set.

### Domain audit — Susan vs candidate vs owner photos

| metric | Susan train | Candidate | Owner photos |
|---|---|---|---|
| dimensions | **100 % 640×640** | 605×1056, 1681×2200, 1075×594 … | 573×1280 ×5, 1280×573 ×1 |
| aspect | **100 % square** | **93 % portrait**, 7 % landscape (0.57–1.81) | 5 portrait / 1 landscape (0.45–2.23) |
| panels / image | 0.81 (mostly 1) | 2.7 | image 6 is a wide array |
| multi-panel images | **4.9 %** | **84 %** | 1 of 6 |
| panel area (median frac) | **0.60** (Roboflow zoomed each image to one panel) | **0.16** (more context) | — |
| boundary-touching boxes | 68 % | 79 % | — |

The candidate matches the owner photos on **two** axes Susan misses — non-square
aspect ratio and multi-panel-with-context composition. But it is **one
installation**, so it broadens *aspect-ratio* coverage, not *scene* coverage.

## 5. Training experiments

| Exp | Model | Data | imgsz | epochs | status |
|---|---|---|---:|---:|---|
| **A** | YOLO11n (existing v1 checkpoint) | Susan only | 640 | — | **DONE** — baseline reproduction check, re-evaluated locally on the Susan test set |
| **B** | YOLO11n (COCO init) | Susan + 125 candidate (train) | 640 | 90 | **BLOCKED (compute)** |
| **C** | YOLO11n (COCO init) | Susan + 125 candidate (train), moderate scale/aspect aug (NO `multi_scale`) | 640 | 90 | **BLOCKED (compute)** |

Frozen across B/C: architecture, COCO init, optimizer `auto`, `cos_lr`, patience
20, seed 42, imgsz 640, batch 16. C aug: scale 0.5, translate 0.15, degrees 7,
mosaic 0.4, close_mosaic 15 — **no `multi_scale`** (Phase 11 v2 showed it 6× the
cost and *worse* OOD).

**Experiment A (local re-eval of v1 on Susan test, 649 images / 1,266 instances):
P 0.876 · R 0.801 · mAP50 0.865 · mAP50-95 0.677 — identical to the recorded v1
metrics.** Reproducibility confirmed (Ultralytics 8.4.118, CPU).

## 6. GPU compute — what happened

| attempt | notebook | result |
|---|---|---|
| 1 (`ground-beast-train` v1) | A + B + C (100 epochs each) **+ a 27-pass validation threshold sweep**, one cell | **hung > 17 h** on the Kaggle GPU with no output. Over-scoped for one session. Cancelled by pushing v2. |
| 2 (`ground-beast-train` v2) | B + C only (90 epochs, batch 16, `workers=4`), threshold sweep moved local | ran **> 9 h** with no output on the Kaggle accelerator (status `RUNNING`, `lastRunTime` null) — far beyond the ~2–4 h expected for two YOLO11n runs. |

Neither run produced a retrievable checkpoint. Kaggle's committed-kernel API
exposes no cancel and no partial output while a kernel runs, so no B/C weights
could be salvaged. **The GPU-training step of Phase 13B is BLOCKED (compute).**

Root-cause hypotheses: a slow / degraded Kaggle accelerator this session, and/or
a dataloader stall on the candidate's large native-resolution frames
(2241×1650). The v1 notebook was additionally over-scoped.

## 7–17. Test metrics / threshold sweep / v1 comparison / six-photo OOD / image 4 / image 6 / background FPs / condition chain / failure analysis

**Not available — B and C did not complete.** No new checkpoint exists to
evaluate on the Susan test set, sweep thresholds on, or run against the six owner
photos. The v1 baselines carried forward (from Phases 4/10/11) are:

- **Susan test:** P 0.876 · R 0.801 · mAP50 0.865 · mAP50-95 0.677 (panel AP50
  0.963, cracked 0.749, dusty 0.882).
- **Six owner photos (full-frame, conf 0.45):** v1 gives a usable panel signal on
  **5 of 6**; image 4's large foreground panel is missed; image 6 gets 2
  overlapping group boxes; images 1 and 3 have `cracked` false positives on the
  black roofing membrane.

## 18. Final model selection

Selection hierarchy (**NOT** highest mAP):
1. no catastrophic real-world regression
2. better real-world panel localization on the six owner photos
3. better held-out Susan test
4. better recall without unacceptable false positives
5. better mAP50-95 / localization quality
6. reasonable training cost

**Decision: C — keep v1.** No completed challenger exists to test against gates
1–2, and — independent of the missing B/C numbers — the intervention is unlikely
to clear them: the candidate adds **≈ 1 independent physical scene** (125 frames
= 2.7 % of training). Phase 11 already established that data/aug changes which do
not add *representative scenes* either fail to help or regress OOD (v2:
+0.011 test mAP50 but **0 panels on all six owner photos**). A single extra
installation cannot re-shape a 4,546-image, 100 %-square, single-panel-per-image
training manifold.

`weights/yolo_solar_ground_research_v1.pt` (SHA `2b26334d…`) stays the ground
PANEL detector. This is the **best research candidate under this evaluation** —
not a perfect, production-ready, or field-validated model.

## 19. Limitations & next step

- The candidate is **one installation, one session** — useful as an
  aspect-ratio-diversity reference, not a scene-diversity source.
- Susan remains 100 % square 640×640 with one frame-filling panel per image —
  the core domain gap (Phases 11/12) is unchanged.
- The six owner photos have no ground truth → all real-world observations are
  qualitative.
- **Next step (unchanged from Phase 12):** the real fix is *representative
  first-party data* — permissioned ground-level rooftop PV photos across many
  independent installations (portrait + landscape, oblique, close-up + array +
  wide, boundary panels), hand-annotated one-box-per-module, native aspect
  ratio kept. When retrying the GPU step: **one training run per Kaggle notebook**,
  threshold sweeps done locally, and pre-resize the candidate frames to ≤ 1280 px
  before upload.

## Integrity

| artifact | SHA-256 | state |
|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **unchanged** |
| `weights/yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` | **unchanged** |
| `weights/yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` | **unchanged** |
| `weights/yolo_solar_condition_research_v1.pt` | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` | **unchanged** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **unchanged** |
| `weights/manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` | **unchanged** |

`models/`, `services/`, `app.py`, `pages/`, `v1.0.0` untouched. No new research
checkpoint was produced. No dataset images, owner photos, or secrets committed.
