# Phase 12 — representative ground-PV data acquisition + provenance audit

**Status: COMPLETE.** Discovery → provenance audit → annotation audit → domain
audit → deduplication → ranking. **No model was trained.** No production artifact
was modified.

**Acquisition recommendation: C — no suitable public dataset found. Pursue
direct / permissioned data collection.**

> This phase exists because of Phase 11: ground detector v2 scored marginally
> *higher* than v1 on the held-out 640×640 test set (mAP50 0.876 vs 0.865) but
> detected **0 panels on all six real owner photos** (v1: a usable signal on
> 5/6). The bottleneck is representative **data**, not augmentation or inference
> config. The next dataset must be judged primarily on **OOD / domain
> representativeness**, not benchmark score.

---

## 1. Target domain definition

Primary objective: **PANEL DETECTION (bounding boxes)** on ground-level / rooftop
RGB imagery.

Qualitative reference set (NOT training data, never uploaded, never
pseudo-labelled): the six owner photos at
`D:\solar_ai_training_scratch\Solar Real Images`.

| property | reference set |
|---|---|
| count | 6 |
| dimensions | 5 × **573×1280 portrait** (aspect 0.448), 1 × **1280×573 landscape** (aspect 2.234) |
| capture | smartphone, WhatsApp-compressed JPEG |
| content | close-up + oblique rooftop PV; large frame-filling panels; one wide multi-panel array; bird droppings / light soiling; black roofing membrane in frame; ordinary daylight |

**Priority HIGH:** RGB ground-level PV · genuine panel bounding boxes ·
smartphone/consumer-camera capture · rooftop installations · non-square native
aspect ratios · large panels / close-ups · multi-panel arrays · partial /
image-boundary panels · oblique viewpoints.

**Priority MEDIUM:** varied weather / illumination · different module
manufacturers · different roof/background materials · different countries.

**Priority LOW / disqualifying:** aerial / satellite / drone-nadir · thermal / IR
/ EL · synthetic · pure classification · no panel-localisation annotation.

A large image count does **not** outweigh domain relevance or provenance.

## 2. Search methodology

Hosts: Roboflow Universe (API + web), Kaggle (API), Zenodo (API + web), Mendeley
Data, GitHub, Nature Scientific Data / ESSD / MDPI Data, arXiv, Dataset Ninja.

Query families (≈ 15 distinct searches): `"solar panel" detection dataset ground
level RGB smartphone bounding boxes`; `rooftop solar panel detection dataset
consumer camera annotated YOLO`; `photovoltaic panel dataset Zenodo/Mendeley
ground-based field images`; `solar panel detection "in the wild" / "handheld" /
"phone camera" individual module annotation`; `roboflow universe solar panel
object detection ground handheld`; `solar photovoltaic panel detection dataset
2025 ground level rooftop oblique`.

APIs: Kaggle (`dataset_list`, `dataset_list_files`, small `dataset_download_files`);
Roboflow (project metadata + small-sample downloads) via the project-root `.env`
`ROBOFLOW_API_KEY` (never printed, never committed).

Download policy honoured: candidates were assessed from metadata + samples first;
only **small** candidates (≤ ~700 images) that passed the preliminary gate (RGB,
plausibly ground/rooftop, panel-localisation annotation) were sampled for the
audit, into `datasets/ground_pv_research/phase12_audit/samples/` (gitignored, not
retained).

## 3. Candidate table

| ID | name (host) | images | panel boxes | modality | viewpoint | annotation | provenance | tier |
|---|---|---:|---:|---|---|---|---|---|
| P12-01 | solar-panel-detection-ubwtz (Roboflow) | 674 | 1,699 | RGB | **ground-level oblique** | **segmentation-polygon** (→ bbox) | UNRESOLVED | **C** |
| P12-02 | fxmikf/solar-panel-bounding-boxes-621 (Kaggle) | 621 | 1,099 | RGB | mixed ground / rooftop / oblique | YOLO bbox | **UNUSABLE (Getty/stock/YouTube scrape)** | D |
| P12-03 | earthbook-zdvbx/solar-panels-rgb (Roboflow) | 374 | 0 panel (3,412 "dirty") | RGB | unknown | bbox | unresolved | D |
| P12-04 | yolo-2uwb2/panel-detection-nfzzu (Roboflow) | 158 | 67 | RGB | aerial / satellite | bbox (mostly unlabelled) | unresolved | D |
| P12-05 | every/solar-panels-r4trk (Roboflow) | 402 | 809 | RGB | **aerial / drone-nadir** | bbox | unresolved | D |
| P12-06 | tensrai/solar-panel-zitzr (Roboflow) | 471 | 4,371 | **grayscale** | aerial farm orthomosaic | bbox | unresolved | D |
| P12-07 | javier-aparicio-gonzlez/panel-solar-yolo (Roboflow) | 102 | 1,571 | **grayscale** | aerial farm orthomosaic | bbox | unresolved | D |
| P12-08 | roboflow-100/solar-panels-taxvb | 161 | — | thermal/EL | cell close-up | bbox (Cell/Shadowing/…) | — | D |
| P12-09 | marius-lee/solar-x0bfa (Roboflow) | 2,234 | — | thermal / IR | module IR | bbox (Hotspot/Cell Fault) | — | D |
| P12-10 | kusanagi31415/labeled-solar-panels-rwanda-rooftops (Kaggle) | many | tiny | RGB | aerial / satellite crops | bbox-in-filename | unresolved | D |
| P12-11 | salimhammadi07/…detection-and-identification (Kaggle) | 3,716 | — | RGB | aerial / satellite | segmentation masks | **resolved (Jiang 2021 CC BY 4.0)** but aerial | D |
| P12-12 | zzzpai/solar-panel-od-yolov8 (Kaggle) | ~2k | — | RGB | stock defect photos | YOLO bbox | **DUPLICATE of pythonafroz / S1** | D |
| P12-13 | aparitosh/solar-panel-dataset (Kaggle) | ~2.5k | — | RGB | stock defect photos | class folders | **DUPLICATE of S3** | D |
| P12-14 | johnsullivan66/rooftop-solar-panel-thailand (Kaggle) | many | — | RGB | aerial / satellite tiles | segmentation masks | unresolved | D |
| P12-15 | rameezakther/ghandinagar-rooftop-segmentation (Kaggle) | many | — | RGB | aerial / satellite tiles | segmentation masks | unresolved | D |
| P12-16 | Villegas-Ceballos RGB electrical (Zenodo 6386767) | 5,211 | 0 | RGB | **1 fixed camera, 1 panel** | none (paired I-V data) | **resolved CC BY 4.0** but no localisation | D |
| P12-17 | Jiang PV01/PV03/PV08 (Zenodo 5171712 / ESSD) | 3,716 | 0 | RGB | satellite + aerial + UAV-nadir | segmentation masks | **resolved CC BY 4.0** but overhead | D |
| P12-18 | BDAPPV crowdsourced aerial (Zenodo 7358126) | large | 0 | RGB | aerial + satellite | segmentation masks | **resolved** but aerial (already our aerial-detector source) | D |

(Also re-confirmed from the Phase 7 audit and not re-listed: S1, S3, S4, S6, C1,
C3 — all Roboflow, all ~640×640, all trace to the `pythonafroz` defect family.)

## 4. Provenance assessment

Applying the Phase 7 rule — **a platform-displayed "CC BY 4.0" does NOT prove the
underlying images are CC BY 4.0**:

- **Original-creator licence, resolved:** only the academic datasets —
  Villegas-Ceballos (P12-16), Jiang PV01/03/08 (P12-17), BDAPPV (P12-18),
  salimhammadi07's PV03 re-host (P12-11). **All are aerial/overhead or have no
  bounding boxes**, so a clean licence does not make them useful here.
- **Platform / re-uploader claim without original provenance ("CC BY 4.0"
  displayed by a Roboflow re-uploader):** P12-01, P12-03, P12-04, P12-05, P12-06,
  P12-07. **Provenance UNRESOLVED** for all. Do not upgrade to CC BY.
- **Unusable / conflicting:** P12-02 (fxmikf-621) — filenames and visible
  watermarks reveal **Getty Images**, **YouTube thumbnails** (`sddefault.jpg`,
  `hqdefault`), **stock sites** (Shutterstock/iStock), **company marketing pages**
  (Frey Energy), **utility press photos** (DTE), plus 294 `pythonafroz`-style
  `_rgb_resized` crops. A large fraction is copyrighted commercial stock imagery.
  **Cannot be redistributed, published, or used commercially. REJECT.**
- **Duplicates of data we already hold:** P12-12 (= pythonafroz/S1 family),
  P12-13 (= S3). Add no new physical scenes.

**No candidate has both (a) resolved original-image provenance and (b)
ground-level RGB panel bounding boxes.**

## 5. Domain assessment

Every downloadable candidate was placed in the Phase 12 viewpoint categories from
metadata + sampled images (or `UNKNOWN` where the data could not support a call):

| category | candidates |
|---|---|
| CLOSE_UP_SINGLE_PANEL | P12-02 (subset), P12-01 (subset) |
| MULTI_PANEL_ARRAY (ground) | P12-01, P12-02 (subset) |
| WIDE_ARRAY (distant, ground) | — (none) |
| PORTRAIT_PHONE | P12-02 (11 images only) |
| LANDSCAPE_PHONE | P12-02 (subset) |
| OBLIQUE (ground) | P12-01, P12-02 (subset) |
| BOUNDARY_PANEL | P12-01, P12-02 (subset) |
| ROOFTOP_BACKGROUND | P12-02 (subset) |
| AERIAL / DRONE-NADIR | P12-04, P12-05, P12-06, P12-07, P12-10, P12-11, P12-14, P12-15, P12-17, P12-18 |
| THERMAL / IR / EL | P12-08, P12-09 |
| OTHER (no localisation / 1 scene) | P12-16 |

**Key structural finding — Roboflow destroys native aspect ratio.** Every
Roboflow-exported candidate (and our own `datasets/ground_pv_research/yolo/`) is
re-encoded to a fixed square/near-square size (640×640, 640×512, 416×416). Even if
a source photo was a portrait phone image, the export is square. Native
aspect-ratio diversity survives **only** in the raw Kaggle sets — and the only one
with real aspect diversity (P12-02, 229 distinct dimensions, aspect 0.47–3.55) has
unusable provenance.

**No public dataset was found that contains a genuine distant wide multi-panel
array photographed from the ground (the image-6 case).** The "wide array" data
that exists is aerial (P12-05) or grayscale farm orthomosaic (P12-06/P12-07).

## 6. Annotation assessment

- **P12-01 (bwpkx):** **YOLO-segmentation polygons** (convertible to bbox), ~3
  per image, one polygon ≈ one module in a ground-mounted rack. Consistent style.
  Small `damaged_physical`/`DUSTY` classes can be dropped for a panel-only
  detector.
- **P12-02 (fxmikf-621):** mixed and inconsistent — some boxes are a whole panel,
  some a module, 155 boxes cover >50 % of the frame (close-ups), a few are large
  scene-spanning boxes. `classes.txt` present. Usable *shape* of annotation, but
  quality is uneven and the images are unusable anyway.
- **P12-03:** the only class is `dirty` — not a panel detector.
- **P12-04:** 158 images, mean 0.4 boxes/image — **mostly unannotated**.
- **P12-06 / P12-07:** module-level boxes in aerial farm orthomosaics (wrong
  domain).
- **P12-11 / P12-14 / P12-15 / P12-17:** segmentation masks only (convertible to
  bbox, but overhead viewpoint).
- **P12-16:** no localisation annotation at all.

**The image-6 question — "does an image with 8 visible panels get 8 boxes, 1 box,
or mixed?"** — cannot be answered from any *ground-level* candidate, because none
contains such an image. In the aerial candidates it is **8 boxes** (P12-06/07,
module-level). In P12-01 the arrays shown have ≤ ~5 racks and get one box per
rack.

## 7. Deduplication / overlap findings

Fingerprinting (exact SHA-256 + 16×16 dHash Hamming ≤ 6 + source basename), using
`training/detection/ground_pv/image_fingerprint.py`:

| candidate | vs `datasets/ground_pv_research/yolo/` (6,492) | notes |
|---|---|---|
| P12-01 bwpkx | 0 % exact / 0 % perceptual | strict 16×16 dHash (Hamming ≤ 6) counts 563 distinct *frames*, but visual inspection shows walk-around video of **≈ 1 ground-mounted installation** — near-zero *scene* diversity despite frame-level distinctness. Labels are **YOLO-segmentation polygons** (505/508 sampled), 3 boxes/image. |
| P12-02 fxmikf-621 | 0 % | **619 distinct scenes** (genuinely the most domain-diverse pool found: aspect 0.47–3.55, 40 % of boxes touch a boundary, 155 boxes > 50 % of frame). 294 of 621 are `_rgb_resized` pythonafroz-style crops. |
| P12-05 r4trk | 0 % | aerial |
| P12-07 panel-solar-yolo | 0 % | grayscale aerial |
| P12-06 solar-panel-zitzr | 0 % | grayscale aerial |
| P12-04 panel-detection-nfzzu | 0 % | aerial |

- **P12-12 (zzzpai)** and **P12-13 (aparitosh)** are re-uploads of S1 / S3 — high
  overlap with `datasets/ground_pv_research/condition/`, add no new scenes.
- **Independent physical scenes are the binding constraint, not image count.**
  P12-01's 674 images ≈ 1 scene. P12-02's ~327 non-crop images are the only pool
  with meaningful independent-scene diversity, and they are unusable.

## 8. Ranking

| tier | candidates |
|---|---|
| **A — strong acquisition target** | *(none)* |
| **B — useful with caveats** | *(none)* |
| **C — research-only** | **P12-01 solar-panel-detection-ubwtz** — right viewpoint (ground-level oblique RGB, module boxes), 0 % overlap with existing data, but ≈ 1 independent physical scene and unresolved provenance |
| **D — reject** | P12-02 (unusable provenance), P12-03 (no panel class), P12-04/05/10/14/15 (aerial), P12-06/07 (grayscale aerial farm), P12-08/09 (thermal), P12-11/17/18 (aerial/overhead, clean licence but wrong domain), P12-12/13 (duplicates), P12-16 (no annotation, 1 scene) |

## 9. Recommended acquisition

**C — no suitable public dataset found. Pursue direct / permissioned data
collection.**

Concretely:

1. **Collect a small first-party set (~150–400 photos).** Ask the owner and any
   consenting installer/partner contacts for ground-level rooftop PV photos:
   phone camera, **portrait *and* landscape**, oblique and straight-on, close-ups
   of single panels, whole arrays, wide shots of multi-panel roofs, partial /
   edge panels, varied weather and time of day, different roofs. Record explicit
   permission per contributor. **Do not** include or pseudo-label the six
   reference photos.
2. **Hand-annotate panel bounding boxes** with a fixed, written convention —
   decide up front: **one box per visible module** (recommended, so image-6 wide
   arrays get per-panel boxes), boundary/partial panels included, minimum
   visible-fraction threshold documented.
3. **Keep native aspect ratio.** Store originals; train with `rect=True` or
   letterbox (not stretch), and evaluate on a held-out slice of the *same*
   first-party data plus the six reference photos.
4. **Optional research-only supplement:** P12-01 (bwpkx) may be added *after* its
   provenance is resolved, tagged `personal-experimental-unverified-provenance`,
   with its ≈ 1-scene redundancy stated honestly and its frames de-duplicated to
   a handful before use. It cannot carry the dataset on its own.

## 10. Rejected datasets and why

- **Wrong viewpoint (aerial / satellite / drone-nadir):** P12-04, P12-05, P12-10,
  P12-11, P12-14, P12-15, P12-17, P12-18. We already have a production aerial
  detector (`weights/yolo_solar.pt`, BDAPPV-trained) for that domain.
- **Wrong modality (grayscale / thermal / IR / EL):** P12-06, P12-07, P12-08,
  P12-09.
- **No panel-localisation annotation:** P12-03 (only `dirty`), P12-16 (paired
  electrical data, no boxes).
- **Unusable provenance (copyrighted stock / scraped):** P12-02.
- **Duplicates of data already held:** P12-12, P12-13 (and, from Phase 7, C1/C3/
  S4/S6 — same pythonafroz family).
- **Near-zero independent physical-scene diversity:** P12-01 (video frames of one
  site) — hence Tier C, not usable alone.

## 11. Gaps that remain

Relative to the six owner photos, the available public data still lacks:

1. **Native portrait/landscape phone aspect ratios** — every usable candidate and
   our own dataset is square-exported.
2. **Distant ground-level wide multi-panel array shots** (image 6) — no public
   dataset contains these.
3. **Rooftop context** — roofing membrane, tiles, mounting rails, chimneys,
   parapets in frame (the source of v1's `cracked` false positives). The one
   ground-level candidate (P12-01) is a ground-mounted rack in grass.
4. **Independent physical-scene diversity at ground level** — the entire usable
   ground-level public pool is ≈ 1–2 installations.
5. **Resolved image-level provenance** for any ground-level RGB detection data.

## 12. Exact next training-data plan

1. **Do not train v3.** v1 (`weights/yolo_solar_ground_research_v1.pt`) remains the
   ground panel detector.
2. **Acquire a first-party ground-level set** per §9 (permissioned collection +
   hand annotation, one-box-per-module convention, native aspect ratio kept).
   Target ≥ ~100 independent physical scenes before any retrain is worthwhile.
3. **Build a leakage-clean split** of that set with the existing family-grouping
   tooling; keep the six owner photos as an untouched qualitative holdout.
4. **Only then** consider a v3 fine-tune of v1: `imgsz` 640 with `rect`/letterbox,
   *light* augmentation (Phase 11 showed heavy scale/mosaic/`multi_scale` hurts
   OOD), validated on held-out real photos — not on a 640×640 benchmark.
5. In parallel, ship the **off-panel `cracked`/`dusty` filter** recommended in
   Phase 10/11 (drop defect boxes with ~0 IoU against any `panel` box) — it needs
   no new data and removes the roofing false positives on images 1 and 3.

## Integrity

| artifact | SHA-256 | state |
|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **unchanged** |
| `weights/yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` | **unchanged** |
| `weights/yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` | unchanged (still NOT the baseline) |
| `weights/yolo_solar_condition_research_v1.pt` | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` | **unchanged** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **unchanged** |
| `weights/manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` | **unchanged** |

Production files modified: **none** (`models/`, `services/`, `app.py`, `pages/`,
`v1.0.0`). No dataset images, owner photos, checkpoints, or secrets committed.
**No model trained.**
