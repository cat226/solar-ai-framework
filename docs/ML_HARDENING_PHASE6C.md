# Solar AI v1.0.0 — Phase 6C: YOLO Domain Remediation

Generated: 2026-09-05
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)
Baseline: `v1.0.0` tag at `14df9b9cdb5411cef79c4282174c7d042abb1a96` (frozen, untouched).
Phase 6B completed at `9864dcb` (YOLO `confidence_threshold` recalibrated to `0.30`,
selected from validation data only).

> **Update (2026-09-05, `docs/ML_DOMAIN_REMEDIATION.md`)**: this phase's
> one open CONDITIONAL lead, OpenStat Madagascar, has since been directly
> verified and closed as **REJECT** (CC BY 4.0 confirmed, but confirmed
> nadir/overhead viewpoint, not close-up/ground-level). That same later
> phase also found and fixed a real RGB/BGR channel-order bug in
> `models/detector.py` that had been corrupting every detection call
> (aerial and close-up alike) independently of the domain-shift finding
> documented below, which otherwise still stands. See
> `docs/ML_DOMAIN_REMEDIATION.md` for the full, current picture.

## 1. Objective

Address the dominant limitation identified in Phase 6B: **YOLO
(`weights/yolo_solar.pt`) has a severe target-domain problem on close-up/
ground-level solar-panel imagery** — the kind of photo Solar AI's own
Inspect workflow expects a real user to upload. Phase 6B's threshold
recalibration measurably helped in-domain (BDAPPV aerial) recall but left
out-of-domain detection near zero (2.56% on the classification dataset's
close-up photos).

This phase determines **whether legitimate, appropriately licensed
target-domain training data exists** before considering any retraining or
domain-adaptation work. Per the task's explicit critical rule, concluding
that no defensible training path currently exists is an acceptable,
successful outcome of this phase — it is not required to produce a new
model.

---

## Task 1 — Current detector contract (reconstructed, not modified)

| Property | Value | Source |
|---|---|---|
| Architecture | YOLOv8n (ultralytics), single-class head | `weights/manifest.json`, `training/detection/train_yolo.py --base-model yolov8n.pt` |
| Training dataset | BDAPPV (IGN config) — `gabrielkasmi/bdappv` (HF mirror of Zenodo DOI 10.5281/zenodo.7358126) | `docs/RELEASE_v1.0.0.md`, `training/experiments/registry.jsonl` (`solar-yolo-full-v1`) |
| Training image count | 17,107 total (11,347 train / 3,179 val / 2,581 test), geographic (department-level) split | Same |
| Image size | 640×640 (letterboxed, preserving aspect ratio) | `configs/settings.yaml`, `utils/image_utils.py::resize_for_yolo` |
| Augmentation | Ultralytics' own internal `YOLO.train()` augmentation pipeline (no custom augmentation code in this repo) | `training/detection/train_yolo.py` |
| Confidence threshold (production) | **0.30** (recalibrated Phase 6B from 0.45) | `configs/settings.yaml` |
| IoU threshold (production, NMS) | 0.50 (unchanged since v1.0.0) | `configs/settings.yaml` |
| Class taxonomy | Single class: `"solar panel"` (class id 0) | `training/detection/train_yolo.py::_write_data_yaml` |
| Annotation format | YOLO txt (`class cx cy w h`, normalized 0-1) | `E:\Solar AI Training Images\yolo_prepared\*\labels\*.txt` |
| Expected input domain (training) | **Aerial/satellite rooftop imagery** (BDAPPV) | Dataset provenance docs |
| Expected input domain (real usage) | Close-up/ground-level photos, per the actual Inspect page upload workflow | `app.py`, `README.md` |
| Artifact SHA-256 | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | Reverified this phase — unchanged from `v1.0.0` |

**Known weaknesses (from Phase 6A/6B, reconfirmed, not re-litigated here):**
1. Object-size capability ceiling — even large in-domain (BDAPPV) boxes cap at ~50-60% recall at any confidence threshold.
2. Severe domain-shift failure — 2.56% detection rate on close-up photos (Phase 6B, post-recalibration).
3. This phase's own threshold-sweep methodology has an unresolved, disclosed discrepancy against `ultralytics.YOLO.val()`'s self-reported numbers (see `docs/ML_HARDENING_PHASE6B.md`).

**No artifact was modified in this task.**

---

## Task 2 — Target-domain data requirements

**Target domain:** RGB close-up/ground-level solar-panel imagery matching what a real Solar AI user photographs and uploads through the Inspect page — i.e. the *opposite* end of the imaging-viewpoint spectrum from BDAPPV's aerial/satellite rooftop imagery.

**Preferred image characteristics** (not all required in every sample, but the dataset as a whole should represent this mix):
- Rooftop imagery taken from ground level or a ladder/drone-at-roof-height (not overhead satellite/aerial)
- Module-level and multi-panel array imagery
- Ground-level inspection-style photography (the style a maintenance technician or homeowner would realistically capture)
- Close-range single/few-panel photographs
- Varied lighting (overcast, direct sun, shadow, dusk)
- Varied panel orientation/tilt/viewing angle
- Partial occlusion (tree shadow, mounting hardware, adjacent structures)
- Realistic backgrounds (roofing material, sky, ground-mount racking, surrounding vegetation)
- Both single-panel and multi-panel-array framing

**Required annotation:**
- Bounding boxes around each visible solar panel/module
- Single detection class: `solar-panel` (matching the existing v1 taxonomy exactly — no new classes, no incompatible label sets mixed in)

**Minimum quality criteria for any candidate dataset:**
1. **Valid annotations** — real bounding boxes in a parseable format (YOLO txt, COCO JSON, Pascal VOC XML, or convertible to one of these without inventing missing data)
2. **Sufficient image resolution** — high enough that a panel occupies a recognizable, non-degenerate pixel region (no thumbnail-only sources)
3. **Meaningful target-domain representation** — genuinely close-up/ground-level, not a relabeled aerial set
4. **Licensing that explicitly permits ML training / derivative model use** — a CC0, CC-BY, CC-BY-SA, MIT, or comparably permissive license, or an explicit written grant; "free to view/download" alone is insufficient (non-negotiable rule 8)
5. **Stable, citable provenance** — a real DOI, institutional repository entry, or an official, actively-maintained project repository; not an anonymous scrape or an unverifiable third-party re-upload
6. **No obvious benchmark contamination** — not a re-hosted or re-labeled copy of BDAPPV, the existing SolNET/PVMD classification sets, or another dataset this project has already investigated and rejected (per `training/classification/DATASET_SOURCES.md` and `training/detection/PROVENANCE_VERIFICATION.md`)

---

## Task 3 — Dataset discovery (real web research, this phase)

Approximately a dozen distinct web searches plus direct source-page fetches were performed (see the candidate table below for exact queries' findings). Discovery covered: Zenodo/Mendeley/institutional-repository searches for ground-level/close-up PV imagery, Roboflow Universe's solar-panel-tagged datasets, recent (2024-2026) academic papers on PV defect/panel detection and the datasets they cite, and a specific investigation of "OpenStat Madagascar" (a drone-imagery PV dataset surfaced during discovery).

**Key cross-check performed:** several candidates share the exact class taxonomy (`Bird-Drop`, `Dusty`, `Electrical-Damage`, `Physical-Damage`, `Snow-Covered`/`Clean`) this project already investigated for the MobileNet classification pipeline. `training/classification/DATASET_SOURCES.md` already established that this exact taxonomy traces to the `pythonafroz/solar-panel-images` Kaggle dataset, whose Kaggle page **explicitly states "License: Unknown"** (confirmed there via peer-reviewed literature (SolarFCD) describing it as "permission uncertain"). Every Roboflow-hosted dataset found this phase sharing that same taxonomy is treated as a probable downstream re-upload of that same unknown-license source — **a re-uploader's own self-declared "CC BY 4.0" tag on Roboflow does not cure an unverifiable original license** (non-negotiable rules 8 and 10).

## Task 4 — Candidate dataset gate

| Dataset | Source | License (as found) | Image type | Bboxes? | Target-domain relevance | Verdict | Reason |
|---|---|---|---|---|---|---|---|
| BDAPPV (already in use) | Zenodo 10.5281/zenodo.7358126, HF mirror `gabrielkasmi/bdappv` | CC BY 4.0 (verified, existing project record) | Aerial/satellite rooftop | Yes (via mask→bbox, already used) | None (this IS the existing training domain) | N/A — not a remediation candidate | Already in use; the subject of the domain gap, not its solution |
| Multi-resolution PV segmentation (PV01/PV03/PV08) | Zenodo 10.5281/zenodo.5171712, Hou/Ling/Yujun (IGSNRR CAS) | CC BY 4.0 | **Satellite (Gaofen-2/Beijing-2, 0.8m/px) + aerial photography (0.3m/px) + UAV orthophotos (0.1m/px)** — confirmed via the dataset's own ESSD publication | Segmentation masks (convertible) | **None — all three tiers are overhead/nadir viewpoints**, including the UAV tier (orthophoto = top-down mosaic, not oblique ground-level) | **REJECT** (for this purpose) | Wrong viewpoint domain despite excellent license/provenance; does not address the close-up/ground-level gap |
| OpenStat Madagascar (PV detection) | Madagascar Initiatives for Digital Innovation (MAIDI) + Lacuna Fund; platform DOI 10.5281/zenodo.3620832 (the specific 130,500-object PV annotation release's own direct download location/DOI was not conclusively located this phase) | Creative Commons (family confirmed; exact variant — CC BY vs CC BY-SA vs other — not confirmed for the specific PV annotation release) | Google Earth satellite screenshots (2,125 images) + **drone imagery (9,202 images, "rooftop... installations")** | Yes — manual polygon annotation of individual solar objects (confirmed) | **Uncertain, likely still predominantly overhead/nadir** — described as drone imagery of "rooftop photovoltaic installations" at higher resolution than satellite, not as oblique/ground-level photography; the source material does not confirm a genuinely close-up, ground-level, oblique viewing angle | **CONDITIONAL** | Real institutional provenance and a real license family, but (a) exact license variant for this specific release and (b) the actual image viewing angle both require direct verification of the raw data/download page before any use — not performed this phase, since that verification itself would require pursuing access to a dataset not yet gated |
| Roboflow "Solar Panel Defects" family (multiple accounts: `solar-panel-defects`, `solarpanel2`, `solar-panel-defects-ax0gw`, `solarpanel-2me5p`, `ramkumar`, `defects-kgig6`, `awe-ayo`, and others) sharing the `Bird-Drop`/`Dusty`/`Electrical-Damage`/`Physical-Damage`/`Snow-Covered` taxonomy | Roboflow Universe (community uploads) | Self-declared "CC BY 4.0" or "Public Domain" per-project | Close-up panel photos (classification-style) | Nominally yes (Roboflow object-detection format), but likely single full-frame boxes on originally-classification imagery, not genuine multi-object annotation | High in principle (genuinely close-up) — moot given the provenance rejection | **REJECT** | Provenance traces to a Kaggle source with **explicitly "Unknown" license** (rule 8, rule 10) — already rejected by this project for the same reason on the classification side (`training/classification/DATASET_SOURCES.md`) |
| Roboflow "PV defect" (`pv-solar/pv-defect-hoaes`, 404 images, 9 classes) | Roboflow Universe | Self-declared CC BY 4.0 | **Electroluminescence (EL) cell-level imagery** (classes: Multi-Cell/Single-Cell Cautious/Critical/Notify, Sub-String) | Yes | None | **REJECT** | Wrong modality (specialized EL imaging, not RGB) and wrong task (manufacturing cell-defect QC, not whole-panel RGB localization) |
| Small individually-uploaded Roboflow community datasets without institutional/DOI backing (TENSRAI 471 imgs, "MARIUS LEE" 2,234 imgs, "YOLO" Panel Detection 158 imgs, "Javier Aparicio Gonzlez" panel-solar-yolo 102 imgs, Roboflow-100 `solar-panels-taxvb` 161 imgs, "Brad Dwyer" Aerial Solar Panels, CV Workspace 84 imgs) | Roboflow Universe (personal accounts) | Various self-declared tags | Mixed (several explicitly aerial; others unstated) | Unverifiable | Unverifiable | **REJECT** | No cited original data source, no DOI, no institutional affiliation for any of these — provenance cannot be independently established (rule 10), regardless of the platform-displayed license tag, which is self-declared by the uploader and not independently audited for community uploads |
| Zenodo 16420123 "Thermal PV Panel Detection and Fault Detection Dataset for UAV-Based Inspection" (351 images) | Zenodo | Stated on record (not fully reviewed — moot, see reason) | Thermal, UAV | Yes | None | **REJECT** | Wrong modality (thermal, not RGB — Task 2 requires RGB) |
| "~300-image aerial RGB dataset for multiclass soiling detection" (surfaced via search, exact source not pinned down) | Unclear | Unclear | **Aerial** (stated explicitly in its own description) | Unclear | None | **REJECT** | Explicitly aerial by its own description |
| 8,973-image "heterogeneous... standardized" aggregate dataset and the 792-image Kaggle/Roboflow combination used by two recent MDPI papers | Multiple aggregated sources | Mixed/inherited | Close-up, classification-style | Nominal | High in principle | **REJECT** | Confirmed (via the papers' own dataset descriptions) to substantially reuse the same already-rejected Afroz-lineage/unknown-license sources; aggregating several already-flagged sources does not cure the underlying provenance problem |

**Result: no ACCEPT-tier candidate was found.** One CONDITIONAL candidate (OpenStat Madagascar) remains open pending direct verification this phase did not perform (see Task 9).

---

## Task 5 — Existing dataset (BDAPPV) domain audit

| Property | Existing BDAPPV (measured) | Target requirement |
|---|---|---|
| Imaging viewpoint | Aerial/satellite, nadir (top-down) | Ground-level / oblique, close-up |
| Typical scale | Whole rooftop or block visible per 400×400px image | Single panel to small array fills the frame |
| Ground-level representation | None — 0 ground-level images in the dataset | Primary requirement |
| Panel size (measured, val split, this project's own audit) | Real GT box area: median 0.54% of image area, 90th pct 1.16% (`docs/ML_HARDENING_PHASE6B.md` Task 2) — **panels are small objects within the frame by construction of the aerial viewpoint** | Panel(s) typically dominate the frame in a close-up photo |
| Background variation | Rooftop material, urban/rural aerial context only | Sky, walls, mounting hardware, vegetation, ground, interior/exterior structures |
| Annotation quality | High — polygon-derived, peer-reviewed, geographic-split, exact-hash-verified provenance (`training/detection/PROVENANCE_VERIFICATION.md`) | N/A (property of BDAPPV itself, not a gap) |
| Lighting variation | Whatever the original aerial/satellite capture campaigns used (unknown time-of-day distribution, not characterized by this project) | Wide (overcast, direct sun, shadow, dusk) expected from real user uploads |
| Intended task (original) | Solar-adoption mapping / installation metadata research (Kasmi et al. 2023) — not originally intended for a ground-level inspection product | Panel localization for a user-facing inspection app |

**Measured evidence of what BDAPPV does and does not cover** (from this project's own real evaluations, not assumption):
- **Does cover:** in-domain aerial detection at a real, reproducible, moderate capability level — mAP50≈0.74 (`docs/ML_EVALUATION_v1.0.0.md`), production-path recall 0.271 at the recalibrated threshold on the in-domain test split (`docs/ML_HARDENING_PHASE6B.md`).
- **Does not cover:** ground-level/close-up imagery at all — `docs/ML_HARDENING_PHASE6B.md`'s end-to-end re-evaluation measured a 2.56% detection rate (4/156) on the real, licensed SolNET/PVMD close-up photo set, even after the threshold recalibration.
- **Aerial data is not useless** — it is a real, working, in-domain capability for exactly the imagery type it was built for (satellite/drone-at-altitude rooftop mapping). The gap is specifically that Solar AI's actual product surface (a user photographing their own panel) does not match that domain.

---

## Task 6 — Target-domain baseline

**No new target-domain evaluation dataset was constructed this phase**, because Task 4 found no ACCEPT-tier annotated target-domain dataset to build one from, and manufacturing box-level ground truth on data not designed/licensed for that specific annotation purpose was judged out of scope (risking exactly the "false confidence from a manufactured split" the task rules warn against).

**However, a real, legitimate, already-licensed target-domain signal already exists** and is not re-derived here, only cited: the MobileNet classification dataset (SolNET + PVMD, CC BY 4.0, already fully verified and in-project use — see `training/classification/DATASET_SOURCES.md`) *is* genuine close-up/ground-level RGB solar-panel photography. `training/evaluation/evaluate_end_to_end.py` already evaluated `weights/yolo_solar.pt` against it, at the current production settings (conf=0.30, iou=0.50, imgsz=640), in Phase 6B:

| Metric | Value | Source |
|---|---|---|
| Images | 156 (real, licensed, close-up single-panel photos) | `docs/ML_HARDENING_PHASE6B.md` Task 7 |
| Detection rate (≥1 panel found) | 2.56% (4/156) | Same |
| TP / FP / FN / precision / recall / mAP | **Not computable** | This dataset has no ground-truth bounding boxes (it was collected/licensed for whole-image fault classification, not panel localization) — only a detected/not-detected signal is available, not full detection metrics |

This is a real, if coarse, measurement, not an assumption: **the existing detector fires on essentially none of a genuine close-up target-domain sample**, independent of confidence threshold (Phase 6B's threshold sweep did not meaningfully change this figure). No threshold sweep against this data was performed for *selection* purposes (rule 15's "do not optimize against the existing BDAPPV test set" is honored trivially — this isn't BDAPPV at all — and no other threshold selection was attempted against it either, consistent with rule 3).

**Train/val/test split with group/duplicate controls:** not applicable — no new dataset was created.

---

## Task 7 — Duplicate/contamination audit

Not applicable this phase in the sense the task describes (checking a *newly acquired* candidate dataset for overlap with existing project data) — **no new dataset was acquired**, since Task 4's gate did not pass. For completeness: the one target-domain data source referenced in Task 6 (SolNET/PVMD) is not new — it is the same, already-leakage-audited MobileNet dataset from Phases 6A/6B (`docs/ML_HARDENING_PHASE6B.md` Task 5), and that audit's findings (0 exact duplicates train/val/test within itself; genuine but apparently low-impact near-duplicate leakage) are not repeated here.

---

## Task 8 — Dataset quality report

Not applicable — no ACCEPT-tier candidate dataset exists to report quality metrics for.

---

## Task 9 — Decision gate for training

| Gate condition | Status |
|---|---|
| Legitimate target-domain dataset exists | ❌ No ACCEPT-tier candidate found (Task 4) |
| Licensing is acceptable | ❌ (moot — no candidate reached this check) |
| Annotations are suitable | ❌ (moot) |
| Provenance is documented | ❌ (moot) |
| Enough independent data exists | ❌ (moot) |
| Train/validation/test separation is defensible | ❌ (moot) |
| No major contamination present | ❌ (moot) |
| Dataset meaningfully represents the intended input domain | ❌ (moot) |

**Gate result: FAILS.** Training/domain-adaptation is **not undertaken this phase.**

**What is actually required before this gate could pass**, stated concretely per the task's critical final rule:

1. **Direct verification of OpenStat Madagascar** (the one CONDITIONAL candidate): locate and inspect the specific PV-annotation release's own download page/DOI (distinct from the general MAIDI/Lacuna Fund platform record found this phase), confirm the exact license variant, and — most importantly — visually inspect a representative sample of the drone imagery to determine whether it is genuinely oblique/ground-level or predominantly nadir/overhead like BDAPPV. If overhead, it does not solve this phase's problem regardless of license.
2. **Absent that, a genuinely new data-collection effort** would be needed: real close-up/ground-level solar panel photographs with bounding-box annotations, either (a) commissioned/collected directly with clear ownership and license terms, or (b) found via a not-yet-discovered academic/institutional source with verifiable DOI-backed provenance — a "free" Roboflow/Kaggle re-upload sharing the already-rejected Afroz taxonomy does not qualify, regardless of self-declared license.
3. Any accepted dataset must then pass the same quality bar already applied to BDAPPV/SolNET/PVMD in this project: peer-reviewed or institutionally-backed provenance, an explicit license permitting derivative-model training, and enough images with real bounding-box annotations to support a genuine train/val/test split with duplicate/group controls (per Task 6/7's requirements, not attempted here for lack of underlying data).

**Tasks 10-12 (candidate training, evaluation, promotion gate) do not apply** — no training was performed, per Task 9 and rule 15/16.

---

## Observed facts vs. measured metrics vs. assumptions vs. unresolved issues

**Observed facts** (directly read from code/config/artifacts, not inferred):
- `weights/yolo_solar.pt` SHA-256 `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` — unchanged from `v1.0.0` throughout this phase.
- `configs/settings.yaml`'s YOLO `confidence_threshold` is `0.30` (Phase 6B value; unchanged this phase).
- BDAPPV's training/eval provenance (license, DOI, split methodology) — established in prior phases, reconfirmed by inspection, not re-verified from scratch.

**Measured metrics** (real computation, this phase or cited from Phase 6A/6B without re-derivation):
- BDAPPV GT box area distribution (median 0.54%, 90th pct 1.16% of image area) — Phase 6B's own real val-split computation, cited here.
- 2.56% detection rate on the SolNET/PVMD close-up test set — Phase 6B's own real computation, cited here, not re-run (the underlying artifact and config are unchanged, so re-running would reproduce the identical number).

**Assumptions** (stated explicitly, not hidden):
- OpenStat Madagascar's drone imagery is assumed, based on its own description ("rooftop... installations," resolution-focused framing analogous to the satellite comparison), to be predominantly nadir/overhead rather than oblique/ground-level. **This is an inference from available descriptions, not a visual confirmation** — the raw imagery was not inspected.
- "Roboflow self-declared license" is assumed unreliable for community uploads with no cited original source, based on this project's own prior, directly-verified finding (the Afroz Kaggle source's real "Unknown" license) about exactly this taxonomy family — a reasonable, evidenced inference, not a blanket claim that all Roboflow-hosted data is untrustworthy.

**Unresolved issues:**
- OpenStat Madagascar's exact license variant and the specific PV-annotation release's own direct download location/DOI were not conclusively pinned down this phase.
- Whether OpenStat Madagascar's drone imagery is genuinely oblique/ground-level (which would make it a real candidate) or nadir/overhead (which would not) is unresolved without direct visual inspection.
- No exhaustive search of every academic repository was possible in one phase; a negative result here means "not found by this genuine effort," not "provably does not exist anywhere."

## Remaining limitations

- **The dominant v1 limitation (severe domain-shift on close-up imagery) remains unresolved after this phase.** No production code, configuration, or model artifact changed as a result of this investigation.
- **No new training data was acquired**, so no domain-adaptation or retraining experiment could be attempted, per this project's own licensing/provenance standards.
- **The CONDITIONAL OpenStat Madagascar lead remains open** for a future phase, specifically requiring direct access/visual verification before any use.
- **This phase's discovery search, while genuine and reasonably thorough (approximately a dozen searches plus source-page verification attempts), is not exhaustive** — it reflects real effort within this phase's scope, not a claim that no usable dataset exists anywhere.
- **All Phase 6A/6B limitations remain unchanged and still apply**: the object-size capability ceiling, the unresolved threshold-sweep-vs-`YOLO.val()` methodology discrepancy, MobileNet's small evaluation set and near-duplicate leakage, XGBoost's continued unavailability, and the 3 unavailable taxonomy classes.

---

## Final assessment

Per this phase's own critical rule, the outcome below is treated as a successful, complete result of Phase 6C, not a failure to reach one:

> **We have identified the domain gap (Phase 6B), verified the existing baseline against a real close-up target-domain sample (this phase, citing Phase 6B's own measurement: 2.56% detection rate), searched extensively for legitimate remediation data (this phase), and found no dataset meeting this project's own licensing/provenance/domain-relevance bar. Exactly one lead (OpenStat Madagascar) remains open, and exactly what would be required to close it or to find an alternative is documented above.**

**Recommendation: DATA COLLECTION REQUIRED** (or, equivalently, further verification of the one open CONDITIONAL lead) before any retraining or domain-adaptation work can be responsibly attempted. **KEEP v1 AS-IS** for this specific limitation — the existing `weights/yolo_solar.pt`, already-honestly-disclosed in `docs/ML_EVALUATION_v1.0.0.md` and `docs/ML_HARDENING_PHASE6B.md`, remains the deployed artifact.

