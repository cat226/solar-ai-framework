# Solar AI v1 — YOLO Close-Up Detection: Domain Remediation

Generated: 2026-09-05
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)
Baseline: `v1.0.0` tag at `14df9b9cdb5411cef79c4282174c7d042abb1a96` — **frozen, untouched by this work.**

## TL;DR

Two separate problems were conflated under "YOLO fails on close-up
images." This phase separated them:

1. **A real, confirmed inference bug** (RGB/BGR channel inversion in
   `models/detector.py`) was silently feeding every production detection
   call color-swapped pixels. **Fixed.** This alone raised in-domain
   (aerial) recall from 25.2% to 86.1% on a 200-image real held-out test
   sample, and raised real close-up detection (on the 156-image
   ground-truthed classification set) from 2.56% to 13.46%.
2. **A genuine, still-unresolved data/domain problem**: even after the
   fix, the detector still misses the large majority of real close-up
   panels, and the specific 16-image ad hoc validation sample from
   `docs/ML_REAL_IMAGE_VALIDATION.md` is unchanged at 0/16. No legitimate,
   appropriately-licensed close-up bounding-box dataset exists to retrain
   against (reconfirmed this phase, including closing Phase 6C's one open
   lead as a definitive reject).

**Nothing here trains, replaces, or overwrites the production YOLO
artifact.** `weights/yolo_solar.pt`'s SHA-256 is unchanged. `v1.0.0` is
untouched. What changed: two production code bugs (fixed), and one
config value (`confidence_threshold`, recalibrated using the same
legitimate, pre-existing, val-split-only methodology that originally set
it — never the test split, never the 16-image sample).

---

## Phase 1 — Pipeline audit

### 1.1 Root cause found: RGB/BGR channel inversion

`models/detector.py` built a raw numpy array from a PIL image
(`pil_to_numpy(resize_for_yolo(image))`) and passed that array directly to
the loaded `ultralytics.YOLO` model. Direct inspection of the installed
Ultralytics package (`ultralytics/data/loaders.py::LoadPilAndNumpy._single_check`)
confirms this is a real, documented behavioral distinction:

```
Notes:
    - PIL inputs are converted to NumPy and returned in OpenCV-compatible BGR order for color images.
    - NumPy color inputs are assumed to use OpenCV-compatible BGR order.
```

- A **PIL.Image** input is explicitly reversed (RGB→BGR) at this stage,
  specifically so that Ultralytics' own later unconditional BGR→RGB flip
  in `Predictor.preprocess()` (`im[..., ::-1]`) restores true RGB before
  the array reaches the network.
- A **raw numpy array** is assumed to already be BGR (the `cv2.imread`
  convention) and is passed through unmodified at this stage — so that
  same later flip inverts *actual* RGB data into BGR before it reaches
  the network.

Reproduced directly (not just reasoned about) with a solid red test image:

| Input given to `LoadPilAndNumpy` | Internal `im0` after loading | Final network input after `preprocess()`'s flip |
|---|---|---|
| `PIL.Image` (red, RGB=(255,0,0)) | BGR=(0,0,255) | RGB=(255,0,0) — **correct** |
| `numpy.ndarray` (red, RGB=(255,0,0)) — what `pil_to_numpy` produced | passthrough (255,0,0) | (0,0,255) — **red became blue** |

Every production detection call therefore fed the network images with red
and blue channels swapped, silently, since this project's YOLO integration
was written. This is unrelated to domain shift — it corrupts *both*
aerial and close-up input identically.

**Fix** (`models/detector.py`): pass the letterboxed `PIL.Image` directly
to the model instead of converting it to a numpy array first. The
now-unused `pil_to_numpy` import was removed from this call site (the
function itself remains — still used elsewhere and by its own tests).
The same bug existed in `training/evaluation/yolo_threshold_sweep.py`
(the tool that selected Phase 6B's `confidence_threshold=0.30`) and was
fixed identically.

**Regression tests added** (`tests/test_detector.py`,
`TestChannelOrderRegression`): assert `detect()` passes ultralytics a
`PIL.Image` instance, never a numpy array, and that the passed image is
still the correctly-letterboxed 640×640 canvas.

**Measured impact** (real re-evaluation, not projected):

| Sample | Metric | Before fix | After fix |
|---|---|---:|---:|
| 200 real held-out **aerial** test images (BDAPPV, conf=0.30 both runs) | Precision | 0.2959 | 0.4783 |
| | Recall | 0.2522 | 0.8609 |
| | TP / FP / FN | 29 / 69 / 86 | 99 / 108 / 16 |
| 156 real, ground-truthed **close-up** classification images (SolNET/PVMD) | Detection rate | 2.56% (4/156, Phase 6B) | 13.46% (21/156) |

The close-up figure combines the code fix with the threshold
recalibration below (both were in effect for that run); the aerial
comparison isolates the code-fix-only effect, since both rows used the
unchanged `conf=0.30`.

### 1.2 Second finding: no EXIF-orientation handling (fixed defensively)

No file in the codebase called `ImageOps.exif_transpose()`. A real phone
photo is frequently stored with orientation metadata rather than
pre-rotated pixels — without this, a portrait photo can reach the models
rotated 90/180/270° from how the user (and the camera's own preview) saw
it.

**Confirmed not the cause of this session's specific 0/16 close-up
result**: none of the 16 real validation images carry a non-default EXIF
orientation tag (checked directly: all `None` or `1`/normal). But this is
a real, latent correctness gap for future user uploads, so it was fixed
regardless:

- `app.py` (the actual production upload path — it opens images itself,
  it does **not** go through `utils.image_utils.load_pil_image()`):
  now applies `ImageOps.exif_transpose()` before `.convert("RGB")`.
- `utils/image_utils.py::load_pil_image()` (used by
  `training/classification/validate_interim_checkpoint.py` and its own
  tests): same fix.
- `training/evaluation/validate_real_images.py`: same fix, so this
  validation tool matches production behavior exactly.

**Regression tests added** (`tests/test_image_utils.py`,
`TestLoadPilImage`): build a JPEG with `Orientation=6` via real EXIF
bytes (not mocked) and confirm the loaded image is actually rotated
(width/height swapped), plus a control case confirming an image with no
EXIF tag is left unchanged.

### 1.3 Confidence-threshold recalibration

Phase 6B's `confidence_threshold=0.30` was itself selected by running
`training/evaluation/yolo_threshold_sweep.py` — which had the *same*
RGB/BGR bug. That sweep's precision/recall curve was therefore built from
color-inverted images; the "best" threshold it found is not trustworthy
once the underlying bug is known.

Re-ran the identical, pre-existing, legitimate methodology (validation
split only — the script itself refuses `--split test`; fixed IoU=0.50;
sweeps confidence 0.05–0.90) on the corrected code, against the full
3,179-image real validation split:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 (previous production value) | 0.5513 | 0.8344 | 0.6639 |
| **0.45 (new production value, best-F1)** | **0.6658** | **0.7536** | **0.7070** |
| 0.50 | 0.7014 | 0.7014 | 0.7014 |

`configs/settings.yaml`'s `models.yolo.confidence_threshold` was updated
`0.30 → 0.45`, with a comment documenting the full before/after evidence.
This is a **recalibration of the existing, unchanged artifact** (same
`weights/yolo_solar.pt`, same SHA-256), not a new model, not a change to
`v1.0.0`, and was **not** selected against the test split or the 16-image
sample — it reuses the same val-only tool and discipline Phase 6B
established. `tests/test_config_validation.py`'s pinned-value assertion
was updated to match.

### 1.4 Everything else checked and found correct

- **MobileNet path** (`models/classifier.py`): uses `torchvision.transforms.ToTensor()`
  directly on a PIL RGB image — no BGR/RGB ambiguity exists in
  torchvision's convention; confirmed correct, not touched.
- **Coordinate scaling / letterbox math** (`utils/image_utils.py::unletterbox_box`,
  `crop_panel`): already covered by existing passing tests
  (`TestUnletterboxBox`, `TestCropPanel`); re-verified by inspection —
  correctly inverts `resize_for_yolo`'s transform. Not the cause.
- **Confidence filtering / NMS**: `conf=`/`iou=` are passed straight
  through to `ultralytics.YOLO.__call__`, which applies both internally;
  confirmed via the library's own documented behavior, not reimplemented
  here. Not the cause.
- **Model loading**: `models/model_manager.py` loads
  `weights/yolo_solar.pt` once, hash-verified; not touched, not the cause.
- **Production vs. training/evaluation pipeline parity**: the production
  path (`models/detector.py`) and the evaluation scripts
  (`training/evaluation/evaluate_yolo.py`, `evaluate_end_to_end.py`) both
  go through the *same* `SolarPanelDetector.detect()` method — so both
  shared the same bug and both now share the same fix. The one true
  divergence found was `evaluate_yolo.py`'s separate `raw_model.val(...)`
  mAP50/mAP50-95 pass, which uses Ultralytics' own file-path-based dataset
  loader (correct BGR handling throughout) — meaning the previously
  reported **mAP50≈0.74 aerial figure was never affected by this bug** and
  remains valid as a citation.

---

## Phase 2 — Supplied real image characterization

Already performed rigorously in `docs/ML_REAL_IMAGE_VALIDATION.md`
(2026-09-05) and re-cited here rather than redone: 16 images, all
decodable RGB after PIL/format normalization; 4 are thermal/FLIR
false-color (not standard RGB photography); at least 2 are watermarked
stock-photo previews; one file has a `.jpg` extension but is actually
AVIF-encoded (handled transparently by PIL regardless of extension).
Resolutions range from 295×325 to 3000×5333. Visual review found several
genuinely multi-panel scenes, several single close-up panels, and no
confirmed zero-panel negative image.

**This set was used only for re-validation in this phase (Task 4/re-run
below), never as training or annotation input** — it remains a
validation-only asset, per the explicit constraint that it must not
become a training set merely because it is available.

---

## Phase 3 — Legitimate close-up dataset search

### 3.1 Resolving Phase 6C's one open lead: OpenStat Madagascar

Phase 6C (`docs/ML_HARDENING_PHASE6C.md`) left one candidate as
**CONDITIONAL**: OpenStat Madagascar's drone imagery, because neither its
exact license variant nor its real image viewpoint (oblique/ground-level
vs. nadir/overhead) had been directly verified.

This phase verified both directly:

- **Specific release located**: Zenodo record **15120875**, "Données sur
  l'énergie solaire et labellisation d'images de panneaux photovoltaïques
  à Madagascar" (distinct from the general MAIDI/Lacuna-Fund platform
  record 3620832 found in Phase 6C).
- **License, confirmed directly on the Zenodo record page**: CC BY 4.0 —
  the license question is resolved, and it is permissive.
- **Viewpoint, confirmed via a downstream academic paper's own reported
  quantitative statistics** on this exact dataset (arXiv 2603.02142,
  "Is Bigger Always Better? Efficiency Analysis in Resource-Constrained
  Small Object Detection"): the drone imagery is explicitly described as
  **overhead/nadir** ("Representative high-resolution drone images... from
  above," figure captions referencing "overhead imagery"), and — most
  tellingly — **64% of annotated bounding boxes have a normalized area
  below 0.01, with a median of 0.006**. That is panels occupying a *small*
  fraction of the frame, the signature of a wide-area aerial survey, not
  close-up photography (for comparison, BDAPPV's own GT box-area median is
  0.0054 — OpenStat Madagascar's panels are, if anything, similarly small
  or smaller in-frame).

**Verdict: REJECT for this purpose.** Real institutional provenance and a
genuinely permissive, now-confirmed license — but the wrong viewpoint
domain, exactly like the previously-rejected PV01/PV03/PV08 dataset. This
closes Phase 6C's one open lead with a definitive answer rather than
leaving it open.

### 3.2 Additional search this phase

One new named lead ("SPID — Solar Panel Image Dataset," surfaced via a
third-party AI-generated summary site) was investigated and found to be
**not a real, independently citable single dataset** — it is an
aggregation label for several different, disparate named projects
(HyperionSolarNet, GloSoFarID, and others), with no primary DOI or paper
of its own. This fails the "stable, citable provenance" requirement
outright (the same standard already applied to reject unverifiable
Roboflow community uploads in Phase 6C) — **REJECT**, not a real dataset.

No other new ACCEPT-tier candidate surfaced.

### 3.3 Net result

Phase 6C's finding stands and is now more complete: **no ACCEPT-tier
close-up/ground-level RGB bounding-box dataset with acceptable license and
provenance has been found**, across two independent investigation passes.
The one previously-open lead is now closed as a firm reject, not merely
unresolved.

---

## Phase 4 & 5 — Dataset pipeline and annotation infrastructure (built, not yet fed data)

Since no dataset passed the gate, per this task's own instruction the
correct action is to build the infrastructure to consume one immediately
once acquired, not to substitute a lesser dataset.

**Built**: `training/detection/prepare_closeup_dataset.py` — a new,
independent, tested ingestion pipeline (24 passing tests in
`tests/test_prepare_closeup_dataset.py`) that:

- Enforces the license/provenance gate as a **hard rejection**, not a
  warning: every image must have a `provenance.json` entry with a license
  drawn from an explicit allow-list (CC0, CC-BY-\*, CC-BY-SA-\*, MIT, or a
  disclosed `explicit-written-grant`) and a `source_url` or
  `rights_holder` — anything else (including "Unknown," or no entry at
  all) is rejected and reported, never defaulted or silently accepted.
- Enforces the single-class taxonomy: rejects any label file using a
  class id other than `0` ("solar_panel") — MobileNet's Clean/Dusty/Hotspot
  labels can never leak into the detection class list through this path.
- Detects exact duplicates (SHA-256) and clusters near-duplicates
  (difference-hash, reusing the same already-audited
  `training.evaluation.common.dhash`/`hamming_distance` functions
  `training/evaluation/leakage_audit.py` uses), and guarantees a whole
  near-duplicate cluster — and any images sharing an explicit
  `group_key` (e.g. same capture session) — lands in exactly one
  train/val/test split, so no same-scene leakage is possible even before
  a leakage audit is run.
- Produces a deterministic split from a fixed seed (verified by test:
  re-running the same inputs/seed reproduces byte-identical split
  assignment and an identical dataset content hash) and a dataset-level
  SHA-256 content hash for reproducibility verification.

**Also written**: `training/detection/CLOSEUP_ANNOTATION_TEMPLATE.md` —
the exact directory layout, YOLO label format, and `provenance.json`
schema a human annotator (or a licensed dataset's own metadata) must
populate. Manual bounding-box annotation is a human judgment task this
project does not, and per its own no-fabrication policy must not,
perform automatically — no annotations were fabricated for the 16
validation images or any other image as part of this work.

**Remaining manual step, stated plainly**: acquire a legitimate,
appropriately-licensed close-up/ground-level image source (or annotate
one already owned with clear rights) and populate it into this exact
format. This script and template make that a data-acquisition problem
only, not also an engineering problem.

---

## Phase 6 & 7 — Training experiments and model comparison

**Not performed.** Per this task's own explicit rule, retraining requires
a legitimate close-up dataset to exist, and Phase 3 found none. Running
"Experiment B" (close-up only) or "Experiment C" (mixed) without real
data would require either fabricating images/boxes or reusing an
already-rejected unknown-license source — both explicitly forbidden.
**Experiment A (the current aerial/BDAPPV baseline) already exists** and
is unchanged: `weights/yolo_solar.pt`, mAP50≈0.74 on the aerial test split
(unaffected by this phase's bug fix, since that figure came from
Ultralytics' own file-path-based `YOLO.val()`, not the buggy code path).

---

## Phase 8 — MobileNet integration

**Question asked**: does a better (bug-fixed) detector → produce useful
close-up crops → that the existing MobileNet classifies well?

Re-ran `training/evaluation/evaluate_end_to_end.py` against the real,
ground-truthed 156-image SolNET/PVMD close-up classification test split
(the same dataset Phase 6B used for its 2.56% figure), under the fixed
code and the recalibrated `conf=0.45`:

| Metric | Value |
|---|---:|
| Detection rate (≥1 panel found) | 13.46% (21/156) — up from 2.56% (4/156) |
| Classification accuracy, **given** detection succeeded | 61.90% (13/21) |
| Classification given detection — macro F1 / weighted F1 | 0.465 / 0.704 |
| End-to-end (detect AND classify correct) | 8.33% (13/156) |
| Whole-image classification accuracy (detection-independent) | 99.36% (155/156) — unchanged; this is MobileNet's own already-established test accuracy, cited not re-derived |

**Answer: partially yes, with a real but moderate signal.** The fix does
produce meaningfully more usable close-up crops (4→21, a >5x increase),
and on those 21 crops MobileNet is correct 61.9% of the time — well above
the 33% three-class chance baseline, so this is a genuine predictive
signal, not noise. It is not strong enough to call the classifier
"working well" on detector-sourced close-up crops, but it is also not the
kind of near-chance or systematically-wrong result that would justify
opening a separate MobileNet remediation task per this task's own
decision rule ("only if there is strong evidence the classifier itself is
failing"). **No MobileNet retraining was performed or recommended** — the
detector's low recall (86.5% of real close-up panels still never reach
the classifier at all) remains the dominant bottleneck, not the
classifier.

**Detector and classifier problems were not mixed**: this result isolates
detection-dependent accuracy (61.9%, n=21) from whole-image accuracy
(99.4%, n=156, no detection required) — the two are reported and
discussed separately throughout, exactly as Phase 6B/6C already
established.

---

## Phase 9 — Thermal/FLIR modality handling

**Current state**: the application has **no reliable modality
detection**. `models/detector.py`/`models/classifier.py` process any
successfully-PIL-decoded 3-channel image identically, whether it is a
genuine RGB photograph or a false-color thermal/FLIR export. Four of the
16 real validation images are thermal, and the pipeline produced
high-confidence "Hotspot" whole-image predictions for all four
(`docs/ML_REAL_IMAGE_VALIDATION.md`) — plausible pattern-matching on a
thermal palette's bright regions, not a validated real hotspot detection.

**A candidate heuristic was investigated, and deliberately not shipped.**
A simple pixel-saturation signal (fraction of pixels with HSV
saturation > 0.5) cleanly separated the 4 known-thermal images
(0.848–0.982) from the 12 known-RGB images (all ≤0.449) in this exact
16-image sample — a promising, large margin. **This was not implemented
in production**, because doing so would mean calibrating a real,
behavior-affecting threshold directly against this task's own frozen
16-image validation sample — precisely the "do not tune against the
16-image validation set" rule this task itself requires, applied here to
a detection heuristic rather than a model. A heuristic this narrowly
tuned (n=16, no independent validation) also risks the explicitly
forbidden failure mode of **silently discarding valid RGB images** — a
real photo taken under unusual colored lighting (sunset, colored roofing,
saturated foliage) could plausibly trigger a naively-tuned saturation
gate.

**No thermal model was built** (no legitimate thermal training data
exists, and no product requirement for thermal support has been
established), per this task's own explicit instruction not to build one
without both.

**Recommendation for a future phase**: collect a real, independently
labeled sample of thermal vs. RGB images (disjoint from this validation
set) before implementing any modality gate, then validate any heuristic's
false-positive rate on real RGB imagery before shipping it — not before.
This remains a documented, real, currently-unmitigated limitation.

---

## Phase 10 — Production promotion gate

No new trained model exists, so the model-promotion gate (10 conditions)
does not apply — `weights/yolo_solar.pt` remains, and continues to be,
the sole production detection artifact, hash-verified unchanged
throughout this work (`0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`).

The one production **configuration** change made (`confidence_threshold`
0.30→0.45) is justified independently, against its own, smaller bar,
consistent with Phase 6B's original precedent for changing this same
value: real held-out validation-split evidence (never test, never the
16-image sample), an unchanged artifact/hash, and a rollback path that is
a one-line config revert (no retraining, no artifact change required to
undo).

---

## Real-image validation — re-run after remediation

Re-ran `training/evaluation/validate_real_images.py` (itself updated with
the EXIF fix) against the same 16 genuine images, after both the code fix
and the threshold recalibration:

| | Before this phase | After this phase |
|---|---:|---:|
| YOLO detections | 0/16 | **0/16 (unchanged)** |
| Whole-image classification | ran independently, 16/16 | ran independently, 16/16 (unchanged predictions) |

**This is reported honestly, not minimized.** The RGB/BGR fix produced
large, real improvements on two other, larger real evaluation samples
(200 aerial images; 156 ground-truthed close-up images) but made no
measurable difference on this specific small (n=16), unusually
challenging sample — which includes 4 thermal images (guaranteed
non-detections regardless of any RGB-domain fix) and several extreme
close-up/high-resolution/stock-photo images. With only 12 genuine
RGB images in this sample, and the 156-image set's own post-fix rate at
13.46%, a handful of additional real detections firing on this exact
16-image set was plausible but not guaranteed — 16 images is too small a
sample for its 0 count to either confirm or contradict the larger, more
statistically meaningful 156-image measurement. Both results are true and
are reported side by side rather than letting the more encouraging number
overshadow the more sobering one.

---

## Phase 12 — Tests and integrity

- **New/updated tests**: `tests/test_detector.py` (+2, channel-order
  regression), `tests/test_image_utils.py` (+3, EXIF orientation),
  `tests/test_prepare_closeup_dataset.py` (+24, new dataset-prep
  infrastructure), `tests/test_config_validation.py` (1 updated pinned
  value).
- **Full suite**: 1197 passed, 4 skipped (symlink-privilege skips,
  pre-existing, environment-only), 4 failed — **the same 4
  pre-existing, environment-only failures present before this session's
  changes** (they assert zero model artifacts exist on disk, which is
  false on this development machine where real artifacts are installed;
  unrelated to any change made here).
- `python -m compileall` and `python verify_imports.py`: clean (14/14).
- `python scripts/verify_model_artifacts.py --manifest weights/manifest.json`:
  **passed** — both `weights/yolo_solar.pt` and
  `weights/mobilenet_solar_v1.pth` SHA-256 match the manifest, unchanged
  throughout this entire phase.
- XGBoost: confirmed still unavailable and fail-closed in every real run
  this phase (`xgboost_available=False`, typed `ModelLoadError`) — never
  bypassed, never fabricated.

---

## Final assessment

**Root cause of the 0/16 result**: two compounding problems, now
separated. (1) A real RGB/BGR channel-inversion bug in
`models/detector.py`, now fixed, which was silently degrading detection
performance on *all* imagery, aerial and close-up alike. (2) A genuine,
still-unresolved close-up/ground-level domain gap: even with the bug
fixed, the detector still misses the large majority of real close-up
panels, and no legitimate, appropriately-licensed close-up bounding-box
dataset exists to retrain against.

**What was fixed**: the RGB/BGR channel-inversion bug
(`models/detector.py`, `training/evaluation/yolo_threshold_sweep.py`);
missing EXIF-orientation handling (`app.py`, `utils/image_utils.py`,
`training/evaluation/validate_real_images.py`); the now-stale
`confidence_threshold` config value, recalibrated using the same
legitimate, pre-existing, val-only methodology that originally set it.

**What could not be fixed**: the underlying close-up/ground-level
detection domain gap itself. This requires real training data this
project does not have and, per its own licensing/provenance discipline,
will not fabricate or substitute with an unverifiable source.

**Dataset**: none acquired or used for training. Investigated and
formally closed this phase: OpenStat Madagascar (Zenodo 15120875, CC BY
4.0, confirmed nadir/overhead viewpoint — REJECT) and "SPID" (not a real
citable dataset — REJECT). Infrastructure (`prepare_closeup_dataset.py`,
`CLOSEUP_ANNOTATION_TEMPLATE.md`) is built and tested, ready to consume a
legitimate dataset immediately once one is acquired.

**Model comparison**: no candidate model exists. `weights/yolo_solar.pt`
remains the sole production artifact, hash-unchanged
(`0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea`).
Its real, held-out aerial mAP50≈0.74 (Phase 6A, unaffected by this
phase's bug — that measurement used Ultralytics' own correct file-path
loader) is unchanged and remains valid.

**Real-image validation**: 0/16 on the specific small ad hoc sample,
unchanged — reported honestly alongside the much more encouraging
156-image ground-truthed result (2.56%→13.46%) rather than letting either
number stand alone.

**MobileNet**: yes, genuinely exercised on real detector-sourced close-up
crops (21 of them, post-fix) — 61.9% accuracy given detection succeeded,
a real but moderate signal. Not retrained; no strong evidence of
classifier failure was found.

**Thermal**: no reliable modality detection exists. A candidate heuristic
was investigated and deliberately not shipped, to avoid tuning production
behavior against this task's own tiny frozen validation sample. Documented
as an open, real limitation.

**Production status: AMBER — engineering/data-collection blocker
remains.**

This is not a demotion from a prior GREEN, nor a claim the ML problem is
solved. Real, verified engineering progress was made (a genuine inference
bug fixed, with strong measured evidence across two independent real
evaluation samples, plus a legitimately recalibrated threshold and two
tested pieces of reusable infrastructure) — but the close-up/ground-level
detection domain gap, the dominant limitation identified across Phases
6B/6C and this real-image validation, is not resolved. Per this task's
own critical rule, that is reported plainly rather than allowed to look
fixed because a different, real bug happened to also be found and fixed
along the way.
