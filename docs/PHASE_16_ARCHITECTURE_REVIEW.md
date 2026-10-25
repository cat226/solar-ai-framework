# Phase 16 — Panel-First Intelligence: Architecture Review

Status: **review only — no training run, no production artifact touched, no
commit/push performed.** Written after reading the full repository (panel
detector, dynamic panel detection, condition pipeline, taxonomy, frontend/
backend wiring, existing experiment reports) and `SOLAR_AI_PROJECT_CONTEXT.md`
in full.

The single most important finding of this review, established below with
evidence, is that **the literal "Experiment 16-A" the project brief asks for
has already been built and run** — not as a one-off test, but as the
project's own standing architecture. Re-running it would waste compute. The
real open question is narrower and more useful than the brief assumes; §14
proposes what to do instead.

---

## 1. What parts of Solar AI are already strong?

- **Dynamic panel detection** (`training/detection/ground_pv/dynamic_panel_detection.py`,
  Phase 14A). Pure-geometry, no hard-coded panel count, deterministic P1..PN
  ordering (verified: 20 shuffles of the same 8 detections → identical
  order), rich diagnostics (overlap, containment, size, aspect, isolation,
  row-geometry) that flag suspicious geometry without ever deleting a
  detection. 37 dedicated tests. This is genuinely good, reusable
  infrastructure — the panel *localization* problem is architecturally
  solved regardless of what happens to condition analysis.
- **The no-fabrication discipline itself.** `condition_taxonomy.py`'s
  `assert_active_taxonomy()` fails closed on any non-4-class model/dataset.
  `panel_condition_pipeline.py`'s `no_prediction` vs `uncertain` vs
  `predicted` status is never collapsed into "Clean". The XGBoost
  unavailable-state handling in `services/pipeline.py` is correct and
  consistently applied. This is not a minor virtue — it is what makes every
  other number in this project trustworthy enough to make decisions from.
- **Experiment hygiene.** Phases 15D-A (resolution) and 15E-A (augmentation)
  each changed one variable, kept the same split/seed/test-set, defined a
  decision rule up front, and *stopped* when the evidence didn't support
  continuing. That discipline is exactly right and should be the template
  for whatever Phase 16 becomes.
- **The four-class taxonomy design itself.** Excluding ELECTRICAL_DAMAGE
  (semantically mislabeled as cell cracks) and HOTSPOT (thermal-only) rather
  than fabricating a mapping is correct and should not be revisited.
- **`docs/CONDITION_MODEL_4CLASS_V1.md` §10 / §12.** The existing
  documentation already states, in writing, that DUSTY's boundaries are
  "visually diffuse with fuzzy, hard-to-box-tightly boundaries, not a
  data-volume problem." That single sentence turns out to be the most
  important clue in the whole codebase for what to do next (see §5).

## 2. What parts are fundamentally weak?

- **The ground panel detector's recall on dominant/foreground panels.**
  `datasets/ground_pv_research/reports/real_world_holdout.md` (uncommitted,
  read in full): image 4's large foreground panel is missed entirely; image
  6 produces two overlapping *group* boxes instead of per-panel boxes. This
  is a real-world domain gap, not a metric artifact — held-out test mAP50 is
  .865 but real-photo behavior is visibly weaker.
- **`cracked` false positives on off-panel texture.** Two of six real
  photos fire `cracked` on black roofing membrane, never on a panel. A
  cheap, well-understood fix (reject defect boxes with ~0 IoU against any
  panel box) exists and has not been applied.
- **The condition model's per-box recall**, confirmed three times now
  (baseline v2, resolution v3, augmentation v4): mAP50 ≈ 0.44–0.45,
  recall ≈ 0.44, and two independently-controlled interventions both failed
  to move it. This is now well-evidenced, not a one-off weak run.
- **DUSTY is the worst-performing class despite being the most common one**
  (AP50-95 = 0.124, 6,050 train instances). Under a bounding-box-detection
  formulation, more data does not fix this, because — per the project's own
  documentation — the underlying defect (a diffuse dust film) does not have
  a tight box to learn in the first place. This is evidence about the
  *problem formulation*, not the *model capacity*.

## 3. What should remain unchanged?

- The production aerial detector and its pipeline — validated for its
  domain, correctly kept separate, not touched by any of this.
- `dynamic_panel_detection.py`'s geometry/ordering/diagnostics logic.
- The active 4-class taxonomy (`condition_taxonomy.py`) — do not add
  ELECTRICAL_DAMAGE, HOTSPOT, SNOW, or DEFECTIVE back.
- The `predicted` / `uncertain` / `no_prediction` status contract in
  `panel_condition_pipeline.py` — this is the right abstraction regardless
  of whether the underlying model stays a detector or becomes a classifier.
- Frozen-artifact and provenance discipline.
- The decision to **not** yet wire any ground-domain research model into
  `app.py` (see §4 — it currently isn't, and that's correct given current
  evidence).

## 4. What should be redesigned?

**Not the panel-first architecture — it already exists.** What needs
redesigning is narrower:

1. **The representation used for condition inference on a panel crop.**
   Bounding-box detection is a poor fit for classes whose defining feature
   is a diffuse, boundary-less texture (DUSTY, and arguably CLEAN as its
   complement). It is a reasonable fit for classes with a genuine
   localized extent (BIRD_DROP, PHYSICAL_DAMAGE). Forcing all four classes
   through one detection head is very likely why targeted interventions
   (resolution, augmentation) both failed to move the needle — they were
   aimed at making small boxes easier to find, but for DUSTY there may not
   be a "correct" box to find.
2. **The cracked-on-roofing false-positive mode** — a cheap geometric guard
   (§2), independent of any retraining.
3. **The absence of an image-quality gate** (Section 43 of the project
   context) — currently nothing distinguishes "the model is uncertain about
   a good photo" from "this photo is a bad capture" (occluded, poor angle,
   dominated by one out-of-frame-edge panel). Given image 4's failure mode
   is arguably a framing/quality issue as much as a detector weakness, this
   gate would pay for itself immediately.
4. **The unresolved gap between the ground-domain research stack and the
   shipped product** — see §4a.

### 4a. Frontend/backend integration gap (new finding, not previously documented)

`app.py` → `services/pipeline.py::run_pipeline` uses **only**
`weights/yolo_solar.pt` (the production *aerial* detector) and
`weights/mobilenet_solar_v1.pth` (the 3-class CLEAN/DUSTY/HOTSPOT
whole-image classifier). None of Phase 14A (dynamic panel detection), Phase
15A (4-class condition pipeline), or any ground-domain checkpoint is
imported anywhere in `app.py`, `services/`, or `pages/`. The entire
ground-domain research line (dynamic detector, condition taxonomy, condition
models v1–v4) exists only as standalone modules, scripts, and tests — it has
never been reachable from the actual product a user would open.

This is not a bug to fix reflexively: given §2's evidence, wiring the current
research stack into the live product now would violate the project's own
"research vs. production" boundary (§53 of the context doc). It is, however,
an important fact for scoping Phase 16 — **the six real images have never
been run through the actual `app.py`**; every real-world evaluation to date
(`real_world_holdout.md`, `CONDITION_MODEL_4CLASS_V1.md §10`) went through
standalone scripts. Any future decision to integrate must happen deliberately,
after condition-model quality improves, not as a side effect of a refactor.

## 5. Should condition analysis become panel-first?

**It already is.** `panel_condition_pipeline.py` implements exactly:

```
full image -> dynamic panel detector -> P1..PN -> crop each panel
           -> 4-class condition model on the crop -> per-panel result
```

and this is not a proposal — it is `scripts/analyze_panel_conditions.py`,
covered by `tests/test_ground_pv_panel_condition_pipeline.py`, documented in
`docs/PANEL_CONDITION_PIPELINE.md`, and it has already been run end-to-end on
the six real images with the real v2 checkpoint
(`docs/CONDITION_MODEL_4CLASS_V1.md` §10: 10 panels → 4 predicted / 4
uncertain / 2 no_prediction). The panel-first *hypothesis* (§39 of the
context doc) is not still open — it's implemented, tested, and has one
real-world qualitative result already on record.

What is still genuinely open is **panel-first using which downstream
representation** — detection-on-crop (current) vs. classification-on-crop
(untested) vs. a hybrid split by class (also untested). §6 and §14 address
this directly.

## 6. Condition-engine alternatives, compared with the evidence now available

| Option | Status | Verdict given the evidence |
|---|---|---|
| **A. Panel crop → classifier** | Not built | Best fit for DUSTY specifically (§2's diffuse-boundary finding). Untested — no crop-level classification dataset or model exists yet. Highest-value unexplored option. |
| **B. Panel crop → defect detector** | **Already built and evaluated 3×** (v2/v3/v4) | mAP50 plateaus ~0.44–0.45 across baseline, resolution, and augmentation variants. Known, documented, evidence-backed ceiling for this formulation as currently specified. |
| **C. Panel crop → segmentation + classifier** | Not built | Rejected for now — no trustworthy segmentation-quality source data exists (per §52 provenance principle), and the annotation cost is not justified before A is tried. |
| **D. Panel crop → tiled condition analysis** | Analogous experiment already run in the *panel-localization* domain (tiled inference, Phase 10) and **not adopted** (worse on 3/6 real images). No reason to expect a better outcome applying the same idea one level down (defect-in-crop), and it multiplies compute for a class of problem (diffuse texture) tiling does not obviously help with. |
| **E. Hybrid classifier + detector** | Not built | Correct medium-term target *if* A shows real signal: classify globally (CLEAN/DUSTY as a whole-crop read) and keep the existing detector for spatially localized classes (BIRD_DROP/PHYSICAL_DAMAGE) where "where" still matters. Premature to build before A is evaluated on its own. |
| **F. Temporal change detection** | Not testable yet | Needs ≥2 real captures of the same array over time. Zero first-party temporal data exists (§38). Architecturally sound future direction, no current action. |

No sixth alternative earns its place ahead of A given current data — e.g. a
learned confidence-calibration layer on top of the existing detector was
considered and rejected: calibration cannot manufacture recall the detector
does not have, and §31/§32/§33's error analyses already show the failures
are dominated by missed detections and localization noise, not a
miscalibrated threshold (the F1 sweep in Phase 15B already found no interior
optimum).

## 7. What can be tested with the current six real images?

Already tested (documented, not hypothetical):
- Ground panel detector alone (`real_world_holdout.md`).
- Full dynamic-panel-detector → crop → condition-model pipeline, smoke
  checkpoint (Phase 15A) and real v2 checkpoint (`CONDITION_MODEL_4CLASS_V1.md`
  §10).
- Production aerial detector, for qualitative comparison (`real_world_holdout.md`
  §"Secondary comparison").

Still untested with these six images, and cheap to test:
- Any future classifier variant (§14, Experiment 16-B) — same six images,
  same qualitative-only caveat (no ground truth, so still no accuracy claim).
- A cracked-false-positive geometric guard (§2) — rerun the existing pipeline
  with the guard added, see if the two roofing-membrane FPs disappear.
- An image-quality gate (§4.3) — score the six images for blur/exposure/
  crop-truncation and check whether image 4 (the clearest detector failure)
  also scores poorly on quality, which would separate "bad detector" from
  "bad capture" as the explanation.

## 8. What can be tested without new training?

The single highest-value zero-training test is proposed in §14 as
Experiment 16-A.1: re-score the **existing** v2 condition model's predictions
on the already-held-out test set at the *image* level (does the model's
highest-confidence box match the image's dominant ground-truth class?)
rather than only the *box* level (current mAP50/recall). This requires no
new model, no new data, and no GPU — only re-reading `v2`'s existing
inference output against the existing test manifest. It directly answers
whether the detector's weakness is (a) a localization/box-drawing problem,
which a classifier would very plausibly fix, or (b) a deeper
feature/semantic problem, which a classifier would not fix — and the answer
determines whether Experiment 16-B is worth running at all.

Also without training: the cracked-false-positive guard (§2, a post-
processing rule) and an image-quality gate (§4.3, pure image statistics —
blur variance, exposure histogram, crop-fraction) are both implementable
and testable today.

## 9. What requires new data?

- A crop-level classification-label dataset for Experiment 16-B. This does
  **not** require new photography: it can be **derived** from the existing,
  already-audited, already-leakage-split `condition_4class_v1` dataset by
  assigning each image a single dominant label (e.g. largest-area box class,
  ties broken by highest confidence/count) — reusing real annotations, never
  fabricating a new one. This must be documented explicitly as a *derived*
  label, not an independently verified image-level label, per §52.
- Any true accuracy/precision/recall claim on the six real images — they have
  no ground truth and are qualitative-only by design (§38). Getting
  quantitative real-world numbers requires first-party annotated captures,
  which do not exist yet.
- Persistent panel identity and temporal monitoring (§41–42) — needs ≥2
  captures of the same physical array over time; none exist.
- Thermal/IR branch — needs a thermal camera and thermal data; explicitly
  out of scope until then (§5, §10 of context doc).

## 10. What should be the minimum viable architecture?

The one that already exists, with two additions, not a redesign:

```
full image -> dynamic panel detector (existing, unchanged)
           -> P1..PN + geometry diagnostics (existing, unchanged)
           -> crop each panel (existing, unchanged)
           -> [NEW] image-quality gate -> NEEDS_RECAPTURE short-circuit
           -> condition inference:
                - IF Experiment 16-A.1 shows the failure is localization-only:
                  keep the detector, but ALSO run a per-crop classifier head
                  and prefer its whole-crop read for DUSTY/CLEAN specifically
                - ELSE: keep the detector as the sole engine, label its
                  output more conservatively (status stays predicted/
                  uncertain/no_prediction as today)
           -> per-panel schema (existing status contract, unchanged)
```

No part of this requires touching the production aerial pipeline, and none
of it should be wired into `app.py` yet (§4a) — it stays a research CLI
path until the condition engine clears a materially better bar than the
current ~0.44 mAP50 / Grade C.

## 11. What should be the long-term architecture?

Unchanged from §59 of the project context: capture → quality gate → panel
detector → panel identity/geometry matching → {RGB engine, thermal engine}
→ fusion/confidence → temporal engine → history → dashboard. Nothing found
in this review contradicts that diagram; the work is filling in the RGB
engine box correctly before building anything above it.

## 12. How should persistent panel identity work?

Given the deployment context (fixed pole-mounted camera, ~5-day capture
interval, no first-party temporal data yet), the pragmatic first
implementation is **geometry matching under a fixed-camera assumption**:
match each new capture's P1..PN to the previous capture's panels by
normalized bounding-box center distance / IoU, with a configurable
tolerance for minor camera drift. This reuses `dynamic_panel_detection.py`'s
existing geometry primitives (`_iou`, center normalization) almost
unchanged. Appearance/feature-based matching or full tracking should not be
built now — they solve a problem (camera movement between captures) that
cannot even be observed yet with zero real temporal captures on hand, and
building it speculatively risks exactly the kind of premature complexity
§54 warns against.

## 13. How should thermal analysis eventually integrate?

As a **separate model and separate inference path**, fused only at the
reporting layer — never inferred from RGB, never blended into the RGB
confidence score. Concretely: `panels[i].condition` (RGB) and
`panels[i].thermal` (future, independent) should be sibling fields in the
per-panel schema, each with their own status/confidence, combined only in
display text ("possible thermal issue, requires inspection" per §5/§47 of
the context doc) — never collapsed into one number. No thermal work should
start before thermal hardware/data exists (§9).

## 14. What are the next three experiments?

1. **Experiment 16-A.1 — image-level agreement analysis (no training, no
   new data).** Re-derive, from the existing v2 condition model's already-
   computed test-set predictions and the existing test-set labels, an
   image-level "does the model's top-confidence box match the image's
   dominant ground-truth class" metric, split by class. This is a few hours
   of analysis work (the pattern already exists in
   `condition_error_analysis.py`). It directly decides whether Experiment
   16-B is worth building at all: if image-level agreement is much higher
   than the current box-level mAP50/recall, the failure is
   localization-shaped and a classifier should help substantially,
   especially for DUSTY; if image-level agreement is similarly weak, the
   problem is more semantic and a classifier reformulation should not be
   expected to fix it on its own.

2. **Experiment 16-B — panel-crop classifier, built on a *derived* label
   set (no new photography).** Only if 16-A.1 supports it: build a
   classification-label variant of `condition_4class_v1` by deriving one
   dominant label per image from its existing boxes (§9), train a
   lightweight classifier (reusing the existing MobileNet pattern from
   `models/classifier.py` rather than inventing a new architecture), and
   evaluate it on the *same* held-out test set at the image level, next to
   the same-level score from the existing detector (from 16-A.1) — an
   apples-to-apples comparison the current mAP50 numbers cannot give,
   because mAP50 is a box metric and a classifier produces no boxes.

3. **Image-quality gate (independent of 1–2, no training).** Implement the
   `NEEDS_RECAPTURE` layer from §43 of the context doc as pure image
   statistics (blur variance — the Laplacian-variance helper already exists
   in `panel_condition_pipeline.py`'s crop-quality check and can be reused
   at the whole-image level — exposure histogram, panel-crop-truncation
   fraction already computed by `check_crop_quality`). Test it on the six
   real images; check specifically whether image 4 (the large missed
   foreground panel) flags as a quality/framing problem, which would
   reframe that failure mode from "detector is bad" to "capture guidance is
   needed" — a materially different, and more actionable, conclusion.

## 15. Which experiments should explicitly NOT be run?

Per §54 of the context doc, unchanged and reaffirmed by this review's
findings:
- No further resolution sweeps (1024/1280) — §32 already found 0/4 classes
  materially improved at 960px, at 2.36× cost.
- No further augmentation sweeps — §33 already found 0/4 classes materially
  improved, and PHYSICAL_DAMAGE regressed on all four official metrics.
- No random optimizer/learning-rate changes without a hypothesis.
- No pseudo-labeling.
- No training on the still-blocked datasets (p-pose — awaiting author
  permission; 6rainstorm raw photos — Roboflow-API-gated, unresolved
  rights).
- No wiring of the current condition model (or any ground-domain model)
  into `app.py`/`services/pipeline.py` (§4a) until it clears a materially
  better real-world bar than today's Grade C.
- No claim of quantitative real-world accuracy from the six real images —
  they remain qualitative-only, permanently, absent independently verified
  ground truth.
- No relabeling of ELECTRICAL_DAMAGE and no reintroduction of HOTSPOT into
  the RGB taxonomy.
- No segmentation experiment (Option C) before Option A (classifier) is
  tried — segmentation's annotation cost is not justified while a much
  cheaper reformulation (derived labels, existing architecture) is still
  untested.

---

## Bottom line

> **BUILD Experiment 16-A.1 (the zero-training image-level agreement
> analysis on the existing v2 condition model) NEXT, BECAUSE the literal
> panel-crop → condition-model pipeline the brief calls "Experiment 16-A"
> already exists, is already tested on the six real images
> (`docs/CONDITION_MODEL_4CLASS_V1.md` §10), and already has a
> well-evidenced weakness (mAP50 ≈ 0.44, DUSTY worst-performing despite
> being most common, two independent interventions already ruled out) — so
> the highest-leverage next step is not to build another detector variant,
> it is to determine, at zero training cost, whether that weakness is a
> localization problem a classifier would fix or a semantic one it would
> not, using data and code that already exist in this repository. Everything
> else in this document (Experiment 16-B, the image-quality gate, persistent
> identity, thermal fusion) is real future work, but 16-A.1 is the cheapest
> experiment that could change what Phase 16 does next, and it should run
> before any new model is trained.**
