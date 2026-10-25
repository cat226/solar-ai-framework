# Four-class condition model — error analysis (Phase 15C)

**Purpose: explain WHERE and WHY `weights/yolo_solar_condition_4class_research_v3.pt`
fails, not to improve it.** No retraining occurred in this phase. No
dataset, label, split, checkpoint, or production artifact was modified.
Every claimed failure mode below is backed by counted instances and/or
directly inspected example crops — never asserted from visual plausibility
alone.

## 1. Executive summary

The model's dominant, evidence-backed weakness is **object size, not
class semantics**. Recall falls monotonically from the largest to the
smallest GT-box-size quartile in **every one of the four active
classes**, with the smallest quartile recalling only 11-35% versus
56-86% for the largest — a pattern too large, too consistent, and too
broadly sampled to be an artifact of any one class's vocabulary or a
single family's data-quality issue. Cross-class confusion remains
negligible (0.8-4.9% of false positives per class involve a different
active class's ground truth), confirming the model discriminates the
four conditions when it fires — its failure is coverage on small
instances, not confusable semantics. Training data already contains
abundant small-object examples (12-26% of instances per class fall in
the same "tiny" bucket that hurts recall at test time), which argues
against "collect more data" and toward a training-recipe change that
improves how the model learns from small objects. A secondary,
genuinely inspected label-quality pattern — dense clusters of many tiny,
near-duplicate same-class boxes describing what looks like one physical
region — was found repeatedly and flagged, but is not large enough
(≈4.6% of instances at most) to be the primary driver. **Recommended
next experiment: C — a targeted augmentation/loss intervention aimed at
small-object learning**, not a data-volume or label-cleanup effort (see
§16).

## 2. Frozen baseline (verbatim from the task's brief, cross-checked against the repo)

- Checkpoint: `weights/yolo_solar_condition_4class_research_v3.pt`
- SHA-256 (verified against the file on disk in this phase): `2e26b6afefffff79f43e75c80996f5fcd2aed435df56a33a1bdc5882d3a53c05`
- Dataset: `datasets/ground_pv_research/condition_4class_v1/` (242 test images / 2,212 test GT instances across the 4 active classes)
- Training: YOLO11n, COCO-pretrained, imgsz 640, epochs 200 (best epoch 189), patience 40, seed 42, optimizer auto, cos_lr, Tesla P100, Python 3.12.13, torch 2.7.1+cu126, torchvision 0.22.1+cu126, Ultralytics 8.4.138 — see [[phase15b1_four_class_condition_duration_ablation]] and `docs/CONDITION_MODEL_4CLASS_TRAINING_COMPARISON.md`.
- v3 official test metrics (Ultralytics, conf floor 0.001): P=0.5848, R=0.4607, mAP50=0.4634, mAP50-95=0.2546.
- **Local inference environment for THIS phase differs slightly**: Python 3.12.10, Ultralytics **8.4.118** (vs 8.4.138 used for training — CPU-only local machine, no CUDA). Recorded explicitly per the reproducibility rule rather than glossed over; see §3 for why this makes a small, expected, well-understood difference to the numbers reported here.
- PHYSICAL_DAMAGE has only 78 test instances — every PHYSICAL_DAMAGE conclusion in this report is treated as lower-confidence and flagged as such.

## 3. Overall error statistics — this phase's own matching vs. the official evaluator

This phase's tooling (`training/detection/ground_pv/condition_error_analysis.py`
+ `scripts/analyze_condition_4class_errors.py`) runs its own greedy,
confidence-ordered IoU matcher (IoU≥0.5) at a fixed confidence threshold
(**0.20 — the existing v3 operating threshold, reused only as a stable
counting point, not a new deployment decision**) to get INSTANCE-LEVEL
TP/FP/FN counts, which Ultralytics' `val()` does not directly expose.
This is deliberately a different measurement from, and a cross-check
against, the official AP/precision/recall:

| | GT | TP | FP | FN | Precision (this phase, conf≥0.20) | Recall (this phase, conf≥0.20) |
|---|---|---|---|---|---|---|
| **Overall** | 2,212 | 1,016 | 952 | 1,196 | 0.516 | 0.459 |

**Cross-check against the official evaluator**: this phase's recall
(0.459) closely matches Ultralytics' official test recall (0.461) —
essentially the same number by two different methods. This phase's
precision (0.516) is lower than Ultralytics' reported 0.585. This gap is
expected and was already documented in `docs/CONDITION_MODEL_4CLASS_V1.md`
§8 for v2: Ultralytics' reported P/R figures come from each class's
own best-F1 point on the full precision-recall curve (constrained to be
at or above the given conf floor), not a literal count of "detections
with confidence ≥ 0.20, one-to-one matched." The two methods measuring
similar-but-not-identical things and landing in the same ballpark is a
reasonable cross-check pass, not a contradiction — the count-based
method here is what enables everything else in this report (categorized
FP/FN instances, size bins, family aggregation), which the official
evaluator's summary metrics cannot provide.

## 4. Per-class breakdown

| class | GT | TP | FP | FN | Precision | Recall | AP50 (official) | AP50-95 (official) |
|---|---|---|---|---|---|---|---|---|
| CLEAN | 545 | 328 | 256 | 217 | 0.562 | 0.602 | 0.550 | 0.369 |
| DUSTY | 863 | 347 | 384 | 516 | 0.475 | 0.402 | 0.393 | 0.194 |
| BIRD_DROP | 726 | 298 | 271 | 428 | 0.524 | 0.410 | 0.393 | 0.176 |
| PHYSICAL_DAMAGE | 78 | 43 | 41 | 35 | 0.512 | 0.551 | 0.517 | 0.280 |

(AP50/AP50-95 columns are the official `test_default` figures from
`training_manifest.json`, reused rather than re-derived — this analysis
does not reimplement AP.)

**Confidence distributions** (median, TP vs FP — full quartiles in the
committed `error_analysis_summary.json`):

| class | TP conf median | FP conf median | TP conf p25 | FP conf p75 |
|---|---|---|---|---|
| CLEAN | 0.780 | 0.378 | 0.483 | 0.586 |
| DUSTY | 0.626 | 0.331 | 0.387 | 0.483 |
| BIRD_DROP | 0.542 | 0.319 | 0.385 | 0.454 |
| PHYSICAL_DAMAGE | 0.828 | 0.389 | 0.758 | 0.735 |

Every class's TPs are, on median, roughly 2× more confident than its
FPs — a real, usable signal — but CLEAN/DUSTY/BIRD_DROP all show
meaningful TP/FP overlap in the interquartile range (TP p25 sits below
FP p75), meaning no single higher threshold would cleanly separate
correct from incorrect detections for these three classes. PHYSICAL_DAMAGE
is the exception (see §7, §I).

**FN "nearby signal"**: for each missed GT instance, this phase checked
whether the model produced ANY same-class detection overlapping it at
ANY confidence (down to the 0.001 floor). Total-miss rate (zero signal
whatsoever) varies by class: CLEAN 14/217 (6.5%), DUSTY 26/516 (5.0%),
BIRD_DROP 79/428 (18.5%), PHYSICAL_DAMAGE 11/35 (31.4%, small-sample
caveat applies). Most CLEAN/DUSTY misses are near-misses (the model
produced *something* there, just too weak/loosely-boxed to count) rather
than complete blindness; BIRD_DROP and especially PHYSICAL_DAMAGE have a
higher share of genuinely signal-free misses.

## 5. DUSTY deep dive

1. **What does the model commonly mistake for DUSTY?** Overwhelmingly,
   nothing semantically different — 45.3% of DUSTY FPs are
   "localization" (a loose/offset box over a REAL dusty region that
   just missed the IoU≥0.5 bar) and 22.9% are duplicate/overlapping
   predictions of an already-counted real instance. Only 0.8% of DUSTY
   FPs overlap a GT box of a *different* active class. Inspected example
   (`gallery/dusty_fp/00_Dust-203-…jpg`): a high-confidence DUSTY box
   drawn over a heavily grime-streaked panel edge that IS visually dusty
   — the annotator's own DUSTY/BIRD_DROP boxes sit nearby — a real
   localization miss, not a hallucination. 10.2% are "background /
   non-panel surface" (a genuine false alarm with zero GT overlap of any
   kind) — a real but secondary pattern. One inspected example
   (`gallery/dusty_fp/06_Cleaan-151-…jpg`) shows a 0.80-confidence DUSTY
   box sitting on a region the annotator labeled CLEAN — a genuine
   "clean panel read as dusty" instance, though this specific case was
   geometrically small enough to also be auto-tagged "tiny/low-information
   region"; the two explanations are not mutually exclusive.
2. **What dusty examples does it miss?** Recall drops sharply with
   object size (§8): 13.7% for the smallest size quartile vs 61.6% for
   the largest. This is the single strongest, most consistent pattern in
   the DUSTY FN population.
3. **Are dusty annotations generally small or large?** DUSTY's own Q1/Q2/Q3
   area-fraction cut points are 0.0105 / 0.0201 / 0.0363 (of image
   area) — smaller on average than CLEAN's (0.0127/0.0391/0.0967),
   meaning a "typical" DUSTY box is already a fairly small region even
   at the median.
4. **TP/FP confidence separation?** Real but moderate (median 0.626 vs
   0.331) with meaningful interquartile overlap (§4) — not clean enough
   to fix via thresholding alone (see §I).
5. **Concentrated in particular families?** Yes, strongly. 20 of the 28
   family/class combinations with recall=0.0 are DUSTY (§9) — more than
   any other class — though most of these are small-sample families
   (gt_count 3-6). The two largest-sample poor DUSTY families are
   `fam00168` (20 GT, 0 TP) and `fam00999` (11 GT, 0 TP); the two
   largest-sample strong DUSTY families are `fam00071` (45 GT, recall
   0.756) and `fam00319` (21 GT, recall 0.762) — DUSTY performance is
   genuinely bimodal by family/source, not uniformly mediocre.
6. **Visually repeated/near-duplicate patterns?** Yes — this phase found
   51 same-image GT-box pairs (across all classes, not DUSTY-specific)
   with IoU≥0.8, consistent with a dense-cluster-of-tiny-boxes annotation
   style rather than one box per physically distinct mark (§12).
7. **Insufficient examples, appearance diversity, annotation quality,
   background confusion, localization, or uncertain?** **Evidence points
   most strongly to a combination of localization/small-object weakness
   (dominant, well-supported) plus family-level appearance diversity
   (supported: the family split above is genuinely bimodal, suggesting
   some DUSTY presentations generalize far better than others) — NOT
   insufficient raw example count** (DUSTY has more training instances
   than any other class, 6,050, and still has the worst recall).
   Background confusion and annotation quality are present but
   secondary contributors, not the dominant one. This phase does not
   select a remediation from this alone — see §16 for why a specific,
   single next step is chosen based on the class-agnostic size effect
   instead.

## 6. BIRD_DROP deep dive

1. **Main FP patterns**: localization (38.4%) and duplicate predictions
   (24.7%) dominate, same as DUSTY — but BIRD_DROP has the **highest
   "background / non-panel surface" share of any class (23.6%)**, a
   real, quantified secondary signal.
2. **Main FN patterns**: identical size-driven story — 13.1% recall at
   the tiny quartile vs 56.0% at the large quartile. BIRD_DROP also has
   the highest "total miss" rate among the three non-rare classes
   (18.5% of FNs have zero model signal at any confidence).
3. **Learning real bird-drop appearance, or generic bright/irregular
   regions? Evidence suggests, cannot fully determine.** The elevated
   background-FP share (23.6%, highest of the 4 classes) is consistent
   with the model sometimes triggering on generic bright/irregular
   texture rather than a confirmed dropping — but this is a minority of
   BIRD_DROP FPs (62% are still localization/duplicate around REAL
   instances), so it is not the dominant failure mode. This phase found
   no tooling capable of confirming a texture-based hypothesis directly
   (e.g. saliency/activation inspection was out of scope) — stated here
   as "evidence suggests a secondary contribution," not proven.
4. **Concentrated in specific families?** Yes — `fam00309` (23 GT
   BIRD_DROP instances, 0 TP, recall 0.0) is the single worst
   meaningful-sample family in the entire dataset for any class, and
   it is a **one-image cluster**, not spread across multiple images
   (see §9, §12 — this is the same image implicated in the
   dense-annotation label-quality flag).
5. **Does object size matter?** Yes, decisively — see §8, the same
   monotonic pattern as every other class.
6. **Are annotations consistent?** Largely yes at a coarse level (no
   cross-class GT conflicts were found anywhere in the test set — §12),
   but the dense-tiny-cluster pattern (§12) specifically implicates
   BIRD_DROP's `fam00309` image, where 23 individually-boxed tiny
   regions appear (on visual inspection) to represent one small textured
   area rather than 23 independently meaningful instances — a genuine,
   inspected label-quality concern, not a fabricated one.

## 7. PHYSICAL_DAMAGE deep dive (78 test instances — low-confidence caveats apply throughout)

1. **TP/FP/FN behavior**: TP=43, FP=41, FN=35 (precision 0.512, recall
   0.551 at conf≥0.20/IoU≥0.5). FP count is nearly as large as TP count,
   but see point 2 — most FPs are not false alarms in the usual sense.
2. **Main visual failure modes**: FP categories are dominated by
   **edge/crop artifact (29.3%)** and **localization (26.8%)** —
   together 56%. Only 4.9% ("other defect mistaken for this class") is
   genuine cross-class confusion. One directly inspected example
   (`gallery/physical_damage_fp/00_Physical-damaged-9-…jpg`) shows a
   classic spiderweb impact-crack with a correctly-drawn GT box; the FP
   is a second, loosely-offset box drawn around the SAME real crack —
   i.e. the model correctly located real damage but produced a
   duplicate/imprecise second box, not a hallucination on an undamaged
   panel.
3. **Object-size distribution**: PHYSICAL_DAMAGE shows the same
   monotonic size-recall pattern as every other class (11.1% tiny →
   78.9% large, §8), though with the smallest per-bin sample sizes
   (as few as 4 instances in the "large" bin) — the direction is
   consistent with the other three classes, but exact percentages here
   carry the most sampling noise of any class in this report.
4. **Is the v2→v3 regression systematic or plausibly sampling
   variance?** The regression (precision −0.094, AP50-95 −0.019,
   `docs/CONDITION_MODEL_4CLASS_TRAINING_COMPARISON.md` §4) is **not
   contradicted, but also not specifically explained, by this phase's
   error analysis** — this phase only analyzed v3 in isolation (per the
   task's scope), not a v2-vs-v3 error diff. Given only 78 test
   instances / 18 unique families, and that PHYSICAL_DAMAGE's FP
   pattern here is dominated by benign localization/edge causes rather
   than new semantic confusion, **sampling variance remains a fully
   plausible explanation and this phase found no positive evidence
   against it.**
5. **Enough evidence to justify a targeted intervention specifically
   for PHYSICAL_DAMAGE?** **No — insufficient evidence for a
   PHYSICAL_DAMAGE-specific intervention.** The class shows the same
   size-driven pattern as every other class (suggesting a shared,
   class-agnostic cause, not something unique to physical damage) and
   the v2→v3 regression is plausibly noise. Per the task's explicit
   instruction, this report does **not** recommend adding PHYSICAL_DAMAGE
   data merely because its metric regressed.

## 8. CLEAN analysis

CLEAN is the best-performing class overall (recall 0.602, AP50-95
0.369) but is not exempt from the size effect: 34.6% recall at the tiny
quartile vs 85.9% at the large quartile — the same pattern, on the
class with the most balanced FP-category mix (localization 44.5%,
duplicate 23.0%, edge 14.1%, tiny 7.8%, ambiguous 5.5%, background
4.3%). CLEAN's background-FP rate (4.3%) is the lowest of the three
common classes, meaning the model rarely calls a genuinely-defective
region "clean" by mistake in the false-positive sense measured here —
consistent with the near-zero cross-class-confusion finding threaded
through this whole report.

## 9. Source/family concentration

177 family×class combinations have ≥3 test GT instances (the minimum
this phase requires before ranking a family, to avoid single-instance
noise dominating a "worst family" list). Of those, 28 have recall=0.0 —
20 DUSTY, 6 BIRD_DROP, 2 CLEAN, **0 PHYSICAL_DAMAGE**. Most zero-recall
families are small-sample (gt_count 3-7); the meaningful-sample
(gt_count≥10) worst cases are:

| family | class | GT | TP | recall |
|---|---|---|---|---|
| fam00309 | BIRD_DROP | 23 | 0 | 0.000 |
| fam00168 | DUSTY | 20 | 0 | 0.000 |
| fam00487 | BIRD_DROP | 13 | 0 | 0.000 |
| fam00999 | DUSTY | 11 | 0 | 0.000 |

...while the same families are not uniformly bad across classes:
`fam00309` scores 0.875 recall on its own CLEAN instances (16 GT) and
`fam00168` scores 0.182 on its own BIRD_DROP instances (11 GT) — **the
failure is class-specific within a family/source, not "this whole
source is unusable."** At the other end, 82 family/class combinations
have ≥10 GT instances; the strongest are `fam00015` and `fam00477`
(CLEAN, recall 1.0, 13-15 GT each) and `fam00351` (BIRD_DROP, recall
0.9, 10 GT). **The same failure mode (poor recall) repeats across many
different families rather than being isolated to one or two sources** —
20+ distinct DUSTY families show weak recall, which argues against a
single bad-source explanation and for the class/size-driven explanation
in §5/§8 instead. No split-crossing information was used to produce any
of this — family IDs and recall are computed entirely within the test
split.

## 10. Object-size analysis

**The single strongest, most consistent finding in this report.**
Bins are Q1/Q2/Q3 of each class's OWN test-set GT box area-fraction
distribution (never a hand-picked absolute threshold, so no class's
bins are gerrymandered to fit a narrative — see
`quartile_cuts`/`size_bin_label` in the analysis module and their unit
tests):

| class | tiny recall | small recall | medium recall | large recall |
|---|---|---|---|---|
| CLEAN | 0.346 | 0.588 | 0.616 | 0.859 |
| DUSTY | 0.137 | 0.350 | 0.482 | 0.616 |
| BIRD_DROP | 0.131 | 0.401 | 0.537 | 0.560 |
| PHYSICAL_DAMAGE | 0.111 | 0.526 | 0.727 | 0.789 |

Recall rises monotonically from tiny→large in **all four classes,
without a single exception**, and the gap is large (typically 3-7×
between the tiny and large bins). This is not a class-specific
vocabulary problem and not confined to one family or source — it holds
across the entire test population.

**Is this a data-scarcity artifact?** Checked directly, not assumed:
this phase counted training instances at or below each class's own
test-derived "tiny" cutoff. Tiny instances make up 25.7% of CLEAN,
24.5% of DUSTY, 25.0% of BIRD_DROP, and 12.7% of PHYSICAL_DAMAGE
training instances — i.e. hundreds to over a thousand small training
examples exist per class already. **Small objects are not rare in
training; the model still fails on them at test time.** This is the
evidentiary basis for §16's recommendation.

## 11. Confidence analysis (diagnostic only — no deployment threshold is redefined here)

See §4's confidence table. Summary: every class shows a real median
TP/FP gap (roughly 2×), meaning confidence is a genuinely informative
signal, but for CLEAN/DUSTY/BIRD_DROP the interquartile ranges overlap
substantially (a meaningful share of real detections sit below where a
meaningful share of false ones sit), so **no single higher threshold
would cleanly trade away FPs without also giving up a comparable share
of TPs for these three classes.** PHYSICAL_DAMAGE is different: its TP
p25 (0.758) sits *above* its FP p75 (0.735) — the cleanest separation of
any class — meaning a higher confidence floor specifically for
PHYSICAL_DAMAGE could plausibly cut a meaningful share of its FPs while
sacrificing at most its bottom quartile of TPs. **This is reported as a
diagnostic observation only; this phase does not set, recommend, or
imply a new operating threshold for any class** — that remains
deployment policy outside this phase's scope, and the existing v3
threshold-sweep finding (operating threshold resolves to 0.20,
`docs/CONDITION_MODEL_4CLASS_TRAINING_COMPARISON.md` §5) is unchanged
by this analysis.

## 12. Data-quality flags

Two flag types fired (a third and fourth — "dense/many-box annotation
above P99" and "overlapping GT boxes of different classes" — were
checked and found **zero** instances of in the test split, itself a
mildly reassuring finding: no ambiguous cross-class label conflicts
exist in this split):

- **51 "near-duplicate GT annotation (same class)" pairs** (IoU≥0.8
  between two same-class GT boxes in the same image) — up to ~102 of
  2,212 test instances (≈4.6% upper bound) are involved. Directly
  inspected example: `Cleaan-6-…jpg` (family `fam00309`) contains 23
  individually-boxed BIRD_DROP instances, all in the "tiny" size bin,
  tightly clustered in one ~60×25px region of the image (raw label
  file inspected directly, not inferred) — visually, at the resolution
  available, this region shows a faint regular dotted texture more
  consistent with normal solar-cell busbar/solder-point highlights than
  with obviously distinct bird droppings, though this cannot be
  stated with certainty from a still image at this resolution.
  **Classified as a possible label-quality issue and left completely
  unedited**, per the task's explicit rule. This same image accounts
  for the single worst meaningful-sample family/class result in the
  entire report (`fam00309`/BIRD_DROP, 23 GT, 0 TP, §9).
- **71 "possibly mislabeled/noise (extremely small GT box)" flags**
  (GT box ≤0.1% of image area) — an absolute, class-independent
  threshold, distinct from the per-class relative "tiny" quartile bin
  used elsewhere in this report.
- **A pervasive, separately-noted visual pattern** (not a formal flag
  in the JSON, since it is not yet counted systematically): multiple
  images inspected during this review (`Cleaan-6-…`, `Dust-203-…`,
  `Cleaan-151-…`, `Bird-162-…`) turned out to be **2×2 mosaic
  composites** of 4 similar-but-not-identical panel photos baked into
  one still image file, rather than a single continuous photograph.
  This was not something this phase set out to look for and is
  reported honestly as an observation from a handful of inspected
  images, not a measured prevalence across the dataset — a concrete,
  cheap follow-up would be to count how common this is.
- **At least one source image was observed to carry a baked-in text
  watermark** ("FONTANA California", diagonally overlaid) — a single
  observed instance, not a counted prevalence; noted rather than
  omitted.

## 13. Error gallery inventory

Rendered under `research_runs/ground_pv/condition_error_analysis_v3/gallery/`
(gitignored — regenerable via `scripts/analyze_condition_4class_errors.py`,
never committed since it is bulky and fully reproducible from the frozen
checkpoint + dataset): up to 12 annotated images per class per FP/FN
category, for DUSTY, BIRD_DROP, PHYSICAL_DAMAGE, and CLEAN (in that
priority order). Each image is annotated with GT boxes (green,
`GT:<class>`) and, for FPs, the flagged prediction (red,
`FP <class> <confidence> [<category>]`); FNs show the full GT context
with a text label recording the nearest same-class model signal, if
any. Full per-instance records (image, class, confidence, bbox,
category, family_id, source) are in the (also gitignored)
`records.json` alongside the gallery.

## 14. Evidence-backed conclusions

1. **Recall degrades monotonically with object size in all four active
   classes** — the strongest, most broadly-sampled finding in this
   report (§10).
2. **Cross-class confusion is negligible** (0.8-4.9% of FPs per class)
   — the model discriminates the four conditions when it commits to a
   label; its failure is coverage, not confusable semantics (§4, §6,
   §7, §8, consistent with the confusion-matrix findings already
   documented for v2/v3).
3. **Most false positives are not hallucinations** — 27-45% of FPs per
   class are "localization" (a loose box over a real same-class
   instance) and 5-25% are duplicate/overlapping predictions of an
   already-detected real instance; only "background / non-panel
   surface" FPs (4-24%, highest for BIRD_DROP) represent genuine false
   alarms with zero real-defect overlap.
4. **Small training-instance scarcity does not explain the size
   effect** — 12-26% of each class's training instances already fall
   in the size range that recalls worst at test time (§10). This is
   the key finding that shapes §16's recommendation.
5. **A genuine, inspected label-granularity pattern exists** (dense
   clusters of many tiny near-duplicate same-class boxes, §12) and
   plausibly inflates the "tiny" bin's apparent difficulty for at least
   the specific families where it was found — but it accounts for at
   most ~4.6% of test instances, not enough to be the primary driver of
   the class-wide, size-driven pattern in §10.
6. **Family/source concentration is real but not monolithic** — the
   same class-level weakness (e.g. DUSTY) repeats across 20+ distinct
   families rather than being isolated to one bad source, and a single
   family can score very differently on different classes (§9).
7. **The v2→v3 PHYSICAL_DAMAGE regression is not explained, or
   contradicted, by this phase's evidence** — plausibly sampling
   variance given only 78 test instances (§7).

## 15. What the evidence does NOT establish

- **Not established**: that the dataset's underlying source images are
  systematically insufficient in *diversity* (only volume was checked,
  not diversity/appearance variety) — this phase counted small-instance
  volume, not visual variety within that volume.
- **Not established**: a confirmed root cause for BIRD_DROP's elevated
  background-FP rate (texture-confusion is plausible, not proven — no
  model-internals tooling was used).
- **Not established**: whether the mosaic-composite image pattern
  noticed in §12 is common or rare across the full dataset, or whether
  it materially affects training — only a handful of specific images
  were inspected.
- **Not established**: any causal explanation for the v2→v3
  PHYSICAL_DAMAGE regression — this phase analyzed v3 alone and found
  no evidence pointing at a specific cause, which is different from
  ruling one out.
- **Not established**: that a higher confidence threshold would be a
  net-positive operating-policy change for any class — §11 reports a
  diagnostic asymmetry for PHYSICAL_DAMAGE only, explicitly not a
  recommendation.
- **Not proven, only suggested**: that the specific 23-box cluster in
  `fam00309` (§12) is actually mislabeled rather than a real dense
  cluster of genuinely separate tiny marks — this phase's tooling
  cannot distinguish the two from geometry and a still-image crop
  alone; it is flagged, not corrected, per the task's rules.

## 16. Recommended next experiment

**C — TARGETED AUGMENTATION/LOSS INTERVENTION, aimed at small-object
learning.**

**Cited evidence**: recall is monotonically and substantially worse for
the smallest GT-box-size quartile in every one of the four active
classes (11-35% tiny-bin recall vs 56-86% large-bin recall, §10),
computed from data-driven per-class quartiles (never hand-picked
thresholds) across the full test population (2,212 instances). This
pattern is class-agnostic (present in all four classes despite very
different per-class FP-category mixes and very different training-data
volumes), and it is not explained by data scarcity: 12-26% of each
class's own training instances already fall in the equivalent small-size
range (§10), so the model has substantial small-object training
exposure and still under-recalls on it at test time. Cross-class
confusion is negligible (§4/§14), ruling out "the model needs to better
tell classes apart" as the lever — the gap is in *finding* small
instances at all, not in what it calls them once found. A pure
data-volume intervention (A) is not well-justified by this evidence,
since more of an already-abundant size range is unlikely to fix a
systematic small-object weakness; a training-recipe change targeting
small-object detection specifically (e.g. small-object-oriented
augmentation such as scale-aware copy-paste, or a loss/IoU-weighting
scheme that up-weights small instances, or reassessing whether imgsz=640
gives small defect regions (many well under 1% of image area) enough
effective resolution for a nano-scale backbone) is the more directly
justified lever.

**Why not the alternatives**: **A (data intervention)** is not chosen
because §10 directly shows small instances are already abundant in
training, undercutting the premise that more raw volume would help.
**B (label-quality review)** is not chosen as the PRIMARY intervention
because, while §12's dense-tiny-cluster pattern is real and genuinely
inspected, it accounts for at most ~4.6% of instances — too small a
share to be the size effect's main driver, though it remains a valid,
cheap, independent follow-up worth doing alongside C, not instead of
it. **D (no intervention)** is not chosen because the size-recall
evidence is unusually strong, consistent, and large-sample for a
"insufficient evidence" verdict — this is exactly the kind of clear,
class-agnostic, quantitatively supported pattern the task's own "GOOD"
example describes.

**What this recommendation is not**: it is not a promise that a
small-object-focused training change will fix DUSTY/BIRD_DROP recall to
an acceptable level, and it is not a claim that PHYSICAL_DAMAGE's
regression or the dense-annotation pattern require no attention — both
remain open, documented, lower-priority observations for a future
phase. This phase does not retrain, does not design the specific
augmentation/loss recipe, and does not commit to running it — that
decision belongs to a future, explicitly-scoped experiment.
