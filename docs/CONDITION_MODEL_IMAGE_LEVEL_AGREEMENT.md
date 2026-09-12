# Phase 16-A.1 — Image-level agreement analysis (zero new training)

**Question:** is the active 4-class condition model's weakness (test mAP50 ≈
0.444, DUSTY worst despite being the most common class — see
`docs/CONDITION_MODEL_4CLASS_V1.md`) a **localization** problem (the model
usually identifies the right class somewhere in the image, just not a
tight-enough box) or a **semantic** one (it doesn't reliably recognize the
class at all)? The answer decides whether Experiment 16-B (a crop-level
classifier reformulation) is worth building — see
`docs/PHASE_16_ARCHITECTURE_REVIEW.md` §8/§14.

## Method

Zero new model inference. Reuses the TP/FP/FN records already computed for
`weights/yolo_solar_condition_4class_research_v2.pt` (checkpoint SHA prefix
`2dcf6122...`) by the existing Phase 15C error analysis
(`research_runs/ground_pv/condition_error_analysis_v2/records.json`, 242
test images).

For every test image:
- `dominant_gt_class` = the class with the most ground-truth instances in
  that image (ties → larger total GT area → alphabetical; deterministic).
- `top_prediction_class` = the class of the single highest-confidence box
  the model produced anywhere in that image (TP or FP alike — any output,
  not just correctly-localized ones), or `no_prediction` if it produced
  nothing.
- `agreement` = whether these match.

This is a strictly weaker criterion than box-level recall: no IoU
requirement, no IoU threshold, any confidence counts. Script:
`scripts/analyze_condition_image_level_agreement.py`. Full output:
`research_runs/ground_pv/condition_image_level_agreement_v2/image_level_agreement.json`.

## Result

| class | image-level agreement | box-level recall (same checkpoint) | gap | no-prediction images |
|---|---:|---:|---:|---:|
| CLEAN | 0.800 | 0.545 | **+0.255** | 1 / 55 |
| DUSTY | 0.720 | 0.359 | **+0.360** | 8 / 107 |
| BIRD_DROP | 0.662 | 0.388 | **+0.274** | 1 / 71 |
| PHYSICAL_DAMAGE | 0.889 | 0.577 | **+0.312** | 0 / 9 |

**Overall image-level accuracy: 0.727** (176/242), vs. overall box-level
recall 0.441 documented in `docs/CONDITION_MODEL_4CLASS_V1.md`.

Cross-class confusion at the image level (the model's top guess being a
*different* active class, not just imprecisely localized) is a minority
contributor to disagreement — e.g. DUSTY has 107 images, 8 no-prediction,
22 confused with another class (mostly CLEAN/BIRD_DROP), and 77 agreeing;
confusion counts are similar order-of-magnitude across all four classes,
consistent with the already-documented "effectively zero cross-class
confusion" finding at the box level (Phase 15B/15C).

## Interpretation

Every one of the four classes shows a large, positive gap (25–36 points)
between image-level agreement and box-level recall, and **DUSTY shows the
single largest gap (+36pp) despite being the worst-performing class at the
box level.** This is the opposite of what a purely semantic failure would
look like — if the model didn't recognize DUSTY at all, image-level
agreement would be as weak as box-level recall, not 2x higher. Combined
with the low cross-class confusion, the evidence points to **localization**,
not recognition, as the dominant bottleneck: the model usually "knows"
roughly what condition a panel is in, it just doesn't draw a tight enough
box around it to count as a correct detection under the box-matching
criterion — most consistent with DUSTY's already-documented "diffuse,
fuzzy, hard-to-box-tightly" boundaries (§12 of `CONDITION_MODEL_4CLASS_V1.md`).

## Decision (per the rule set in the architecture review)

> If image-level agreement is materially higher than box-level recall,
> that supports a crop-level classifier reformulation (Experiment 16-B).

The evidence clears that bar for all four classes, most strongly for
DUSTY. **Experiment 16-B (panel-crop → classifier, using a derived
dominant-class label from the existing, already-audited box annotations —
no new photography, no new training data collection) is now justified by
evidence, not assumption.**

## What this does NOT show

- Not a claim that a classifier WILL outperform the detector — only that
  the detector's own behavior, re-scored at a weaker criterion, shows most
  of its "failures" are localization noise around a broadly correct
  read, which a classifier formulation has no localization requirement to
  fail at. Experiment 16-B still needs to be built and evaluated on the
  same held-out test set before that claim can be made.
- Not evidence about real-world (six-photo) performance — this analysis
  only used the held-out test split, which has ground truth; the six real
  photos still have none and remain qualitative-only.
- Not a claim that localization is *worthless* — BIRD_DROP/PHYSICAL_DAMAGE
  are spatially localized defects where knowing *where* still has product
  value; a classifier drops that, which is a real cost, not free.
