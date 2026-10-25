# Active 4-class condition model — `condition_yolo11n_4class_baseline_v1` (Phase 15B)

**This model is not production validated.**
**Electrical Damage is not part of the active RGB condition taxonomy.**
**Hotspot detection requires thermal/IR data.**

Research-only. Branch `feat/ground-pv-research-detector`, not merged, not
deployed. Does not replace, modify, or touch the production detector
(`weights/yolo_solar.pt`), the ground panel detector
(`weights/yolo_solar_ground_research_v1.pt`), the frozen 5-class condition
checkpoint (`weights/yolo_solar_condition_research_v1.pt`), or the Phase
15A CPU smoke checkpoint
(`weights/yolo_solar_condition_4class_research_v1.pt`) — all verified
byte-identical before and after this work (§checkpoint SHA below).

## 1. Dataset

`datasets/ground_pv_research/condition_4class_v1/` — 2,411 images, built in
Phase 15A by removing every ELECTRICAL_DAMAGE annotation (never
relabelled) from the existing 5-class research dataset. Split preserved
exactly from the 5-class dataset's own leakage-clean, family-grouped
split: train 1,686 / valid 483 / test 242. Re-verified before this
training run: zero missing images, zero malformed labels, zero invalid
class IDs, zero cross-split exact duplicates, zero cross-split
near-duplicate families. See [[phase15a_four_class_condition_pipeline]]
and docs/PANEL_CONDITION_PIPELINE.md.

Provenance grade **C** — personal, non-commercial research only. The
underlying images ultimately trace to a Kaggle dataset with an Unknown
license. Do not redistribute, publish, use commercially, or claim
verified image rights.

## 2. Taxonomy

```
0 = CLEAN
1 = DUSTY
2 = BIRD_DROP
3 = PHYSICAL_DAMAGE
```

**Electrical Damage is not part of the active RGB condition taxonomy** —
removed entirely (not present in this dataset at all), never relabelled
as PHYSICAL_DAMAGE or anything else. SNOW and generic "DEFECTIVE" are
likewise absent. **Hotspot detection requires thermal/IR data** and is
out of scope for any RGB model, this one included.

## 3. Training configuration

| | |
|---|---|
| architecture | YOLO11n (Ultralytics), COCO-pretrained base — never `weights/yolo_solar.pt` |
| dataset | `edithstark/solar-ai-ground-condition-4class-research-v1` (Kaggle, private) |
| dataset content SHA-256 | `6637e6361343e7bb17d1703e705bffc0cb733a5428f48ce681fa28bc8826bf92` |
| split manifest SHA-256 | `600752137a4cbf8b24f506339b662ea0d0dbb67d1fc897f52fef83d6eb1d893e` |
| data.yaml SHA-256 | `0731819484bb3b07c87a2ced8e79997d380fa5a34a4701fdb3845c8fa03cffe4` |
| imgsz | 640 |
| epochs (budget / actual) | 100 / 100 (patience=20 never triggered) |
| best epoch | 97 |
| batch | -1 (Ultralytics auto) |
| optimizer | auto (AdamW-family) |
| cos_lr | true |
| seed | 42 |
| pretrained | true |

**Augmentation** (conservative, no semantics-erasing transforms):
`hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.3,
shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0,
copy_paste=0.0`.

## 4. Environment

| | |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| Ultralytics | 8.4.138 |
| CUDA | 12.6 |
| GPU | Tesla P100-PCIE-16GB (16.9 GB, compute capability 6.0) |
| Training duration | 2,797.4 s (~46.6 min) |

## 5. Overall test-set metrics (held-out, never used for training/validation/threshold tuning)

Standard reporting (conf floor 0.001, threshold-independent mAP):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.592 | 0.440 | 0.444 | 0.236 |

For comparison, the frozen 5-class predecessor
(`condition_yolo11n_baseline_v1`, [[ground_pv_condition_model_baseline]])
scored test mAP50 **0.472** and was itself flagged UNDERFIT. This 4-class
run (mAP50 0.444) is in the same modest range — removing ELECTRICAL_DAMAGE
did not meaningfully change overall difficulty; the remaining four classes
were already the harder part of the task.

## 6. Per-class test-set results (at the chosen operating threshold, conf=0.20 — see §8)

| class | Precision | Recall | AP50 | AP50-95 | test instances | test images |
|---|---|---|---|---|---|---|
| CLEAN | 0.627 | 0.518 | 0.468 | 0.315 | 545 | 95 |
| DUSTY | 0.467 | 0.324 | 0.257 | 0.124 | 863 | 144 |
| BIRD_DROP | 0.595 | 0.342 | 0.288 | 0.126 | 726 | 100 |
| PHYSICAL_DAMAGE | 0.679 | 0.577 | 0.493 | 0.291 | 78 | 30 |

## 7. Rare-class analysis — PHYSICAL_DAMAGE and BIRD_DROP (never hidden behind aggregate mAP)

**PHYSICAL_DAMAGE** is the rarest active class (314 images / 725 instances
across all splits, only 78 instances / 30 images in test). Despite this,
it has the **best recall (0.577) and best AP50-95 (0.291) of all four
classes** — better than DUSTY or BIRD_DROP, which have 8-10x more training
data. The confusion matrix (§9) shows PHYSICAL_DAMAGE also has the
**lowest false-positive-on-background rate (3%)** of any class. This is a
genuinely encouraging, non-obvious result: scarcity did not translate into
the weakest performance here — physical damage instances (cracks,
fractures) are apparently more visually distinct than diffuse
dust/dropping marks, at least given the label set present in this
provenance-grade-C source data. **This should not be over-interpreted**:
78 test instances is a small sample, so this recall estimate carries real
variance, and it says nothing about damage severity or types not
represented in the source data.

**BIRD_DROP** recall (0.342) and AP50-95 (0.126) are weak, comparable to
DUSTY's weakness, despite having the second-most training data (5,500
instances). Do not treat a BIRD_DROP miss as reliable evidence of a clean
panel.

## 8. Threshold analysis (validation-only; test set touched exactly once, afterward)

Swept conf ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60} on the
**validation split only**:

| conf | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 0.20 | 0.551 | 0.467 | 0.393 | 0.219 |
| 0.30 | 0.638 | 0.409 | 0.359 | 0.207 |
| 0.40 | 0.721 | 0.356 | 0.324 | 0.192 |
| 0.50 | 0.780 | 0.313 | 0.290 | 0.179 |
| 0.60 | 0.833 | 0.261 | 0.247 | 0.161 |

**Honest limitation of this sweep**: F1 (harmonic mean of P/R) is
monotonically *highest* at the *lowest* tested threshold (0.20) and
decreases steadily as confidence rises — the sweep never found an interior
peak. The selection rule (smallest conf within 2% of best F1) therefore
picked **0.20**, the edge of the tested range, not a genuine sweet spot.
**A real optimum may lie below 0.20**, which this sweep did not test — a
concrete, cheap follow-up for any future run of this exact recipe.

Test-set evaluation at the chosen operating threshold (0.20) is the table
in §5/§6 above (`test_operating_metrics`) — Precision/Recall are
numerically identical to the default-threshold test evaluation here
(Ultralytics reports P/R at each class's own best-F1 point on the curve,
which was already above 0.20 in both cases); mAP differs because the
default evaluation integrates the full curve down to near-zero confidence
while the operating-threshold evaluation truncates it.

Default documented threshold for future qualitative/deployment runs of
this specific checkpoint: **`detection_floor=0.25`, `predicted_threshold=0.45`**
(the two documented in docs/PANEL_CONDITION_PIPELINE.md, reused for
consistency across the project rather than re-tuned per checkpoint).

## 9. Confusion / error analysis

Full images: `training/cloud/runs/condition_yolo11n_4class_baseline_v1/output/confusion_matrix.png`
and `confusion_matrix_normalized.png` (not committed — see §Retrieve).

**Zero confusion between the four active classes.** No CLEAN instance was
ever predicted as DUSTY, no BIRD_DROP as PHYSICAL_DAMAGE, etc. — the
entire off-diagonal error mass is either:
- **missed detections** (recall gap — the class column shows a
  "background" row entry): CLEAN 48% missed, DUSTY 63% missed, BIRD_DROP
  65% missed, PHYSICAL_DAMAGE 39% missed, or
- **background false positives** (a region with no real object gets
  labelled): 35% of background regions predicted CLEAN, 32% DUSTY, 30%
  BIRD_DROP, only **3%** PHYSICAL_DAMAGE.

**Interpretation**: the model can genuinely tell the four classes apart
when it commits to a label — its dominant failure mode is *coverage*
(missing real instances, especially DUSTY and BIRD_DROP at ~63-65% miss
rate) and *background false positives* on the three common classes, not
*confusable semantics* between classes.

## 10. Six real-world images — qualitative OOD reference only

`D:\solar_ai_training_scratch\Solar Real Images` — NOT used for training,
validation, or threshold tuning. **No ground truth exists for these
images; no accuracy is claimed or computable.**

Pipeline: full image → dynamic panel detector (unchanged,
`yolo_solar_ground_research_v1.pt`) → panel crop → this 4-class model.

| image | panels | per-panel result |
|---|---|---|
| …56 | 1 | P1: det 0.874, **predicted BIRD_DROP** (0.729) |
| …57(1) | 2 | P1: det 0.505, no_prediction · P2: det 0.866, uncertain BIRD_DROP (0.390) |
| …57 | 2 | P1: det 0.848, **predicted CLEAN** (0.616) · P2: det 0.676, no_prediction |
| …58 | 1 | P1: det 0.580, uncertain CLEAN (0.303) |
| …59(1) | 2 | P1: det 0.635, **predicted CLEAN** (0.578) · P2: det 0.589, uncertain CLEAN (0.382) |
| …59 | 2 | P1: det 0.665, uncertain DUSTY (0.251) · P2: det 0.750, **predicted CLEAN** (0.834) |

Totals: 10 panels detected (identical to Phase 14A/15A — the panel
detector is unchanged), **4 predicted / 4 uncertain / 2 no_prediction**.
Zero PHYSICAL_DAMAGE predictions in this small sample (consistent with the
class's rarity and no obvious physical damage described in the prior
qualitative notes for these photos).

**One notable, unverified-but-plausible match**: image …56 was previously
described (Phase 14A holdout notes, [[ground_pv_research_detector]]) as
"heavily marked with bird droppings" — the model's top prediction on that
exact panel is BIRD_DROP at 0.729 confidence. This is an encouraging
qualitative anecdote, **not evidence of accuracy** — one match with no
verified ground truth proves nothing statistically, and every other
prediction here is equally unverified.

`no_prediction` was never treated as CLEAN anywhere in this report or in
the underlying pipeline.

## 11. End-to-end integration result vs. condition-model result (kept separate)

**A. Condition-model held-out performance** (§5-9 above) is a property of
this checkpoint alone, measured on `condition_4class_v1/test` crops that
were already correctly localized.

**B. End-to-end qualitative performance** (§10) additionally depends on
the dynamic panel detector's own accuracy — a missed panel upstream (see
docs/DYNAMIC_PANEL_DETECTION.md's documented image-4/image-6 failure
modes) can never be classified by this condition model, no matter how
good the condition model itself is. The two are separate limitations that
compound in deployment; **condition-model mAP is not a claim about
complete-pipeline real-world accuracy**, and this report makes no such
claim.

## 12. Limitations

- **The training run had not converged at epoch 100.** mAP50 was still
  rising at the last logged epoch (0.4578→0.4643→0.4648 across epochs
  91/98/100) with `patience=20` never triggering — this checkpoint is
  genuinely undertrained relative to what the same recipe could likely
  achieve with more epochs, not a converged ceiling for this architecture/
  dataset combination.
- **Recall is weak across the three common classes** (32-52%) — roughly
  half or more of real DUSTY/BIRD_DROP instances go undetected at the
  documented operating point.
- **DUSTY has the worst AP50-95 (0.124)** despite being the single most
  common class (6,050 train instances) — likely because dust deposits are
  visually diffuse with fuzzy, hard-to-box-tightly boundaries, not a data-
  volume problem.
- **The threshold sweep found no interior optimum** — see §8; the true
  best operating point may be below the tested range.
- **Provenance grade C**, personal non-commercial research use only,
  unverified image-level rights — unchanged from the 5-class dataset.
- **Panel-detector-dependent** — see §11B; this model's own metrics say
  nothing about complete-pipeline field accuracy.
- **No condition ground truth exists for the six real-world reference
  photos** — §10 is illustrative only.

## 13. Research-only status

`weights/yolo_solar_condition_4class_research_v2.pt`. **Not production
validated. Not promoted. Not wired into `models/`, `services/`, or the
Streamlit app.** The production detector, the ground panel detector, the
5-class condition checkpoint, and the Phase 15A smoke checkpoint all
remain frozen and untouched (SHA-256 re-verified — see
tests/test_ground_pv_frozen_artifacts_integrity.py, which now enforces
this automatically going forward).

## 14. Checkpoint

- Path: `weights/yolo_solar_condition_4class_research_v2.pt`
- SHA-256: `2dcf6122c18590f33f3b44ad88e054d409509274193977046ad2076d368de28e`
- `last.pt` SHA-256: `87df2c75edb589be0abee9711dd6aa5f93ab887f1f9dbb7e4101c0f631decdbc`
- Verified via `scripts/verify_condition_4class_checkpoint.py` against the
  retrieved `training_manifest.json`: internally consistent (recorded
  hash matches the actual downloaded file, all reproducibility fields
  present, taxonomy exactly the active 4-class set).

## 15. Go / No-Go grade: **C — weak research candidate**

**Why not D (unusable)**: the model demonstrably learned the task. Zero
confusion between active classes (§9) shows real semantic discrimination,
not noise. PHYSICAL_DAMAGE — the rarest class — achieved the best recall
and lowest false-positive rate of any class, a genuine, non-obvious
positive signal. The qualitative BIRD_DROP match on a previously-described
image (§10) is a plausible (not proven) anecdotal success.

**Why not B or A**: overall test mAP50-95 (0.236) and per-class recall
(32-58%) are modest; the two most common classes (DUSTY, BIRD_DROP) have
weak localization quality (AP50-95 ≈ 0.12); and — most importantly — the
run was cut off while metrics were still improving, meaning THIS
checkpoint under-represents what the recipe is capable of. A model that
hasn't finished learning cannot be called a solid usable baseline yet,
even though the signal it does show is real.

## 16. Recommendation for Phase 15B (continued) / 15C

1. **Re-run with a larger epoch budget** (e.g. 200-300 epochs, or
   `patience` raised so training runs until it genuinely plateaus) — the
   single cheapest, most likely lever given mAP was still rising at the
   epoch-100 cutoff.
2. **Extend the threshold sweep below 0.20** to find whether a true
   interior optimum exists.
3. **Investigate DUSTY/BIRD_DROP localization specifically** (AP50-95 ≈
   0.12) — possibly via examining false-positive/false-negative crops
   directly (`val_batch*_pred.jpg` vs `val_batch*_labels.jpg` in the
   retrieved run output) before assuming more data alone would fix it.
4. Continue treating PHYSICAL_DAMAGE's small sample size (78 test
   instances) as a variance risk even though its point estimate looks
   good — more data for this class would tighten confidence, not
   necessarily change the estimate.
5. Do not promote this or any successor checkpoint to production without
   a proper held-out evaluation against genuinely independent,
   human-verified deployment-domain images (the six-image reference set
   is not, and cannot become, that evaluation).
