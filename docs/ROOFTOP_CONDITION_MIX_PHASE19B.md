# Rooftop condition classifier MIX retrain — Phase 19-B — NOT ADOPTED

Research only. Branch `feat/ground-pv-research-detector`, not merged. Answers
the second half of the owner's Phase 19 request (retrain the CLEAN/DUSTY/
BIRD_DROP/PHYSICAL_DAMAGE condition model on the real rooftop photos).
**Result: a real, measured regression on the actual benchmark. Not adopted.
The active classifier (`weights/mobilenet_condition_4class_research_v1.pth`,
Phase 16-B, 0.752 test accuracy) is unchanged.**

## 1. Why labels first, not a straight retrain

Panel *location* is a geometric, visually-obvious property a segmentation
model + visual review can reasonably read off a photo (Phase 19, Section 3).
Panel *condition* is not: dust is often invisible at this resolution
([[condition_model_image_level_agreement]]), and only the property owner
knows their panels' real history. With no more photos available and the
owner unable to hand-label right now, I did the only honest thing available:
visually reviewed all 242 exported crops myself, disclosed as AI review (not
owner-verified), and recorded only what was actually visible — see the Phase
19 commit for the full account. Result: 235 CLEAN, 3 BIRD_DROP, 2 DUSTY,
1 PHYSICAL_DAMAGE, 1 unusable (skipped).

`scripts/build_rooftop_condition_classifier_dataset.py` correctly refused
(exit 2) to build a standalone classifier from this — nowhere near enough
non-CLEAN examples for a real held-out benchmark. That refusal is itself the
honest finding: this rooftop, in these photos, just doesn't show much
condition diversity.

## 2. MIX strategy (same idea that worked for the panel detector) — tried, and rejected

Packaged the 241 labelled crops as `edithstark/solar-ai-rooftop-condition-topup-v1`
(`scripts/build_rooftop_condition_mix_package.py`) to blend into the much
larger, balanced `solar-ai-ground-condition-4class-classifier-v1` (Susan-
derived) training set — oversampled in train only, selection/early-stopping
kept on the ORIGINAL Susan-derived valid split so the comparison stays fair.

**Real bug found and fixed before any real number came out**: the rooftop
top-up folder legitimately has zero examples of some classes in some splits
(e.g. no DUSTY at all in train). `torchvision.datasets.ImageFolder` raises
`FileNotFoundError` on an empty class directory, and creating an empty
placeholder directory doesn't fix it either — it still raises. The real fix
was a small custom `Dataset` that borrows the BASE dataset's fixed
`class_to_idx` instead of letting either side infer classes from what happens
to be present on disk (`scripts/make_rooftop_condition_mix_notebook.py`).
Verified locally before spending a second Kaggle run.

**Results — two oversample factors tried, both regress the real benchmark:**

| | Susan-derived TEST accuracy (n=242) | vs. baseline (0.752) |
|---|---|---|
| baseline (`..._research_v1.pth`, Phase 16-B) | 0.752 | — |
| MIX, oversample x10 | 0.715 | **-0.037** |
| MIX, oversample x1 | 0.731 | **-0.021** |

Per-class recall, oversample x1 vs. baseline: BIRD_DROP 0.690 vs 0.676
(+0.014, noise-level), **CLEAN 0.673 vs 0.764 (-0.091)**, DUSTY 0.794 vs
0.794 (unchanged), **PHYSICAL_DAMAGE 0.667 vs 0.778 (-0.111)**. Full
confusion matrices and manifests:
`datasets/ground_pv_research/rooftop_condition_v1/reports/mix_oversample{10,1}/training_manifest.json`.

Lowering the oversample factor (my hypothesis: x10 over-amplified the
rooftop set's 97%-CLEAN skew) recovered about half the gap but did not close
it. Per the project's own established discipline (one clearly-motivated
ablation, report, stop — see [[phase15d_a_resolution_ablation]],
[[phase15e_a_augmentation_ablation]]), this phase stops here rather than
sweeping further oversample values chasing a number.

**Plausible honest explanation, not further chased**: even the "light" blend
adds 149 visually-uniform rooftop CLEAN crops (bright, clean, one camera,
one lighting condition) against only 1 new PHYSICAL_DAMAGE and 3 new
BIRD_DROP examples. On the already CLEAN-heavy base training set this likely
narrows the model's learned notion of "what CLEAN looks like" toward this
one rooftop's visual style, at the cost of generalizing to the far more
visually diverse Susan-derived CLEAN/PHYSICAL_DAMAGE examples the real
benchmark is built from. This is a class-imbalance/domain-narrowing effect,
not a bug — small, skewed top-up data can measurably hurt a class-imbalanced
classifier even when every added label is correct.

## 3. Rooftop "real-world check" numbers — explicitly not a benchmark

The rooftop crops the model was trained ON (train/valid, oversampled) score
near-perfectly (0.94–1.0) — this is expected and uninformative (it is
graded on data it just memorized), not evidence of anything. The one
genuinely held-out rooftop slice (test, 48 crops: 46 CLEAN + 2 DUSTY, never
seen in training) scored 44/46 CLEAN correct and 1/2 DUSTY correct at
oversample x1 — a sample of 2 DUSTY examples is far too small to read into
either way, positive or negative, and is reported here only for completeness,
never as a validated capability.

## 4. Decision

**Not adopted.** `services/ground_pipeline.py`'s `CLASSIFIER_WEIGHTS` remains
`weights/mobilenet_condition_4class_research_v1.pth` — unchanged, still the
active condition classifier the app runs. Both experimental checkpoints are
kept for the record, never promoted:
`weights/mobilenet_condition_4class_rooftop_mix_oversample10_v1.pth`,
`weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth`.

## 5. What would actually move this number

Not more retraining tricks on the same 6 non-CLEAN examples — more real,
owner-labelled condition diversity. `scripts/export_rooftop_condition_labeling_sheet.py`
already exported all 242 crops + a blank labeling CSV
(`datasets/ground_pv_research/rooftop_condition_v1/labeling_sheet.csv`,
currently filled with my own AI-reviewed labels as a placeholder — the
property owner correcting/replacing those with real, physically-verified
knowledge of each panel is the honest next step, not a bigger oversample
sweep.
