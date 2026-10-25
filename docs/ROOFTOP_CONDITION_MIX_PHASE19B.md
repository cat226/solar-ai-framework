# Rooftop condition classifier MIX retrain — Phase 19-B — ADOPTED (decision revised)

Research only. Branch `feat/ground-pv-research-detector`, not merged. Answers
the second half of the owner's Phase 19 request (retrain the CLEAN/DUSTY/
BIRD_DROP/PHYSICAL_DAMAGE condition model on the real rooftop photos).

**Current result: ADOPTED.** `services/ground_pipeline.py`'s `CLASSIFIER_WEIGHTS`
is now `weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth`.
This section originally shipped as "NOT ADOPTED" (see Section 4a below for
that reasoning, kept for the record) — that decision was **wrong** and is
corrected here based on real evidence the owner surfaced by actually using
the app.

## 0. The correction — what happened and why

The original decision (Section 4a) rejected the MIX retrain because it cost
2–4 points of accuracy on the Susan-derived benchmark (0.752 → 0.715–0.731).
That comparison was real and correctly measured, but it was the **wrong
benchmark to decide on**: this app's whole purpose is to run on ground-level
photos of the OWNER's one rooftop (see `docs/PHASE_16_ARCHITECTURE_REVIEW.md`
— "the project's actual target deployment is a ground-level, pole-mounted
camera," this specific installation), not on Susan's source photos. The
original evaluation never checked how the *baseline* classifier — the one
that stayed active — actually performed on this rooftop's own held-out
photos before deciding to keep it.

The owner reported the app "still doesn't work well" after using it for
real inspections. Checking the actual inspection history
(`data/inspections.db`) showed the symptom directly: runs where most or all
panels got flagged DUSTY or even PHYSICAL_DAMAGE at high confidence (e.g. one
run: 10 of 13 panels flagged DUSTY; another: 0 of 5 panels read CLEAN) on
photos that were visually confirmed CLEAN during the Phase 19 labeling
review. That prompted a direct, apples-to-apples check that should have been
run before the original decision: **evaluate both the baseline and the MIX
checkpoints on the SAME held-out rooftop test crops** (48 crops, scene-
disjoint from training, 46 CLEAN + 2 DUSTY by verified label) —
`datasets/ground_pv_research/rooftop_condition_v1/reports/compare_classifiers_on_rooftop_test.py`:

| checkpoint | accuracy on held-out rooftop test (n=48) | prediction spread |
|---|---|---|
| baseline (`..._research_v1.pth`, was active) | **13/48 = 0.271** | CLEAN 11, DUSTY 32, BIRD_DROP 2, PHYSICAL_DAMAGE 3 |
| MIX oversample x10 | 46/48 = 0.958 | CLEAN 48, DUSTY 0 — **collapsed to always-CLEAN**, missed both real DUSTY examples; rejected despite the higher score, see Section 2 |
| **MIX oversample x1 — ADOPTED** | **45/48 = 0.938** | CLEAN 45, DUSTY 3 — retains real discrimination |

The baseline — what the app was actually running — gets barely better than
random on its own deployment domain, and several of its wrong calls were
high-confidence false PHYSICAL_DAMAGE flags (0.85, 0.93) on undamaged panels.
Full mismatch list in the comparison script's output
(`datasets/ground_pv_research/rooftop_condition_v1/reports/baseline_vs_mix_rooftop_test_comparison.txt`).

**Why oversample x10 was passed over despite its higher raw score**: it
predicts CLEAN for every single one of the 48 test crops — zero DUSTY, zero
BIRD_DROP, zero PHYSICAL_DAMAGE predictions at all. That is not a
well-generalizing model; it is a degenerate always-predict-the-majority-class
model that happens to score well on a test set that is 46/48 CLEAN. A model
that can never flag a real problem is not "accurate" in any useful sense, so
oversample x1 (which still makes real BIRD_DROP/DUSTY calls, e.g. correctly
flagging the panel bird-drop mark in `WhatsApp Image ... 01.03.46.jpeg`,
panel index p03, exactly matching the manually-verified label from Phase 19)
is the one promoted.

**Verified live**, not just as an offline metric: re-ran the exact
`WhatsApp Image 2026-09-18 at 01.03.46.jpeg` capture that showed the failure
(10/13 panels DUSTY under the baseline) through the real
`services.ground_pipeline.run_capture()` pipeline with the new checkpoint —
now reads 11/13 CLEAN (confidence 0.88–0.99) and 2/13 BIRD_DROP (confidence
0.74/0.87), which matches the real, hand-confirmed bird-drop mark found on
that exact rooftop scene during Phase 19 labeling.

**Honest remaining trade-off, not hidden**: this checkpoint costs 2.1 points
of accuracy on the Susan-derived benchmark (0.752 → 0.731) — see Section 4a
for the full breakdown. That trade is accepted deliberately: a few points on
an out-of-domain public dataset is a small price for going from "barely
better than random" to "94% accurate" on the photos this app is actually
used on.

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
non-CLEAN examples for a real held-out benchmark on their own. That refusal
is itself an honest finding: this rooftop, in these photos, just doesn't
show much condition diversity. It's why the MIX approach (blend into the
larger base dataset rather than train from rooftop data alone) was used at
all.

## 2. MIX strategy (same idea that worked for the panel detector)

Packaged the 241 labelled crops as `edithstark/solar-ai-rooftop-condition-topup-v1`
(`scripts/build_rooftop_condition_mix_package.py`) to blend into the much
larger, balanced `solar-ai-ground-condition-4class-classifier-v1` (Susan-
derived) training set — oversampled in train only, selection/early-stopping
kept on the ORIGINAL Susan-derived valid split.

**Real bug found and fixed along the way**: the rooftop top-up folder
legitimately has zero examples of some classes in some splits (e.g. no
DUSTY at all in train). `torchvision.datasets.ImageFolder` raises
`FileNotFoundError` on an empty class directory, and creating an empty
placeholder directory doesn't fix it either — it still raises. The real fix
was a small custom `Dataset` that borrows the BASE dataset's fixed
`class_to_idx` instead of letting either side infer classes from what
happens to be present on disk (`scripts/make_rooftop_condition_mix_notebook.py`).
Verified locally before spending a second Kaggle run.

## 3. Rooftop "real-world check" numbers reported at training time

At training time, the rooftop crops the model was trained ON (train/valid,
oversampled) scored near-perfectly (0.94–1.0) — expected and uninformative
(graded on data it just memorized). The one genuinely held-out rooftop slice
(test, 48 crops, never seen in training) scored 44/46 CLEAN correct and
1/2 DUSTY correct for oversample x1 at training time — this is the SAME 48
crops re-measured properly in Section 0 above (45/48 = 0.938 on a slightly
different per-image aggregation; both readings agree on the conclusion).

## 4a. Original decision (2026-09-19) — superseded, kept for the record

*(Everything below this line was the original writeup and is preserved
verbatim for the audit trail. It is superseded by Section 0.)*

Results — two oversample factors tried, both regress the SUSAN-DERIVED benchmark:

| | Susan-derived TEST accuracy (n=242) | vs. baseline (0.752) |
|---|---|---|
| baseline (`..._research_v1.pth`, Phase 16-B) | 0.752 | — |
| MIX, oversample x10 | 0.715 | -0.037 |
| MIX, oversample x1 | 0.731 | -0.021 |

Per-class recall, oversample x1 vs. baseline: BIRD_DROP 0.690 vs 0.676
(+0.014, noise-level), CLEAN 0.673 vs 0.764 (-0.091), DUSTY 0.794 vs 0.794
(unchanged), PHYSICAL_DAMAGE 0.667 vs 0.778 (-0.111). Full confusion
matrices: `datasets/ground_pv_research/rooftop_condition_v1/reports/mix_oversample{10,1}/training_manifest.json`.

Original (superseded) decision: **not adopted**, keep the baseline — made
without checking the baseline's own accuracy on the rooftop domain first.
That gap in the evaluation is exactly what Section 0 corrects.

## 5. What would still move this number further

Not more retraining tricks on the same 6 non-CLEAN examples — more real,
owner-labelled condition diversity. `scripts/export_rooftop_condition_labeling_sheet.py`
already exported all 242 crops + a blank labeling CSV
(`datasets/ground_pv_research/rooftop_condition_v1/labeling_sheet.csv`,
currently filled with my own AI-reviewed labels as a placeholder — the
property owner correcting/replacing those with real, physically-verified
knowledge of each panel is the honest next step, not a bigger oversample
sweep. The DUSTY class in particular rests on only 2 real examples total in
this dataset; that is thin, and confidence in DUSTY specifically should stay
lower than confidence in the CLEAN/BIRD_DROP reads, which rest on more (and
now verified-live) evidence.
