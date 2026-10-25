# Unified 4-class condition dataset + baseline classifier (v1)

Built 2026-10-24 from three owner-supplied public sources that re-ship the same
photos: `Faulty_solar_panel` (pythonafroz), Roboflow `ahmed-tamer/solar-panels-dust-detection`
v2, GitHub `suhasmaddali/Solar-Panels-Dust-Detection` (originals only).
**Research only: every source has an unresolved licence. Do not redistribute.**

## Dataset (`scripts/build_unified_condition_dataset.py`)
- Independence unit = content **family** (flip/rot90-invariant dHash within 4 bits). 11,322 decodable
  images -> 3,457 families. Splits are per family, stratified by class (70/15/15, seed 42).
- Taxonomy = the 4 active classes. Electrical-damage / Snow-Covered excluded (rejected classes;
  172 families + 6 that also matched an active class).
- Conflicts: 31 families with irreconcilable labels dropped. A **coarse-to-fine override** keeps
  Faulty's BIRD_DROP (56 families) / PHYSICAL_DAMAGE (3) over a binary source's "DUSTY"
  (those sets only say "dirty"). CLEAN is never overridden.
- The Roboflow single-panel macro session (`phone2021_macro`, 630 families) is excluded by default
  (`--include-lineage phone2021_macro` re-adds it).
- GitHub `Augmented_images`, `*_Discarded`, `*_temporary`, `*_with_Labels` never used.
- Result: 2,618 images. CLEAN 1,559 / DUSTY 883 / BIRD_DROP 128 / PHYSICAL_DAMAGE 48 families.
  Bulk images: `F:\Solar AI Datasets\condition_unified_v1\{train,valid,test}\<CLASS>\`.
  Committed here: `manifest_representatives.csv`, `build_report.json`.

## Baseline (`scripts/train_unified_condition_classifier.py`, CPU, 11 epochs, best = epoch 6)
MobileNetV2, whole-frame 224x224, class-weighted CE, selection on val macro-F1.
Class index order = canonical `CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE`.

| test (family-disjoint, same web distribution) | n | recall [Wilson 95%] | precision |
|---|---|---|---|
| CLEAN | 234 | 0.855 [0.80, 0.89] | 0.862 |
| DUSTY | 132 | 0.674 [0.59, 0.75] | 0.742 |
| BIRD_DROP | 19 | 0.842 [0.62, 0.95] | 0.485 |
| PHYSICAL_DAMAGE | 7 | 0.714 [0.36, 0.92] | 0.714 |

accuracy 0.791, balanced accuracy 0.771, macro-F1 0.724. Per source: Faulty 0.79 (n=80), Img*-web 0.79 (n=305).
**BIRD_DROP / PHYSICAL_DAMAGE test n is 19 / 7: the intervals are the result, not the point estimates.**

## Real-world check (qualitative, NOT an accuracy figure)
48 pristine owner rooftop photos (`rooftop48_predictions_unified_v1.csv`): CLEAN 19 / DUSTY 21 /
BIRD_DROP 3 / PHYSICAL_DAMAGE 5, mostly low confidence (0.35-0.55). PHYSICAL_DAMAGE fires (up to 0.90)
on panels that look intact; of the 6 owner-marked defect photos only 2 were flagged non-CLEAN, and the
4 marked bird-drop photos were mostly called CLEAN. **The web-photo classifier does not transfer to the
rooftop camera domain**, so this model must not be promoted or used for alerts.

**Panel-crop follow-up** (rooftop detector `yolo_solar_rooftop_v1.pt`, conf 0.25 -> 291 crops from the 48
photos; scratch run, not committed): the unified model calls 60/291 crops (21%) PHYSICAL_DAMAGE and 107
DUSTY (mean conf 0.65), while the promoted rooftop MIX classifier calls 277/291 CLEAN (mean conf 0.94);
they agree on only 125/291. Visually the crops are intact, uniform panels, so the unified PHYSICAL_DAMAGE
calls are false positives (multi-cell grid texture reads as cracks to a model trained on close-up web
shots). Cropping does NOT fix the transfer gap. The MIX model is not an independent check - it was trained
on this rooftop's own (AI-labelled) photos.

## AI-proposed rooftop labels (`rooftop48_ai_proposed_labels.csv`)
At the owner's request the 48 pristine photos were labelled by visual review (Claude). **These are NOT ground
truth.** 6 rows are the owner's own marks (`confidence=owner`, override everything). The rest are
`ai_visual_review`: 2 high, 6 medium, 34 low confidence, 9 `UNSURE`. Defects on this roof are a few pixels
wide in wide shots, so "no defect seen" means unconfirmed, not clean. Counts: BIRD_DROP 17 / CLEAN 16 /
UNSURE 9 / PHYSICAL_DAMAGE 6. Rules: do not train on the non-owner rows, do not score a model against them as
if verified; the owner reviews/corrects the file. The unified model agrees with only 1 of the 14 owner/high/
medium-confidence labels (it calls the bird-drop photos CLEAN), which confirms the web-to-rooftop gap.

## Owner ground truth restored: `owner_mark_boxes.json`
The owner's burned-in marks were diffed against the pristine originals
(`scripts/extract_owner_mark_boxes.py`, tests in `tests/test_owner_mark_boxes.py`): 6 photos / 7 boxes
(black = BIRD_DROP, red = PHYSICAL_DAMAGE), each verified by overlay. This is the only machine-readable
owner ground truth for the rooftop. Review sheets for the AI-proposed labels:
`scripts/make_label_review_sheets.py` -> `F:\Solar AI Datasets\condition_unified_v1\review_sheets\`.
New-photo protocol: `docs/ROOFTOP_PHOTO_COLLECTION_GUIDE.md`.

## Direct test against the owner's boxes (2026-10-24, scratch run, n=7 boxes / 6 photos / ~4 scenes)
Each owner box -> the detected panel covering it (7/7 covered) -> whole-panel crop -> classifier.
Exact-class correct: unified web-trained **0/7**, currently promoted rooftop MIX classifier **0/7**.
MIX calls all 5 bird-drop panels CLEAN at 0.96-1.00 confidence and flags only the 2 physical-damage panels
(as BIRD_DROP). Unified flags non-CLEAN on 4/7 but also calls ~21% of unmarked panels PHYSICAL_DAMAGE.
So the earlier "MIX 94% on the rooftop" figure was against AI-assigned labels that the owner's marks
contradict - it does NOT mean MIX finds the owner's defects. Panel-level labels for unmarked panels are
unconfirmed, so this measures recall on marked defects only; n is tiny.

## Deliberately NOT done: defect-level (patch) model
Needs defect locations, and only 7 owner boxes in ~4 scenes exist. A scene-held-out evaluation on that would
be noise, so no patch model was trained. Trigger to build it: >= ~10 owner-confirmed defect scenes from the
fixed camera spot (see the collection guide). Plan then: native-resolution patches of detected panels,
pretrain on this pool, fine-tune on owner labels, leave-one-scene-out evaluation.

## Next (needs owner decision)
In-domain data is the bottleneck: owner-confirmed labels for all 48 photos (explicit no-defect tags) and
new defect photos in new scenes; consider panel-crop classification via the Phase 14A detector instead of
whole frames; use this pool only for pretraining.
