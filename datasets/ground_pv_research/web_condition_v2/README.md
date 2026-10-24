# Web-only condition pool v2 (fresh build, 2026-10-25)

Owner decision (2026-10-25): delete every manually captured rooftop photo and rebuild from scratch using ONLY
the three public sources. This pipeline contains **no first-party / rooftop image at any stage** (the
Faulty_solar_panel, Roboflow ahmed-tamer v2 and suhasmaddali GitHub repos). Research only: all three sources have
an unresolved licence, and every one is web-scraped.

## Dataset
`scripts/build_unified_condition_dataset.py --out-dir "F:/Solar AI Datasets/condition_web_v2" --report-dir
datasets/ground_pv_research/web_condition_v2` reproduced v1 exactly (deterministic): 11,322 images -> 3,457
content families -> 2,618 kept (CLEAN 1,559 / DUSTY 883 / BIRD_DROP 128 / PHYSICAL_DAMAGE 48). See
`../unified_condition_v1/README.md` for the dedup / conflict / coarse-to-fine rules.

## Evaluation protocol (changed from v1)
v1 used one 15% test split (BIRD_DROP n=19, PHYSICAL_DAMAGE n=7: uselessly wide intervals). v2 uses 5-fold
**family-grouped** stratified CV so every image is predicted once out-of-fold (BIRD n=128, PHYS n=48), and no
photo family ever sits on both sides of a fold. Frozen ImageNet embeddings (whole-frame, + hflip average) ->
standardised -> balanced logistic regression (C=0.05, fixed a priori).

| backbone @ input | accuracy | balanced acc | macro-F1 |
|---|---|---|---|
| mobilenet_v2 @224 | 0.783 | 0.728 | 0.724 |
| efficientnet_b0 @224 | 0.785 | 0.698 | 0.700 |
| resnet50 @224 | 0.765 | 0.701 | 0.711 |
| convnext_tiny @224 | 0.827 | 0.777 | 0.776 |
| **convnext_tiny @320** | **0.830** | **0.792** | **0.790** |

Best (convnext_tiny@320), recall with Wilson 95% intervals: CLEAN 0.876 [0.858, 0.891]; DUSTY 0.763 [0.734, 0.790];
BIRD_DROP 0.758 [0.677, 0.824]; PHYSICAL_DAMAGE 0.771 [0.635, 0.867]. Accuracy by source: Faulty 0.824,
Imgclean/Imgdirty web 0.828. Confusion (rows true; CLEAN, DUSTY, BIRD, PHYS):
`[1365,190,3,1] [165,674,39,5] [3,27,97,1] [3,3,5,37]` - by far the main error is CLEAN<->DUSTY (355 images),
then DUSTY->BIRD_DROP (39).

## What this number is and is not
* It is a **web-photo, same-distribution, family-disjoint** estimate. It is NOT a rooftop-camera accuracy:
  in v1 a web-trained MobileNet got 0/7 exact on the owner's marked rooftop defects while scoring 79% on web test.
* Backbone choice was selected on the OOF scores, so the best number is mildly optimistic.
* The gain over v1's fine-tuned MobileNet (macro-F1 ~0.72 on a different, smaller split) comes from the stronger
  backbone and the larger-n protocol, not from any new data.
* With the rooftop photos deleted there is no in-domain evaluation left in this repo. Any deployment claim needs
  fresh fixed-camera photos (see `docs/ROOFTOP_PHOTO_COLLECTION_GUIDE.md`).

## Artifacts
`scripts/eval_backbones_grouped_cv.py` -> `backbone_grouped_cv.json`; `scripts/train_final_web_condition.py` ->
`weights/convnext_lr_condition_web_v2.joblib` (+ `.json`, gitignored bundle; fit on all 2,618 images; class order
CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE). Embedding caches live next to the images on F:.
