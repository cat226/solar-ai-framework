# Ground-level PV research dataset — audit report

Dataset directory: `datasets\ground_pv_research\raw\panel_solar_bw945_v1`

**Overall: FAIL (see fail-closed reasons)**

## Fail-closed reasons

- 1 exact-duplicate image(s) span multiple splits (train/val/test leakage) - a leakage-clean split is required before training
- 614 YOLO-segmentation polygon line(s) across 410 file(s) (166 mixed with boxes, 244 polygon-only) - the dataset mixes YOLO-seg and YOLO-detect formats; the normalized copy must convert polygons to bounding boxes before detection training

## 1. Structure

- data.yaml present: True
- splits found: ['test', 'train', 'valid']
- source `names:` = {0: 'bird_drop', 1: 'cracked', 2: 'dusty', 3: 'panel'}
- class-id remap (source→research): {0: 1, 1: 2, 2: 3, 3: 0} (identity: False)

## Totals

- images: 6493   label files: 6493
- annotations: 15791  (bbox: 15177, polygon: 614)
- corrupt images: 0
- exact-duplicate groups: 1
- near-duplicate pairs (Hamming ≤ 5): 41166
- cross-split exact leakage: 1
- cross-split near leakage: 18204

## Annotation format

- bbox lines: 15177   polygon (YOLO-seg) lines: 614
- files bbox-only: 6083   polygon-only: 244   mixed: 166
- pure-detection dataset: False
- polygon lines are valid YOLO-segmentation annotations, converted here to their bounding box for counting; a pure-detection training run needs the normalized copy to bake that conversion in.

## Per-split

| split | images | labels | missing labels | no-parseable-annot | truly empty | polygon-only | label w/o image | corrupt | defect-only (no panel) | panel (no defect) |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 648 | 648 | 0 | 0 | 0 | 29 | 0 | 0 | 124 | 261 |
| train | 4546 | 4546 | 0 | 0 | 0 | 175 | 0 | 0 | 898 | 1822 |
| valid | 1299 | 1299 | 0 | 0 | 0 | 40 | 0 | 0 | 244 | 551 |

## Class distribution

| class | total instances | bbox | polygon | images containing |
|---|---|---|---|---|
| panel | 6319 | 6310 | 9 | 5227 |
| bird_drop | 81 | 81 | 0 | 9 |
| cracked | 9040 | 8780 | 260 | 3694 |
| dusty | 351 | 6 | 345 | 156 |

- images with >1 distinct class: 2593
- combinations: {'cracked+panel': 2428, 'dusty+panel': 156, 'bird_drop+panel': 9}

## Geometry

- image width dist: {'count': 6493, 'min': 640.0, 'max': 640.0, 'mean': 640.0, 'median': 640.0, 'p05': 640.0, 'p95': 640.0}
- image height dist: {'count': 6493, 'min': 640.0, 'max': 640.0, 'mean': 640.0, 'median': 640.0, 'p05': 640.0, 'p95': 640.0}
- aspect ratio dist: {'count': 6493, 'min': 1.0, 'max': 1.0, 'mean': 1.0, 'median': 1.0, 'p05': 1.0, 'p95': 1.0}
- box area (frac of image) dist: {'count': 15791, 'min': 0.0001171875, 'max': 1.0, 'mean': 0.21087641510631133, 'median': 0.05220703125000001, 'p05': 0.0010668945312499998, 'p95': 0.9982953125}
- tiny objects (area < 0.001): 731

## Duplication & leakage

- exact-duplicate groups: 1
- near-duplicate pairs: 41166
- cross-split exact leakage: 1
- cross-split near leakage: 18204
- likely augmentation families: 804
  - EXACT: ['test/Screenshot-2023-04-02-002436_png_jpg.rf.ff762fb15d12276b24f64f2a12789ba5.jpg', 'train/Screenshot-2023-04-02-002436_png_jpg.rf.0f280b7f6b16e81aff4adef3b7fa13ef.jpg']
  - NEAR (h=3): test/1677069178370_jpg.rf.f93f981a3bea73b446345479013dc1a1.jpg  <->  valid/1677069178370_jpg.rf.27c245745f537d94d3d32132ff0c5332.jpg
  - NEAR (h=2): test/1677069179152_jpg.rf.50cb7e223d9f180e57f71c954dd0571f.jpg  <->  train/1677069179152_jpg.rf.38464a0dbdd0eae150a5f47b9c657dc3.jpg
  - NEAR (h=2): test/1677069179152_jpg.rf.50cb7e223d9f180e57f71c954dd0571f.jpg  <->  train/1677069179152_jpg.rf.4107e6b8d2dfd53812d57548230fd4c1.jpg
  - NEAR (h=1): test/1677069179152_jpg.rf.50cb7e223d9f180e57f71c954dd0571f.jpg  <->  train/1677069179152_jpg.rf.8c2241743ac4733fad9fd7fe2697d894.jpg
  - NEAR (h=4): test/1677069179152_jpg.rf.50cb7e223d9f180e57f71c954dd0571f.jpg  <->  train/1677069179152_jpg.rf.e4dc0eb863cc9e515dbf163df5a3d492.jpg
  - NEAR (h=3): test/1677069179169_jpg.rf.6e323beb9cbed9b5b1e128b3603ef93d.jpg  <->  train/1677069179169_jpg.rf.258ed6d7f08288e54cfafaf7d8218f48.jpg
  - NEAR (h=5): test/1677069179169_jpg.rf.6e323beb9cbed9b5b1e128b3603ef93d.jpg  <->  valid/1677069179169_jpg.rf.b463a855ff4a70d0445afd9c4ef52606.jpg
  - NEAR (h=1): test/Crack_126_jpg.rf.8bec07d27e57e19756b6c411f6b522fd.jpg  <->  train/Crack_126_jpg.rf.dde818463ccf014bd97b573add6ee498.jpg
  - NEAR (h=3): test/Crack_127_jpg.rf.cc39b8dedb2c1d0758e43b5ae7a25e46.jpg  <->  train/Crack_127_jpg.rf.dbc592d0177a6ca0ede44fde229fc8b9.jpg
  - NEAR (h=1): test/Crack_132_jpg.rf.d3597679d24c5bd0c33d6eff10acbf15.jpg  <->  train/Crack_132_jpg.rf.2ce0d56742fcfa4435e387f2e31e486f.jpg
  - NEAR (h=2): test/Crack_132_jpg.rf.d3597679d24c5bd0c33d6eff10acbf15.jpg  <->  train/Crack_132_jpg.rf.87503b19415060366b425625c0bf1545.jpg
  - NEAR (h=4): test/Crack_133_jpg.rf.4e8f12efdc5ccd134a95026d3ee980e2.jpg  <->  train/Crack_133_jpg.rf.0f8399e34ab569163d0e9fe8856dbbce.jpg
  - NEAR (h=2): test/Crack_133_jpg.rf.4e8f12efdc5ccd134a95026d3ee980e2.jpg  <->  valid/Crack_133_jpg.rf.5b419eb6b96188b928edc96acb092b76.jpg
  - NEAR (h=1): test/Crack_138_jpg.rf.981b357a1cb1509baa779eb884567ab5.jpg  <->  train/Crack_138_jpg.rf.c085dcc2afaa35ee8de4d931b14edafb.jpg
  - NEAR (h=1): test/Crack_196_jpg.rf.8cdb5f5f85cde59187d5f0d9b99ba418.jpg  <->  train/Crack_196_jpg.rf.d585288532b82a73d948fa7bc899d8a9.jpg
  - NEAR (h=1): test/Crack_196_jpg.rf.8cdb5f5f85cde59187d5f0d9b99ba418.jpg  <->  train/Crack_196_jpg.rf.ea375a4f4f0f212b818e9fee3d31f80e.jpg
  - NEAR (h=2): test/Crack_197_jpg.rf.1da96ca4dd34a70087229051f3fe416c.jpg  <->  train/Crack_197_jpg.rf.5005a1e471abd58a9a0872542eea58c0.jpg
  - NEAR (h=1): test/Crack_197_jpg.rf.1da96ca4dd34a70087229051f3fe416c.jpg  <->  train/Crack_197_jpg.rf.c5aaa01562d0b05639de73f157a2e5bd.jpg
  - NEAR (h=3): test/Crack_197_jpg.rf.4fa6dc19c40edb0dc2685940635953e6.jpg  <->  train/Crack_197_jpg.rf.0de1316ea4c76dd2b948479e349a3abb.jpg
  - NEAR (h=2): test/Crack_197_jpg.rf.4fa6dc19c40edb0dc2685940635953e6.jpg  <->  train/Crack_197_jpg.rf.13f2f323974c31b3490c4937ecfa8aad.jpg

## Annotation issues

- issue counts by kind: {}

