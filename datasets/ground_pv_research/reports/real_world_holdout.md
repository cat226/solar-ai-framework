# Real-world rooftop holdout — qualitative evaluation

**Qualitative only. No ground-truth annotations exist for these 6 images, so
NO accuracy / precision / recall / F1 / mAP is computed and no prediction is
called "correct" merely because it looks plausible.**

## Input

6 owner-provided close-up / ground-level RGB photographs of rooftop PV,
copied off the (unstable) removable drive to
`D:\solar_ai_training_scratch\Solar Real Images\`. Source images not
modified. All JPEG, RGB.

| # | filename | size (bytes) | dimensions | SHA-256 |
|---|---|---|---|---|
| 1 | WhatsApp Image 2026-09-07 at 15.25.56.jpeg | 158,669 | 573×1280 | `583f063090d74ba32783951cab3aca28501b5c9f28bee5e0b1adb5a7fdf8080d` |
| 2 | WhatsApp Image 2026-09-07 at 15.25.57 (1).jpeg | 163,303 | 573×1280 | `24eb0006e67bfa5f4482ef02f673a3fd810914b6183851533aecfd7aeef1b9a9` |
| 3 | WhatsApp Image 2026-09-07 at 15.25.57.jpeg | 137,436 | 573×1280 | `f6a6024376444fe7bcae425c4d3028a6b2e889407b6d701df768b8f158194ecf` |
| 4 | WhatsApp Image 2026-09-07 at 15.25.58.jpeg | 143,812 | 573×1280 | `a8243d233c7be3dba7fc776c9f75e40d4255405dba015f123642fd6884b2dd24` |
| 5 | WhatsApp Image 2026-09-07 at 15.25.59 (1).jpeg | 155,230 | 573×1280 | `0863acbca61c7ebf88d07b09bc0aacd5e7c687b6b78ff44efcfb3a66e10e14c0` |
| 6 | WhatsApp Image 2026-09-07 at 15.25.59.jpeg | 171,216 | 1280×573 | `7bf163746890f2e245b622a12f82c2c0ae94fba8c33d06e57c37929302dd99eb` |

## Model & configuration

- Checkpoint: `weights/yolo_solar_ground_research_v1.pt`
  SHA-256 `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50`
  (verified before inference).
- Classes the model can predict: **`panel`, `cracked`, `dusty`** — nothing else.
- `scripts/infer_ground_pv.py --imgsz 640 --conf 0.45 --iou 0.50
  --min-assoc-iou 0.10 --no-condition`
- Outputs (gitignored):
  `research_runs/ground_pv/real_world_predictions/ground_research_v1_conf045/`
  — annotated JPEGs, per-image JSON, `detections.csv`, `inference_summary.json`.
- **Absence of a detection is reported as "0 detections" — it is NOT
  "Clean".** No `Clean` / `BirdDrop` / `Hotspot` / electrical-damage label is
  produced or inferred.

## Per-image detection results (raw model output)

| img | dims | dets | panel | cracked | dusty | confidence range |
|---|---|---|---|---|---|---|
| 1 (…56) | 573×1280 | 2 | 1 (0.87) | 1 (0.61) | 0 | 0.61–0.87 |
| 2 (…57 (1)) | 573×1280 | 2 | 2 (0.87, 0.51) | 0 | 0 | 0.51–0.87 |
| 3 (…57) | 573×1280 | 3 | 2 (0.85, 0.68) | 1 (0.50) | 0 | 0.50–0.85 |
| 4 (…58) | 573×1280 | 1 | 1 (0.58) | 0 | 0 | 0.58 |
| 5 (…59 (1)) | 573×1280 | 2 | 2 (0.64, 0.59) | 0 | 0 | 0.59–0.64 |
| 6 (…59) | 1280×573 | 2 | 2 (0.75, 0.67) | 0 | 0 | 0.67–0.75 |

Totals: 12 detections across 6 images — **10 `panel`, 2 `cracked`, 0
`dusty`**. 0 images with zero detections. Exact bounding boxes: `detections.csv`.

## Per-image qualitative assessment

### 1 — …15.25.56 (573×1280)
Close-up of one panel heavily marked with bird droppings + some soiling;
adjacent panels partly visible behind; black roofing membrane at the bottom.
- **panel localization: good.** The 0.87 box bounds the main foreground panel
  tightly.
- **missed panels:** the partial background panels above are not boxed
  (acceptable — they are cropped).
- **false positive:** the `cracked` 0.61 box sits on the **black roofing
  membrane** at the bottom-left (printed "Sika" branding / texture), **not on
  a panel**. Obvious FP.
- **dusty:** none, despite visible bird-splatter and light soiling (bird
  droppings are not a model class; the soiling is mild).
- no scene-spanning / nonsensical boxes.

### 2 — …15.25.57 (1) (573×1280)
Same style — one panel with bird droppings, a row of panels behind.
- **panel localization: good.** 0.87 box on the main panel; the 0.51 box is a
  reasonable strip over the background row.
- **missed panels:** left/right edge panels (heavily cropped) not boxed.
- **false positives:** none.
- **dusty / cracked:** none.

### 3 — …15.25.57 (573×1280)
Two adjacent panels seen at a steep near-edge-on angle; roofing membrane below.
- **panel localization: good.** Both panels boxed (0.85, 0.68), tight.
- **missed panels:** none of note.
- **false positive:** `cracked` 0.50 box on the **roofing membrane** bottom-
  right — same failure mode as image 1. Obvious FP.
- **dusty:** none.

### 4 — …15.25.58 (573×1280)
A single large panel fills ~90 % of the frame; small background panels at top.
- **panel localization: poor.** The only detection (0.58) is a thin strip
  over the **background** panels at the top.
- **obvious missed panel:** the large foreground panel that dominates the
  image is **not detected at all** — the clearest failure in the set. Likely
  a domain-scale issue: a panel that essentially *is* the whole frame, with a
  flat texture and no fully-visible border, is unlike the more-bounded panels
  in the (640×640, stretch-resized) training data.
- **false positives / dusty / cracked:** none.

### 5 — …15.25.59 (1) (573×1280)
A small panel at the top, a large panel filling the rest; roofing at right.
- **panel localization: good / partial.** Top panel boxed well (0.64); the
  large panel is boxed (0.59) but the box is a little narrow on the right and
  short at the bottom — captures most of it.
- **missed panels:** none.
- **false positives / dusty / cracked:** none.

### 6 — …15.25.59 (1280×573, landscape)
A wide view of a multi-panel rooftop array at an angle; buildings behind.
- **panel localization: partial.** Two large boxes (0.75, 0.67) each cover a
  *group* of ~2–3 panels; they **overlap** in the middle. No per-panel boxes.
  Not nonsensical (everything inside is panels) but not usable for
  per-panel condition assignment.
- **missed panels:** the top strip of the array is partly outside both boxes.
- **false positives / dusty / cracked:** none.

## Failure modes observed

1. **`cracked` false positives on dark textured roofing membrane** — 2 of 6
   images. The model fires `cracked` (conf ~0.5–0.6) on the black
   waterproofing sheet with printed text at the image edge, never on a panel.
   A simple guard (ignore `cracked` boxes whose IoU with any `panel` box is
   ~0, or require defects to fall inside a panel) would remove both.
2. **Large frontal panels missed** — image 4 (clear miss), image 5/6
   (partial). When a single panel dominates the frame with no visible full
   border, recall drops. Consistent with the training data's 640×640
   stretch-resize and more-bounded panels.
3. **Group boxes instead of per-panel boxes** on wide array shots (image 6).
4. **No soiling signal** — 0 `dusty` detections at conf 0.45 across all 6,
   including panels with visible bird-splatter and a dust film. The panels
   here are not heavily dust-caked, so this is not necessarily wrong, but the
   model offered no "needs cleaning" cue on any image.

Confidence ranges: `panel` 0.51–0.87 (mostly 0.58–0.87); `cracked` 0.50–0.61;
`dusty` — none.

## Secondary comparison — production aerial detector (QUALITATIVE ONLY)

`weights/yolo_solar.pt` (SHA `0b58609905…`, single class `solar panel`, prod
config conf 0.45 / iou 0.50 / imgsz 640) on the same 6 images:

| img | aerial `yolo_solar.pt` detections |
|---|---|
| 1 | 1 × `solar panel` 0.47 — a small sliver box (0,248–89,507), not a panel outline |
| 2–6 | **0 detections** |

The aerial detector is effectively blind on these close-up photos (1
low-confidence sliver across 6 images). The ground research model produced a
plausible `panel` box on 5 of 6. **This is a qualitative observation, not a
scored comparison.** It confirms the two models serve different domains and
neither substitutes for the other. Outputs:
`research_runs/ground_pv/real_world_predictions/production_aerial_yolo_solar_conf045/`.

## Bottom line (qualitative)

On the owner's real close-up rooftop photos the research model gives a
**usable panel-localisation signal on ~5 of 6 images**, with two recurring
weaknesses: `cracked` false-positives on off-panel roofing texture, and
missed panels when one panel fills the whole frame. It produced **no `dusty`
detections** here. It is a reasonable **assistive** close-up panel detector;
it is not reliable enough for unattended per-panel condition assessment, and
it remains a **research artifact — not production, not a replacement for the
aerial detector, no hotspot capability, no `bird_drop` capability.**
