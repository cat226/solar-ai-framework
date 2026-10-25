# Tiled ground-panel inference — real-world re-evaluation (Phase 10)

**Conclusion: C — TILED INFERENCE IMPROVES SOME CASES BUT INTRODUCES TRADE-OFFS.**

> RESEARCH EXPERIMENT. No retraining, no checkpoint change, no production
> change. **Qualitative only** — there is no ground truth for the six owner
> photos, so **no accuracy / precision / recall / F1 / mAP / confusion matrix**
> is computed. A prediction is a raw model output, not a fact. **Absence of a
> condition prediction does not mean Clean.** ELECTRICAL_DAMAGE is **not**
> validated electrical-fault detection (the source audit found those boxes
> visually correspond to cell cracks). Hotspot is out of scope and never
> inferred from RGB.

---

## 1. Objective

Test whether overlapping-tile inference improves the **first stage** of the
two-stage ground-PV research pipeline
(`photo → ground panel detector → panel crops → condition/defect detector`).
The full-frame ground detector `weights/yolo_solar_ground_research_v1.pt` is
known to (a) miss a large foreground panel on image 4, (b) emit
overlapping/group boxes on image 6, and (c) fire `cracked` false positives on
black roofing membrane on images 1 and 3.

## 2. Existing full-frame baseline

`weights/yolo_solar_ground_research_v1.pt` (SHA
`2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50`), classes
`panel / cracked / dusty`, run whole-frame at `conf 0.45 · iou 0.50 · imgsz
640` — the configuration documented in `real_world_holdout.md`.

| # | image (dims) | full-frame panels | conf | non-panel |
|---|---|---|---|---|
| 1 | …56 (573×1280) | 1 | 0.87 | `cracked` 0.61 on roofing |
| 2 | …57 (1) (573×1280) | 2 | 0.51–0.87 | — |
| 3 | …57 (573×1280) | 2 | 0.68–0.85 | `cracked` 0.50 on roofing |
| 4 | …58 (573×1280) | 1 (wrong — a strip over background panels; **large foreground panel missed**) | 0.58 | — |
| 5 | …59 (1) (573×1280) | 2 | 0.59–0.63 | — |
| 6 | …59 (1280×573) | 2 (overlapping group boxes) | 0.67–0.75 | — |

## 3. Tiled configuration

`scripts/infer_ground_pv_tiled.py` → `training/detection/ground_pv/tiled_inference.py`.

| parameter | value |
|---|---|
| tile size | 640 × 640 |
| overlap | 25 % (step 480 px) |
| tiling of 573×1280 | 1 column × 3 rows → 3 tiles |
| tiling of 1280×573 | 3 columns × 1 row → 3 tiles |
| edge handling | final row/column window clamped to the image edge; the cropped region is **padded** (not resized) onto a 640×640 grey (114) canvas anchored top-left. The whole image is **never** globally resized before tiling. |
| detector | the SAME `yolo_solar_ground_research_v1.pt`, per tile, `conf 0.45` |
| per-tile NMS | IoU 0.50 (Ultralytics default) |
| global de-duplication | after remapping to original coordinates: per-class NMS at IoU 0.50, plus "mostly-contained" suppression (a lower-confidence same-class box ≥ 90 % inside a higher-confidence box is dropped) |
| coordinate remap | detection clipped to the un-padded crop extent, then offset by the tile origin |
| determinism | tile grid + merge order are deterministic (ties broken by −confidence then coordinates) |

## 4. Per-image panel comparison

| # | full-frame | tiled | winner | key observation |
|---|---:|---:|---|---|
| 1 | 1 | 3 | tie | tiled adds the partial background panel (0.89) **and fragments the main panel into 2 bands** (0.79 + 0.62); `cracked` roofing FP persists (0.48) |
| 2 | 2 | 3 | tie | tiled keeps the top strip, **fragments the main panel into mid + lower bands** |
| 3 | 2 | 4 | **full-frame** | tiled **fragments both panels into 4 pieces, loses the top half of the left panel**, and the `cracked` roofing FP gets **worse** (0.65, full image-width, vs 0.50 a corner) |
| 4 | 1 (wrong) | 3 | **tiled** | tiled **recovers the large foreground panel** (missed entirely by full-frame) as a mid band (0.92) + lower band (0.93); also keeps the top strip (0.93) |
| 5 | 2 | 2 | **full-frame** | tiled **truncates the large panel** — it is represented only by its lower half (y 640–1280); the middle band (y 185–640) is lost. Full-frame's box covered y 174–1274 |
| 6 | 2 (group) | 1 | **full-frame** | **tiled catastrophically fails** — tiles 2 and 3 (centre + right of the array) detect **nothing**; only the left tile fires. The right two-thirds of the array is unboxed. Full-frame's two group boxes at least covered the whole array |

Full-frame wins 3 (images 3, 5, 6), tiled wins 1 (image 4), tie 2 (images 1, 2).

**Confidence:** tiled boxes are frequently *higher* confidence (0.90–0.96 vs
0.58–0.87) because a whole panel that fits a single tile at training-like scale
is an easy detection — but a higher score on a **fragment** is not a better
result.

**Root cause.** The detector was trained on views where a panel is reasonably
bounded within the frame. A 640 px tile landing **entirely inside** a large or
oblique panel shows only repeating cell texture with no panel boundary, so the
detector fires nothing (images 5, 6) or, near a boundary, fires a band
(images 1–4). Tiling helps only when it happens to bring a whole panel into one
tile at training-like scale (image 4's foreground panel).

## 5. Visual findings

Side-by-side `original | full-frame | tiled` for all six images:
`research_runs/ground_pv/tiled_real_world/compare/` (gitignored).

- **Image 4** — the foreground panel that full-frame ignored is clearly boxed
  by tiling (two stacked bands).
- **Image 6** — tiling's single box covers only the left ~third of a rooftop
  array that has ~8 visible panels.
- **Image 3** — tiling cuts each tall panel into halves and the left panel's
  upper half is unboxed; the roofing `cracked` box spans the full image width.
- **Images 1, 2** — the heavily bird-dropped foreground panel is split into two
  boxes.

## 6. Two-stage condition comparison

Each detected panel (both chains) was cropped (4 % pad) and run through
`weights/yolo_solar_condition_research_v1.pt` (SHA
`b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578`) at
`conf 0.25 · iou 0.50` (the Phase 9 configuration). Classes
`CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE / ELECTRICAL_DAMAGE`.

| | full-frame chain | tiled chain |
|---|---:|---:|
| panel crops | 10 | 16 |
| crops with ≥ 1 condition prediction | 8 | 11 |
| CLEAN | 6 | 5 |
| DUSTY | 2 | 3 |
| BIRD_DROP | 1 | **4** |
| PHYSICAL_DAMAGE | 0 | **1** |
| ELECTRICAL_DAMAGE | 0 | 0 |
| condition confidence range | 0.27–0.81 | 0.29–0.94 |

Per image:

| # | full-frame chain | tiled chain | change |
|---|---|---|---|
| 1 | DUSTY ×1 | **BIRD_DROP ×2** + DUSTY ×1 | tiled's tighter crops let the condition model localise the droppings — arguably a genuine improvement |
| 2 | BIRD_DROP ×1, DUSTY ×1 | BIRD_DROP ×2 (up to 0.78), DUSTY ×2, **PHYSICAL_DAMAGE ×1 (0.29)** | stronger bird-drop signal, but a new low-confidence PHYSICAL_DAMAGE box on a panel with no visible damage |
| 3 | CLEAN ×2 | CLEAN ×1 (0.94, one fragment) | tiled lost 3 of the 4 fragment crops' predictions |
| 4 | (nothing — no panel detected) | **CLEAN ×2 (0.80, 0.83)** on the recovered foreground panel | tiled gives image 4 a "panel present, looks clean" signal where full-frame gave nothing |
| 5 | CLEAN ×2 | CLEAN ×1 | tiled's truncated panel → fewer/worse crops |
| 6 | CLEAN ×2 | CLEAN ×1 | tiled's single partial box → one crop |

**Net:** the tiled chain finds more bird-drop on the two dirty panels
(images 1–2) and rescues image 4, but produces fewer/worse crops on images 3,
5, 6 and adds a weak PHYSICAL_DAMAGE box. **Absence of a condition prediction
does not mean the panel is Clean.**

## 7. Image 4 investigation — recovered

Full-frame produced a single 0.58 box that is a thin strip over the
*background* panels at the top; the large foreground panel that fills ~90 % of
the frame was **not detected at all**.

Tiling **recovers it.** Tile 0 (rows 0–640) detects the top strip (0.93) and a
band down to y≈639 (0.92); tiles 1–2 (rows 480–1280) detect a band y≈481–1120
(0.93). Together the two lower bands cover most of the foreground panel. It is
**two bands, not one clean box**, but the panel is no longer missing, and the
condition model then labels those crops CLEAN (0.80, 0.83).

## 8. Image 6 investigation — not separated, made worse

Full-frame produced two overlapping group boxes that between them covered the
whole rooftop array. Tiling produced **one** box covering only the left tile's
content. Tiles 1 and 2 (centre and right of the array) returned **zero
detections** — a 640 px tile landing inside the obliquely-viewed array sees
only cell lattice and no panel outline. Tiling did **not** separate the array
into per-panel boxes; it lost most of it.

## 9. Crack false-positive investigation — not fixed, partly worse

The `cracked`-on-roofing-membrane false positive is a **class/data problem, not
a resolution problem**, so tiling does not fix it:

| # | full-frame `cracked` | tiled `cracked` |
|---|---|---|
| 1 | 0.61, corner box on the "Sika" roofing text | 0.48, same location (lower conf, still present) |
| 3 | 0.50, bottom-right corner of roofing | **0.65, spanning the full image width** along the roofing strip |

On image 3 the tile that focuses on the lower roofing region gives the spurious
"crack" texture more relative prominence, so the FP gets *larger and more
confident*. The right fix remains a **panel-containment guard** (drop `cracked`
/ `dusty` boxes that do not overlap any `panel` box) or a cleaner training set
— not tiling.

## 10. Panel-overlap analysis (diagnostic)

For every condition detection in both chains, the condition box lies **within**
its panel crop (30/30 full-frame, all tiled) — trivially, because the condition
model is run *on* the crop. This confirms the two-stage plumbing is sound
(condition boxes are always reported in-crop and were remapped correctly) but
carries **no** signal about whether a detection corresponds to a real feature
on that panel — there is no ground truth. **No production panel-overlap guard
is added** (Phase 10 §8): the experiment does not demonstrate a clear benefit
that would justify it, and the more valuable guard is the `cracked`/`dusty`
off-panel filter noted in §9.

## 11. Limitations

- Six images, one rooftop, one photographer, one device — not a representative
  sample.
- No ground truth → every judgement above is a **visual** judgement.
- The condition model is itself a weak, underfit research baseline
  (test mAP50 0.47) on unverified-provenance data; its predictions on real
  crops are indicative at best.
- Tiling parameters (640 / 25 %) were not swept; a different tile size might
  behave differently, but the failure mode (interior tiles with no panel
  boundary) is structural.
- The detector's training data was 640×640 stretch-resized; tiled inference
  padding (not stretching) is *more* faithful to that only when a whole panel
  fits a tile.

## 12. Recommendation

**C — TILED INFERENCE IMPROVES SOME CASES BUT INTRODUCES TRADE-OFFS.**

- **Keep the full-frame detector as the default first stage.** It is more
  robust on 4 of 6 images and never loses a panel it previously found.
- **Do not adopt naive fixed-grid tiling.** It fragments panels that span
  tiles, silently drops panels whose tile shows no boundary (images 5, 6), and
  makes the roofing `cracked` FP worse (image 3).
- **The one real win (image 4) points at the actual problem:** the detector's
  recall collapses when a single panel dominates the frame. The better
  investments are (a) a `cracked`/`dusty` **off-panel filter** (fixes images 1
  and 3 cheaply and safely), and (b) **retraining / fine-tuning the panel
  detector with more large-frontal-panel and wide-array examples**, or a
  boundary-aware tiling that only keeps tiles containing a panel edge.
- Tiling could be offered as an **opt-in fallback** for images where the
  full-frame detector returns 0–1 panels (like image 4) — but only with
  post-merge band-stitching to avoid fragmented crops, and never as the
  default.

No production change. This remains a research experiment.
