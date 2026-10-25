# Ground-PV defect dataset expansion — candidate comparison

Generated 2026-09-08. **DATASET DISCOVERY + PROVENANCE AUDIT ONLY.**
No training, no fine-tuning, no dataset merging, no production changes.

**Target taxonomy (RGB):** `panel · clean/non-defective · dusty · bird_drop ·
physical_damage · electrical_damage`. **Hotspot stays OUT** of the RGB
taxonomy (thermal/IR only, separate pipeline).

**Existing data — not touched:**
`datasets/ground_pv_research/raw/panel_solar_bw945_v1` (6,493 raw) and
`datasets/ground_pv_research/yolo` (6,492 remediated leakage-clean, classes
`panel/cracked/dusty`).

---

## The one thing that matters

There is **no fully legally-clean candidate**. Every downloadable candidate's
imagery traces to a Kaggle dataset (`pythonafroz/solar-panel-images`) whose
Kaggle licence reads **"Unknown"**; the "CC BY 4.0" shown on Roboflow is a
re-uploader's dataset-level declaration, not a verified image-level right
(see `overlap/overlap_report.md` §4). The candidates are also **heavily
overlapping** — the ~6,700 downloaded images represent only **~1,900–2,100
genuinely independent** photos, and the rare classes we most need
(**electrical_damage ~200, physical_damage ~350**) are the scarcest.

The audit's job was to find the *largest genuinely useful, independent,
legally-usable* pool. The honest finding: the largest *useful* pool is one
dataset (**S1**) plus a partial top-up from one other (**S3**), all at
**provenance grade C**.

---

## Candidate table

| id | dataset | ws / project / ver | task | fmt | images (unique src) | licence shown | prov. | our-taxonomy coverage | ground-level RGB | overlap w/ existing | recommended use |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | Detection for defects in solar panels | `defects-kgig6 / detection-for-defects-in-solar-panels / v2` | detection (boxes) | yolov8 | 1,963 (**~817**) | CC BY 4.0 | **C** | **CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE, ELECTRICAL_DAMAGE** (exact 6-class, + snow) | yes | 1.6 % | **BEST training candidate** |
| **S3** | dusty and bird-droppings | `solar-panel-a5cyn / dusty-and-bird-droppings / v1` | detection (boxes) | yolov8 | 710 (**710**, no aug) | CC BY 4.0 | **C** | DUSTY, BIRD_DROP only | yes | 0.3 % | **SECONDARY** — the only mostly-independent top-up (dusty/bird-drop) |
| **C3** | solar panel4 | `solar-panel-4isfg / solar-panel4 / v2` | **classification** | folder | 1,738 (~720) | CC BY 4.0 | **C** | CLEAN, DUSTY, BIRD_DROP, PHYSICAL_DAMAGE, ELECTRICAL_DAMAGE (image-level) | yes | 1.3 % | **CLASSIFICATION-ONLY** — but ~99 % of its images are already in S1 |
| C1 | SOLAR PANEL DEFECT | `solar-panel-defects-ax0gw / solar-panel-defect-w1dfi / v8` | detection (boxes) | yolov8 | 1,368 (~570) | CC BY 4.0 | **C** | Non-Defective, Dusty, Bird-drop, Physical/Electrical-Damage, + ambiguous **Defective**, + Snow | yes (macro-heavy) | 2.0 % | **RESEARCH-ONLY / provenance-risk** — Pool B, ~74–92 % overlaps S6, ambiguous "Defective" |
| S6 | Solar Panel defects (SolarPanel2) | `solarpanel2 / solar-panel-defects-b17u9 / v4` | detection (boxes) | yolov8 | 780 (~691) | CC BY 4.0 | **C** | same 7-class scheme as C1 (Defective/Non-Defective + defects + snow) | yes | 3.9 % | **RESEARCH-ONLY** — Pool B twin of C1 (byte-identical images, re-numbered) |
| S4 | Solar Panel Anomalies Bird Drop | `mrudula-tqirx / solar-panel-anomalies-bird-drop-69qgu / v1` | detection (boxes) | yolov8 | 120 (120) | CC BY 4.0 | **C** | BIRD_DROP (+ snow) | yes | 0.0 % | **RESEARCH-ONLY** — 96 % of its images are already in S1 / C3; adds ~nothing |
| S2 | Solar Panel Damage Detection | `solar-panel-defects-ax0gw / solar-panel-damage-detection-vfg96` | detection | — | — | Public Domain | **C** | 7-class (same as C1) | — | — | **BLOCKED** — 0 generated versions → undownloadable |
| S5 | Solar Panel defects | `solarpanel-2me5p / solar-panel-defects-lnge0` | detection | — | — | CC BY 4.0 | **C** | chaotic (18 spelling variants of ~6 classes) | — | — | **BLOCKED** — 0 versions + unmerged taxonomy |
| **C2** | SolarPanelPakZidni 2 | `ganibs-workspace / solarpanelpakzidni-2` | ? | — | — | **unverifiable** | **D** | — | — | — | **BLOCKED** — HTTP 404, cannot verify it exists (all slug variants + Universe search failed) |

> C3's requested project slug `solarpanel4` did not resolve; the working
> slug is `solar-panel4` (hyphen). C2 was tried as `solarpanelpakzidni-2`,
> `solarpanelpakzidni2`, `solarpanelpakzidni-2-yfjqz` — all 404.

## Class balance (downloaded candidates)

Per-class **unique-source-image** estimates (image count ÷ aug factor;
detection = images-containing-class; classification = folder counts):

| unified class | S1 (det) | S3 (det) | C3 (cls) | C1 (det) | S6 (det) |
|---|---|---|---|---|---|
| CLEAN | ~382 | — | ~207 | ~179 (Non-Defective) | ~199 (Non Defective) |
| DUSTY | ~377 | **~469** | ~227 | ~76 | ~124 |
| BIRD_DROP | ~335 | ~238 | ~163 | ~51 | ~69 |
| PHYSICAL_DAMAGE | ~131 | — | ~53 | ~275 | **~312** |
| ELECTRICAL_DAMAGE | **~199** | — | ~85 | ~58 | ~71 |
| — Defective (ambiguous, unmapped) — | — | — | — | ~198 | ~338 |
| — Snow (out of scope, unmapped) — | ~252 | — | ~111 | ~41 | ~50 |

Splits (as downloaded, incl. augmentation):
S1 1713/162/88 · S3 600/60/50 · C3 1521/141/76 · C1 1197/114/57 · S6 499/174/107 · S4 84/24/12.

Rare-class reality (independent, deduped, licence-passing):
**BIRD_DROP ~500–550 · PHYSICAL_DAMAGE ~350–400 · ELECTRICAL_DAMAGE ~200–230.**
Electrical is the bottleneck, and S1's "electrical-damage" boxes visually
correspond to **cell cracks / microcracks**, not always electrical failure.

## Class taxonomy mapping (proposed)

| unified | maps from (aliases) |
|---|---|
| **CLEAN** | `Non-Defective`, `Non Defective`, `clean`, `Clean` |
| **DUSTY** | `Dusty`, `dusty`, `dusdusty` (S3 typo) |
| **BIRD_DROP** | `Bird-drop`, `bird-drop`, `Bird Drop`, `Bird-Drop`, `bird-droppings` (S3), `Bird_drop` |
| **PHYSICAL_DAMAGE** | `Physical-Damage`, `physical-damage`, `Physical Damage`, `Physical_Damge` (typo) |
| **ELECTRICAL_DAMAGE** | `Electrical-Damage`, `electrical-damage`, `Electrical Damage`, `Electrical-damage`, `Electrical_Damge` (typo) |
| **PANEL** | (none of these candidates provide panel boxes) |
| **— do NOT map —** | `Defective` (ambiguous — keep as its own source label), `Snow` / `snow-covered` / `Snow-Covered` (out of scope) |

## Per-candidate suitability

Suitability score = 0–5 (independent useful images + taxonomy fit + annotation
type + ground-level fit + provenance ceiling of grade C).

| id | score | why |
|---|---|---|
| **S1** | **3.5 / 5** | exact 6-class detection taxonomy, ~817 unique ground-level RGB with boxes, best electrical/physical coverage of any candidate, near-zero overlap with existing data. Capped at grade C. |
| **S3** | **2.5 / 5** | ~710 truly independent, un-augmented, ground-level RGB with boxes — but only dusty + bird-drop, and it's the *only* candidate that meaningfully tops up S1. Grade C. |
| **C3** | **2.0 / 5** | clean 6-class *labels* (good for a classifier), ~720 images — but ~99 % are the same photos as S1, and it has no boxes. Grade C. |
| **S6** | **1.5 / 5** | ~691 images with boxes and decent physical-damage — but ~74–92 % overlap C1, ambiguous "Defective", Pool B (grade C). |
| **C1** | **1.5 / 5** | ~570 images with boxes — Pool B twin of S6, physical-damage class skews to extreme cracked-cell macro shots, ambiguous "Defective". Grade C. |
| **S4** | **0.5 / 5** | 120 images, 96 % already in S1/C3 — adds essentially nothing. Grade C. |
| **S2 / S5** | **0 / 5** | undownloadable (0 versions). |
| **C2** | **0 / 5** | cannot verify it exists (grade D). |

---

## FINAL RANKING

1. **BEST TRAINING CANDIDATE — S1 `defects-kgig6/detection-for-defects-in-solar-panels` v2.**
   Exact 6-class detection taxonomy, ~817 unique ground-level RGB photos with
   bounding boxes, the best electrical- and physical-damage coverage of any
   candidate, and it is disjoint from the existing Solar AI ground data.
   *Caveat:* provenance grade **C** (Unknown-licence Kaggle ancestry); use for
   research with that documented.
2. **SECONDARY CANDIDATE — S3 `solar-panel-a5cyn/dusty-and-bird-droppings` v1.**
   The only candidate that meaningfully *adds independent images* on top of S1
   (~500+ net-new dusty / bird-drop photos, un-augmented, with boxes). Only two
   classes; provenance grade **C**.
3. **CLASSIFICATION-ONLY CANDIDATE — C3 `solar-panel-4isfg/solar-panel4` v2.**
   Clean 6-class image-level labels for training/validating a *classifier*
   (not the detector). Its images are ~99 % already in S1, so it contributes
   no new pixels — value is purely the label taxonomy. Grade **C**.
4. **RESEARCH-ONLY / PROVENANCE RISK — C1, S6** (Pool B; overlapping, ambiguous
   "Defective", macro-heavy damage classes), and **S4** (redundant with S1).
5. **BLOCKED — C2** (unverifiable, grade D), **S2** and **S5** (0 downloadable
   versions).

## Recommendation for the next phase

**Proceed with S1 as the primary expansion source, topped up by S3 for
dusty/bird-drop — both flagged provenance grade C.** Concretely:

- Take **S1 v2** (least-augmented; de-augment to its ~817 unique source
  images) as the multi-class defect base: `clean, dusty, bird_drop,
  physical_damage, electrical_damage` (drop `snow-covered`).
- Add **S3** (~710 unique) for `dusty` + `bird_drop` only, after a
  cross-dedup against S1 (~9 % overlap to remove).
- Keep **C3** aside as a *classification* validation set if a soiling/defect
  classifier is later built.
- **Do not** bring in C1 / S6 / S4 — they add almost no independent images and
  carry the ambiguous "Defective" label.
- `panel` boxes must continue to come from the existing Susan detector data;
  none of these candidates supply them.
- **Before any training or redistribution:** resolve the `pythonafroz` Kaggle
  source licence (it is "Unknown"). Until then this expansion is
  research-internal only.
- Realistic post-merge dataset: **existing 6,492 (panel/cracked/dusty)** +
  **~1,300–1,500 new independent multi-class defect images** → a 6-class
  research detector, with `electrical_damage` (~200 imgs) and
  `physical_damage` (~350 imgs) still thin enough to warrant a caveat like the
  one applied to `bird_drop` in the first experiment.
