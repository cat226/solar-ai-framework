# Candidate-dataset overlap / provenance audit

Generated 2026-09-08. **Discovery + audit only — no training, no merging.**

## Method

For every image: exact **SHA-256**, a **16×16 difference hash** (256-bit
perceptual near-dup, Hamming ≤ 6 counted as a match), the **source basename**
(Roboflow `_jpg.rf.<hash>` export suffix stripped), and pixel **dimensions**.
"Overlap" of candidate A against a reference B = the fraction of A's images
that match ≥ 1 image in B by *any* of those signals.

Reference sets (read-only, **not modified**):
- `existing_raw_susan_bw945_v1` — 6,493 images (the raw Roboflow "panel solar" v1)
- `existing_derived_yolo_6492` — 6,492 images (the remediated leakage-clean 3-class set)

## 1. Candidate vs EXISTING Solar AI ground data

| candidate | images | vs raw Susan | vs derived 6492 | exact | shared basename |
|---|---|---|---|---|---|
| C1 SOLAR PANEL DEFECT | 1,368 | 2.0 % | 2.0 % | 0 | 0 |
| C3 solar panel4 | 1,738 | 1.3 % | 1.3 % | 0 | 0 |
| S1 detection-for-defects | 1,963 | 1.6 % | 1.6 % | 0 | 0 |
| S3 dusty-and-bird-droppings | 710 | 0.3 % | 0.3 % | 0 | 0 |
| S4 anomalies-bird-drop | 120 | 0.0 % | 0.0 % | 0 | 0 |
| S6 solar-panel-defects (b17u9) | 780 | 3.9 % | 3.9 % | 0 | 0 |

**Every candidate is essentially disjoint from the existing Solar AI ground
data** — 0–4 %, no exact duplicates, no shared filenames. The few percent
perceptual hits are coincidental (generic close-up polycrystalline-panel
texture → similar 16×16 hash); one genuine thread is a handful of crack
photos shared between S1 (`Physical-damaged-*`) and Susan (`pv_crack-*`,
`Crack_*`) at Hamming 1–2 (≈ 31 / 1,963 for S1). **Merging any candidate
would add genuinely new images** relative to the current set — but a fresh
combined leakage-clean split would still be required (see §3).

## 2. Candidate ↔ candidate overlap (the "shared source" problem)

Fraction of the ROW dataset's images that near-duplicate an image in the COLUMN dataset:

| | C1 | C3 | S1 | S3 | S4 | S6 |
|---|---|---|---|---|---|---|
| **C1** | — | 20 % | 19 % | 5 % | 3 % | **74 %** |
| **C3** | 16 % | — | **99 %** | 9 % | 13 % | 16 % |
| **S1** | 14 % | **89 %** | — | 9 % | 12 % | 14 % |
| **S3** | 11 % | 24 % | 26 % | — | 8 % | 11 % |
| **S4** | 34 % | **93 %** | **96 %** | 40 % | — | 34 % |
| **S6** | **92 %** | 41 % | 39 % | 11 % | 6 % | — |

Two overlapping source pools plus one mostly-independent set:

- **Pool A — `S1 ⊇ C3 ⊇ S4`.** S1 (817 unique source images) *superset-contains*
  C3 (99 %) and S4 (96 %). C3 is the same images as a **classification** re-cut;
  S4 is a 120-image bird-drop slice. **C3 and S4 add ≈ 0 independent images
  over S1.**
- **Pool B — `C1 ≈ S6`.** 74–92 % shared; the audit found **byte-identical
  images with different filenames** (`Physical_Damge_103` in C1 == SHA-256 of
  `Physical_Damge_104` in S6) — the same photos re-numbered. Both use the
  ambiguous "Defective / Non-Defective" scheme. Pool B overlaps Pool A by
  ~14–19 %.
- **S3 — mostly independent.** 8–26 % overlap with everything else; distinct
  `Imgdirty_*` filenames and a visibly different panel style (monocrystalline).
  Only 2 classes (dusty + bird-drop), no augmentation.

### Within-candidate augmentation families

| candidate | images | distinct basenames (≈ unique source) | distinct SHA-256 | within-set exact dups | aug factor |
|---|---|---|---|---|---|
| C1 | 1,368 | 570 | 1,368 | 0 | ×2.40 |
| C3 | 1,738 | 720 | 1,737 | 1 | ×2.41 |
| S1 | 1,963 | 817 | 1,961 | 2 | ×2.40 |
| S3 | 710 | 710 | 705 | 5 | ×1.00 |
| S4 | 120 | 120 | 119 | 1 | ×1.00 |
| S6 | 780 | 691 | 779 | 1 | ×1.13 |

## 3. Independent-image estimate (license-passing candidates only)

Deduplicating across pools, the realistic pool of **independent** ground-level
RGB PV images is **≈ 1,900–2,100 unique**, concentrated in the common classes:

| unified class | best independent unique-image estimate | drawn mainly from |
|---|---|---|
| CLEAN | ~450–480 | S1 (~382) + S6/C1 net-new |
| DUSTY | ~750–800 | S1 (~377) + S3 (~400 net-new) |
| BIRD_DROP | ~500–550 | S1 (~335) + S3 (~200 net-new) |
| PHYSICAL_DAMAGE | ~350–400 | S6 (~312) + S1 (~131), many are macro shots of shattered cells |
| **ELECTRICAL_DAMAGE** | **~200–230** ← binding constraint | S1 (~199) + S6/C1 (~50 net) |
| PANEL (boxes) | 0 from these candidates | must still come from the existing Susan detector data |

## 4. Provenance

The 6-class RGB taxonomy (Clean / Dusty / Bird-drop / Electrical-Damage /
Physical-Damage / Snow-Covered) and the exact filename conventions
(`Bird_drop_N`, `Clean_N`, `Physical_Damge_N`, `Electrical_Damge_N`) trace to
a widely-cited Kaggle dataset — **"Solar Panel Images Clean and Faulty
Images" / "Solar panel clean and faulty images" by Kaggle user
`pythonafroz`** (kaggle.com/datasets/pythonafroz/solar-panel-images). Its
Kaggle **licence is "Unknown"** (confirmed via the Kaggle dataset-view API).

**A Roboflow re-uploader cannot upgrade an "Unknown"-licence source to "CC BY
4.0".** The "CC BY 4.0" shown on every candidate's Roboflow page is a
**dataset-level re-declaration**, not a verified image-level right.

| grade | meaning |
|---|---|
| A | strong / documented original source + clear licence |
| B | reasonable but incomplete provenance |
| C | unresolved / high-risk (Unknown-licence Kaggle ancestry, re-declared CC BY) |
| D | blocked — cannot verify the dataset exists |

**Every downloadable candidate is grade C. C2 is grade D (unresolvable).**
None is licence-cleared. Usable for internal **research** with the risk
documented; **not** for redistribution or commercial use without
establishing the `pythonafroz` (or other) source licence.
