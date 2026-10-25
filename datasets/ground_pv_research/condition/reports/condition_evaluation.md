# Ground-PV CONDITION/DEFECT detector — evaluation

> EXPERIMENTAL research model. Second-stage condition detector (runs on panel
> crops). NOT a panel detector, NOT production. Provenance grade C (`personal-experimental-unverified-provenance`).

Weights: `weights\yolo_solar_condition_research_v1.pt`  
SHA-256: `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578`  
Ultralytics 8.4.118 · imgsz=640 · iou=0.5

## Split: test

- images: 250

**Overall** — precision 0.623 · recall 0.4491 · mAP50 0.4718 · mAP50-95 0.254

| class | precision | recall | AP50 | AP50-95 | test imgs w/ class | test instances | unique families |
|---|---|---|---|---|---|---|---|
| CLEAN | 0.5897 | 0.4967 | 0.5162 | 0.3348 | 95 | 545 | 53 |
| DUSTY | 0.4925 | 0.3476 | 0.3577 | 0.1627 | 144 | 863 | 90 |
| BIRD_DROP | 0.5654 | 0.3203 | 0.374 | 0.1494 | 100 | 726 | 56 |
| PHYSICAL_DAMAGE | 0.6595 | 0.5256 | 0.5228 | 0.275 | 30 | 78 | 18 |
| ELECTRICAL_DAMAGE | 0.8079 | 0.5551 | 0.5882 | 0.348 | 44 | 144 | 31 |

### ELECTRICAL_DAMAGE — ⚠️ HIGH SEMANTIC RISK (test)

Every ELECTRICAL_DAMAGE label is S1's `electrical-damage` class. The provenance
audit found these boxes visually correspond to **cell cracks**, not confirmed
electrical failure. Labels were preserved unchanged (not relabelled). **A high
score here is NOT an electrical-fault diagnosis** — it may simply reflect the
source dataset labelling cell cracks as `electrical-damage`.

- precision 0.8079 · recall 0.5551 · AP50 0.5882 · AP50-95 0.348
- test images with class: 44 · instances: 144 · unique families: 31

## Split: val

- images: 502

**Overall** — precision 0.6237 · recall 0.4793 · mAP50 0.5205 · mAP50-95 0.2792

| class | precision | recall | AP50 | AP50-95 | test imgs w/ class | test instances | unique families |
|---|---|---|---|---|---|---|---|
| CLEAN | 0.655 | 0.5254 | 0.5763 | 0.3581 | 184 | 1246 | 106 |
| DUSTY | 0.5053 | 0.3427 | 0.3829 | 0.1711 | 261 | 1637 | 163 |
| BIRD_DROP | 0.5287 | 0.3355 | 0.3469 | 0.1394 | 194 | 1516 | 118 |
| PHYSICAL_DAMAGE | 0.7254 | 0.6205 | 0.6738 | 0.3563 | 79 | 166 | 52 |
| ELECTRICAL_DAMAGE | 0.7042 | 0.5723 | 0.6226 | 0.3712 | 95 | 208 | 61 |

### ELECTRICAL_DAMAGE — ⚠️ HIGH SEMANTIC RISK (val)

Every ELECTRICAL_DAMAGE label is S1's `electrical-damage` class. The provenance
audit found these boxes visually correspond to **cell cracks**, not confirmed
electrical failure. Labels were preserved unchanged (not relabelled). **A high
score here is NOT an electrical-fault diagnosis** — it may simply reflect the
source dataset labelling cell cracks as `electrical-damage`.

- precision 0.7042 · recall 0.5723 · AP50 0.6226 · AP50-95 0.3712
- val images with class: 95 · instances: 208 · unique families: 61

> CLEAN: trained only on explicit clean-region annotations. Absence of a defect
> detection is NOT proof of Clean. PHYSICAL_DAMAGE / ELECTRICAL_DAMAGE families are
> few — per-class metrics are high-variance; do not over-interpret.

