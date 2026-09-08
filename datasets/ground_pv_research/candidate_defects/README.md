# candidate_defects/ — ground-PV defect dataset expansion: discovery + audit

**This phase is DISCOVERY + PROVENANCE AUDIT ONLY.** No training, no
fine-tuning, no dataset merging, no production changes. The existing
`datasets/ground_pv_research/{raw,yolo}` data was not modified.

Goal: expand the ground-PV research taxonomy to
`panel · clean/non-defective · dusty · bird_drop · physical_damage ·
electrical_damage`. **Hotspot stays out** of the RGB taxonomy (thermal only).

## Layout

```
candidate_defects/
├── raw/          immutable Roboflow exports of the licence-passing candidates   (gitignored)
│                 each dir has the source README + data.yaml + a MANIFEST.json
├── metadata/     Roboflow API metadata for every candidate (incl. blocked ones)  (committed)
├── manifests/    per-download provenance + candidate_manifest.json               (committed)
├── overlap/      overlap_report.{md,json} + overlap_matrix.json                  (committed)
│   └── _fingerprints/   per-image sha256 / 16x16 dHash dumps                     (gitignored)
└── reports/      candidate_comparison.{md,json}                                  (committed)
```

## Read these

- **`reports/candidate_comparison.md`** — the ranked comparison + the
  recommendation for the next phase.
- **`overlap/overlap_report.md`** — the duplicate/overlap audit and the
  provenance finding (all candidates trace to an "Unknown"-licence Kaggle
  dataset).

## One-line result

Every downloadable candidate is **provenance grade C** (Unknown-licence
Kaggle ancestry re-declared "CC BY 4.0" on Roboflow). The largest genuinely
independent, useful pool is **S1** (`defects-kgig6/detection-for-defects-in-solar-panels`
v2, ~817 unique images, exact 6-class detection taxonomy) topped up by
**S3** (`solar-panel-a5cyn/dusty-and-bird-droppings` v1, ~710 unique, dusty +
bird-drop only). Nothing is licence-cleared; use is research-internal until
the `pythonafroz` Kaggle source licence is resolved.

## Reproduce

Metadata + downloads + audit were run with the Roboflow API
(`ROBOFLOW_API_KEY` via the project `.env`). Fingerprint / overlap logic:
`training/detection/ground_pv/image_fingerprint.py`.
