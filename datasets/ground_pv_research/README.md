# Ground-level PV defect research dataset

Research / non-commercial. This directory holds the data pipeline for a
**separate research object detector** for ground-level / close-up RGB
photographs of solar panels — distinct from the production aerial detector
(`weights/yolo_solar.pt`), which is **not** modified, replaced, or retrained
by any of this work.

## Classes (canonical research taxonomy, fixed order)

| id | name       | meaning                                   |
|----|------------|-------------------------------------------|
| 0  | panel      | individual solar-panel boundary           |
| 1  | bird_drop  | bird-dropping contamination region        |
| 2  | cracked    | crack / physical damage region            |
| 3  | dusty      | dust / soiling region                     |

A panel may carry multiple defects (`panel + bird_drop`, `panel + dusty`, …).
**Hotspot is deliberately not a class** — RGB photographs cannot validate
thermal-hotspot semantics (see the project's existing Hotspot data audit).

## Source dataset

> Susan. *"panel solar Dataset."* Roboflow Universe.
> https://universe.roboflow.com/susan-ifblr/panel-solar-bw945

- workspace / project / version: `susan-ifblr` / `panel-solar-bw945` / `1`
- export format: YOLOv8
- license **as displayed by Roboflow**: CC BY 4.0
- source class names: `panel, bird_drop, cracked, dusty`
- published (unverified) size: ~6,493 images; instance counts cracked 6233 /
  panel 4477 / dusty 235 / bird_drop 60

Published counts are **not trusted** — `scripts/audit_ground_pv_dataset.py`
performs an independent audit.

## Directory layout

Per the project storage policy (large local data lives on the E: drive,
not the low-space C:/D: drives — see
`training/cloud/base/storage_paths.py` and
`training/detection/ground_pv/paths.py`), the **bulk image data** resolves
to `E:/Solar AI Training Images/ground_pv_research/...` when that drive is
available, falling back to the repo-relative paths below otherwise (CI, a
machine with no E: drive). Only the small `reports/` tree is always kept in
the repo and committed.

```
<bulk data root>/ground_pv_research/           # E:/Solar AI Training Images/... or <repo>/datasets/...
├── raw/panel_solar_bw945_v1/     # immutable Roboflow download          (never committed)
├── yolo/                         # class-normalized Ultralytics copy    (never committed)
├── splits/
│   ├── official_split/           # the source's own train/valid/test    (never committed)
│   └── leakage_clean_split/      # only if cross-split leakage is found  (never committed)
└── real_world_holdout/images/    # real rooftop photos (qualitative only, never committed)

<repo>/datasets/ground_pv_research/
├── reports/                      # dataset_audit.md/json, evaluation, manifests  (COMMITTED)
└── real_world_holdout/README.md  # provenance record                    (COMMITTED)
```

## Workflow

```bash
export ROBOFLOW_API_KEY=<key>                      # required — never commit it
python scripts/download_ground_pv_dataset.py       # -> raw/panel_solar_bw945_v1/
python scripts/audit_ground_pv_dataset.py          # -> reports/dataset_audit.{md,json}
# (normalize + Kaggle packaging + Kaggle GPU training — see kaggle/README.md)
python scripts/evaluate_ground_pv.py  --weights weights/yolo_solar_ground_research_v1.pt \
                                      --data-yaml datasets/ground_pv_research/yolo/data.yaml
python scripts/infer_ground_pv.py     --weights weights/yolo_solar_ground_research_v1.pt \
                                      --images datasets/ground_pv_research/real_world_holdout/images \
                                      --output-dir research_runs/ground_pv/real_world_predictions
```

See `docs/GROUND_PV_RESEARCH.md` for the full reproducibility record.
