# kaggle/ground_pv_dataset/ — Kaggle Dataset staging for the ground PV research detector

This folder is the **local staging area** for the Kaggle Dataset that the
GPU training notebook (`kaggle/ground_pv_train.ipynb`) consumes. Its bulk
contents (`images/`, `labels/`) are **gitignored** — never commit the
dataset to this repository.

## What goes here (built by the prep step, not by hand)

```
kaggle/ground_pv_dataset/
├── dataset-metadata.json     # committed template — edit the "id" to <your-username>/solar-ai-ground-pv-research-v1
├── README.md                 # this file (committed)
├── data.yaml                 # generated: research taxonomy (0 panel / 1 bird_drop / 2 cracked / 3 dusty)
├── images/ train|valid|test  # generated: byte-copies from datasets/ground_pv_research/yolo/
├── labels/ train|valid|test  # generated: class-remapped YOLO labels
├── dataset_audit.md/json     # copied from datasets/ground_pv_research/reports/
└── NORMALIZATION.json        # copied from datasets/ground_pv_research/yolo/
```

## Build + upload

```bash
# 1. acquire + audit + normalize (needs ROBOFLOW_API_KEY)
python scripts/download_ground_pv_dataset.py
python scripts/audit_ground_pv_dataset.py
python -m training.detection.ground_pv.build_kaggle_dataset \
    --normalized-dir datasets/ground_pv_research/yolo \
    --staging-dir kaggle/ground_pv_dataset

# 2. set the dataset id
#    edit kaggle/ground_pv_dataset/dataset-metadata.json  "id": "<username>/solar-ai-ground-pv-research-v1"

# 3. validate, then create (first time) or version (subsequent)
kaggle datasets create   -p kaggle/ground_pv_dataset --dir-mode zip
kaggle datasets version  -p kaggle/ground_pv_dataset -m "audited v1" --dir-mode zip
```

**Credentials**: `kaggle` reads `~/.kaggle/kaggle.json` (or
`KAGGLE_USERNAME` / `KAGGLE_KEY`, or `~/.kaggle/access_token`). Never commit
any of them.

## License / provenance

CC BY 4.0 as displayed by the source
(<https://universe.roboflow.com/susan-ifblr/panel-solar-bw945>). Solar AI
did not create the photographs; attribution is preserved in
`dataset-metadata.json` and `data.yaml`. Image-level provenance of the
individual photographs is **not** independently verified — see the audit
report bundled into the dataset.
