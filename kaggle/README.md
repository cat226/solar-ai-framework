# kaggle/ — GPU training for the ground-level PV defect research detector

## Why Kaggle

The dev machine has no usable GPU and local YOLO training on CPU is
impractical (~1.5 h/epoch was measured for the aerial detector). Kaggle
gives a free NVIDIA GPU (T4 preferred; P100 also handled). All data
acquisition, auditing, normalization, evaluation, and real-world testing
run **locally**; only the training loop runs on Kaggle.

## What's here

| path | purpose |
|---|---|
| `ground_pv_train.ipynb` | the reproducible training + evaluation + packaging notebook |
| `ground_pv_dataset/` | local staging area for the Kaggle Dataset (bulk contents gitignored) |
| `ground_pv_dataset/dataset-metadata.json` | Kaggle Dataset metadata template — edit `id` before upload |

## Dataset source & license

> Susan. *"panel solar Dataset."* Roboflow Universe.
> <https://universe.roboflow.com/susan-ifblr/panel-solar-bw945>
> workspace/project/version `susan-ifblr` / `panel-solar-bw945` / `1`, YOLOv8 export.

License **as displayed by Roboflow**: CC BY 4.0. Solar AI did not create the
photographs; attribution is preserved. Image-level provenance is not
independently verified.

## Classes

`0 = panel`, `1 = bird_drop`, `2 = cracked`, `3 = dusty`. Hotspot is
excluded (RGB cannot validate thermal-hotspot semantics).

## Training configuration

| | |
|---|---|
| experiment | `ground_yolo11n_baseline_v1` |
| base model | `yolo11n.pt` (COCO-pretrained) — **never** `weights/yolo_solar.pt` |
| imgsz | 640 |
| epochs / patience | 100 / 20 |
| batch | `-1` (auto) |
| seed | 42 |
| optimizer | `auto` |
| augmentation | conservative — hflip, mild rotation/scale/translate, mild HSV; no vertical flip, no mixup/copy-paste |
| second experiment | `ground_yolo11n_balanced_v1` — train-only weighted image sampling for the rare classes (bird_drop, dusty); val/test never oversampled |

Expected GPU: NVIDIA T4. The notebook auto-detects (`torch.cuda.get_device_name(0)`)
and **fails closed** if CUDA is unavailable. `torch==2.7.1` / `torchvision==0.22.1`
are pinned so a Pascal P100 (sm_60) also works.

## Reproduce

1. **Local**: `python scripts/download_ground_pv_dataset.py` (needs
   `ROBOFLOW_API_KEY`) → `python scripts/audit_ground_pv_dataset.py` →
   normalize → `python -m training.detection.ground_pv.build_kaggle_dataset`.
2. Edit `ground_pv_dataset/dataset-metadata.json` `id` to
   `<your-username>/solar-ai-ground-pv-research-v1`.
3. `kaggle datasets create -p kaggle/ground_pv_dataset --dir-mode zip`.
4. New Kaggle Notebook → upload `ground_pv_train.ipynb` → Settings →
   Accelerator = GPU → attach the dataset → Run All.
5. Download `/kaggle/working/output/` (`best.pt`, `last.pt`, `results.*`,
   confusion matrices, PR/F1 curves, `predictions/`, `evaluation.json`,
   `training_manifest.json`). Optionally save it as a new Kaggle Dataset
   version for durability and record that dataset id.

## Retrieve the checkpoint locally

```bash
cp <downloaded>/best.pt weights/yolo_solar_ground_research_v1.pt   # NEVER weights/yolo_solar.pt
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/yolo_solar_ground_research_v1.pt').read_bytes()).hexdigest())"
python scripts/evaluate_ground_pv.py --weights weights/yolo_solar_ground_research_v1.pt \
    --data-yaml datasets/ground_pv_research/yolo/data.yaml --splits val test
python scripts/infer_ground_pv.py --weights weights/yolo_solar_ground_research_v1.pt \
    --images datasets/ground_pv_research/real_world_holdout/images \
    --output-dir research_runs/ground_pv/real_world_predictions
```

Record the SHA-256, training args, and evaluation in
`datasets/ground_pv_research/reports/ground_research_model_manifest.json`.

## Limitations

- `bird_drop` (~60 instances) and `dusty` (~235) are severely
  underrepresented — their metrics are high-variance; do not over-interpret.
- Source image-level provenance is unverified.
- Domain gap: the source images may differ from the owner's rooftop photos;
  the real-world holdout is qualitative only.
- No RGB hotspot detection — out of scope by design.
- A baseline run is **not** production-ready.
