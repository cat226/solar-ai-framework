"""Generate kaggle/ground_condition_train.ipynb — the Kaggle GPU notebook that
trains the YOLO11n CONDITION/DEFECT baseline on
edithstark/solar-ai-ground-condition-research-v1.

Regenerable: `python scripts/make_condition_notebook.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "kaggle" / "ground_condition_train.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").splitlines()]}


CELLS = [
    md(
        "# Solar AI — ground-PV CONDITION/DEFECT research detector (YOLO11n)",
        "",
        "Trains `condition_yolo11n_baseline_v1` on a Kaggle GPU. **5-class second-stage condition model:**",
        "`0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE 4=ELECTRICAL_DAMAGE`.",
        "",
        "- **Experimental research model only.** NOT the panel detector. Does NOT touch / replace",
        "  the production aerial detector `weights/yolo_solar.pt` or the ground panel detector",
        "  `weights/yolo_solar_ground_research_v1.pt`.",
        "- Base model: pretrained **YOLO11n** COCO checkpoint (never a Solar AI model).",
        "- Dataset: `edithstark/solar-ai-ground-condition-research-v1` (PRIVATE). Built from Roboflow",
        "  S1 `defects-kgig6/detection-for-defects-in-solar-panels` v2 + S3",
        "  `solar-panel-a5cyn/dusty-and-bird-droppings` v1, de-duplicated, leakage-safe split.",
        "",
        "## ⚠️ PROVENANCE — grade C, UNVERIFIED",
        "The underlying images ultimately trace to Kaggle `pythonafroz/solar-panel-images`",
        "(licence **Unknown**). Tag: `personal-experimental-unverified-provenance`. Do NOT",
        "redistribute / publish / use commercially / claim verified CC BY image rights.",
        "",
        "## ⚠️ ELECTRICAL_DAMAGE — HIGH SEMANTIC RISK",
        "Every ELECTRICAL_DAMAGE annotation is S1's `electrical-damage` class. The provenance",
        "audit found these boxes visually correspond to **cell cracks**, not confirmed electrical",
        "failure. Labels are preserved unchanged (not relabelled). A high ELECTRICAL_DAMAGE score",
        "is **not** evidence of electrical-fault diagnosis. Metrics for this class are reported",
        "separately and prominently.",
        "",
        "## CLEAN — semantic limitation",
        "CLEAN is trained only where the source carried explicit clean-region annotations. Absence",
        "of a defect detection is **not** proof an image/panel is Clean. No CLEAN boxes were",
        "fabricated.",
        "",
        "imgsz=640, epochs=100, patience=20, seed=42, cos_lr, optimizer=auto. Attach the dataset",
        "in the sidebar and set Accelerator = GPU before running.",
    ),
    md("## 1. Environment"),
    code(
        "import sys, platform, subprocess, json, os, time, hashlib, shutil\n"
        "from pathlib import Path\n"
        "print('Python :', platform.python_version())"
    ),
    md(
        "### CUDA — fail closed if unavailable",
        "Kaggle's free GPU pool can hand out a Pascal **Tesla P100 (sm_60)**. Pin",
        "`torch==2.7.1` / `torchvision==0.22.1` (CUDA 12.6 wheels, last line retaining Pascal",
        "support) — the pins proven on this account's prior ground-detector run.",
    ),
    code(
        "PINNED = ['torch==2.7.1', 'torchvision==0.22.1', 'ultralytics==8.4.138', 'PyYAML>=6.0.1']\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINNED], check=True)\n"
        "\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA unavailable — set Settings -> Accelerator -> GPU.'\n"
        "import torchvision, ultralytics\n"
        "ENV = dict(\n"
        "    python=platform.python_version(), torch=torch.__version__,\n"
        "    torchvision=torchvision.__version__, cuda=torch.version.cuda,\n"
        "    ultralytics=ultralytics.__version__,\n"
        "    gpu=torch.cuda.get_device_name(0),\n"
        "    gpu_capability='.'.join(map(str, torch.cuda.get_device_capability(0))),\n"
        "    gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),\n"
        ")\n"
        "print(json.dumps(ENV, indent=2))"
    ),
    md("## 2. Locate the attached Kaggle dataset + verify provenance"),
    code(
        "candidates = list(Path('/kaggle/input').rglob('data.yaml'))\n"
        "assert candidates, 'No data.yaml under /kaggle/input — attach the dataset in the sidebar.'\n"
        "SRC_DATA_YAML = None\n"
        "for c in candidates:\n"
        "    if 'condition' in str(c).lower():\n"
        "        SRC_DATA_YAML = c; break\n"
        "SRC_DATA_YAML = SRC_DATA_YAML or candidates[0]\n"
        "DATA_ROOT = SRC_DATA_YAML.parent\n"
        "print('mounted data.yaml :', SRC_DATA_YAML)\n"
        "print(SRC_DATA_YAML.read_text())\n"
        "\n"
        "import yaml as _yaml\n"
        "_cfg = _yaml.safe_load(SRC_DATA_YAML.read_text())\n"
        "_cfg['path'] = str(DATA_ROOT.resolve())\n"
        "_cfg['train'], _cfg['val'], _cfg['test'] = 'images/train', 'images/valid', 'images/test'\n"
        "DATA_YAML = Path('/kaggle/working/ground_condition_data.yaml')\n"
        "DATA_YAML.write_text(_yaml.safe_dump(_cfg, sort_keys=False))\n"
        "print('training data.yaml (absolute):'); print(DATA_YAML.read_text())\n"
        "for _s in ('images/train','images/valid','images/test','labels/train','labels/valid','labels/test'):\n"
        "    assert (DATA_ROOT / _s).is_dir(), f'missing {_s} under {DATA_ROOT}'\n"
        "print('all split dirs present')\n"
        "\n"
        "# provenance shipped inside the dataset\n"
        "PROV = DATA_ROOT / 'provenance'\n"
        "assert PROV.is_dir(), 'no provenance/ directory in the dataset'\n"
        "pkg_summary = json.loads((DATA_ROOT / 'PACKAGE_SUMMARY.json').read_text())\n"
        "print('provenance_grade :', pkg_summary['provenance_grade'])\n"
        "print('provenance_tag   :', pkg_summary['provenance_tag'])\n"
        "print('electrical_damage_semantic_risk :', pkg_summary['electrical_damage_semantic_risk']['flag'])\n"
        "assert pkg_summary['provenance_grade'] == 'C'\n"
        "assert pkg_summary['provenance_tag'] == 'personal-experimental-unverified-provenance'\n"
        "assert pkg_summary['electrical_damage_semantic_risk']['flag'] == 'HIGH'\n"
        "assert pkg_summary['no_family_crosses_splits'] is True\n"
        "\n"
        "def _sha_file(p):\n"
        "    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()\n"
        "SPLIT_CSV = PROV / 'condition_split_manifest.csv'\n"
        "TAXO_JSON = PROV / 'condition_taxonomy.json'\n"
        "DATASET_HASHES = dict(\n"
        "    split_manifest_sha256=_sha_file(SPLIT_CSV),\n"
        "    taxonomy_sha256=_sha_file(TAXO_JSON),\n"
        "    package_dataset_content_sha256=pkg_summary['dataset_content_sha256'],\n"
        "    package_data_yaml_sha256=pkg_summary['data_yaml_sha256'],\n"
        ")\n"
        "print(json.dumps(DATASET_HASHES, indent=2))\n"
        "assert DATASET_HASHES['split_manifest_sha256'] == pkg_summary['split_manifest_sha256']\n"
        "assert DATASET_HASHES['taxonomy_sha256'] == pkg_summary['taxonomy_sha256']"
    ),
    md("## 3. Validate taxonomy + per-split, per-class statistics (no training yet)"),
    code(
        "import yaml, csv\n"
        "from collections import Counter, defaultdict\n"
        "cfg = yaml.safe_load(DATA_YAML.read_text())\n"
        "names = cfg['names']\n"
        "if isinstance(names, list): names = {i: n for i, n in enumerate(names)}\n"
        "NAMES = {int(k): v for k, v in names.items()}\n"
        "EXPECTED = {0:'CLEAN', 1:'DUSTY', 2:'BIRD_DROP', 3:'PHYSICAL_DAMAGE', 4:'ELECTRICAL_DAMAGE'}\n"
        "assert NAMES == EXPECTED, f'taxonomy mismatch: {NAMES} != {EXPECTED}'\n"
        "assert 'panel' not in [v.lower() for v in NAMES.values()], 'PANEL must not be a class'\n"
        "assert not any('snow' in v.lower() for v in NAMES.values())\n"
        "print('taxonomy OK:', NAMES)\n"
        "\n"
        "# unique source-family counts per class, per split, from the bundled split manifest\n"
        "rows = list(csv.DictReader(open(SPLIT_CSV, encoding='utf-8')))\n"
        "fam_by_split_class = defaultdict(lambda: defaultdict(set))\n"
        "img_by_split_class = defaultdict(Counter)\n"
        "for r in rows:\n"
        "    s = r['derived_split']\n"
        "    for c in r['unified_classes'].split(';'):\n"
        "        if c and c != '(none)':\n"
        "            fam_by_split_class[s][c].add(r['family_id'])\n"
        "            img_by_split_class[s][c] += 1\n"
        "\n"
        "STATS = {}\n"
        "for split_key in ('train', 'val', 'test'):\n"
        "    rel = cfg[split_key]; sname = {'train':'train','val':'valid','test':'test'}[split_key]\n"
        "    img_dir = DATA_ROOT / rel\n"
        "    lbl_dir = DATA_ROOT / rel.replace('images/', 'labels/')\n"
        "    imgs = [p for p in img_dir.glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]\n"
        "    inst, img_with = Counter(), Counter()\n"
        "    for im in imgs:\n"
        "        lp = lbl_dir / (im.stem + '.txt')\n"
        "        if not lp.is_file(): continue\n"
        "        seen = set()\n"
        "        for line in lp.read_text().splitlines():\n"
        "            p = line.split()\n"
        "            if len(p) == 5:\n"
        "                c = EXPECTED[int(p[0])]; inst[c] += 1; seen.add(c)\n"
        "        for c in seen: img_with[c] += 1\n"
        "    STATS[sname] = dict(\n"
        "        images=len(imgs), instances=dict(inst), images_with_class=dict(img_with),\n"
        "        unique_families_per_class={c: len(f) for c, f in fam_by_split_class[sname].items()},\n"
        "    )\n"
        "    print(f'  {sname:5s} images={len(imgs):5d} instances={dict(inst)}')\n"
        "    print(f'        images_with={dict(img_with)}  fam={STATS[sname][\"unique_families_per_class\"]}')"
    ),
    md(
        "## 4. Train `condition_yolo11n_baseline_v1`",
        "Base `yolo11n.pt` (COCO-pretrained). Reproducible baseline, not leaderboard tuning.",
        "Augmentation matches the prior ground-detector experiment (conservative — must not",
        "distort defect semantics).",
    ),
    code(
        "from ultralytics import YOLO\n"
        "PROJECT = '/kaggle/working/research_runs/ground_condition'\n"
        "RUN_NAME = 'condition_yolo11n_baseline_v1'\n"
        "BASE_MODEL = 'yolo11n.pt'\n"
        "SEED = 42\n"
        "\n"
        "train_args = dict(\n"
        "    data=str(DATA_YAML), epochs=100, imgsz=640, patience=20, seed=SEED,\n"
        "    batch=-1, device=0, project=PROJECT, name=RUN_NAME, exist_ok=True,\n"
        "    pretrained=True, cos_lr=True, optimizer='auto', verbose=True, val=True,\n"
        "    hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.3,\n"
        "    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0,\n"
        ")\n"
        "print(json.dumps({k: str(v) for k, v in train_args.items()}, indent=2))\n"
        "\n"
        "t0 = time.time()\n"
        "model = YOLO(BASE_MODEL)\n"
        "results = model.train(**train_args)\n"
        "train_seconds = time.time() - t0\n"
        "RUN_DIR = Path(results.save_dir)\n"
        "print('run dir:', RUN_DIR, ' elapsed(s):', round(train_seconds))"
    ),
    md("## 5. Evaluate — validation and the HELD-OUT leakage-clean test set, per class"),
    code(
        "best = RUN_DIR / 'weights' / 'best.pt'\n"
        "assert best.is_file(), best\n"
        "eval_model = YOLO(str(best))\n"
        "\n"
        "def evaluate(split):\n"
        "    m = eval_model.val(data=str(DATA_YAML), split=split, imgsz=640, iou=0.5,\n"
        "                       plots=True, verbose=False)\n"
        "    box = m.box\n"
        "    per_class = {}\n"
        "    for pos, cidx in enumerate(list(box.ap_class_index)):\n"
        "        p, r, ap50, ap = box.class_result(pos)\n"
        "        per_class[EXPECTED[int(cidx)]] = dict(precision=float(p), recall=float(r),\n"
        "                                              AP50=float(ap50), AP50_95=float(ap))\n"
        "    return dict(\n"
        "        overall=dict(precision=float(box.mp), recall=float(box.mr),\n"
        "                     mAP50=float(box.map50), mAP50_95=float(box.map)),\n"
        "        per_class=per_class, save_dir=str(m.save_dir))\n"
        "\n"
        "evaluation = {}\n"
        "for split in ('val', 'test'):\n"
        "    evaluation[split] = evaluate(split)\n"
        "    print('\\n==', split, '==')\n"
        "    print(json.dumps(evaluation[split]['overall'], indent=2))\n"
        "    print('per-class:', json.dumps(evaluation[split]['per_class'], indent=2))\n"
        "\n"
        "# merge in the test image / instance / family counts computed in cell 3\n"
        "for c, d in evaluation['test']['per_class'].items():\n"
        "    d['test_images_with_class'] = STATS['test']['images_with_class'].get(c, 0)\n"
        "    d['test_instances'] = STATS['test']['instances'].get(c, 0)\n"
        "    d['test_unique_families'] = STATS['test']['unique_families_per_class'].get(c, 0)\n"
        "\n"
        "print('\\n*** ELECTRICAL_DAMAGE — HIGH SEMANTIC RISK ***')\n"
        "print('These metrics may simply reflect the source dataset labelling cell cracks as')\n"
        "print('\\'electrical-damage\\'. A high score is NOT electrical-fault diagnosis.')\n"
        "print(json.dumps(evaluation['test']['per_class'].get('ELECTRICAL_DAMAGE', {}), indent=2))"
    ),
    md("## 6. Package artifacts + training manifest + SHA-256"),
    code(
        "OUT = Path('/kaggle/working/output'); OUT.mkdir(exist_ok=True)\n"
        "for name in ('best.pt', 'last.pt'):\n"
        "    src = RUN_DIR / 'weights' / name\n"
        "    if src.is_file(): shutil.copy2(src, OUT / name)\n"
        "for pat in ('results.csv','results.png','args.yaml',\n"
        "            'confusion_matrix.png','confusion_matrix_normalized.png',\n"
        "            'BoxPR_curve.png','BoxF1_curve.png','BoxP_curve.png','BoxR_curve.png',\n"
        "            'PR_curve.png','F1_curve.png','P_curve.png','R_curve.png',\n"
        "            'labels.jpg','labels_correlogram.jpg',\n"
        "            'train_batch*.jpg','val_batch*.jpg'):\n"
        "    for f in RUN_DIR.rglob(pat): shutil.copy2(f, OUT / f.name)\n"
        "for sd in {e['save_dir'] for e in evaluation.values()}:\n"
        "    sd = Path(sd)\n"
        "    if sd.is_dir(): shutil.copytree(sd, OUT / ('eval_' + sd.name), dirs_exist_ok=True)\n"
        "\n"
        "pred_dir = OUT / 'predictions'; pred_dir.mkdir(exist_ok=True)\n"
        "test_imgs = sorted((DATA_ROOT / 'images/test').glob('*'))[:24]\n"
        "if test_imgs:\n"
        "    YOLO(str(best)).predict(source=[str(p) for p in test_imgs], conf=0.25, save=True,\n"
        "                            project=str(pred_dir), name='test_examples', exist_ok=True)\n"
        "\n"
        "sha = hashlib.sha256((OUT / 'best.pt').read_bytes()).hexdigest()\n"
        "last_sha = hashlib.sha256((OUT / 'last.pt').read_bytes()).hexdigest() if (OUT/'last.pt').is_file() else None\n"
        "\n"
        "import csv as _csv\n"
        "results_rows = list(_csv.DictReader(open(RUN_DIR / 'results.csv')))\n"
        "best_epoch = None\n"
        "if results_rows:\n"
        "    key = next((k for k in results_rows[0] if 'mAP50-95' in k and 'B' in k), None)\n"
        "    if key:\n"
        "        best_epoch = max(range(len(results_rows)), key=lambda i: float(results_rows[i][key])) + 1\n"
        "\n"
        "manifest = dict(\n"
        "    experiment_name=RUN_NAME,\n"
        "    model_role='EXPERIMENTAL RESEARCH condition/defect detector (second stage, on panel crops) — NOT a panel detector, NOT production',\n"
        "    dataset_kaggle='edithstark/solar-ai-ground-condition-research-v1',\n"
        "    dataset_sources=['Roboflow defects-kgig6/detection-for-defects-in-solar-panels v2',\n"
        "                     'Roboflow solar-panel-a5cyn/dusty-and-bird-droppings v1'],\n"
        "    provenance_grade='C', provenance_tag='personal-experimental-unverified-provenance',\n"
        "    provenance_warning='underlying images trace to Kaggle pythonafroz/solar-panel-images (licence Unknown); do NOT redistribute/publish/use commercially/claim verified CC BY',\n"
        "    electrical_damage_semantic_risk='HIGH — S1 electrical-damage boxes visually correspond to cell cracks; label preserved, not relabelled; high score is NOT electrical-fault diagnosis',\n"
        "    clean_semantic_limitation='CLEAN trained only on explicit clean-region annotations; absence of a defect detection is NOT proof of Clean; no CLEAN boxes fabricated',\n"
        "    classes=EXPECTED,\n"
        "    base_model=BASE_MODEL, pretrained_source='YOLO11n COCO-pretrained (Ultralytics)',\n"
        "    epochs=train_args['epochs'], imgsz=train_args['imgsz'], batch=train_args['batch'],\n"
        "    patience=train_args['patience'], optimizer='auto', cos_lr=True, seed=SEED,\n"
        "    augmentation={k: train_args[k] for k in ('hsv_h','hsv_s','hsv_v','degrees','translate',\n"
        "        'scale','shear','perspective','flipud','fliplr','mosaic','mixup','copy_paste')},\n"
        "    full_train_args={k: str(v) for k, v in train_args.items()},\n"
        "    device='cuda:0', env=ENV,\n"
        "    dataset_hashes=DATASET_HASHES,\n"
        "    dataset_stats=STATS,\n"
        "    train_seconds=train_seconds, best_epoch=best_epoch, total_epochs=train_args['epochs'],\n"
        "    run_dir=str(RUN_DIR),\n"
        "    val_metrics=evaluation['val']['overall'], test_metrics=evaluation['test']['overall'],\n"
        "    per_class=dict(val=evaluation['val']['per_class'], test=evaluation['test']['per_class']),\n"
        "    checkpoint_sha256=sha, last_checkpoint_sha256=last_sha,\n"
        ")\n"
        "(OUT / 'training_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))\n"
        "(OUT / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, default=str))\n"
        "(OUT / 'environment.json').write_text(json.dumps(ENV, indent=2))\n"
        "print('best.pt sha256:', sha)\n"
        "print(json.dumps(manifest, indent=2, default=str))\n"
        "print('\\nArtifacts:', sorted(p.name for p in OUT.iterdir()))"
    ),
    md(
        "## 7. Retrieve",
        "Download `/kaggle/working/output/`. Locally save `best.pt` as",
        "`weights/yolo_solar_condition_research_v1.pt` (gitignored research artifact) — **never**",
        "overwrite `weights/yolo_solar.pt` or `weights/yolo_solar_ground_research_v1.pt`.",
        "",
        "This condition model is a research artifact only. It is a **second-stage** model that",
        "runs on panel crops. It is not a panel detector, makes no RGB-hotspot claim, and a high",
        "ELECTRICAL_DAMAGE score is not an electrical-fault diagnosis (source labels look like",
        "cell cracks).",
    ),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB, "-", len(CELLS), "cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
