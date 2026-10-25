"""Generate kaggle/ground_beast_train.ipynb — Phase 13B "beast" ground PANEL
detector training. LEAN v2: the original 3-experiment + 27-pass threshold-sweep
notebook hung on Kaggle (>17 h). This version trains only the two experiments
that need a GPU and does the threshold sweep + baseline re-eval locally.

  B  Susan + scene-capped candidate      (data_BC, v1 conservative augmentation)
  C  Susan + candidate, scale/aspect aug  (data_BC, moderate scale/translate/
                                           mosaic - NO multi_scale (v2 lesson))

Experiment A (Susan-only baseline reproduction) is done by re-evaluating the
existing v1 checkpoint locally on the identical Susan test set.

Frozen B/C: YOLO11n + COCO init, optimizer=auto, cos_lr, epochs 90, patience 20,
seed 42, imgsz 640. Susan valid/test identical -> fair held-out comparison.
The six owner photos are NOT here.

Regenerable: `python scripts/make_beast_notebook.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "kaggle" / "ground_beast_train.ipynb"

V1_TEST = {
    "overall": {"precision": 0.8756, "recall": 0.8014, "mAP50": 0.8647, "mAP50_95": 0.6775},
    "per_class": {
        "panel": {"precision": 0.9465, "recall": 0.9250, "AP50": 0.9628, "AP50_95": 0.8427},
        "cracked": {"precision": 0.8090, "recall": 0.6531, "AP50": 0.7495, "AP50_95": 0.5296},
        "dusty": {"precision": 0.8712, "recall": 0.8261, "AP50": 0.8819, "AP50_95": 0.6601},
    },
}


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").splitlines()]}


CELLS = [
    md(
        "# Solar AI - ground PANEL detector, Phase 13B \"beast\" (YOLO11n) - LEAN",
        "",
        "Trains **B** and **C** only. Experiment A (Susan-only baseline) is re-evaluated",
        " locally from the existing v1 checkpoint. **Research only** - does not touch",
        " `weights/yolo_solar.pt`, `weights/yolo_solar_ground_research_v1.pt`, or the",
        " condition model.",
        "",
        "| exp | data | augmentation |",
        "|---|---|---|",
        "| **B** | Susan + 125 scene-capped candidate frames (train only) | v1 conservative aug |",
        "| **C** | Susan + candidate (train only) | moderate scale/translate/mosaic - **NO multi_scale** |",
        "",
        "Frozen: YOLO11n + COCO init, optimizer `auto`, `cos_lr`, epochs 90, patience 20,",
        " seed 42, imgsz 640. Susan valid/test unchanged. Candidate = ONE ground-mounted",
        " installation, ONE session, `personal-experimental-unverified-provenance`, research-only.",
        "",
        "Attach `edithstark/solar-ai-ground-pv-research-v1` and",
        " `edithstark/solar-ai-ground-beast-candidate-v1`. Accelerator = GPU. Run All.",
    ),
    md("## 1. Environment (pinned to v1/v2 versions)"),
    code(
        "import sys, platform, subprocess, json, os, time, hashlib, shutil\n"
        "from pathlib import Path\n"
        "PINNED = ['torch==2.7.1', 'torchvision==0.22.1', 'ultralytics==8.4.138', 'PyYAML>=6.0.1']\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINNED], check=True)\n"
        "import torch, torchvision, ultralytics\n"
        "assert torch.cuda.is_available(), 'GPU required'\n"
        "ENV = dict(python=platform.python_version(), torch=torch.__version__,\n"
        "           torchvision=torchvision.__version__, cuda=torch.version.cuda,\n"
        "           ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0),\n"
        "           gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory/1e9,1))\n"
        "print(json.dumps(ENV, indent=2))"
    ),
    md("## 2. Locate mounts + build data_BC.yaml"),
    code(
        "import yaml\n"
        "INPUT = Path('/kaggle/input')\n"
        "susan_yaml = next((p for p in INPUT.rglob('data.yaml')\n"
        "                   if 'ground-pv-research' in str(p) and 'beast' not in str(p)), None)\n"
        "assert susan_yaml, 'Susan dataset not attached'\n"
        "SUSAN = susan_yaml.parent\n"
        "for s in ('images/train','images/valid','images/test','labels/train','labels/valid','labels/test'):\n"
        "    assert (SUSAN / s).is_dir(), f'missing {s}'\n"
        "cand_imgs = next((p for p in INPUT.rglob('candidate/train/images') if p.is_dir()), None)\n"
        "assert cand_imgs, 'candidate dataset not attached'\n"
        "CAND_TRAIN = cand_imgs.parent\n"
        "n_cand = len(list(cand_imgs.glob('*')))\n"
        "assert n_cand == 125, f'expected 125 candidate frames, got {n_cand}'\n"
        "print('Susan:', SUSAN, '| candidate:', CAND_TRAIN, f'({n_cand})')\n"
        "\n"
        "NAMES = {0: 'panel', 1: 'cracked', 2: 'dusty'}\n"
        "DATA_A = Path('/kaggle/working/data_A.yaml')  # for split='test'/'val' pathing\n"
        "DATA_A.write_text(yaml.safe_dump({\n"
        "    'path': str(SUSAN.resolve()),\n"
        "    'train': 'images/train', 'val': 'images/valid', 'test': 'images/test',\n"
        "    'names': NAMES}, sort_keys=False))\n"
        "DATA_BC = Path('/kaggle/working/data_BC.yaml')\n"
        "DATA_BC.write_text(yaml.safe_dump({\n"
        "    'train': [str((SUSAN/'images/train').resolve()), str((CAND_TRAIN/'images').resolve())],\n"
        "    'val':  str((SUSAN/'images/valid').resolve()),\n"
        "    'test': str((SUSAN/'images/test').resolve()),\n"
        "    'names': NAMES}, sort_keys=False))\n"
        "print(DATA_BC.read_text())\n"
        "rr = next((p for p in SUSAN.rglob('remediation_report.json')), None)\n"
        "if rr:\n"
        "    r = json.loads(rr.read_text()); assert r.get('no_group_crosses_splits') is True\n"
        "    print('Susan leakage-clean split verified:', r.get('final_split_counts'))"
    ),
    md("## 3. Train B and C"),
    code(
        "from ultralytics import YOLO\n"
        "PROJECT = '/kaggle/working/research_runs/ground_beast'\n"
        "SEED = 42\n"
        "V1_AUG = dict(hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.3,\n"
        "              shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.5,\n"
        "              mixup=0.0, copy_paste=0.0)\n"
        "C_AUG = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=7.0, translate=0.15, scale=0.5,\n"
        "             shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.4,\n"
        "             mixup=0.0, copy_paste=0.0, close_mosaic=15)\n"
        "EXPERIMENTS = [\n"
        "    ('B', V1_AUG, 'Susan + scene-capped candidate, v1 aug'),\n"
        "    ('C', C_AUG,  'Susan + candidate, moderate scale/aspect aug'),\n"
        "]\n"
        "COMMON = dict(data=str(DATA_BC), epochs=90, imgsz=640, patience=20, seed=SEED,\n"
        "              batch=16, device=0, workers=4, cache=False,\n"
        "              project=PROJECT, exist_ok=True, pretrained=True, cos_lr=True,\n"
        "              optimizer='auto', verbose=False, val=True, plots=True)\n"
        "runs = {}\n"
        "for tag, aug, desc in EXPERIMENTS:\n"
        "    args = dict(name=f'beast_{tag}', **COMMON, **aug)\n"
        "    print(f'\\n===== EXPERIMENT {tag}: {desc} =====')\n"
        "    print(json.dumps({k: str(v) for k, v in args.items()}, indent=1))\n"
        "    t0 = time.time()\n"
        "    res = YOLO('yolo11n.pt').train(**args)\n"
        "    dt = time.time() - t0\n"
        "    runs[tag] = dict(save_dir=str(res.save_dir), seconds=dt, args=args, desc=desc)\n"
        "    print(f'  {tag} done in {dt/60:.1f} min')"
    ),
    md("## 4. Evaluate on the HELD-OUT Susan test set (identical for B and C)"),
    code(
        "V1_TEST = " + json.dumps(V1_TEST) + "\n"
        "def evaluate(best_pt):\n"
        "    m = YOLO(str(best_pt)).val(data=str(DATA_A), split='test', imgsz=640, iou=0.5,\n"
        "                               plots=True, verbose=False)\n"
        "    b = m.box\n"
        "    pc = {}\n"
        "    for pos, ci in enumerate(list(b.ap_class_index)):\n"
        "        p, r, ap50, ap = b.class_result(pos)\n"
        "        pc[NAMES[int(ci)]] = dict(precision=float(p), recall=float(r), AP50=float(ap50), AP50_95=float(ap))\n"
        "    return dict(overall=dict(precision=float(b.mp), recall=float(b.mr),\n"
        "                             mAP50=float(b.map50), mAP50_95=float(b.map)),\n"
        "                per_class=pc, save_dir=str(m.save_dir))\n"
        "for tag in runs:\n"
        "    best = Path(runs[tag]['save_dir']) / 'weights' / 'best.pt'\n"
        "    runs[tag]['eval_test'] = evaluate(best)\n"
        "    runs[tag]['best_sha256'] = hashlib.sha256(best.read_bytes()).hexdigest()\n"
        "    o = runs[tag]['eval_test']['overall']\n"
        "    d = {k: round(o[k] - V1_TEST['overall'][k], 4) for k in o}\n"
        "    print(f\"{tag}: mAP50={o['mAP50']:.4f} mAP50-95={o['mAP50_95']:.4f} \"\n"
        "          f\"P={o['precision']:.4f} R={o['recall']:.4f}   vs v1 delta={d}\")"
    ),
    md("## 5. Package"),
    code(
        "OUT = Path('/kaggle/working/output'); OUT.mkdir(exist_ok=True)\n"
        "import csv as _csv\n"
        "def best_epoch(sd):\n"
        "    rows = list(_csv.DictReader(open(Path(sd)/'results.csv')))\n"
        "    if not rows: return None\n"
        "    k = next((c for c in rows[0] if 'mAP50-95' in c and 'B' in c), None)\n"
        "    return (max(range(len(rows)), key=lambda i: float(rows[i][k])) + 1) if k else None\n"
        "for tag in runs:\n"
        "    sd = Path(runs[tag]['save_dir']); tdir = OUT / f'beast_{tag}'; tdir.mkdir(exist_ok=True)\n"
        "    for name in ('best.pt', 'last.pt'):\n"
        "        if (sd/'weights'/name).is_file(): shutil.copy2(sd/'weights'/name, tdir/name)\n"
        "    for pat in ('results.csv','results.png','args.yaml','confusion_matrix*.png',\n"
        "                'BoxPR_curve.png','BoxF1_curve.png','labels.jpg'):\n"
        "        for f in sd.rglob(pat): shutil.copy2(f, tdir / f.name)\n"
        "    ev = runs[tag].get('eval_test', {}).get('save_dir')\n"
        "    if ev and Path(ev).is_dir(): shutil.copytree(ev, tdir/'eval_test', dirs_exist_ok=True)\n"
        "    runs[tag]['best_epoch'] = best_epoch(sd)\n"
        "manifest = dict(\n"
        "    phase='13B beast (lean B+C)', role='RESEARCH ground PANEL detector - NOT production',\n"
        "    env=ENV, seed=SEED, v1_test_metrics=V1_TEST,\n"
        "    candidate_provenance='personal-experimental-unverified-provenance; ONE installation, ONE session; research-only',\n"
        "    experiments={tag: dict(desc=runs[tag]['desc'], train_seconds=runs[tag]['seconds'],\n"
        "        best_epoch=runs[tag]['best_epoch'], best_sha256=runs[tag]['best_sha256'],\n"
        "        args={k: str(v) for k, v in runs[tag]['args'].items()},\n"
        "        eval_test=runs[tag]['eval_test']) for tag in runs})\n"
        "(OUT/'beast_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))\n"
        "print(json.dumps({tag: {'sha': runs[tag]['best_sha256'][:16],\n"
        "                        'test': manifest['experiments'][tag]['eval_test']['overall'],\n"
        "                        'min': round(runs[tag]['seconds']/60,1)} for tag in runs}, indent=2))\n"
        "print('Artifacts:', sorted(p.name for p in OUT.iterdir()))"
    ),
    md("## 6. Retrieve\n"
       "Download `/kaggle/working/output/`. Locally: save each `best.pt` as"
       " `weights/yolo_solar_ground_research_beast_<tag>.pt` (gitignored). Then run the"
       " threshold sweep on Susan val, the v1-vs-best six-photo OOD comparison, failure"
       " analysis and condition chain **locally**. Model selection is NOT by mAP alone"
       " - a v2-style real-world regression -> REJECT."),
]


def main() -> int:
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB, "-", len(CELLS), "cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
