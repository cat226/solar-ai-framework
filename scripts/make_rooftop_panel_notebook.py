"""Generate kaggle/rooftop_panel_train.ipynb - Phase 19 rooftop PANEL detector (1 class).

Two YOLO11n runs, same recipe, differing only in data:
  MIX          Susan panel-only (all Susan images that contain a `panel` box)  +
               first-party rooftop TRAIN scenes oversampled x10
  ROOFTOP_ONLY first-party rooftop TRAIN scenes only (COCO init) - shows whether Susan helps

Model selection / early stopping uses the rooftop VALID scenes (target domain). Final
numbers are reported on the scene-disjoint rooftop TEST scenes and on Susan test
(panel class only) for the record. Research only.

Regenerable: `python scripts/make_rooftop_panel_notebook.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle" / "rooftop_panel_train.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").splitlines()]}


CELLS = [
    md(
        "# Solar AI - rooftop PANEL detector (Phase 19, YOLO11n, 1 class)",
        "",
        "Research only - does not touch production weights. Attach datasets",
        "`edithstark/solar-ai-ground-pv-research-v1` (Susan) and `edithstark/solar-ai-rooftop-panel-v1`.",
        "Accelerator = GPU. Run All.",
        "",
        "Rooftop labels are `annotated_ai_assisted_unverified` (SAM proposals corrected by an AI",
        "reviewer), NOT human-verified. All rooftop images are ONE rooftop; splits are scene-disjoint.",
    ),
    code(
        "import sys, platform, subprocess, json, os, time, hashlib, shutil, random\n"
        "from pathlib import Path\n"
        "PINNED = ['torch==2.7.1', 'torchvision==0.22.1', 'ultralytics==8.4.138', 'PyYAML>=6.0.1']\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINNED], check=True)\n"
        "import torch, ultralytics, yaml\n"
        "assert torch.cuda.is_available(), 'GPU required'\n"
        "ENV = dict(python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__,\n"
        "           gpu=torch.cuda.get_device_name(0))\n"
        "print(ENV)"
    ),
    md("## Build datasets"),
    code(
        "INPUT = Path('/kaggle/input')\n"
        "susan_yaml = next(p for p in INPUT.rglob('data.yaml') if 'ground-pv-research' in str(p) and 'beast' not in str(p))\n"
        "SUSAN = susan_yaml.parent\n"
        "roof_sum = next(p for p in INPUT.rglob('dataset_summary.json') if 'rooftop' in str(p))\n"
        "ROOF = roof_sum.parent\n"
        "print('Susan:', SUSAN, '| rooftop:', ROOF)\n"
        "print(json.dumps({k: v for k, v in json.loads(roof_sum.read_text()).items() if k in ('train','valid','test')}, indent=1))\n"
        "\n"
        "W = Path('/kaggle/working'); SP = W / 'susan_panel'\n"
        "def build_susan_split(split):\n"
        "    (SP/'images'/split).mkdir(parents=True, exist_ok=True); (SP/'labels'/split).mkdir(parents=True, exist_ok=True)\n"
        "    n = 0\n"
        "    for lab in sorted((SUSAN/'labels'/split).glob('*.txt')):\n"
        "        keep = [l for l in lab.read_text().splitlines() if l.strip() and l.split()[0] == '0']\n"
        "        if not keep: continue\n"
        "        img = next((SUSAN/'images'/split/(lab.stem+e) for e in ('.jpg','.jpeg','.png') if (SUSAN/'images'/split/(lab.stem+e)).exists()), None)\n"
        "        if img is None: continue\n"
        "        (SP/'labels'/split/lab.name).write_text('\\n'.join(keep)+'\\n')\n"
        "        dst = SP/'images'/split/img.name\n"
        "        if not dst.exists(): os.symlink(img, dst)\n"
        "        n += 1\n"
        "    return n\n"
        "print('Susan panel-only images:', {s: build_susan_split(s) for s in ('train','valid','test')})\n"
        "\n"
        "OVERSAMPLE = 10\n"
        "roof_train = sorted((ROOF/'images'/'train').glob('*.jpg'))\n"
        "lst = W/'mix_train.txt'\n"
        "lst.write_text('\\n'.join([str(p) for p in sorted((SP/'images'/'train').glob('*'))] + [str(p) for p in roof_train]*OVERSAMPLE))\n"
        "roof_lst = W/'roof_train.txt'\n"
        "roof_lst.write_text('\\n'.join(str(p) for p in roof_train*OVERSAMPLE))\n"
        "NAMES = {0: 'panel'}\n"
        "def y(name, **kw): (W/name).write_text(yaml.safe_dump({**kw, 'names': NAMES}, sort_keys=False))\n"
        "y('data_mix.yaml',  train=str(lst), val=str(ROOF/'images'/'valid'))\n"
        "y('data_roof.yaml', train=str(roof_lst), val=str(ROOF/'images'/'valid'))\n"
        "y('eval_roof_valid.yaml', train=str(ROOF/'images'/'train'), val=str(ROOF/'images'/'valid'), test=str(ROOF/'images'/'valid'))\n"
        "y('eval_roof_test.yaml',  train=str(ROOF/'images'/'train'), val=str(ROOF/'images'/'test'),  test=str(ROOF/'images'/'test'))\n"
        "y('eval_susan_test.yaml', train=str(SP/'images'/'train'), val=str(SP/'images'/'valid'), test=str(SP/'images'/'test'))\n"
        "print('rooftop train images', len(roof_train), 'x', OVERSAMPLE)"
    ),
    md("## Train"),
    code(
        "from ultralytics import YOLO\n"
        "PROJECT = '/kaggle/working/research_runs/rooftop_panel'\n"
        "AUG = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5, shear=0.0,\n"
        "           perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=10)\n"
        "COMMON = dict(imgsz=640, patience=30, seed=42, batch=16, device=0, workers=4, cache=False, project=PROJECT,\n"
        "              exist_ok=True, pretrained=True, cos_lr=True, optimizer='auto', verbose=False, val=True, plots=True)\n"
        "RUNS = [('MIX', 'data_mix.yaml', 60), ('ROOFTOP_ONLY', 'data_roof.yaml', 100)]\n"
        "runs = {}\n"
        "for tag, data, epochs in RUNS:\n"
        "    args = dict(data=str(W/data), epochs=epochs, name=f'rooftop_{tag}', **COMMON, **AUG)\n"
        "    t0 = time.time()\n"
        "    res = YOLO('yolo11n.pt').train(**args)\n"
        "    runs[tag] = dict(save_dir=str(res.save_dir), seconds=time.time()-t0, epochs=epochs)\n"
        "    print(tag, 'done in %.1f min' % (runs[tag]['seconds']/60))"
    ),
    md("## Evaluate (rooftop valid, rooftop TEST scenes, Susan test - panel only)"),
    code(
        "def evaluate(best, yml, split):\n"
        "    m = YOLO(str(best)).val(data=str(W/yml), split=split, imgsz=640, iou=0.5, plots=False, verbose=False)\n"
        "    b = m.box\n"
        "    return dict(precision=float(b.mp), recall=float(b.mr), mAP50=float(b.map50), mAP50_95=float(b.map))\n"
        "for tag in runs:\n"
        "    sd = Path(runs[tag]['save_dir'])\n"
        "    runs[tag]['best_sha256'] = hashlib.sha256((sd/'weights'/'best.pt').read_bytes()).hexdigest()\n"
        "    runs[tag]['eval'] = {\n"
        "        'rooftop_valid': evaluate(sd/'weights'/'best.pt', 'eval_roof_valid.yaml', 'val'),\n"
        "        'rooftop_test':  evaluate(sd/'weights'/'best.pt', 'eval_roof_test.yaml', 'test'),\n"
        "        'susan_test_panel_only': evaluate(sd/'weights'/'best.pt', 'eval_susan_test.yaml', 'test'),\n"
        "    }\n"
        "    print(tag, json.dumps(runs[tag]['eval'], indent=1))"
    ),
    md("## Package"),
    code(
        "OUT = W/'output'; OUT.mkdir(exist_ok=True)\n"
        "for tag in runs:\n"
        "    sd = Path(runs[tag]['save_dir']); t = OUT/f'rooftop_{tag}'; t.mkdir(exist_ok=True)\n"
        "    for n in ('best.pt', 'last.pt'):\n"
        "        if (sd/'weights'/n).is_file(): shutil.copy2(sd/'weights'/n, t/n)\n"
        "    for pat in ('results.csv', 'results.png', 'args.yaml'):\n"
        "        for f in sd.glob(pat): shutil.copy2(f, t/f.name)\n"
        "(OUT/'rooftop_panel_manifest.json').write_text(json.dumps(dict(\n"
        "    phase='19 rooftop panel detector', env=ENV, oversample=OVERSAMPLE,\n"
        "    label_provenance='annotated_ai_assisted_unverified (AI-reviewed SAM proposals); one rooftop',\n"
        "    runs=runs), indent=2, default=str))\n"
        "print(sorted(p.name for p in OUT.iterdir()))"
    ),
]


def main() -> int:
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": "edithstark/rooftop-panel-yolo11n-v1", "title": "rooftop-panel-yolo11n-v1",
            "code_file": "rooftop_panel_train.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["edithstark/solar-ai-ground-pv-research-v1", "edithstark/solar-ai-rooftop-panel-v1"],
            "competition_sources": [], "kernel_sources": []}
    kdir = ROOT / "kaggle" / "rooftop_panel_kernel"
    kdir.mkdir(exist_ok=True)
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (kdir / NB.name).write_text(NB.read_text(encoding="utf-8"), encoding="utf-8")
    print("wrote", NB, "and", kdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
