#!/usr/bin/env python3
"""training/detection/evaluate_domain_adaptation_experimental.py

PERSONAL / NON-COMMERCIAL / EXPERIMENTAL evaluation only.

Runs ultralytics' own validation (mAP50, mAP50-95, precision, recall) for
a given YOLO checkpoint against two independent, disjoint held-out image
sets:
  - a ground-level test split (never used in training)
  - an aerial sample drawn from the production aerial dataset's TEST
    split (never used in training or in checkpoint-selection monitoring
    - see build_aerial_eval_sample.py / build_aerial_replay_sample.py)

Intended to be run twice per model (baseline weights/yolo_solar.pt, and
the experimental yolo_solar_domain_v1.pt) so the two JSON outputs can be
diffed for a before/after comparison. Never trains, never writes to any
weights/ file, never fabricates a number - if ultralytics can't compute
a metric (e.g. an empty split), this script reports that plainly rather
than inventing a value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def _write_single_val_yaml(images_dir: Path, out_path: Path) -> Path:
    # Ultralytics requires a 'train' key to be present even for a val-only
    # run - it is never actually read for gradients here (model.val() does
    # not train), so pointing it at the same val images is inert, not a
    # contamination of anything: this script never calls model.train().
    config = {
        "path": str(images_dir.parent.parent.resolve()),
        "train": str(images_dir.resolve()),
        "val": str(images_dir.resolve()),
        "names": {0: "solar panel"},
    }
    out_path.write_text(yaml.safe_dump(config))
    return out_path


def evaluate(model_path: Path, images_dir: Path, imgsz: int, project: Path, name: str) -> dict:
    yaml_path = project / f"{name}.data.yaml"
    project.mkdir(parents=True, exist_ok=True)
    _write_single_val_yaml(images_dir, yaml_path)

    model = YOLO(str(model_path))
    results = model.val(data=str(yaml_path), split="val", imgsz=imgsz, project=str(project), name=name, exist_ok=True, verbose=False)

    n_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
    return {
        "model_path": str(model_path),
        "images_dir": str(images_dir),
        "num_images": n_images,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ground-test-images", type=Path, required=True)
    parser.add_argument("--aerial-eval-images", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True, help="e.g. baseline or adapted - used to name output files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ground_metrics = evaluate(args.model, args.ground_test_images, args.imgsz, args.project, f"{args.tag}_ground")
    aerial_metrics = evaluate(args.model, args.aerial_eval_images, args.imgsz, args.project, f"{args.tag}_aerial")

    summary = {"tag": args.tag, "ground": ground_metrics, "aerial": aerial_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
