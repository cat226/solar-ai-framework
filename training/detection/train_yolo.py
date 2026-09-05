"""Train the YOLO solar-panel detector on the prepared BDAPPV IGN dataset.

Single class ("solar panel", id 0). Trains at imgsz=640 to match the
production detector's configured image_size (configs/settings.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Local training runs (checkpoints, result plots/logs) default to the E:
# Solar AI data drive on this machine, per this project's storage policy
# (large local data off the low-space C:/D: drives - see
# training/cloud/base/storage_paths.py for the same convention used
# elsewhere). Deliberately NOT imported from that module: this script is
# invoked on Kaggle as a bare `python /path/to/train_yolo.py` subprocess
# (see training/cloud/kaggle/entrypoints/yolo_detection.py) with no cwd set
# to the repo root, so sys.path[0] is just this file's own directory and a
# `training.cloud.*` package import would fail there with
# ModuleNotFoundError - this stays a self-contained, dependency-free
# default instead. On any platform/environment without the override env
# var and without Windows (in particular Kaggle's ephemeral Linux
# container, which has no persistent drives of its own), falls back to the
# original repo-relative location unchanged.
def _compute_default_project_dir(env: dict, platform: str) -> Path:
    """Pure function so the platform/env-var branching can be unit tested
    without needing an actual E: drive or Windows to run the test suite."""
    if env.get("SOLAR_AI_DATA_ROOT"):
        return Path(env["SOLAR_AI_DATA_ROOT"]) / "local_training_runs" / "detection"
    if platform == "win32":
        return Path("E:/Solar AI Training Images/local_training_runs/detection")
    return Path("training/detection/runs")


_DEFAULT_PROJECT_DIR = _compute_default_project_dir(os.environ, sys.platform)


def _write_data_yaml(data_root: Path, out_path: Path) -> Path:
    config = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "solar panel"},
    }
    out_path.write_text(yaml.safe_dump(config))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Output of prepare_dataset.py")
    parser.add_argument("--output", type=Path, required=True, help="Where to copy the final best.pt checkpoint")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=_DEFAULT_PROJECT_DIR)
    parser.add_argument("--name", type=str, default="solar_yolo")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("epochs and batch must be positive")

    args.project.mkdir(parents=True, exist_ok=True)
    data_yaml = _write_data_yaml(args.data_root, args.project / "data.yaml")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"training did not produce a checkpoint at {best_pt}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(best_pt.read_bytes())

    summary = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "checkpoint_saved_to": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
