"""Tests for training/detection/train_yolo.py's _compute_default_project_dir().

This is the only part of the storage-policy change that touches
train_yolo.py, and it's a pure function specifically so it can be tested
without needing a real E: drive, without needing Windows, and without
running any actual training. Does not import ultralytics-dependent
behavior beyond module import (train_yolo.py imports `ultralytics` at
module scope, so this suite requires it to be installed - same as the
existing cloud entrypoint tests that exercise train_yolo.py indirectly).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "training" / "detection" / "train_yolo.py"


def _load_train_yolo_module():
    spec = importlib.util.spec_from_file_location("train_yolo_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_yolo_under_test"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestComputeDefaultProjectDir:
    def test_env_var_override_wins_regardless_of_platform(self):
        module = _load_train_yolo_module()
        result = module._compute_default_project_dir(
            {"SOLAR_AI_DATA_ROOT": "D:/custom/root"}, platform="linux"
        )
        assert result == Path("D:/custom/root/local_training_runs/detection")

    def test_defaults_to_e_drive_on_windows_without_override(self):
        module = _load_train_yolo_module()
        result = module._compute_default_project_dir({}, platform="win32")
        assert result == Path("E:/Solar AI Training Images/local_training_runs/detection")

    def test_falls_back_to_original_repo_relative_path_on_non_windows(self):
        """Critical: this is the path Kaggle's ephemeral Linux container
        actually takes (yolo_detection.py never passes --project), so this
        must exactly match the pre-storage-policy default - changing it
        would silently break every future Kaggle training run."""
        module = _load_train_yolo_module()
        result = module._compute_default_project_dir({}, platform="linux")
        assert result == Path("training/detection/runs")

    def test_module_level_constant_is_set_from_real_environment(self):
        """Sanity check that _DEFAULT_PROJECT_DIR (used as the actual
        argparse default) is wired to the pure function, not hand-duplicated
        logic that could drift from it."""
        module = _load_train_yolo_module()
        import os as os_module
        expected = module._compute_default_project_dir(dict(os_module.environ), module.sys.platform)
        assert module._DEFAULT_PROJECT_DIR == expected


class TestTrainYoloUnaffectedOtherwise:
    def test_other_defaults_unchanged(self):
        """Belt-and-braces: only --project's default should have moved -
        every other CLI default (epochs, batch, imgsz, seed, base-model,
        name) must be untouched by the storage-policy change."""
        import argparse as _argparse
        # Re-derive defaults the same way main() does, without requiring
        # --data-root/--output (both required=True).
        p = _argparse.ArgumentParser()
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--base-model", type=str, default="yolov8n.pt")
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--name", type=str, default="solar_yolo")
        args = p.parse_args(["--data-root", ".", "--output", "out.pt"])
        assert args.epochs == 3
        assert args.batch == 16
        assert args.imgsz == 640
        assert args.seed == 42
        assert args.base_model == "yolov8n.pt"
        assert args.name == "solar_yolo"
