"""Tests for training/cloud/kaggle/build_yolo_smoke_package.py.

Focused mainly on the Kaggle data_root path construction, since that's
where a real bug was found twice against the live Kaggle platform: Kaggle
mounts an attached dataset at /kaggle/input/datasets/<owner>/<slug>/, not
the flat /kaggle/input/<slug>/ this module used to assume. No test here
touches the real Kaggle CLI - prepare()/dry_run() are pure local file I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.cloud.kaggle.build_yolo_smoke_package import build


@pytest.fixture
def smoke_dataset_root(tmp_path):
    root = tmp_path / "smoke_dataset"
    root.mkdir()
    manifest = {
        "source_manifest_hash": "deadbeef" * 8,
        "class_names": ["solar panel"],
        "counts": {"train": 30, "val": 30, "test": 30},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class TestKaggleDataRootConstruction:
    def test_with_dataset_ref_uses_nested_datasets_path(self, tmp_path, smoke_dataset_root):
        package_dir = tmp_path / "package"
        job_spec, _ = build(
            experiment_id="test-exp",
            smoke_dataset_root=smoke_dataset_root,
            package_dir=package_dir,
            registry_path=tmp_path / "registry.jsonl",
            kaggle_dataset_ref="edithstark/solar-ai-yolo-smoke-001",
        )
        rendered = (package_dir / "train.py").read_text(encoding="utf-8")
        assert "'/kaggle/input/datasets/edithstark/solar-ai-yolo-smoke-001'" in rendered
        # The old, incorrect flat form must not appear as the configured data_root.
        assert "'data_root': '/kaggle/input/solar-ai-yolo-smoke-001'" not in rendered

    def test_without_dataset_ref_uses_honest_placeholder(self, tmp_path, smoke_dataset_root):
        package_dir = tmp_path / "package"
        build(
            experiment_id="test-exp",
            smoke_dataset_root=smoke_dataset_root,
            package_dir=package_dir,
            registry_path=tmp_path / "registry.jsonl",
            kaggle_dataset_ref=None,
        )
        rendered = (package_dir / "train.py").read_text(encoding="utf-8")
        assert "__NOT_YET_UPLOADED__" in rendered

    def test_title_equals_slug_to_avoid_kaggle_rename(self, tmp_path, smoke_dataset_root):
        package_dir = tmp_path / "package"
        build(
            experiment_id="test-exp",
            smoke_dataset_root=smoke_dataset_root,
            package_dir=package_dir,
            registry_path=tmp_path / "registry.jsonl",
            kaggle_dataset_ref="edithstark/solar-ai-yolo-smoke-001",
        )
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["title"] == metadata["id"].split("/", 1)[1]

    def test_dry_run_passes_with_real_dataset_ref(self, tmp_path, smoke_dataset_root):
        package_dir = tmp_path / "package"
        from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run

        build(
            experiment_id="test-exp",
            smoke_dataset_root=smoke_dataset_root,
            package_dir=package_dir,
            registry_path=tmp_path / "registry.jsonl",
            kaggle_dataset_ref="edithstark/solar-ai-yolo-smoke-001",
        )
        config_data = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        owner, slug = config_data["id"].split("/", 1)
        config = KaggleKernelConfig(
            owner=owner, slug=slug, title=config_data["title"], code_file=config_data["code_file"],
            enable_gpu=config_data["enable_gpu"], enable_internet=config_data["enable_internet"],
            dataset_sources=config_data.get("dataset_sources", []),
        )
        result = dry_run(package_dir, config)
        assert result.passed, result.errors
