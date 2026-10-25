"""Tests for training/cloud/kaggle/build_mobilenet_package.py.

No test here touches the real Kaggle CLI or uploads anything -
prepare()/dry_run() are pure local file I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.cloud.kaggle.build_mobilenet_package import build


@pytest.fixture
def dataset_manifest(tmp_path):
    manifest = {
        "classes": ["Clean", "Dusty", "Hotspot"],
        "is_production_class_set": False,
        "seed": 42,
        "counts": {"train": {"Clean": 570, "Dusty": 508, "Hotspot": 149}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class TestBuildMobilenetPackage:
    def test_data_root_uses_nested_datasets_path(self, tmp_path, dataset_manifest):
        package_dir = tmp_path / "package"
        build(
            experiment_id="solar-mobilenet-3class-smoke",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-mobilenet-3class-smoke",
            registry_path=tmp_path / "registry.jsonl",
        )
        rendered = (package_dir / "train.py").read_text(encoding="utf-8")
        assert "'/kaggle/input/datasets/edithstark/solar-mobilenet-3class-smoke'" in rendered

    def test_classes_from_manifest_are_passed_through(self, tmp_path, dataset_manifest):
        package_dir = tmp_path / "package"
        build(
            experiment_id="exp",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/x",
            registry_path=tmp_path / "registry.jsonl",
        )
        rendered = (package_dir / "train.py").read_text(encoding="utf-8")
        assert "'Clean,Dusty,Hotspot'" in rendered

    def test_job_spec_records_class_order_and_hyperparameters(self, tmp_path, dataset_manifest):
        package_dir = tmp_path / "package"
        job_spec, _ = build(
            experiment_id="exp",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/x",
            epochs=7,
            batch_size=16,
            lr=1e-4,
            seed=99,
            registry_path=tmp_path / "registry.jsonl",
        )
        assert job_spec.class_order == ("Clean", "Dusty", "Hotspot")
        assert job_spec.epochs == 7
        assert job_spec.batch_size == 16
        assert job_spec.learning_rate == 1e-4
        assert job_spec.random_seed == 99
        assert job_spec.image_size == 224
        assert job_spec.model == "mobilenet_classification"

    def test_dataset_manifest_hash_computed_from_real_file(self, tmp_path, dataset_manifest):
        from training.cloud.base.artifact_validation import sha256_file
        package_dir = tmp_path / "package"
        job_spec, _ = build(
            experiment_id="exp",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/x",
            registry_path=tmp_path / "registry.jsonl",
        )
        assert job_spec.dataset_manifest_hash == sha256_file(dataset_manifest)

    def test_registry_records_non_production_class_set(self, tmp_path, dataset_manifest):
        from training.cloud.base.registry import get_experiment
        registry_path = tmp_path / "registry.jsonl"
        build(
            experiment_id="exp-3class",
            dataset_manifest_path=dataset_manifest,
            package_dir=tmp_path / "package",
            kaggle_dataset_ref="edithstark/x",
            registry_path=registry_path,
        )
        record = get_experiment("exp-3class", registry_path=registry_path)
        assert record["status_detail"]["is_production_class_set"] is False

    def test_dry_run_passes(self, tmp_path, dataset_manifest):
        from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run
        package_dir = tmp_path / "package"
        build(
            experiment_id="exp",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/x",
            registry_path=tmp_path / "registry.jsonl",
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

    def test_title_equals_slug(self, tmp_path, dataset_manifest):
        package_dir = tmp_path / "package"
        build(
            experiment_id="exp",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/x",
            registry_path=tmp_path / "registry.jsonl",
        )
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["title"] == metadata["id"].split("/", 1)[1]

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no dataset manifest"):
            build(
                experiment_id="exp",
                dataset_manifest_path=tmp_path / "does_not_exist.json",
                package_dir=tmp_path / "package",
                kaggle_dataset_ref="edithstark/x",
                registry_path=tmp_path / "registry.jsonl",
            )
