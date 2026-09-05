"""Tests for training/cloud/kaggle/build_yolo_full_training_package.py.

No test here touches the real Kaggle CLI or uploads anything -
prepare()/dry_run() are pure local file I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.cloud.kaggle.build_yolo_full_training_package import build


@pytest.fixture
def dataset_manifest(tmp_path):
    manifest = {
        "source": "gabrielkasmi/bdappv",
        "class_names": ["solar panel"],
        "counts": {
            "train": {"images": 11347},
            "val": {"images": 3179},
            "test": {"images": 2581},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class TestBuildFullTrainingPackage:
    def test_requires_real_dataset_ref(self, tmp_path, dataset_manifest):
        """Unlike the smoke package, full training only ever makes sense
        once the full dataset is actually uploaded - there is no honest
        placeholder path to fall back to."""
        import inspect
        sig = inspect.signature(build)
        assert sig.parameters["kaggle_dataset_ref"].default is inspect.Parameter.empty

    def test_data_root_uses_nested_datasets_path(self, tmp_path, dataset_manifest):
        package_dir = tmp_path / "package"
        build(
            experiment_id="solar-yolo-full-v1",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
            registry_path=tmp_path / "registry.jsonl",
        )
        rendered = (package_dir / "train.py").read_text(encoding="utf-8")
        assert "'/kaggle/input/datasets/edithstark/solar-ai-bdappv-ign-yolo-v1'" in rendered

    def test_default_hyperparameters_match_train_yolo_defaults(self, tmp_path, dataset_manifest):
        """Per instruction: use train_yolo.py's existing defaults (epochs=3,
        batch=16) unless there's a documented reason to deviate."""
        package_dir = tmp_path / "package"
        job_spec, _ = build(
            experiment_id="solar-yolo-full-v1",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
            registry_path=tmp_path / "registry.jsonl",
        )
        assert job_spec.epochs == 3
        assert job_spec.batch_size == 16
        assert job_spec.image_size == 640
        assert job_spec.random_seed == 42

    def test_dataset_manifest_hash_computed_from_real_file(self, tmp_path, dataset_manifest):
        from training.cloud.base.artifact_validation import sha256_file
        package_dir = tmp_path / "package"
        job_spec, _ = build(
            experiment_id="solar-yolo-full-v1",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
            registry_path=tmp_path / "registry.jsonl",
        )
        assert job_spec.dataset_manifest_hash == sha256_file(dataset_manifest)

    def test_dry_run_passes(self, tmp_path, dataset_manifest):
        from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run

        package_dir = tmp_path / "package"
        build(
            experiment_id="solar-yolo-full-v1",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
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
            experiment_id="solar-yolo-full-v1",
            dataset_manifest_path=dataset_manifest,
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
            registry_path=tmp_path / "registry.jsonl",
        )
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["title"] == metadata["id"].split("/", 1)[1]

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no dataset manifest"):
            build(
                experiment_id="solar-yolo-full-v1",
                dataset_manifest_path=tmp_path / "does_not_exist.json",
                package_dir=tmp_path / "package",
                kaggle_dataset_ref="edithstark/solar-ai-bdappv-ign-yolo-v1",
                registry_path=tmp_path / "registry.jsonl",
            )
