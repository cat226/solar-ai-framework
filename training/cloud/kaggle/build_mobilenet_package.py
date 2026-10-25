"""training/cloud/kaggle/build_mobilenet_package.py — Build (locally only)
the Kaggle kernel package for MobileNetV2 classification training (smoke or
full - same shape, just a different dataset manifest/ref and hyperparameters).

Same structure as build_yolo_full_training_package.py: captures a
TrainingJobSpec, renders the mobilenet_classification entrypoint template
with real values, builds the kernel package via the Kaggle adapter's
prepare(), validates it with dry_run(). Never calls launch() - this script
cannot push a kernel or consume GPU time by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.cloud.base.artifact_validation import sha256_file
from training.cloud.base.job_spec import TrainingJobSpec, capture_environment
from training.cloud.base.registry import DEFAULT_REGISTRY_PATH, record_experiment
from training.cloud.base.storage_paths import default_kaggle_package_dir
from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run, prepare
from training.cloud.kaggle.entrypoints.render import render_entrypoint

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TEMPLATE = _REPO_ROOT / "training" / "cloud" / "kaggle" / "entrypoints" / "mobilenet_classification.py"


def build(
    *,
    experiment_id: str,
    dataset_manifest_path: Path,
    package_dir: Path,
    kaggle_dataset_ref: str,
    owner: str = "edithstark",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[TrainingJobSpec, Path]:
    """Returns (job_spec, package_dir). Local only - never pushes anything."""
    if not dataset_manifest_path.is_file():
        raise RuntimeError(f"no dataset manifest at {dataset_manifest_path}")
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(dataset_manifest_path)
    classes: list[str] = manifest["classes"]

    env = capture_environment(["torch", "torchvision"], cwd=str(_REPO_ROOT))
    job_spec = TrainingJobSpec(
        experiment_id=experiment_id,
        model="mobilenet_classification",
        git_sha=env["git_sha"],
        dataset_id="SolNET (Clean/Dusty, MDPI Energies 2023, CC BY 4.0) + PVMD (Hotspot, Mendeley, CC BY 4.0)",
        dataset_revision="main",
        dataset_manifest_hash=manifest_hash,
        class_order=tuple(classes),
        image_size=224,
        batch_size=batch_size,
        epochs=epochs,
        optimizer="AdamW",
        learning_rate=lr,
        random_seed=seed,
        requested_gpu="kaggle-default",
        python_version=env["python_version"],
        package_versions=env["package_versions"],
    )

    # Same nested mount convention confirmed empirically on the YOLO Kaggle
    # runs - see build_yolo_full_training_package.py.
    kaggle_data_root = f"/kaggle/input/datasets/{kaggle_dataset_ref}"

    rendered_entrypoint = package_dir / "_rendered_mobilenet_classification.py"
    render_entrypoint(
        _TEMPLATE,
        {
            "git_sha": job_spec.git_sha,
            "data_root": kaggle_data_root,
            "output_path": "/kaggle/working/mobilenet_solar_candidate.pth",
            "classes": ",".join(classes),
            "epochs": str(epochs),
            "batch_size": str(batch_size),
            "lr": str(lr),
            "seed": str(seed),
        },
        rendered_entrypoint,
    )

    config = KaggleKernelConfig(
        owner=owner,
        slug=experiment_id,
        # title == slug avoids Kaggle silently renaming the kernel - see
        # build_yolo_smoke_package.py for the real failure this fixes.
        title=experiment_id,
        code_file="train.py",
        enable_gpu=True,
        enable_internet=True,  # required: git clone + pip install
        dataset_sources=[kaggle_dataset_ref],
    )
    prepare(config, rendered_entrypoint, package_dir)
    rendered_entrypoint.unlink()

    record_experiment({
        "experiment_id": experiment_id,
        "model": "mobilenet_classification",
        "status": "prepared_local",  # built and dry-run locally; not launched
        "git_sha": job_spec.git_sha,
        "job_spec_hash": job_spec.spec_hash(),
        "dataset": job_spec.dataset_id,
        "dataset_hash": job_spec.dataset_manifest_hash,
        "configuration": {
            "classes": classes, "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        },
        "hardware": {"requested_gpu": job_spec.requested_gpu},
        "status_detail": {
            "kaggle_dataset_ref": kaggle_dataset_ref,
            "is_production_class_set": classes == [
                "Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Hotspot",
            ],
        },
        "metrics": {},
        "checkpoint": "",
        "artifact_hash": "",
    }, registry_path=registry_path)

    return job_spec, package_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset-manifest-path", type=Path, required=True)
    parser.add_argument(
        "--package-dir", type=Path, default=None,
        help="Defaults to the E: Solar AI data drive (see training/cloud/base/storage_paths.py) "
             "under kaggle_runs/<experiment-id> when available.",
    )
    parser.add_argument("--kaggle-dataset-ref", required=True, help="owner/slug of the uploaded Kaggle dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    package_dir = args.package_dir or default_kaggle_package_dir(args.experiment_id)

    job_spec, package_dir = build(
        experiment_id=args.experiment_id,
        dataset_manifest_path=args.dataset_manifest_path,
        package_dir=package_dir,
        kaggle_dataset_ref=args.kaggle_dataset_ref,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"job spec hash: {job_spec.spec_hash()}")
    print(f"package built at: {package_dir}")

    config_path = package_dir / "kernel-metadata.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    owner, slug = config_data["id"].split("/", 1)
    kernel_config = KaggleKernelConfig(
        owner=owner, slug=slug, title=config_data["title"], code_file=config_data["code_file"],
        enable_gpu=config_data["enable_gpu"], enable_internet=config_data["enable_internet"],
        dataset_sources=config_data.get("dataset_sources", []),
    )
    result = dry_run(package_dir, kernel_config)
    print(f"DRY RUN: {'PASS' if result.passed else 'FAIL'}")
    for error in result.errors:
        print(f"  - {error}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
