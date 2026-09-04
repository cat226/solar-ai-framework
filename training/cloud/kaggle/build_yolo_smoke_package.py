"""training/cloud/kaggle/build_yolo_smoke_package.py — Build (locally only) the
Kaggle kernel package for the YOLO detection smoke test.

Ties together everything from Phases 3-4: captures a TrainingJobSpec,
renders the yolo_detection entrypoint template with real values, builds the
kernel package via the Kaggle adapter's prepare(), and validates it with
dry_run(). Never calls launch() - this script cannot push a kernel or
consume GPU time by construction (it never imports/calls that function).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.cloud.base.job_spec import TrainingJobSpec, capture_environment
from training.cloud.base.registry import DEFAULT_REGISTRY_PATH, record_experiment
from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run, prepare
from training.cloud.kaggle.entrypoints.render import render_entrypoint

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TEMPLATE = _REPO_ROOT / "training" / "cloud" / "kaggle" / "entrypoints" / "yolo_detection.py"


def build(
    *,
    experiment_id: str,
    smoke_dataset_root: Path,
    package_dir: Path,
    kaggle_dataset_ref: str | None,
    owner: str = "edithstark",
    epochs: int = 1,
    batch: int = 8,
    imgsz: int = 640,
    seed: int = 42,
    base_model: str = "yolov8n.pt",
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[TrainingJobSpec, Path]:
    """Returns (job_spec, package_dir). Local only - never pushes anything."""
    smoke_manifest_path = smoke_dataset_root / "manifest.json"
    if not smoke_manifest_path.is_file():
        raise RuntimeError(f"no smoke-dataset manifest at {smoke_manifest_path} - run create_smoke_dataset.py first")
    smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))

    env = capture_environment(["ultralytics", "torch", "PyYAML"], cwd=str(_REPO_ROOT))
    job_spec = TrainingJobSpec(
        experiment_id=experiment_id,
        model="yolo_detection",
        git_sha=env["git_sha"],
        dataset_id="gabrielkasmi/bdappv (IGN config) - local smoke subset",
        dataset_revision="main",
        dataset_manifest_hash=smoke_manifest["source_manifest_hash"],
        class_order=tuple(smoke_manifest["class_names"]),
        image_size=imgsz,
        batch_size=batch,
        epochs=epochs,
        optimizer="auto",
        learning_rate=0.01,
        random_seed=seed,
        requested_gpu="kaggle-default",
        python_version=env["python_version"],
        package_versions=env["package_versions"],
    )

    # Kaggle mounts an attached dataset under /kaggle/input/datasets/<owner>/<slug>/,
    # NOT the flat /kaggle/input/<slug>/ that older Kaggle docs/examples show -
    # confirmed empirically via a real read-only diagnostic kernel
    # (solar-yolo-smoke-001-dataset-mount-diagnostic, 2026-09-04): the
    # declared dataset edithstark/solar-ai-yolo-smoke-001 was present, but
    # only under /kaggle/input/datasets/edithstark/solar-ai-yolo-smoke-001/,
    # which is why the first two real launches failed the data_root.is_dir()
    # check even though dataset_sources was correctly attached both times.
    kaggle_data_root = (
        f"/kaggle/input/datasets/{kaggle_dataset_ref}"
        if kaggle_dataset_ref
        else "__NOT_YET_UPLOADED__/kaggle/input/datasets/<owner>/<dataset-slug-once-uploaded>"
    )

    rendered_entrypoint = package_dir / "_rendered_yolo_detection.py"
    render_entrypoint(
        _TEMPLATE,
        {
            "git_sha": job_spec.git_sha,
            "data_root": kaggle_data_root,
            "output_path": "/kaggle/working/yolo_solar_candidate.pt",
            "epochs": str(epochs),
            "batch": str(batch),
            "imgsz": str(imgsz),
            "seed": str(seed),
            "base_model": base_model,
        },
        rendered_entrypoint,
    )

    config = KaggleKernelConfig(
        owner=owner,
        slug=experiment_id,
        # Kaggle derives the kernel's actual URL slug from the title, not
        # from the "id" field in kernel-metadata.json, and silently renames
        # the kernel (with only a warning on stdout) if the two don't match
        # - discovered for real when solar-yolo-smoke-001's first launch
        # came back as edithstark/solar-ai-yolo-smoke-test-solar-yolo-smoke-001
        # instead of the requested slug. Since experiment_id is already a
        # lowercase alnum+hyphen string, using it verbatim as the title
        # makes Kaggle's title-derived slug equal the requested slug.
        title=experiment_id,
        code_file="train.py",
        enable_gpu=True,
        enable_internet=True,  # required: git clone + pip install + base-model download
        dataset_sources=[kaggle_dataset_ref] if kaggle_dataset_ref else [],
    )
    prepare(config, rendered_entrypoint, package_dir)
    # The rendered file was copied in as config.code_file by prepare(); the
    # loose staging copy above is no longer needed.
    rendered_entrypoint.unlink()

    record_experiment({
        "experiment_id": experiment_id,
        "model": "yolo_detection",
        "status": "prepared_local",  # built and dry-run locally; not launched
        "git_sha": job_spec.git_sha,
        "job_spec_hash": job_spec.spec_hash(),
        "dataset": job_spec.dataset_id,
        "dataset_hash": job_spec.dataset_manifest_hash,
        "configuration": {
            "epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed, "base_model": base_model,
        },
        "hardware": {"requested_gpu": job_spec.requested_gpu},
        "status_detail": {"kaggle_dataset_ref": kaggle_dataset_ref or "not_yet_uploaded"},
        "metrics": {},
        "checkpoint": "",
        "artifact_hash": "",
    }, registry_path=registry_path)

    return job_spec, package_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--smoke-dataset-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--kaggle-dataset-ref", default=None, help="owner/slug once uploaded to Kaggle (Step 4C)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    job_spec, package_dir = build(
        experiment_id=args.experiment_id,
        smoke_dataset_root=args.smoke_dataset_root,
        package_dir=args.package_dir,
        kaggle_dataset_ref=args.kaggle_dataset_ref,
        epochs=args.epochs,
        batch=args.batch,
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
