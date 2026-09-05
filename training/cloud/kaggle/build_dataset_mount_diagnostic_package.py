"""training/cloud/kaggle/build_dataset_mount_diagnostic_package.py — Build
(locally only) the Kaggle kernel package for the dataset-mount diagnostic.

Mirrors build_yolo_smoke_package.py's structure, but for the diagnostic
entrypoint: no GPU requested, no internet requested (the diagnostic never
clones the repo - it's a single self-contained stdlib script), and the
declared dataset_sources entry is the only thing under test. Never calls
launch() - this script cannot push a kernel or consume GPU time by
construction (it never imports/calls that function).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.cloud.base.job_spec import capture_environment
from training.cloud.base.registry import DEFAULT_REGISTRY_PATH, record_experiment
from training.cloud.base.storage_paths import default_kaggle_package_dir
from training.cloud.kaggle.adapter import KaggleKernelConfig, dry_run, prepare
from training.cloud.kaggle.entrypoints.render import render_entrypoint

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TEMPLATE = _REPO_ROOT / "training" / "cloud" / "kaggle" / "entrypoints" / "dataset_mount_diagnostic.py"


def build(
    *,
    experiment_id: str,
    package_dir: Path,
    kaggle_dataset_ref: str,
    owner: str = "edithstark",
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Returns package_dir. Local only - never pushes anything."""
    env = capture_environment([], cwd=str(_REPO_ROOT))

    rendered_entrypoint = package_dir / "_rendered_dataset_mount_diagnostic.py"
    render_entrypoint(
        _TEMPLATE,
        {
            "git_sha": env["git_sha"],
            "declared_dataset_ref": kaggle_dataset_ref,
        },
        rendered_entrypoint,
    )

    config = KaggleKernelConfig(
        owner=owner,
        slug=experiment_id,
        # See build_yolo_smoke_package.py for why title must equal the slug
        # exactly - Kaggle derives the real URL slug from the title and
        # silently renames the kernel (warning only, not an error) if they
        # differ from the declared "id".
        title=experiment_id,
        code_file="diagnostic.py",
        enable_gpu=False,       # no GPU workload, by design
        enable_internet=False,  # self-contained script, no repo clone needed
        dataset_sources=[kaggle_dataset_ref],
    )
    prepare(config, rendered_entrypoint, package_dir)
    rendered_entrypoint.unlink()

    record_experiment({
        "experiment_id": experiment_id,
        "model": "diagnostic",
        "status": "prepared_local",
        "git_sha": env["git_sha"],
        "job_spec_hash": "",
        "dataset": kaggle_dataset_ref,
        "dataset_hash": "",
        "configuration": {},
        "hardware": {"requested_gpu": "none"},
        "status_detail": {
            "purpose": "dataset-mount diagnostic, not a smoke test or training run",
            "kaggle_dataset_ref": kaggle_dataset_ref,
        },
        "metrics": {},
        "checkpoint": "",
        "artifact_hash": "",
    }, registry_path=registry_path)

    return package_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--package-dir", type=Path, default=None,
        help="Defaults to the E: Solar AI data drive (see training/cloud/base/storage_paths.py) "
             "under kaggle_runs/<experiment-id> when available.",
    )
    parser.add_argument("--kaggle-dataset-ref", required=True)
    args = parser.parse_args()
    package_dir = args.package_dir or default_kaggle_package_dir(args.experiment_id)

    package_dir = build(
        experiment_id=args.experiment_id,
        package_dir=package_dir,
        kaggle_dataset_ref=args.kaggle_dataset_ref,
    )
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
