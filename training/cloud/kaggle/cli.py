"""training/cloud/kaggle/cli.py — Command-line entry point tying together
TrainingJobSpec, the Kaggle adapter, and the experiment registry.

Three explicit subcommands, matching the three real operations:

    prepare   - local only. Writes the kernel package + records an
                experiment in the registry with status "prepared".
    dry-run   - local only. Validates an already-prepared package.
    launch    - IRREVERSIBLE. Pushes the package and starts Kaggle GPU
                execution. Requires --yes; refuses to run otherwise.

Example (does not execute anything by itself - these are the commands a
human or Claude would run):

    python -m training.cloud.kaggle.cli prepare \\
        --experiment-id exp-0001 --model yolo_detection \\
        --entrypoint training/detection/train_yolo_kaggle_entry.py \\
        --package-dir training/cloud/runs/exp-0001 \\
        --dataset-source gabrielkasmi/bdappv --gpu

    python -m training.cloud.kaggle.cli dry-run --package-dir training/cloud/runs/exp-0001

    python -m training.cloud.kaggle.cli launch --package-dir training/cloud/runs/exp-0001 --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.cloud.base.registry import record_experiment, update_experiment_status
from training.cloud.kaggle.adapter import (
    KaggleKernelConfig,
    dry_run,
    launch,
    prepare,
)


def _cmd_prepare(args: argparse.Namespace) -> int:
    config = KaggleKernelConfig(
        owner=args.owner,
        slug=args.experiment_id,
        title=args.title or args.experiment_id,
        code_file=Path(args.entrypoint).name,
        enable_gpu=args.gpu,
        enable_internet=args.internet,
        dataset_sources=args.dataset_source or [],
    )
    package_dir = prepare(config, Path(args.entrypoint), Path(args.package_dir))
    print(f"prepared kernel package at {package_dir}")

    record_experiment({
        "experiment_id": args.experiment_id,
        "model": args.model,
        "status": "prepared",
        "kernel_id": config.kernel_id,
        "package_dir": str(package_dir),
        "configuration": {
            "enable_gpu": config.enable_gpu,
            "enable_internet": config.enable_internet,
            "dataset_sources": config.dataset_sources,
        },
        "hardware": {"requested_gpu": "kaggle-default" if config.enable_gpu else "none"},
        "metrics": {},
        "checkpoint": "",
        "artifact_hash": "",
    })
    print(f"recorded experiment '{args.experiment_id}' with status=prepared")
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir)
    metadata_path = package_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        print(f"ERROR: no kernel-metadata.json in {package_dir} - run 'prepare' first", file=sys.stderr)
        return 1
    import json
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    owner, slug = data["id"].split("/", 1)
    config = KaggleKernelConfig(
        owner=owner, slug=slug, title=data["title"], code_file=data["code_file"],
        enable_gpu=data["enable_gpu"], enable_internet=data["enable_internet"],
        dataset_sources=data.get("dataset_sources", []),
    )
    result = dry_run(package_dir, config)
    if result.passed:
        print("DRY RUN: PASS")
        return 0
    print("DRY RUN: FAIL")
    for error in result.errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def _cmd_launch(args: argparse.Namespace) -> int:
    if not args.yes:
        print(
            "ERROR: launch requires --yes. This starts Kaggle GPU execution "
            "immediately and cannot be cancelled via the Kaggle CLI.",
            file=sys.stderr,
        )
        return 1

    package_dir = Path(args.package_dir)
    import json
    metadata_path = package_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        print(f"ERROR: no kernel-metadata.json in {package_dir} - run 'prepare' first", file=sys.stderr)
        return 1
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    owner, slug = data["id"].split("/", 1)
    config = KaggleKernelConfig(
        owner=owner, slug=slug, title=data["title"], code_file=data["code_file"],
        enable_gpu=data["enable_gpu"], enable_internet=data["enable_internet"],
        dataset_sources=data.get("dataset_sources", []),
    )

    result = launch(package_dir, config, confirm=True)
    print(f"launched: {result.kernel_id}")
    print(result.stdout)

    if args.experiment_id:
        try:
            update_experiment_status(args.experiment_id, "launched", kernel_id=result.kernel_id)
        except KeyError:
            print(f"WARNING: no existing experiment record for '{args.experiment_id}' to update", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Build the kernel package locally (no network call).")
    p_prepare.add_argument("--experiment-id", required=True)
    p_prepare.add_argument("--model", required=True, choices=["yolo_detection", "mobilenet_classification"])
    p_prepare.add_argument("--owner", default="edithstark")
    p_prepare.add_argument("--title", default=None)
    p_prepare.add_argument("--entrypoint", required=True)
    p_prepare.add_argument("--package-dir", required=True)
    p_prepare.add_argument("--gpu", action="store_true")
    p_prepare.add_argument("--internet", action="store_true")
    p_prepare.add_argument("--dataset-source", action="append", default=[])
    p_prepare.set_defaults(func=_cmd_prepare)

    p_dry_run = sub.add_parser("dry-run", help="Validate a prepared package locally (no network call).")
    p_dry_run.add_argument("--package-dir", required=True)
    p_dry_run.set_defaults(func=_cmd_dry_run)

    p_launch = sub.add_parser("launch", help="IRREVERSIBLE: push and start Kaggle GPU execution.")
    p_launch.add_argument("--package-dir", required=True)
    p_launch.add_argument("--experiment-id", default=None)
    p_launch.add_argument("--yes", action="store_true", help="Required to actually launch.")
    p_launch.set_defaults(func=_cmd_launch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
