#!/usr/bin/env python3
"""scripts/download_ground_pv_dataset.py — Acquire the ground-level PV defect
RESEARCH dataset from Roboflow Universe.

    Dataset : "panel solar"  (owner: Susan)
    URL     : https://universe.roboflow.com/susan-ifblr/panel-solar-bw945
    Workspace / project / version : susan-ifblr / panel-solar-bw945 / 1
    Export format : yolov8
    License (as displayed by Roboflow) : CC BY 4.0

This script is deliberately fail-closed and non-destructive:

  * The Roboflow API key is read ONLY from the ROBOFLOW_API_KEY environment
    variable. It is never hard-coded, never prompted for, never written to
    disk, never logged. If it is missing, the script exits with a clear
    message naming exactly that variable and does NOT proceed.
  * If a raw download already exists at the target directory, it is NOT
    overwritten. Re-running is a no-op unless --force is given.
  * The raw download is treated as immutable. Normalization to the research
    taxonomy happens later, in a separate step, into a separate directory.
  * A download manifest (counts, sha256 of data.yaml, timestamps, the
    resolved Roboflow version/id) is written next to the raw data.

Nothing here touches the production model or the aerial dataset.

Examples
--------
    export ROBOFLOW_API_KEY=xxxxxxxx            # (Windows: set ROBOFLOW_API_KEY=xxxx)
    python scripts/download_ground_pv_dataset.py
    python scripts/download_ground_pv_dataset.py --dest datasets/ground_pv_research/raw/panel_solar_bw945_v1
    python scripts/download_ground_pv_dataset.py --check-only     # report state, download nothing
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = "susan-ifblr"
PROJECT = "panel-solar-bw945"
VERSION = 1
EXPORT_FORMAT = "yolov8"
DATASET_URL = "https://universe.roboflow.com/susan-ifblr/panel-solar-bw945"
LICENSE_AS_DISPLAYED = "CC BY 4.0"
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

DEFAULT_DEST = _REPO_ROOT / "datasets" / "ground_pv_research" / "raw" / "panel_solar_bw945_v1"

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_split(root: Path) -> dict[str, dict[str, int]]:
    """Count images/labels under each of train/valid/test (and their
    common aliases) for whatever the export actually produced."""
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "val", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() and not labels_dir.is_dir():
            continue
        n_images = sum(1 for p in images_dir.glob("*") if p.suffix.lower() in _IMAGE_EXT) if images_dir.is_dir() else 0
        n_labels = sum(1 for _ in labels_dir.glob("*.txt")) if labels_dir.is_dir() else 0
        out[split] = {"images": n_images, "labels": n_labels}
    return out


def _read_data_yaml(root: Path) -> dict:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return {}
    import yaml
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def _describe_existing(dest: Path) -> dict:
    return {
        "path": str(dest),
        "exists": dest.is_dir(),
        "has_data_yaml": (dest / "data.yaml").is_file(),
        "splits": _count_split(dest) if dest.is_dir() else {},
        "data_yaml": _read_data_yaml(dest) if dest.is_dir() else {},
    }


def _require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        sys.stderr.write(
            "\n".join([
                "",
                "=" * 72,
                f"BLOCKED: the {API_KEY_ENV_VAR} environment variable is not set.",
                "=" * 72,
                "",
                "The Roboflow Universe dataset",
                f"    {DATASET_URL}",
                "requires an API key for programmatic download (the public API",
                "endpoints return HTTP 401/403 without one). This script will not",
                "invent, prompt for, or hard-code a credential.",
                "",
                "To proceed:",
                "  1. Sign in at https://app.roboflow.com and open Settings -> API Keys",
                "     (or the 'Private API Key' shown on your workspace settings page).",
                "  2. Export it (do NOT commit it anywhere):",
                "       Windows (PowerShell):  $env:ROBOFLOW_API_KEY = \"<key>\"",
                "       Windows (cmd):          set ROBOFLOW_API_KEY=<key>",
                "       macOS/Linux:            export ROBOFLOW_API_KEY=<key>",
                "  3. Re-run:  python scripts/download_ground_pv_dataset.py",
                "",
                "No dataset was downloaded. Nothing downstream (audit, Kaggle",
                "packaging, training) can run until the raw dataset exists.",
                "",
            ])
        )
        raise SystemExit(2)
    return key


def _download_with_roboflow(api_key: str, dest: Path) -> Path:
    """Use the official roboflow package. Returns the directory the export
    landed in (roboflow chooses a name; we then move it into `dest`)."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"the 'roboflow' package is not installed ({exc}). "
            "Install it:  pip install roboflow"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    staging = dest.parent / f".{dest.name}.download_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # roboflow downloads into location/<Project Name-vN>/; ask it to use our staging dir.
    print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} as {EXPORT_FORMAT} ...", flush=True)
    ds = version.download(EXPORT_FORMAT, location=str(staging), overwrite=True)
    export_dir = Path(ds.location)
    if not export_dir.is_dir():
        raise SystemExit(f"roboflow reported location {export_dir} but it does not exist")
    return export_dir


def _finalize(export_dir: Path, dest: Path, api_version_meta: dict | None) -> dict:
    if dest.exists():
        raise SystemExit(f"destination {dest} already exists - refusing to overwrite (use --force)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(export_dir), str(dest))
    # clean up the now-empty staging parent if roboflow made one
    staging = dest.parent / f".{dest.name}.download_staging"
    if staging.is_dir() and not any(staging.iterdir()):
        staging.rmdir()

    data_yaml = dest / "data.yaml"
    splits = _count_split(dest)
    total_images = sum(s["images"] for s in splits.values())
    total_labels = sum(s["labels"] for s in splits.values())

    manifest = {
        "dataset": "panel solar",
        "owner": "Susan",
        "source": "Roboflow Universe",
        "url": DATASET_URL,
        "workspace": WORKSPACE,
        "project": PROJECT,
        "version": VERSION,
        "export_format": EXPORT_FORMAT,
        "license_as_displayed_by_source": LICENSE_AS_DISPLAYED,
        "downloaded_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "raw_dir": str(dest),
        "data_yaml_sha256": _sha256_file(data_yaml) if data_yaml.is_file() else None,
        "data_yaml_contents": _read_data_yaml(dest),
        "split_counts": splits,
        "total_images": total_images,
        "total_labels": total_labels,
        "roboflow_version_metadata": api_version_meta or {},
        "notes": [
            "Raw download - immutable. Do not edit in place.",
            "Published counts are NOT verified here; run scripts/audit_ground_pv_dataset.py.",
        ],
    }
    (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Raw download directory (default: {DEFAULT_DEST}).")
    parser.add_argument("--check-only", action="store_true",
                        help="Report the current state and required credentials; download nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Allow replacing an existing raw download (moves the old one aside first).")
    args = parser.parse_args(argv)

    dest: Path = args.dest
    state = _describe_existing(dest)

    if args.check_only:
        state["api_key_present"] = bool(os.environ.get(API_KEY_ENV_VAR, "").strip())
        state["api_key_env_var"] = API_KEY_ENV_VAR
        print(json.dumps(state, indent=2))
        return 0

    if dest.is_dir() and (dest / "data.yaml").is_file() and not args.force:
        print(f"Raw dataset already present at {dest} (data.yaml found). Nothing to do.")
        print("Re-run with --force to replace it, or run the audit next:")
        print("  python scripts/audit_ground_pv_dataset.py")
        print(json.dumps(state, indent=2))
        return 0

    api_key = _require_api_key()

    if dest.exists() and args.force:
        aside = dest.with_name(dest.name + ".replaced-" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S"))
        print(f"--force: moving existing {dest} -> {aside}")
        shutil.move(str(dest), str(aside))

    export_dir = _download_with_roboflow(api_key, dest)
    manifest = _finalize(export_dir, dest, api_version_meta=None)

    print("\nDownload complete.")
    print(f"  raw dir      : {manifest['raw_dir']}")
    print(f"  total images : {manifest['total_images']}")
    print(f"  total labels : {manifest['total_labels']}")
    print(f"  split counts : {manifest['split_counts']}")
    print(f"  data.yaml    : names={manifest['data_yaml_contents'].get('names')}")
    print("\nNext: python scripts/audit_ground_pv_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
