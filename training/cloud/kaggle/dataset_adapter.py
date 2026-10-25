"""training/cloud/kaggle/dataset_adapter.py — Kaggle Datasets adapter.

Mirrors the safety pattern already established in kaggle/adapter.py (the
kernel adapter): prepare() and dry_run() are fully local and touch nothing
on Kaggle's servers; create() is the one irreversible, network-touching
step and refuses to run without confirm=True.

`kaggle datasets create -p <folder>` uploads every file found under
<folder> (recursively), alongside a dataset-metadata.json placed directly
inside that same folder — there is no way to point the metadata file at a
separate data directory. To honor this project's rule that an audited
source dataset must never be modified, callers must never point
package_dir directly at an audited source directory. Two supported patterns:

  - small derived datasets (e.g. the smoke subset, itself already a copy
    produced by create_smoke_dataset.py): package_dir may be the dataset
    directory itself; prepare() just writes metadata.json into it.
  - large audited datasets: build a separate staging directory whose
    contents are OS-level junctions/symlinks to the audited directory's
    children (see stage_via_links()), so nothing is copied and the
    audited directory itself is never written to.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from training.cloud.base.artifact_validation import ValidationResult

# Kaggle's accepted license identifiers for dataset-metadata.json, per
# https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md
VALID_LICENSES = frozenset({
    "CC0-1.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "CC-BY-NC-SA-4.0", "GPL-2.0",
    "ODbL-1.0", "DbCL-1.0", "copyright-authors", "other", "unknown",
    "CC-BY-4.0", "CC-BY-NC-4.0", "PDDL", "CC-BY-3.0", "CC-BY-3.0-IGO",
    "US-Government-Works", "CC-BY-NC-SA-3.0-IGO", "CDLA-Permissive-1.0",
    "CDLA-Sharing-1.0", "CC-BY-ND-4.0", "CC-BY-NC-ND-4.0", "ODC-BY-1.0",
    "LGPL-3.0", "AGPL-3.0", "FDL-1.3", "EU-ODP-Legal-Notice", "apache-2.0",
    "GPL-3.0",
})


class KaggleCLIError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"kaggle CLI command failed (exit {returncode}): {' '.join(command)}\n{stderr or stdout}"
        )


class CreateNotConfirmedError(RuntimeError):
    """create() was called without confirm=True."""


@dataclass
class KaggleDatasetConfig:
    owner: str
    slug: str
    title: str
    license_name: str = "CC-BY-4.0"
    subtitle: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.license_name not in VALID_LICENSES:
            raise ValueError(f"{self.license_name!r} is not a Kaggle-recognized license identifier")

    @property
    def dataset_id(self) -> str:
        return f"{self.owner}/{self.slug}"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "id": self.dataset_id,
            "licenses": [{"name": self.license_name}],
        }


def stage_via_links(source_dir: Path, package_dir: Path, *, entries: Optional[list[str]] = None) -> Path:
    """Populate package_dir with OS-level links to source_dir's children,
    without copying data and without writing anything into source_dir.

    Uses os.symlink; falls back to a Windows directory junction (via
    mklink /J, which does not require admin/developer-mode privileges,
    unlike symlinks, and works across drives) for directories when symlink
    creation is not permitted. For a *file* entry, falls back to a hardlink
    (still no copy) when that's possible (same drive), and only as a last
    resort - when the source and package_dir are on different drives, so
    neither a symlink nor a hardlink can be created - copies the file. This
    only ever applies to small top-level files (e.g. manifest.json); the
    bulk data (train/val/test image directories) is always linked/junctioned,
    never copied.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    names = entries if entries is not None else sorted(p.name for p in source_dir.iterdir())
    for name in names:
        src = source_dir / name
        dst = package_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            if src.is_dir():
                os.symlink(src, dst, target_is_directory=True)
            else:
                os.symlink(src, dst)
        except OSError:
            if src.is_dir():
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    capture_output=True, text=True, check=True,
                )
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    # Cross-drive: neither symlink nor hardlink is possible
                    # for a file. Last resort only - never used for the bulk
                    # image directories above, which always junction.
                    shutil.copyfile(src, dst)
    return package_dir


def prepare(config: KaggleDatasetConfig, package_dir: Path) -> Path:
    """Write dataset-metadata.json into package_dir. package_dir must
    already contain the data files to upload (via stage_via_links() for a
    large audited dataset, or directly for a small derived one). Pure
    local file I/O — no Kaggle API call."""
    package_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = package_dir / "dataset-metadata.json"
    metadata_path.write_text(json.dumps(config.to_metadata_dict(), indent=2), encoding="utf-8")
    return package_dir


def dry_run(package_dir: Path, config: KaggleDatasetConfig, *, required_entries: Optional[list[str]] = None) -> ValidationResult:
    """Validate a prepared dataset package without touching the Kaggle API."""
    result = ValidationResult.start()

    metadata_path = package_dir / "dataset-metadata.json"
    if not metadata_path.is_file():
        result.add("metadata_file_exists", False, str(metadata_path))
        result.errors.append(f"dataset-metadata.json not found in {package_dir}")
        return result
    result.add("metadata_file_exists", True)

    try:
        on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.add("metadata_valid_json", False, str(exc))
        result.errors.append(f"dataset-metadata.json is not valid JSON: {exc}")
        return result
    result.add("metadata_valid_json", True)

    expected = config.to_metadata_dict()
    matches_config = on_disk == expected
    result.add("metadata_matches_config", matches_config, {"expected": expected, "on_disk": on_disk})
    if not matches_config:
        result.errors.append("dataset-metadata.json on disk does not match the provided config")

    dataset_id = on_disk.get("id", "")
    valid_id = (
        isinstance(dataset_id, str)
        and "/" in dataset_id
        and "INSERT_" not in dataset_id
        and all(part for part in dataset_id.split("/", 1))
    )
    result.add("dataset_id_well_formed", valid_id, dataset_id)
    if not valid_id:
        result.errors.append(f"dataset id is missing, malformed, or a leftover template placeholder: {dataset_id!r}")

    licenses = on_disk.get("licenses", [])
    license_ok = bool(licenses) and all(lic.get("name") in VALID_LICENSES for lic in licenses)
    result.add("license_recognized", license_ok, licenses)
    if not license_ok:
        result.errors.append(f"license(s) missing or not in Kaggle's recognized set: {licenses}")

    for name in (required_entries or []):
        entry_path = package_dir / name
        exists = entry_path.exists()
        result.add(f"entry_exists:{name}", exists, str(entry_path))
        if not exists:
            result.errors.append(f"required entry missing from package: {name}")

    return result


def _run_kaggle(args: list[str], *, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    command = ["kaggle", *args]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise KaggleCLIError(command, proc.returncode, proc.stdout, proc.stderr)
    return proc


def create(package_dir: Path, config: KaggleDatasetConfig, *, confirm: bool, public: bool = False, timeout: float = 1800.0) -> subprocess.CompletedProcess:
    """Upload package_dir as a new Kaggle dataset. IRREVERSIBLE (creates a
    real resource under the account) — refuses without confirm=True.
    Private by default; public=True must be passed explicitly."""
    if not confirm:
        raise CreateNotConfirmedError(
            "create() requires confirm=True - kaggle datasets create uploads real data "
            "and creates a persistent resource under the Kaggle account."
        )
    validation = dry_run(package_dir, config)
    if not validation.passed:
        raise ValueError(f"refusing to create: dry_run validation failed: {validation.errors}")

    args = ["datasets", "create", "-p", str(package_dir), "-r", "zip"]
    if public:
        args.append("-u")
    return _run_kaggle(args, timeout=timeout)


def status(dataset_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    proc = _run_kaggle(["datasets", "status", dataset_id], timeout=timeout)
    return {"dataset_id": dataset_id, "raw_output": proc.stdout.strip()}


def list_files(dataset_id: str, *, timeout: float = 60.0) -> str:
    proc = _run_kaggle(["datasets", "files", dataset_id], timeout=timeout)
    return proc.stdout
