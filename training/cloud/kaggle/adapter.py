"""training/cloud/kaggle/adapter.py — Kaggle kernel adapter.

Kaggle has no separate "create then run" operation for kernels: `kaggle
kernels push` uploads the code AND immediately starts execution on Kaggle's
infrastructure. There is no dry-run mode on Kaggle's side and no native
cancel/stop command (verified against the real CLI - see
training/cloud/README.md). This module makes that explicit rather than
pretending a safer-looking API exists:

- prepare()  — fully local. Writes kernel-metadata.json + copies the entry
               script into a package directory. No network call.
- dry_run()  — fully local. Validates the prepared package's structure and
               metadata without invoking the Kaggle CLI at all.
- launch()   — THE irreversible step: shells out to `kaggle kernels push`,
               which uploads and starts GPU execution immediately. Refuses
               to run unless called with confirm=True.
- status() / logs() / outputs() — read-only Kaggle CLI calls against an
               already-launched kernel. Safe to call at any time; consume
               no additional GPU time themselves (they only report on
               execution already happening).

All Kaggle CLI invocations use subprocess with an explicit argument list
(never shell=True) and no secrets are ever passed as command-line
arguments - the CLI reads ~/.kaggle credentials itself.
"""
from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from training.cloud.base.artifact_validation import ValidationResult


class KaggleCLIError(RuntimeError):
    """A `kaggle` CLI invocation failed or returned an error."""

    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"kaggle CLI command failed (exit {returncode}): {' '.join(command)}\n{stderr or stdout}"
        )


class LaunchNotConfirmedError(RuntimeError):
    """launch() was called without confirm=True. Kaggle kernel execution is
    irreversible (no cancel command exists) and starts GPU billing/quota
    consumption immediately - this must never happen by accident."""


@dataclass
class KaggleKernelConfig:
    """Maps directly onto Kaggle's kernel-metadata.json schema."""

    owner: str
    slug: str
    title: str
    code_file: str  # filename within the package dir, e.g. "train.py"
    kernel_type: str = "script"  # "script" or "notebook"
    language: str = "python"
    is_private: bool = True
    enable_gpu: bool = False
    enable_tpu: bool = False
    enable_internet: bool = False  # off by default - only turn on when genuinely needed
    dataset_sources: list[str] = field(default_factory=list)  # ["owner/dataset-slug", ...]
    competition_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)

    @property
    def kernel_id(self) -> str:
        return f"{self.owner}/{self.slug}"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "id": self.kernel_id,
            "title": self.title,
            "code_file": self.code_file,
            "language": self.language,
            "kernel_type": self.kernel_type,
            "is_private": self.is_private,
            "enable_gpu": self.enable_gpu,
            "enable_tpu": self.enable_tpu,
            "enable_internet": self.enable_internet,
            "dataset_sources": list(self.dataset_sources),
            "competition_sources": list(self.competition_sources),
            "kernel_sources": list(self.kernel_sources),
            "model_sources": list(self.model_sources),
        }


@dataclass
class LaunchResult:
    kernel_id: str
    package_dir: str
    command: list[str]
    stdout: str
    stderr: str


def prepare(
    config: KaggleKernelConfig,
    entrypoint_script: Path,
    package_dir: Path,
    *,
    extra_files: Optional[list[Path]] = None,
) -> Path:
    """Build the local kernel package: kernel-metadata.json + the entry
    script (+ any extra files, e.g. a shared module the entry script
    imports). Pure local file I/O - no Kaggle API call.

    Returns package_dir.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = package_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(config.to_metadata_dict(), indent=2), encoding="utf-8")

    dest_entry = package_dir / config.code_file
    shutil.copyfile(entrypoint_script, dest_entry)

    for extra in extra_files or []:
        shutil.copyfile(extra, package_dir / extra.name)

    return package_dir


def dry_run(package_dir: Path, config: KaggleKernelConfig) -> ValidationResult:
    """Validate a prepared package without touching the Kaggle API.

    Checks: metadata.json exists and is valid JSON matching config, the
    entry script exists inside the package, the kernel id is well-formed
    (owner/slug, no placeholder text left over from a template), and GPU
    is only enabled when the caller actually asked for it (catches an
    accidental default flipping on billed compute).
    """
    result = ValidationResult.start()

    metadata_path = package_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        result.add("metadata_file_exists", False, str(metadata_path))
        result.errors.append(f"kernel-metadata.json not found in {package_dir}")
        return result
    result.add("metadata_file_exists", True)

    try:
        on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.add("metadata_valid_json", False, str(exc))
        result.errors.append(f"kernel-metadata.json is not valid JSON: {exc}")
        return result
    result.add("metadata_valid_json", True)

    expected = config.to_metadata_dict()
    matches_config = on_disk == expected
    result.add("metadata_matches_config", matches_config, {"expected": expected, "on_disk": on_disk})
    if not matches_config:
        result.errors.append("kernel-metadata.json on disk does not match the provided config")

    kernel_id = on_disk.get("id", "")
    valid_id = (
        isinstance(kernel_id, str)
        and "/" in kernel_id
        and "INSERT_" not in kernel_id
        and all(part for part in kernel_id.split("/", 1))
    )
    result.add("kernel_id_well_formed", valid_id, kernel_id)
    if not valid_id:
        result.errors.append(f"kernel id is missing, malformed, or a leftover template placeholder: {kernel_id!r}")

    entry_path = package_dir / config.code_file
    entry_exists = entry_path.is_file()
    result.add("entry_script_exists", entry_exists, str(entry_path))
    if not entry_exists:
        result.errors.append(f"entry script not found in package: {entry_path}")

    result.add("gpu_matches_request", on_disk.get("enable_gpu") == config.enable_gpu)
    result.add("internet_matches_request", on_disk.get("enable_internet") == config.enable_internet)

    return result


def _run_kaggle(args: list[str], *, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    command = ["kaggle", *args]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise KaggleCLIError(command, proc.returncode, proc.stdout, proc.stderr)
    return proc


def launch(package_dir: Path, config: KaggleKernelConfig, *, confirm: bool, timeout: float = 300.0) -> LaunchResult:
    """Push the prepared package and start execution. IRREVERSIBLE: this
    begins consuming Kaggle GPU quota/time immediately, and there is no
    Kaggle-native cancel command.

    Raises LaunchNotConfirmedError unless confirm=True is passed explicitly
    by the caller - this function must never be reachable from a code path
    that doesn't deliberately intend to spend compute.
    """
    if not confirm:
        raise LaunchNotConfirmedError(
            "launch() requires confirm=True - kaggle kernels push starts GPU "
            "execution immediately and cannot be cancelled via the Kaggle CLI."
        )
    validation = dry_run(package_dir, config)
    if not validation.passed:
        raise ValueError(f"refusing to launch: dry_run validation failed: {validation.errors}")

    command = ["kernels", "push", "-p", str(package_dir)]
    proc = _run_kaggle(command, timeout=timeout)
    return LaunchResult(
        kernel_id=config.kernel_id,
        package_dir=str(package_dir),
        command=["kaggle", *command],
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def status(kernel_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Read-only: current status of an already-launched kernel."""
    proc = _run_kaggle(["kernels", "status", kernel_id], timeout=timeout)
    return {"kernel_id": kernel_id, "raw_output": proc.stdout.strip()}


def logs(kernel_id: str, *, timeout: float = 60.0) -> str:
    """Read-only: execution logs from the latest run of kernel_id."""
    proc = _run_kaggle(["kernels", "logs", kernel_id], timeout=timeout)
    return proc.stdout


def outputs(kernel_id: str, dest_dir: Path, *, timeout: float = 300.0) -> list[Path]:
    """Read-only (network download, not compute): retrieve output files from
    the latest run of kernel_id into dest_dir. Returns the list of files
    found there afterward (best-effort listing, not necessarily exhaustive
    if the CLI's own output format changes)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_kaggle(["kernels", "output", kernel_id, "-p", str(dest_dir)], timeout=timeout)
    return sorted(p for p in dest_dir.rglob("*") if p.is_file())
