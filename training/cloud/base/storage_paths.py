"""training/cloud/base/storage_paths.py — Canonical locations for large,
locally-generated Solar AI data.

Policy (2026-09-04): this project's other local drives (C:, D: — where the
repository itself lives) run low on space, so all large/local
Solar-AI-generated data (downloaded/prepared datasets, Kaggle dataset
staging, retrieved training outputs, local training run directories,
caches, temp files) defaults to the E: drive instead. Small, git-tracked
metadata (the experiment registry, kernel-metadata.json templates, source
code) stays in the repository as before - this module is only about where
*large* data lives.

The root is environment-variable-overridable (`SOLAR_AI_DATA_ROOT`) so this
stays portable to a machine that doesn't have an E: drive at all (a CI
runner, another contributor's machine) - callers that need E: specifically
on this machine get it as the default, but nothing here hardcodes it as
the only possible location.

This module never creates directories or moves data on import - see
ensure_dir() / ensure_free_space() for the explicit, opt-in operations.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def _default_data_root() -> Optional[Path]:
    """The E: convention only makes sense on this Windows machine - on any
    other platform (in particular the ephemeral Linux container a Kaggle
    kernel runs in, which has no persistent drives of its own and is
    unaffected by this machine's local disk pressure) there is no sensible
    default, so callers must fall back to their own prior behavior. Never
    guesses a path that would silently create a literal "E:" subdirectory
    on a non-Windows filesystem."""
    if os.environ.get("SOLAR_AI_DATA_ROOT"):
        return Path(os.environ["SOLAR_AI_DATA_ROOT"])
    if sys.platform == "win32":
        return Path("E:/Solar AI Training Images")
    return None


SOLAR_AI_DATA_ROOT = _default_data_root()

# Existing, already-in-use dataset locations under the root (do not rename -
# real audited data already lives here; see docs/DATASET_SOURCES.md and
# training/cloud's own experiment records for provenance tied to these
# exact paths):
#   <root>/_raw_downloads, source, prepared, yolo_source, yolo_prepared,
#   yolo_smoke_dataset
#
# New subdirectories for large data this project generates locally, added
# under the same root rather than scattered elsewhere. Both are Optional -
# None on a platform/environment with no SOLAR_AI_DATA_ROOT (see
# _default_data_root() above) - callers must handle that case explicitly
# (typically: fall back to whatever repo-relative default existed before
# this policy).
KAGGLE_RUNS_DIR: Optional[Path] = SOLAR_AI_DATA_ROOT / "kaggle_runs" if SOLAR_AI_DATA_ROOT else None
"""Kaggle kernel packages built locally and outputs retrieved from Kaggle
(checkpoints, result plots, logs) - was previously written under the
repository's own training/cloud/runs/, which lives on the low-space drive."""

LOCAL_TRAINING_RUNS_DIR: Optional[Path] = SOLAR_AI_DATA_ROOT / "local_training_runs" if SOLAR_AI_DATA_ROOT else None
"""Output directory for training run locally (e.g. train_yolo.py's
--project default, or a local model.val() results directory) - checkpoints,
result images/plots, logs."""


def default_kaggle_package_dir(experiment_id: str) -> Path:
    """Where a Kaggle kernel package (and any outputs retrieved from it)
    should live by default for a given experiment id - under
    KAGGLE_RUNS_DIR when available, otherwise the original repo-relative
    location (training/cloud/runs/<experiment_id>), unchanged from before
    this storage policy existed."""
    if KAGGLE_RUNS_DIR is not None:
        return KAGGLE_RUNS_DIR / experiment_id
    return Path("training/cloud/runs") / experiment_id


class InsufficientSpaceError(RuntimeError):
    """Raised instead of silently falling back to a different (low-space)
    drive - see storage policy rule: stop and report the space requirement."""


def ensure_dir(path: Path) -> Path:
    """mkdir -p and return path. Pure convenience - never silently
    redirects elsewhere on failure."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_free_space(path: Path, required_bytes: int, *, label: str = "") -> None:
    """Check that the drive containing `path` has at least `required_bytes`
    free. Raises InsufficientSpaceError (never silently proceeds, never
    silently redirects to another drive) if not - the caller decides what
    to do next (e.g. ask the user, or point at a different explicit path).

    `path` does not need to exist yet - only its drive/mount point does
    (shutil.disk_usage resolves through non-existent subdirectories fine on
    Windows and POSIX as long as an ancestor exists).
    """
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise InsufficientSpaceError(f"cannot determine free space: no existing ancestor of {path}")
        probe = parent
    usage = shutil.disk_usage(str(probe))
    if usage.free < required_bytes:
        required_gb = required_bytes / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        detail = f" ({label})" if label else ""
        raise InsufficientSpaceError(
            f"insufficient free space at {path}{detail}: need {required_gb:.2f} GB, "
            f"only {free_gb:.2f} GB free. Not falling back to another drive - "
            f"free up space or point this operation at an explicit alternate path."
        )
