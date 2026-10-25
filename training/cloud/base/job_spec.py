"""training/cloud/base/job_spec.py — Serializable, reproducible training-job specification.

A TrainingJobSpec fully describes one training run: what code, what data,
what hyperparameters, on what hardware. It is provider-independent - the
same spec could in principle be handed to a Kaggle adapter, a Colab
adapter, or run locally, though only Kaggle is implemented today.

Serialization is deterministic (sorted keys, stable formatting) so two
specs with identical content always hash identically - this is what makes
an experiment's job_spec_hash a meaningful audit key, not just a
convenience field.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class TrainingJobSpec:
    """Full reproducibility record for one training run.

    All fields are required except where a default is given - a spec
    missing dataset/config identity is not reproducible and should not be
    constructed with silent defaults for those fields.
    """

    experiment_id: str
    model: str  # e.g. "yolo_detection", "mobilenet_classification"
    git_sha: str
    dataset_id: str
    dataset_revision: str
    dataset_manifest_hash: str
    class_order: tuple[str, ...]
    image_size: int
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    random_seed: int
    requested_gpu: str  # e.g. "P100", "T4x2", "none" (CPU)
    python_version: str
    package_versions: dict[str, str] = field(default_factory=dict)
    scheduler: str = "none"
    augmentation: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dict form - tuples become sorted-stable lists, ready for JSON."""
        d = asdict(self)
        d["class_order"] = list(self.class_order)
        return d

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, fixed separators, no NaN/Infinity."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def spec_hash(self) -> str:
        """SHA-256 of the deterministic JSON form - the audit key for this exact configuration."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJobSpec":
        data = dict(data)
        if "class_order" in data:
            data["class_order"] = tuple(data["class_order"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, text: str) -> "TrainingJobSpec":
        return cls.from_dict(json.loads(text))


def _git_sha(cwd: Optional[str] = None) -> str:
    """Current git commit SHA. Raises if not in a git repo or git is unavailable -
    a job spec must never silently record a fabricated/placeholder SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not determine git SHA: {result.stderr.strip()}")
    return result.stdout.strip()


def _package_versions(package_names: list[str]) -> dict[str, str]:
    """Best-effort installed-version lookup for the given packages. A package that
    isn't installed is recorded as 'not-installed' rather than omitted, so the
    absence itself is part of the reproducibility record, not silently lost."""
    from importlib import metadata as importlib_metadata

    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def capture_environment(package_names: list[str], *, cwd: Optional[str] = None) -> dict[str, Any]:
    """Capture the auto-detectable parts of a job spec (git SHA, Python version,
    package versions) from the current environment. Callers still supply the
    experiment-specific fields (dataset, hyperparameters, etc.) themselves -
    this never guesses at those.
    """
    return {
        "git_sha": _git_sha(cwd=cwd),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "package_versions": _package_versions(package_names),
    }
