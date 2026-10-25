"""training/detection/ground_pv/verify_condition_checkpoint.py — Phase 15B
post-retrieval verification for a trained 4-class condition checkpoint +
its Kaggle training manifest.

Checks the reproducibility fields Phase 15B requires are actually present
and internally consistent (`checkpoint_sha256` in the manifest matches the
ACTUAL downloaded file's hash — catching a corrupted/partial/wrong-file
download — and the recorded taxonomy is exactly the active 4-class set),
without needing a GPU or Ultralytics installed (pure JSON + hashlib).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_NAMES

# Every field Phase 15B's reproducibility requirement (req 5) asks to be
# recorded. Nested paths use "." (e.g. "env.python").
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "base_model", "dataset_kaggle", "epochs", "imgsz", "patience", "seed",
    "optimizer", "augmentation", "env.python", "env.torch", "env.torchvision",
    "env.ultralytics", "env.cuda", "env.gpu", "dataset_hashes",
    "train_seconds", "best_epoch", "checkpoint_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def verify_training_manifest(
    manifest: dict[str, Any],
    checkpoint_path: Path,
    *,
    required_fields: tuple[str, ...] = REQUIRED_MANIFEST_FIELDS,
) -> list[str]:
    """Returns a list of problems found (empty list == everything checks
    out). Never raises — the caller decides what a non-empty list means."""
    problems: list[str] = []

    for field in required_fields:
        if _get_nested(manifest, field) in (None, ""):
            problems.append(f"missing or empty required field: {field}")

    taxonomy = manifest.get("active_taxonomy")
    if taxonomy is not None:
        normalized = {int(k): v for k, v in taxonomy.items()} if isinstance(taxonomy, dict) else None
        if normalized != CONDITION_4CLASS_NAMES:
            problems.append(f"active_taxonomy mismatch: {taxonomy} != {CONDITION_4CLASS_NAMES}")
    else:
        problems.append("missing active_taxonomy field")

    recorded_sha = manifest.get("checkpoint_sha256")
    if recorded_sha:
        if not checkpoint_path.is_file():
            problems.append(f"checkpoint file not found: {checkpoint_path}")
        else:
            actual_sha = _sha256_file(checkpoint_path)
            if actual_sha != recorded_sha:
                problems.append(
                    f"checkpoint_sha256 mismatch: manifest says {recorded_sha}, "
                    f"actual downloaded file is {actual_sha} — possible corrupted/wrong download"
                )

    if manifest.get("not_production_validated") is not True:
        problems.append("manifest must explicitly record not_production_validated=True")

    return problems


def verify_from_files(manifest_path: Path, checkpoint_path: Path) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return verify_training_manifest(manifest, Path(checkpoint_path))
