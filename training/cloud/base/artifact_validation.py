"""training/cloud/base/artifact_validation.py — Reusable checkpoint/artifact validation.

Used both after a local training run and after retrieving a checkpoint from
a remote worker, before it is ever considered for production promotion.
Never fabricates a passing result - every check either genuinely passes,
genuinely fails, or raises because it cannot be evaluated (e.g. missing
file), and callers must treat "could not evaluate" as a failure, not a skip.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: Any = None) -> None:
        self.checks[name] = ok
        if detail is not None:
            self.details[name] = detail
        if not ok:
            self.passed = False

    @classmethod
    def start(cls) -> "ValidationResult":
        return cls(passed=True)


def validate_file_exists(path: Path) -> ValidationResult:
    result = ValidationResult.start()
    exists = path.is_file()
    result.add("file_exists", exists, str(path))
    if not exists:
        result.errors.append(f"file does not exist: {path}")
    return result


def compute_and_check_sha256(path: Path, expected_sha256: Optional[str] = None) -> ValidationResult:
    result = validate_file_exists(path)
    if not result.passed:
        return result
    actual = sha256_file(path)
    result.details["sha256"] = actual
    if expected_sha256 is not None:
        matches = actual.lower() == expected_sha256.lower()
        result.add("sha256_matches", matches, {"expected": expected_sha256, "actual": actual})
        if not matches:
            result.errors.append(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return result


def validate_torch_checkpoint_integrity(path: Path) -> ValidationResult:
    """Confirm a PyTorch checkpoint (e.g. MobileNet state dict) loads cleanly
    with weights_only=True - the same safe-loading contract the production
    ModelManager uses. Does not inspect architecture; see
    validate_mobilenet_class_head for that."""
    result = validate_file_exists(path)
    if not result.passed:
        return result
    try:
        import torch
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any load failure is a real validation failure
        result.add("torch_load_succeeds", False, str(exc))
        result.errors.append(f"torch.load(weights_only=True) failed: {exc}")
        return result
    is_state_dict = isinstance(state_dict, dict) and len(state_dict) > 0
    result.add("torch_load_succeeds", True)
    result.add("is_nonempty_state_dict", is_state_dict, {"num_keys": len(state_dict) if isinstance(state_dict, dict) else None})
    if not is_state_dict:
        result.errors.append("loaded object is not a non-empty state dict")
    return result


def validate_mobilenet_class_head(path: Path, expected_num_classes: int) -> ValidationResult:
    """Confirm the checkpoint's final classifier layer has exactly
    expected_num_classes output units - catches an interim/subset checkpoint
    (e.g. the 3-class experiment) being mistaken for the production 6-class
    artifact before it ever reaches configs/settings.yaml's num_classes."""
    result = validate_torch_checkpoint_integrity(path)
    if not result.passed:
        return result
    import torch
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    # MobileNetV2's replaced classifier head, per models/model_manager.py's
    # construction pattern: model.classifier[1] = nn.Linear(...)
    key = "classifier.1.weight"
    if key not in state_dict:
        result.add("classifier_head_present", False, list(state_dict.keys())[:10])
        result.errors.append(f"expected key '{key}' not found in state dict")
        return result
    actual_classes = int(state_dict[key].shape[0])
    matches = actual_classes == expected_num_classes
    result.add("class_count_matches", matches, {"expected": expected_num_classes, "actual": actual_classes})
    if not matches:
        result.errors.append(f"class count mismatch: expected {expected_num_classes}, got {actual_classes}")
    return result


def validate_ultralytics_checkpoint_integrity(path: Path) -> ValidationResult:
    """Confirm a YOLO (Ultralytics) checkpoint loads and reports the expected
    single-class ("solar panel") detection head - this project's YOLO stage
    is single-class panel localization, not multi-class."""
    result = validate_file_exists(path)
    if not result.passed:
        return result
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
    except Exception as exc:  # noqa: BLE001
        result.add("ultralytics_load_succeeds", False, str(exc))
        result.errors.append(f"YOLO(path) failed to load: {exc}")
        return result
    result.add("ultralytics_load_succeeds", True)
    names = getattr(model, "names", None) or {}
    result.details["names"] = names
    single_class = len(names) == 1
    result.add("single_class_head", single_class, names)
    if not single_class:
        result.errors.append(f"expected exactly 1 class, found {len(names)}: {names}")
    return result
