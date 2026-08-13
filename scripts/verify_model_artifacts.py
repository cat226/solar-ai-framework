#!/usr/bin/env python3
"""Verify supplied model artifacts against a reviewed SHA-256 manifest.

The repository intentionally does not contain trained model weights. This tool
is for deployment environments where trusted artifacts are supplied separately.
It never downloads or creates model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_MANIFEST = Path("weights/manifest.json")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(manifest_path: Path, relative_path: str) -> Path:
    """Resolve an artifact path while preventing manifest-directory escapes."""
    root = manifest_path.parent.resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("artifact path must be relative to the manifest directory")

    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes the manifest directory") from exc
    return resolved


def verify_manifest(manifest_path: Path) -> tuple[bool, list[str]]:
    if not manifest_path.is_file():
        return False, [f"manifest not found: {manifest_path}"]

    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"invalid manifest: {type(exc).__name__}"]

    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        return False, ["manifest must contain a non-empty 'artifacts' list"]

    errors: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, dict):
            errors.append("artifact entry must be an object")
            continue

        relative_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append("artifact entry is missing a valid 'path'")
            continue
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            errors.append(f"{relative_path}: missing valid SHA-256 digest")
            continue

        try:
            path = _resolve_artifact_path(manifest_path, relative_path)
        except ValueError as exc:
            errors.append(f"{relative_path}: {exc}")
            continue

        if not path.is_file():
            errors.append(f"{relative_path}: artifact missing")
            continue

        actual = _sha256(path)
        if actual.lower() != expected.lower():
            errors.append(f"{relative_path}: SHA-256 mismatch")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    ok, errors = verify_manifest(args.manifest)
    if ok:
        print("Model artifact integrity verification passed.")
        return 0

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
