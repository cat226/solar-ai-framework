"""Regression tests for model artifact supply-chain verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_model_artifacts import verify_manifest


def _write_manifest(tmp_path: Path, artifact: Path, digest: str) -> Path:
    manifest = tmp_path / "manifest.json"
    artifact_path = artifact.relative_to(tmp_path)
    manifest.write_text(
        json.dumps({"artifacts": [{"path": str(artifact_path), "sha256": digest}]}),
        encoding="utf-8",
    )
    return manifest


def test_verify_manifest_accepts_matching_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"trusted model bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    ok, errors = verify_manifest(_write_manifest(tmp_path, artifact, digest))

    assert ok is True
    assert errors == []


def test_verify_manifest_resolves_relative_paths_from_manifest_directory(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    artifact = weights / "model.bin"
    artifact.write_bytes(b"trusted model bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = weights / "manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": [{"path": artifact.name, "sha256": digest}]}),
        encoding="utf-8",
    )

    ok, errors = verify_manifest(manifest)

    assert ok is True
    assert errors == []


def test_verify_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"tampered model bytes")

    ok, errors = verify_manifest(_write_manifest(tmp_path, artifact, "0" * 64))

    assert ok is False
    assert "SHA-256 mismatch" in errors[0]


def test_verify_manifest_rejects_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "missing.bin"

    ok, errors = verify_manifest(_write_manifest(tmp_path, artifact, "0" * 64))

    assert ok is False
    assert "artifact missing" in errors[0]


def test_verify_manifest_rejects_missing_or_non_hex_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": [{"path": artifact.name, "sha256": "g" * 64}]}),
        encoding="utf-8",
    )

    ok, errors = verify_manifest(manifest)

    assert ok is False
    assert "missing valid SHA-256 digest" in errors[0]


def test_verify_manifest_rejects_absolute_artifact_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": [{"path": str(artifact), "sha256": digest}]}),
        encoding="utf-8",
    )

    ok, errors = verify_manifest(manifest)

    assert ok is False
    assert "artifact path must be relative" in errors[0]


def test_verify_manifest_rejects_paths_that_escape_manifest_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside artifact")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    weights = tmp_path / "weights"
    weights.mkdir()
    manifest = weights / "manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": [{"path": "../outside.bin", "sha256": digest}]}),
        encoding="utf-8",
    )

    ok, errors = verify_manifest(manifest)

    assert ok is False
    assert "escapes the manifest directory" in errors[0]


def test_verify_manifest_does_not_create_missing_files(tmp_path: Path) -> None:
    artifact = tmp_path / "never-created.bin"
    manifest = _write_manifest(tmp_path, artifact, "0" * 64)

    ok, _ = verify_manifest(manifest)

    assert ok is False
    assert not artifact.exists()
