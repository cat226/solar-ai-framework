"""tests/test_ground_pv_verify_condition_checkpoint.py — Phase 15B
post-retrieval checkpoint/manifest verification."""
from __future__ import annotations

import hashlib
import json

import pytest

from training.detection.ground_pv.verify_condition_checkpoint import (
    verify_from_files,
    verify_training_manifest,
)


def _valid_manifest(checkpoint_sha256: str) -> dict:
    return {
        "base_model": "yolo11n.pt",
        "dataset_kaggle": "edithstark/solar-ai-ground-condition-4class-research-v1",
        "active_taxonomy": {"0": "CLEAN", "1": "DUSTY", "2": "BIRD_DROP", "3": "PHYSICAL_DAMAGE"},
        "epochs": 100, "imgsz": 640, "patience": 20, "seed": 42, "optimizer": "auto",
        "augmentation": {"hsv_h": 0.015},
        "env": {"python": "3.11.0", "torch": "2.7.1", "torchvision": "0.22.1",
                "ultralytics": "8.4.138", "cuda": "12.6", "gpu": "Tesla P100"},
        "dataset_hashes": {"split_manifest_sha256": "abc123"},
        "train_seconds": 5760.0, "best_epoch": 88,
        "checkpoint_sha256": checkpoint_sha256,
        "not_production_validated": True,
    }


class TestVerifyTrainingManifest:
    def test_valid_manifest_and_matching_checkpoint_no_problems(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake checkpoint bytes")
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        problems = verify_training_manifest(_valid_manifest(sha), ckpt)
        assert problems == []

    def test_missing_required_field_flagged(self):
        manifest = _valid_manifest("x" * 64)
        del manifest["epochs"]
        problems = verify_training_manifest(manifest, __import__("pathlib").Path("/nonexistent"))
        assert any("epochs" in p for p in problems)

    def test_missing_nested_env_field_flagged(self):
        manifest = _valid_manifest("x" * 64)
        del manifest["env"]["cuda"]
        problems = verify_training_manifest(manifest, __import__("pathlib").Path("/nonexistent"))
        assert any("env.cuda" in p for p in problems)

    def test_wrong_taxonomy_flagged(self):
        manifest = _valid_manifest("x" * 64)
        manifest["active_taxonomy"] = {"0": "CLEAN", "1": "DUSTY", "2": "BIRD_DROP",
                                       "3": "PHYSICAL_DAMAGE", "4": "ELECTRICAL_DAMAGE"}
        problems = verify_training_manifest(manifest, __import__("pathlib").Path("/nonexistent"))
        assert any("active_taxonomy mismatch" in p for p in problems)

    def test_missing_taxonomy_flagged(self):
        manifest = _valid_manifest("x" * 64)
        del manifest["active_taxonomy"]
        problems = verify_training_manifest(manifest, __import__("pathlib").Path("/nonexistent"))
        assert any("active_taxonomy" in p for p in problems)

    def test_checkpoint_sha_mismatch_flagged(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"actual bytes")
        wrong_sha = "0" * 64
        problems = verify_training_manifest(_valid_manifest(wrong_sha), ckpt)
        assert any("checkpoint_sha256 mismatch" in p for p in problems)

    def test_missing_checkpoint_file_flagged(self, tmp_path):
        problems = verify_training_manifest(_valid_manifest("a" * 64), tmp_path / "does_not_exist.pt")
        assert any("not found" in p for p in problems)

    def test_not_production_validated_must_be_true(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        manifest = _valid_manifest(sha)
        manifest["not_production_validated"] = False
        problems = verify_training_manifest(manifest, ckpt)
        assert any("not_production_validated" in p for p in problems)

    def test_verify_from_files(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"content")
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        manifest_path = tmp_path / "training_manifest.json"
        manifest_path.write_text(json.dumps(_valid_manifest(sha)), encoding="utf-8")
        problems = verify_from_files(manifest_path, ckpt)
        assert problems == []
