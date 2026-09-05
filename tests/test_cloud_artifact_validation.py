"""Tests for training/cloud/base/artifact_validation.py.

Core coverage uses synthetic fixtures so it runs reliably in CI (which
never has real training artifacts - weights/ is gitignored). A few tests
additionally exercise real local checkpoints from this machine's actual
training runs when present, skipping cleanly when they are not (e.g. in
CI or a fresh checkout) - real-artifact coverage is a bonus, not a
requirement.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from training.cloud.base.artifact_validation import (
    compute_and_check_sha256,
    sha256_file,
    validate_file_exists,
    validate_mobilenet_class_head,
    validate_torch_checkpoint_integrity,
    validate_ultralytics_checkpoint_integrity,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_MOBILENET_3CLASS = _REPO_ROOT / "weights" / "mobilenet_interim_3class.pth"
_REAL_YOLO_CHECKPOINT = _REPO_ROOT / "weights" / "yolo_solar_candidate_epoch2.pt"


class TestFileExists:
    def test_existing_file_passes(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        result = validate_file_exists(f)
        assert result.passed

    def test_missing_file_fails_with_error_message(self, tmp_path):
        result = validate_file_exists(tmp_path / "missing.bin")
        assert not result.passed
        assert result.errors


class TestSha256:
    def test_sha256_matches_hashlib(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world" * 1000)
        expected = hashlib.sha256(b"hello world" * 1000).hexdigest()
        assert sha256_file(f) == expected

    def test_check_with_correct_expected_passes(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"content")
        expected = hashlib.sha256(b"content").hexdigest()
        result = compute_and_check_sha256(f, expected_sha256=expected)
        assert result.passed

    def test_check_with_wrong_expected_fails(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"content")
        result = compute_and_check_sha256(f, expected_sha256="0" * 64)
        assert not result.passed
        assert "mismatch" in result.errors[0].lower()

    def test_check_without_expected_just_reports_hash(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"content")
        result = compute_and_check_sha256(f)
        assert result.passed
        assert "sha256" in result.details

    def test_check_on_missing_file_fails(self, tmp_path):
        result = compute_and_check_sha256(tmp_path / "missing.bin", expected_sha256="0" * 64)
        assert not result.passed


class TestTorchCheckpointIntegrity:
    def test_valid_state_dict_passes(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        torch.save({"layer.weight": torch.randn(3, 4)}, f)
        result = validate_torch_checkpoint_integrity(f)
        assert result.passed

    def test_empty_state_dict_fails(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        torch.save({}, f)
        result = validate_torch_checkpoint_integrity(f)
        assert not result.passed

    def test_corrupt_file_fails_cleanly_not_a_crash(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        f.write_bytes(b"this is not a torch checkpoint at all")
        result = validate_torch_checkpoint_integrity(f)
        assert not result.passed
        assert result.errors

    def test_missing_file_fails(self, tmp_path):
        result = validate_torch_checkpoint_integrity(tmp_path / "missing.pth")
        assert not result.passed

    @pytest.mark.skipif(not _REAL_MOBILENET_3CLASS.is_file(), reason="real local checkpoint not present (expected in CI)")
    def test_real_interim_mobilenet_checkpoint_loads(self):
        result = validate_torch_checkpoint_integrity(_REAL_MOBILENET_3CLASS)
        assert result.passed


class TestMobilenetClassHead:
    def test_matching_class_count_passes(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        torch.save({"classifier.1.weight": torch.randn(6, 1280), "classifier.1.bias": torch.randn(6)}, f)
        result = validate_mobilenet_class_head(f, expected_num_classes=6)
        assert result.passed

    def test_mismatched_class_count_fails(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        torch.save({"classifier.1.weight": torch.randn(3, 1280), "classifier.1.bias": torch.randn(3)}, f)
        result = validate_mobilenet_class_head(f, expected_num_classes=6)
        assert not result.passed
        assert "mismatch" in result.errors[0].lower()

    def test_missing_classifier_key_fails(self, tmp_path):
        f = tmp_path / "ckpt.pth"
        torch.save({"some.other.key": torch.randn(3, 3)}, f)
        result = validate_mobilenet_class_head(f, expected_num_classes=6)
        assert not result.passed

    @pytest.mark.skipif(not _REAL_MOBILENET_3CLASS.is_file(), reason="real local checkpoint not present (expected in CI)")
    def test_real_interim_checkpoint_is_3_class_not_6(self):
        """This is the exact scenario the project's rules exist to catch: an
        interim/subset checkpoint must never pass as the production 6-class
        artifact."""
        result_3 = validate_mobilenet_class_head(_REAL_MOBILENET_3CLASS, expected_num_classes=3)
        assert result_3.passed
        result_6 = validate_mobilenet_class_head(_REAL_MOBILENET_3CLASS, expected_num_classes=6)
        assert not result_6.passed


class TestUltralyticsCheckpointIntegrity:
    def test_missing_file_fails(self, tmp_path):
        result = validate_ultralytics_checkpoint_integrity(tmp_path / "missing.pt")
        assert not result.passed

    def test_corrupt_file_fails_cleanly(self, tmp_path):
        f = tmp_path / "ckpt.pt"
        f.write_bytes(b"not a real ultralytics checkpoint")
        result = validate_ultralytics_checkpoint_integrity(f)
        assert not result.passed
        assert result.errors

    @pytest.mark.skipif(not _REAL_YOLO_CHECKPOINT.is_file(), reason="real local checkpoint not present (expected in CI)")
    def test_real_yolo_checkpoint_is_single_class(self):
        result = validate_ultralytics_checkpoint_integrity(_REAL_YOLO_CHECKPOINT)
        assert result.passed
        assert result.details["names"] == {0: "solar panel"}
