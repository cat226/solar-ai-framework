"""Tests for training/cloud/base/registry.py."""
from __future__ import annotations

import json

import pytest

from training.cloud.base.registry import (
    SecretFieldError,
    get_experiment,
    load_experiments,
    record_experiment,
    update_experiment_status,
)


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "registry.jsonl"


def _basic_record(experiment_id="exp-1", **overrides):
    record = {
        "experiment_id": experiment_id,
        "model": "yolo_detection",
        "git_sha": "a" * 40,
        "dataset": "gabrielkasmi/bdappv",
        "dataset_hash": "b" * 64,
        "configuration": {"epochs": 5},
        "hardware": {"gpu": "P100"},
        "status": "queued",
        "metrics": {},
        "checkpoint": "",
        "artifact_hash": "",
    }
    record.update(overrides)
    return record


class TestRecordAndLoad:
    def test_record_then_load_round_trips(self, registry_path):
        record_experiment(_basic_record(), registry_path=registry_path)
        loaded = load_experiments(registry_path=registry_path)
        assert len(loaded) == 1
        assert loaded[0]["experiment_id"] == "exp-1"

    def test_load_empty_registry_returns_empty_list(self, registry_path):
        assert load_experiments(registry_path=registry_path) == []

    def test_multiple_records_preserve_order(self, registry_path):
        record_experiment(_basic_record("exp-1"), registry_path=registry_path)
        record_experiment(_basic_record("exp-2"), registry_path=registry_path)
        loaded = load_experiments(registry_path=registry_path)
        assert [r["experiment_id"] for r in loaded] == ["exp-1", "exp-2"]

    def test_file_is_valid_jsonl_one_record_per_line(self, registry_path):
        record_experiment(_basic_record(), registry_path=registry_path)
        lines = registry_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        json.loads(lines[0])  # must not raise

    def test_malformed_line_is_skipped_not_fatal(self, registry_path):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("not valid json\n", encoding="utf-8")
        record_experiment(_basic_record(), registry_path=registry_path)
        loaded = load_experiments(registry_path=registry_path)
        assert len(loaded) == 1


class TestGetExperiment:
    def test_get_missing_experiment_returns_none(self, registry_path):
        assert get_experiment("does-not-exist", registry_path=registry_path) is None

    def test_get_returns_most_recent_record_for_id(self, registry_path):
        record_experiment(_basic_record("exp-1", status="queued"), registry_path=registry_path)
        record_experiment(_basic_record("exp-1", status="running"), registry_path=registry_path)
        result = get_experiment("exp-1", registry_path=registry_path)
        assert result["status"] == "running"

    def test_get_does_not_confuse_different_ids(self, registry_path):
        record_experiment(_basic_record("exp-1", status="queued"), registry_path=registry_path)
        record_experiment(_basic_record("exp-2", status="running"), registry_path=registry_path)
        assert get_experiment("exp-1", registry_path=registry_path)["status"] == "queued"


class TestUpdateStatus:
    def test_update_appends_new_record_not_mutates(self, registry_path):
        record_experiment(_basic_record("exp-1", status="queued"), registry_path=registry_path)
        update_experiment_status("exp-1", "running", registry_path=registry_path)
        loaded = load_experiments(registry_path=registry_path)
        assert len(loaded) == 2
        assert loaded[0]["status"] == "queued"
        assert loaded[1]["status"] == "running"

    def test_update_raises_for_unknown_experiment(self, registry_path):
        with pytest.raises(KeyError):
            update_experiment_status("does-not-exist", "running", registry_path=registry_path)

    def test_update_can_set_additional_fields(self, registry_path):
        record_experiment(_basic_record("exp-1"), registry_path=registry_path)
        update_experiment_status("exp-1", "completed", registry_path=registry_path, checkpoint="weights/best.pt")
        result = get_experiment("exp-1", registry_path=registry_path)
        assert result["status"] == "completed"
        assert result["checkpoint"] == "weights/best.pt"


class TestSecretRejection:
    @pytest.mark.parametrize("bad_key", ["api_key", "kaggle_token", "password", "secret_value", "credential_blob"])
    def test_rejects_records_with_credential_like_keys(self, registry_path, bad_key):
        record = _basic_record()
        record[bad_key] = "should-never-be-here"
        with pytest.raises(SecretFieldError):
            record_experiment(record, registry_path=registry_path)

    def test_rejects_nested_credential_like_keys(self, registry_path):
        record = _basic_record()
        record["hardware"]["api_key"] = "nope"
        with pytest.raises(SecretFieldError):
            record_experiment(record, registry_path=registry_path)

    def test_clean_record_is_not_rejected(self, registry_path):
        record_experiment(_basic_record(), registry_path=registry_path)  # must not raise
