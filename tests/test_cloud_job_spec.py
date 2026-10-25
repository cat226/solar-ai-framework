"""Tests for training/cloud/base/job_spec.py."""
from __future__ import annotations

import pytest

from training.cloud.base.job_spec import TrainingJobSpec, capture_environment, _git_sha, _package_versions


def _make_spec(**overrides) -> TrainingJobSpec:
    defaults = dict(
        experiment_id="exp-0001",
        model="yolo_detection",
        git_sha="a" * 40,
        dataset_id="gabrielkasmi/bdappv",
        dataset_revision="main",
        dataset_manifest_hash="b" * 64,
        class_order=("solar panel",),
        image_size=640,
        batch_size=16,
        epochs=5,
        optimizer="auto",
        learning_rate=0.01,
        random_seed=42,
        requested_gpu="P100",
        python_version="3.12.10",
        package_versions={"ultralytics": "8.4.118"},
    )
    defaults.update(overrides)
    return TrainingJobSpec(**defaults)


class TestSerialization:
    def test_to_dict_has_list_not_tuple_for_class_order(self):
        spec = _make_spec()
        d = spec.to_dict()
        assert isinstance(d["class_order"], list)
        assert d["class_order"] == ["solar panel"]

    def test_to_json_is_deterministic_across_calls(self):
        spec = _make_spec()
        assert spec.to_json() == spec.to_json()

    def test_to_json_is_deterministic_across_equal_specs(self):
        spec1 = _make_spec()
        spec2 = _make_spec()
        assert spec1.to_json() == spec2.to_json()

    def test_field_order_does_not_affect_json(self):
        # Constructing with kwargs in a different order must not change output -
        # sort_keys=True should make this true regardless of dataclass field order.
        spec1 = _make_spec(epochs=5, batch_size=16)
        spec2 = _make_spec(batch_size=16, epochs=5)
        assert spec1.to_json() == spec2.to_json()

    def test_round_trip_via_json(self):
        spec = _make_spec()
        restored = TrainingJobSpec.from_json(spec.to_json())
        assert restored == spec

    def test_round_trip_preserves_class_order_as_tuple(self):
        spec = _make_spec(class_order=("Clean", "Dusty", "Bird-Drop"))
        restored = TrainingJobSpec.from_json(spec.to_json())
        assert restored.class_order == ("Clean", "Dusty", "Bird-Drop")

    def test_from_dict_ignores_unknown_fields(self):
        d = _make_spec().to_dict()
        d["some_future_field"] = "ignored"
        restored = TrainingJobSpec.from_dict(d)
        assert not hasattr(restored, "some_future_field")


class TestSpecHash:
    def test_hash_is_sha256_hex(self):
        h = _make_spec().spec_hash()
        assert len(h) == 64
        int(h, 16)  # raises if not valid hex

    def test_identical_specs_hash_identically(self):
        assert _make_spec().spec_hash() == _make_spec().spec_hash()

    def test_different_seed_changes_hash(self):
        h1 = _make_spec(random_seed=42).spec_hash()
        h2 = _make_spec(random_seed=43).spec_hash()
        assert h1 != h2

    def test_different_dataset_hash_changes_hash(self):
        h1 = _make_spec(dataset_manifest_hash="b" * 64).spec_hash()
        h2 = _make_spec(dataset_manifest_hash="c" * 64).spec_hash()
        assert h1 != h2


class TestImmutability:
    def test_spec_is_frozen(self):
        spec = _make_spec()
        with pytest.raises(Exception):
            spec.epochs = 10  # type: ignore[misc]


class TestCaptureEnvironment:
    def test_git_sha_returns_plausible_sha_in_this_repo(self):
        # This test runs inside the actual solar-ai-framework git repo.
        sha = _git_sha()
        assert len(sha) == 40
        int(sha, 16)

    def test_git_sha_raises_outside_a_repo(self, tmp_path):
        with pytest.raises(RuntimeError):
            _git_sha(cwd=str(tmp_path))

    def test_package_versions_reports_installed_package(self):
        versions = _package_versions(["pytest"])
        assert versions["pytest"] != "not-installed"

    def test_package_versions_reports_missing_package_explicitly(self):
        versions = _package_versions(["definitely-not-a-real-package-xyz"])
        assert versions["definitely-not-a-real-package-xyz"] == "not-installed"

    def test_capture_environment_shape(self):
        env = capture_environment(["pytest"])
        assert set(env.keys()) == {"git_sha", "python_version", "package_versions"}
        assert env["python_version"].count(".") == 2
