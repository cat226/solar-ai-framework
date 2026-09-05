"""Tests for training/cloud/kaggle/dataset_adapter.py.

No test here invokes a real Kaggle upload or calls the real `kaggle` CLI -
every subprocess boundary is mocked. prepare()/dry_run()/stage_via_links()
are pure local file operations and are tested for real (no mocking needed,
since they never shell out).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.cloud.kaggle.dataset_adapter import (
    VALID_LICENSES,
    CreateNotConfirmedError,
    KaggleCLIError,
    KaggleDatasetConfig,
    create,
    dry_run,
    list_files,
    prepare,
    stage_via_links,
    status,
)


def _make_config(**overrides) -> KaggleDatasetConfig:
    defaults = dict(
        owner="edithstark",
        slug="solar-ai-yolo-smoke-001",
        title="Solar AI YOLO smoke dataset",
        license_name="CC-BY-4.0",
    )
    defaults.update(overrides)
    return KaggleDatasetConfig(**defaults)


class TestKaggleDatasetConfig:
    def test_rejects_unrecognized_license(self):
        with pytest.raises(ValueError, match="not a Kaggle-recognized license"):
            _make_config(license_name="Fully-Made-Up-License")

    def test_accepts_every_documented_license(self):
        for lic in VALID_LICENSES:
            _make_config(license_name=lic)  # must not raise

    def test_dataset_id_property(self):
        config = _make_config()
        assert config.dataset_id == "edithstark/solar-ai-yolo-smoke-001"

    def test_to_metadata_dict_shape(self):
        config = _make_config()
        metadata = config.to_metadata_dict()
        assert metadata["id"] == "edithstark/solar-ai-yolo-smoke-001"
        assert metadata["title"] == "Solar AI YOLO smoke dataset"
        assert metadata["licenses"] == [{"name": "CC-BY-4.0"}]


class TestPrepare:
    def test_writes_metadata_into_existing_package_dir(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        (package_dir / "train").mkdir()
        config = _make_config()
        result = prepare(config, package_dir)
        assert result == package_dir
        metadata = json.loads((package_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == config.dataset_id

    def test_creates_package_dir_if_missing(self, tmp_path):
        package_dir = tmp_path / "does_not_exist_yet"
        prepare(_make_config(), package_dir)
        assert package_dir.is_dir()
        assert (package_dir / "dataset-metadata.json").is_file()


class TestStageViaLinks:
    def test_links_children_without_copying(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "manifest.json").write_text('{"a": 1}', encoding="utf-8")
        (source_dir / "train").mkdir()
        (source_dir / "train" / "images").mkdir()
        (source_dir / "train" / "images" / "x.png").write_bytes(b"fake-bytes")

        package_dir = tmp_path / "staged"
        stage_via_links(source_dir, package_dir)

        assert (package_dir / "manifest.json").is_file()
        assert (package_dir / "manifest.json").read_text(encoding="utf-8") == '{"a": 1}'
        assert (package_dir / "train" / "images" / "x.png").is_file()

    def test_never_writes_into_source_dir(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
        before = sorted(p.name for p in source_dir.iterdir())

        stage_via_links(source_dir, tmp_path / "staged")

        after = sorted(p.name for p in source_dir.iterdir())
        assert before == after  # source directory contents unchanged

    def test_respects_explicit_entries_subset(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "keep_me.json").write_text("{}", encoding="utf-8")
        (source_dir / "skip_me.json").write_text("{}", encoding="utf-8")

        package_dir = tmp_path / "staged"
        stage_via_links(source_dir, package_dir, entries=["keep_me.json"])

        assert (package_dir / "keep_me.json").exists()
        assert not (package_dir / "skip_me.json").exists()

    def test_falls_back_to_copy_when_file_cannot_be_linked_or_hardlinked(self, tmp_path):
        """Cross-drive staging (the real full-dataset case: E:\\... source,
        D:\\...\\runs\\... package_dir) can't symlink (no privilege on this
        box) or hardlink (different volume) a *file* entry like
        manifest.json. Must fall back to a plain copy rather than crash -
        only ever for small top-level files, never the bulk image dirs."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "manifest.json").write_text('{"real": "content"}', encoding="utf-8")

        package_dir = tmp_path / "staged"
        with patch("training.cloud.kaggle.dataset_adapter.os.symlink", side_effect=OSError("no privilege")), \
             patch("training.cloud.kaggle.dataset_adapter.os.link", side_effect=OSError("different disk drive")):
            stage_via_links(source_dir, package_dir, entries=["manifest.json"])

        assert (package_dir / "manifest.json").read_text(encoding="utf-8") == '{"real": "content"}'

    def test_directory_still_junctions_when_file_fallback_would_be_copy(self, tmp_path):
        """The copy fallback must never silently apply to directories - a
        directory always uses symlink or mklink /J, so bulk data is never
        duplicated on disk even when a sibling file entry has to be copied."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (source_dir / "train").mkdir()
        (source_dir / "train" / "images.txt").write_text("data", encoding="utf-8")

        package_dir = tmp_path / "staged"
        with patch("training.cloud.kaggle.dataset_adapter.os.symlink", side_effect=OSError("no privilege")), \
             patch("training.cloud.kaggle.dataset_adapter.os.link", side_effect=OSError("different disk drive")):
            stage_via_links(source_dir, package_dir)

        # The directory must exist and be readable (via junction), not a copy -
        # verified indirectly: its child file is reachable through the link.
        assert (package_dir / "train" / "images.txt").read_text(encoding="utf-8") == "data"


class TestDryRun:
    def test_valid_prepared_package_passes(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        (package_dir / "train").mkdir()
        config = _make_config()
        prepare(config, package_dir)
        result = dry_run(package_dir, config, required_entries=["train"])
        assert result.passed, result.errors

    def test_missing_metadata_file_fails(self, tmp_path):
        package_dir = tmp_path / "empty_package"
        package_dir.mkdir()
        result = dry_run(package_dir, _make_config())
        assert not result.passed

    def test_missing_required_entry_fails(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        result = dry_run(package_dir, config, required_entries=["train", "val", "test"])
        assert not result.passed
        assert any("train" in e for e in result.errors)

    def test_tampered_metadata_fails(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        metadata_path = package_dir / "dataset-metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["title"] = "Something else entirely"
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed

    def test_placeholder_dataset_id_fails(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        metadata_path = package_dir / "dataset-metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["id"] = "edithstark/INSERT_SLUG_HERE"
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed
        assert any("placeholder" in e for e in result.errors)

    def test_invalid_json_fails_cleanly(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        (package_dir / "dataset-metadata.json").write_text("{not valid json", encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed

    def test_unrecognized_license_on_disk_fails(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        metadata_path = package_dir / "dataset-metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["licenses"] = [{"name": "Not-A-Real-License"}]
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed
        assert any("license" in e for e in result.errors)


class TestCreateRefusesWithoutConfirmation:
    def test_raises_without_confirm(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run") as mock_run:
            with pytest.raises(CreateNotConfirmedError):
                create(package_dir, config, confirm=False)
            mock_run.assert_not_called()

    def test_default_confirm_is_required_explicitly(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        with pytest.raises(TypeError):
            create(package_dir, config)  # type: ignore[call-arg]


class TestCreateWithConfirmation:
    def test_confirmed_create_invokes_kaggle_with_correct_args_private_by_default(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)

        mock_result = MagicMock(returncode=0, stdout="Your private Dataset is being created", stderr="")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result) as mock_run:
            proc = create(package_dir, config, confirm=True)

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args == ["kaggle", "datasets", "create", "-p", str(package_dir), "-r", "zip"]
        assert "-u" not in called_args
        assert proc.returncode == 0

    def test_public_flag_only_when_explicitly_requested(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result) as mock_run:
            create(package_dir, config, confirm=True, public=True)
        assert "-u" in mock_run.call_args[0][0]

    def test_refuses_to_create_when_dry_run_would_fail(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        (package_dir / "dataset-metadata.json").unlink()  # corrupt after prepare()

        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="dry_run validation failed"):
                create(package_dir, config, confirm=True)
            mock_run.assert_not_called()

    def test_nonzero_exit_raises_kaggle_cli_error(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)

        mock_result = MagicMock(returncode=1, stdout="", stderr="The dataset title must be between 6 and 50 characters")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result):
            with pytest.raises(KaggleCLIError, match="50 characters"):
                create(package_dir, config, confirm=True)


class TestReadOnlyOperations:
    def test_status_calls_correct_command(self):
        mock_result = MagicMock(returncode=0, stdout="ready", stderr="")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result) as mock_run:
            result = status("edithstark/solar-ai-yolo-smoke-001")
        assert mock_run.call_args[0][0] == ["kaggle", "datasets", "status", "edithstark/solar-ai-yolo-smoke-001"]
        assert result["raw_output"] == "ready"

    def test_list_files_calls_correct_command(self):
        mock_result = MagicMock(returncode=0, stdout="manifest.json\n", stderr="")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result) as mock_run:
            result = list_files("edithstark/solar-ai-yolo-smoke-001")
        assert mock_run.call_args[0][0] == ["kaggle", "datasets", "files", "edithstark/solar-ai-yolo-smoke-001"]
        assert "manifest.json" in result

    def test_status_raises_on_cli_failure(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="dataset not found")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result):
            with pytest.raises(KaggleCLIError):
                status("edithstark/does-not-exist")


class TestNeverInvokesRealSubprocessByAccident:
    def test_module_never_calls_subprocess_run_without_kaggle_prefix(self, tmp_path):
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        config = _make_config()
        prepare(config, package_dir)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("training.cloud.kaggle.dataset_adapter.subprocess.run", return_value=mock_result) as mock_run:
            create(package_dir, config, confirm=True)
            status("a/b")
            list_files("a/b")
        for call in mock_run.call_args_list:
            args = call[0][0]
            assert isinstance(args, list)
            assert args[0] == "kaggle"
