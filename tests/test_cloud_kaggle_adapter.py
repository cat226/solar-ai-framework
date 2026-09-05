"""Tests for training/cloud/kaggle/adapter.py.

No test in this file invokes a real Kaggle kernel or calls the real
`kaggle` CLI - every subprocess boundary is mocked. prepare()/dry_run() are
pure local file operations and are tested for real (no mocking needed,
since they never shell out).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.cloud.kaggle.adapter import (
    KaggleCLIError,
    KaggleKernelConfig,
    LaunchNotConfirmedError,
    dry_run,
    launch,
    logs,
    outputs,
    prepare,
    status,
)


def _make_config(**overrides) -> KaggleKernelConfig:
    defaults = dict(
        owner="edithstark",
        slug="solar-yolo-test",
        title="Solar YOLO Test",
        code_file="train.py",
        enable_gpu=True,
        enable_internet=False,
        dataset_sources=["gabrielkasmi/bdappv"],
    )
    defaults.update(overrides)
    return KaggleKernelConfig(**defaults)


@pytest.fixture
def entrypoint(tmp_path):
    script = tmp_path / "train_entry.py"
    script.write_text("print('training script')\n", encoding="utf-8")
    return script


class TestPrepare:
    def test_writes_metadata_and_copies_entry_script(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        result = prepare(config, entrypoint, package_dir)
        assert result == package_dir
        assert (package_dir / "kernel-metadata.json").is_file()
        assert (package_dir / "train.py").is_file()
        assert (package_dir / "train.py").read_text(encoding="utf-8") == "print('training script')\n"

    def test_metadata_content_matches_config(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config(enable_gpu=True, dataset_sources=["a/b"])
        prepare(config, entrypoint, package_dir)
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == "edithstark/solar-yolo-test"
        assert metadata["enable_gpu"] is True
        assert metadata["dataset_sources"] == ["a/b"]

    def test_internet_defaults_to_disabled(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()  # enable_internet not overridden -> default False
        prepare(config, entrypoint, package_dir)
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["enable_internet"] is False

    def test_copies_extra_files(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        helper = tmp_path / "helper.py"
        helper.write_text("HELPER = 1\n", encoding="utf-8")
        prepare(_make_config(), entrypoint, package_dir, extra_files=[helper])
        assert (package_dir / "helper.py").read_text(encoding="utf-8") == "HELPER = 1\n"


class TestDryRun:
    def test_valid_prepared_package_passes(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        result = dry_run(package_dir, config)
        assert result.passed, result.errors

    def test_missing_metadata_file_fails(self, tmp_path):
        package_dir = tmp_path / "empty_package"
        package_dir.mkdir()
        result = dry_run(package_dir, _make_config())
        assert not result.passed

    def test_missing_entry_script_fails(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        (package_dir / "train.py").unlink()
        result = dry_run(package_dir, config)
        assert not result.passed
        assert any("entry script" in e for e in result.errors)

    def test_tampered_metadata_fails(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        metadata_path = package_dir / "kernel-metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["enable_gpu"] = not data["enable_gpu"]
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed

    def test_placeholder_kernel_id_fails(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        metadata_path = package_dir / "kernel-metadata.json"
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["id"] = "edithstark/INSERT_KERNEL_SLUG_HERE"
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed
        assert any("placeholder" in e for e in result.errors)

    def test_invalid_json_fails_cleanly(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        (package_dir / "kernel-metadata.json").write_text("{not valid json", encoding="utf-8")
        result = dry_run(package_dir, config)
        assert not result.passed


class TestLaunchRefusesWithoutConfirmation:
    def test_raises_without_confirm(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        with patch("training.cloud.kaggle.adapter.subprocess.run") as mock_run:
            with pytest.raises(LaunchNotConfirmedError):
                launch(package_dir, config, confirm=False)
            mock_run.assert_not_called()

    def test_default_confirm_is_required_explicitly(self, tmp_path, entrypoint):
        """confirm has no default - a caller must actively pass True."""
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        with pytest.raises(TypeError):
            launch(package_dir, config)  # type: ignore[call-arg]


class TestLaunchWithConfirmation:
    def test_confirmed_launch_invokes_kaggle_push_with_correct_args(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)

        mock_result = MagicMock(returncode=0, stdout="Kernel version ... successfully pushed", stderr="")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result) as mock_run:
            result = launch(package_dir, config, confirm=True)

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args == ["kaggle", "kernels", "push", "-p", str(package_dir)]
        assert result.kernel_id == "edithstark/solar-yolo-test"
        assert "successfully pushed" in result.stdout

    def test_refuses_to_launch_when_dry_run_would_fail(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        (package_dir / "train.py").unlink()  # corrupt the package after prepare()

        with patch("training.cloud.kaggle.adapter.subprocess.run") as mock_run:
            with pytest.raises(ValueError, match="dry_run validation failed"):
                launch(package_dir, config, confirm=True)
            mock_run.assert_not_called()

    def test_nonzero_exit_raises_kaggle_cli_error(self, tmp_path, entrypoint):
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)

        mock_result = MagicMock(returncode=1, stdout="", stderr="403 Forbidden")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result):
            with pytest.raises(KaggleCLIError, match="403 Forbidden"):
                launch(package_dir, config, confirm=True)


class TestReadOnlyOperations:
    def test_status_calls_correct_command(self):
        mock_result = MagicMock(returncode=0, stdout="Kernel is running", stderr="")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result) as mock_run:
            result = status("edithstark/solar-yolo-test")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["kaggle", "kernels", "status", "edithstark/solar-yolo-test"]
        assert result["raw_output"] == "Kernel is running"

    def test_logs_calls_correct_command(self):
        mock_result = MagicMock(returncode=0, stdout="epoch=1 loss=1.2\n", stderr="")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result) as mock_run:
            result = logs("edithstark/solar-yolo-test")
        assert mock_run.call_args[0][0] == ["kaggle", "kernels", "logs", "edithstark/solar-yolo-test"]
        assert "epoch=1" in result

    def test_outputs_calls_correct_command_and_lists_files(self, tmp_path):
        dest = tmp_path / "out"
        mock_result = MagicMock(returncode=0, stdout="Output downloaded", stderr="")

        def _fake_run(cmd, **kwargs):
            # Simulate the CLI actually depositing a file, like the real `kernels output` would.
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "best.pt").write_bytes(b"fake checkpoint bytes")
            return mock_result

        with patch("training.cloud.kaggle.adapter.subprocess.run", side_effect=_fake_run) as mock_run:
            files = outputs("edithstark/solar-yolo-test", dest)

        assert mock_run.call_args[0][0] == ["kaggle", "kernels", "output", "edithstark/solar-yolo-test", "-p", str(dest)]
        assert files == [dest / "best.pt"]

    def test_status_raises_on_cli_failure(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="kernel not found")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result):
            with pytest.raises(KaggleCLIError):
                status("edithstark/does-not-exist")


class TestNeverInvokesRealSubprocessByAccident:
    def test_module_never_calls_subprocess_run_without_kaggle_prefix(self, tmp_path, entrypoint):
        """Belt-and-braces: confirm every subprocess.run call this module makes
        is prefixed with the literal 'kaggle' executable name, never a bare
        shell string and never anything else."""
        package_dir = tmp_path / "package"
        config = _make_config()
        prepare(config, entrypoint, package_dir)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result) as mock_run:
            launch(package_dir, config, confirm=True)
            status("a/b")
            logs("a/b")
        for call in mock_run.call_args_list:
            args = call[0][0]
            assert isinstance(args, list)
            assert args[0] == "kaggle"
