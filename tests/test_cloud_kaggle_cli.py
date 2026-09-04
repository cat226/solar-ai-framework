"""Tests for training/cloud/kaggle/cli.py — CLI wiring of job spec, adapter, registry.

Mocks the Kaggle CLI subprocess boundary same as test_cloud_kaggle_adapter.py;
never invokes a real kernel.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.cloud.base import registry as registry_module
from training.cloud.base.registry import get_experiment, load_experiments
from training.cloud.kaggle import cli as kaggle_cli


@pytest.fixture
def entrypoint(tmp_path):
    script = tmp_path / "train_entry.py"
    script.write_text("print('training')\n", encoding="utf-8")
    return script


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.jsonl"
    monkeypatch.setattr(
        kaggle_cli, "record_experiment",
        lambda record: registry_module.record_experiment(record, registry_path=registry_path),
    )
    monkeypatch.setattr(
        kaggle_cli, "update_experiment_status",
        lambda *a, **kw: registry_module.update_experiment_status(*a, registry_path=registry_path, **kw),
    )
    return registry_path


class TestPrepareCommand:
    def test_prepare_writes_package_and_registers_experiment(self, tmp_path, entrypoint, isolated_registry):
        package_dir = tmp_path / "pkg"
        rc = kaggle_cli.main([
            "prepare", "--experiment-id", "exp-1", "--model", "yolo_detection",
            "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
            "--gpu", "--dataset-source", "gabrielkasmi/bdappv",
        ])
        assert rc == 0
        assert (package_dir / "kernel-metadata.json").is_file()
        record = get_experiment("exp-1", registry_path=isolated_registry)
        assert record is not None
        assert record["status"] == "prepared"
        assert record["configuration"]["enable_gpu"] is True

    def test_prepare_rejects_unknown_model(self, tmp_path, entrypoint):
        package_dir = tmp_path / "pkg"
        with pytest.raises(SystemExit):
            kaggle_cli.main([
                "prepare", "--experiment-id", "exp-1", "--model", "not-a-real-model",
                "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
            ])


class TestDryRunCommand:
    def test_dry_run_passes_on_valid_package(self, tmp_path, entrypoint, capsys):
        package_dir = tmp_path / "pkg"
        kaggle_cli.main([
            "prepare", "--experiment-id", "exp-1", "--model", "yolo_detection",
            "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
        ])
        rc = kaggle_cli.main(["dry-run", "--package-dir", str(package_dir)])
        assert rc == 0
        assert "PASS" in capsys.readouterr().out

    def test_dry_run_fails_on_missing_package(self, tmp_path, capsys):
        rc = kaggle_cli.main(["dry-run", "--package-dir", str(tmp_path / "does-not-exist")])
        assert rc == 1

    def test_dry_run_fails_when_entry_script_removed(self, tmp_path, entrypoint, capsys):
        package_dir = tmp_path / "pkg"
        kaggle_cli.main([
            "prepare", "--experiment-id", "exp-1", "--model", "yolo_detection",
            "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
        ])
        (package_dir / entrypoint.name).unlink()
        rc = kaggle_cli.main(["dry-run", "--package-dir", str(package_dir)])
        assert rc == 1
        assert "FAIL" in capsys.readouterr().out


class TestLaunchCommand:
    def test_launch_without_yes_refuses_and_calls_no_subprocess(self, tmp_path, entrypoint, capsys):
        package_dir = tmp_path / "pkg"
        kaggle_cli.main([
            "prepare", "--experiment-id", "exp-1", "--model", "yolo_detection",
            "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
        ])
        with patch("training.cloud.kaggle.adapter.subprocess.run") as mock_run:
            rc = kaggle_cli.main(["launch", "--package-dir", str(package_dir)])
        assert rc == 1
        mock_run.assert_not_called()
        assert "--yes" in capsys.readouterr().err

    def test_launch_with_yes_invokes_kaggle_and_updates_registry(self, tmp_path, entrypoint, isolated_registry):
        package_dir = tmp_path / "pkg"
        kaggle_cli.main([
            "prepare", "--experiment-id", "exp-1", "--model", "yolo_detection",
            "--entrypoint", str(entrypoint), "--package-dir", str(package_dir),
        ])
        mock_result = MagicMock(returncode=0, stdout="pushed successfully", stderr="")
        with patch("training.cloud.kaggle.adapter.subprocess.run", return_value=mock_result) as mock_run:
            rc = kaggle_cli.main([
                "launch", "--package-dir", str(package_dir), "--experiment-id", "exp-1", "--yes",
            ])
        assert rc == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][:3] == ["kaggle", "kernels", "push"]
        records = load_experiments(registry_path=isolated_registry)
        assert records[-1]["status"] == "launched"

    def test_launch_missing_package_fails_without_subprocess_call(self, tmp_path):
        with patch("training.cloud.kaggle.adapter.subprocess.run") as mock_run:
            rc = kaggle_cli.main([
                "launch", "--package-dir", str(tmp_path / "nope"), "--yes",
            ])
        assert rc == 1
        mock_run.assert_not_called()
