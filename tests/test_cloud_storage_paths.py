"""Tests for training/cloud/base/storage_paths.py.

Covers the platform/env-var branching (the "E: drive on this Windows
machine, but never on another platform" rule) and ensure_free_space()'s
fail-loud behavior, all without touching the real filesystem beyond
tmp_path.
"""
from __future__ import annotations

import importlib

import pytest


def _reload_with(monkeypatch, *, env_value: str | None, platform: str):
    import training.cloud.base.storage_paths as module

    if env_value is None:
        monkeypatch.delenv("SOLAR_AI_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("SOLAR_AI_DATA_ROOT", env_value)
    monkeypatch.setattr(module.sys, "platform", platform, raising=False)
    return importlib.reload(module)


class TestDefaultDataRoot:
    def test_env_var_override_wins_on_any_platform(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value="D:/somewhere/else", platform="linux")
        assert module.SOLAR_AI_DATA_ROOT == module.Path("D:/somewhere/else")

    def test_defaults_to_e_drive_on_windows_without_override(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value=None, platform="win32")
        assert module.SOLAR_AI_DATA_ROOT == module.Path("E:/Solar AI Training Images")

    def test_none_on_non_windows_without_override(self, monkeypatch):
        """Must never guess a path that would create a literal 'E:'
        subdirectory on a platform with no such drive concept (e.g. the
        Linux container a Kaggle kernel runs in)."""
        module = _reload_with(monkeypatch, env_value=None, platform="linux")
        assert module.SOLAR_AI_DATA_ROOT is None

    def test_subdirs_are_none_when_root_is_none(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value=None, platform="linux")
        assert module.KAGGLE_RUNS_DIR is None
        assert module.LOCAL_TRAINING_RUNS_DIR is None

    def test_subdirs_derived_from_root_when_present(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value="E:/Solar AI Training Images", platform="win32")
        assert module.KAGGLE_RUNS_DIR == module.SOLAR_AI_DATA_ROOT / "kaggle_runs"
        assert module.LOCAL_TRAINING_RUNS_DIR == module.SOLAR_AI_DATA_ROOT / "local_training_runs"

    @pytest.fixture(autouse=True)
    def _restore_real_module_after_each_test(self):
        """Every test above reloads the module with a monkeypatched
        env/platform - reload it once more for real afterwards so later
        test files that import it see the genuine environment, not a
        leftover patched state."""
        yield
        import training.cloud.base.storage_paths as module
        importlib.reload(module)


class TestDefaultKagglePackageDir:
    def test_under_kaggle_runs_dir_when_available(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value="E:/Solar AI Training Images", platform="win32")
        result = module.default_kaggle_package_dir("solar-yolo-full-v1")
        assert result == module.Path("E:/Solar AI Training Images/kaggle_runs/solar-yolo-full-v1")

    def test_falls_back_to_repo_relative_path_when_unavailable(self, monkeypatch):
        module = _reload_with(monkeypatch, env_value=None, platform="linux")
        result = module.default_kaggle_package_dir("solar-yolo-full-v1")
        assert result == module.Path("training/cloud/runs/solar-yolo-full-v1")


class TestEnsureFreeSpace:
    def test_passes_when_enough_space(self, tmp_path):
        from training.cloud.base.storage_paths import ensure_free_space
        ensure_free_space(tmp_path, required_bytes=1)  # 1 byte - always available

    def test_raises_when_not_enough_space(self, tmp_path):
        from training.cloud.base.storage_paths import InsufficientSpaceError, ensure_free_space
        huge = 10 ** 18  # an exabyte - no real drive has this free
        with pytest.raises(InsufficientSpaceError, match="insufficient free space"):
            ensure_free_space(tmp_path, required_bytes=huge, label="test dataset")

    def test_error_message_includes_label(self, tmp_path):
        from training.cloud.base.storage_paths import InsufficientSpaceError, ensure_free_space
        with pytest.raises(InsufficientSpaceError, match="my-label"):
            ensure_free_space(tmp_path, required_bytes=10 ** 18, label="my-label")

    def test_works_for_a_not_yet_existing_path(self, tmp_path):
        """required_bytes check should walk up to an existing ancestor
        rather than requiring the exact target directory to pre-exist."""
        from training.cloud.base.storage_paths import ensure_free_space
        not_yet_created = tmp_path / "a" / "b" / "c"
        ensure_free_space(not_yet_created, required_bytes=1)


class TestEnsureDir:
    def test_creates_nested_directory(self, tmp_path):
        from training.cloud.base.storage_paths import ensure_dir
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result == target
        assert target.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path):
        from training.cloud.base.storage_paths import ensure_dir
        ensure_dir(tmp_path)  # already exists - must not raise
