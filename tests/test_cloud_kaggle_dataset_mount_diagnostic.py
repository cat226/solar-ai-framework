"""Tests for training/cloud/kaggle/entrypoints/dataset_mount_diagnostic.py
and build_dataset_mount_diagnostic_package.py.

No test here invokes a real Kaggle kernel or calls the real `kaggle` CLI.
The rendered script is executed for real (it's pure stdlib, read-only
filesystem inspection) against a temp directory standing in for
/kaggle/input, so its actual reporting logic is exercised, not just mocked.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from training.cloud.kaggle.build_dataset_mount_diagnostic_package import build
from training.cloud.kaggle.entrypoints.render import render_entrypoint

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "training" / "cloud" / "kaggle" / "entrypoints" / "dataset_mount_diagnostic.py"
)

_FAKE_VALUES = {
    "git_sha": "a" * 40,
    "declared_dataset_ref": "edithstark/solar-ai-yolo-smoke-001",
}


def _load_rendered_module(path: Path):
    module_name = f"rendered_diagnostic_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestTemplateIsValidPython:
    def test_template_itself_parses_as_valid_python(self):
        ast.parse(_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def test_template_contains_no_training_or_gpu_logic(self):
        text = _TEMPLATE_PATH.read_text(encoding="utf-8")
        for forbidden in ("YOLO(", "model.train(", "import torch", "import ultralytics", "subprocess"):
            assert forbidden not in text, f"diagnostic template contains non-diagnostic logic: {forbidden!r}"


class TestRenderEntrypoint:
    def test_render_produces_valid_python(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        ast.parse(out.read_text(encoding="utf-8"))

    def test_render_embeds_actual_values(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        text = out.read_text(encoding="utf-8")
        assert _FAKE_VALUES["git_sha"] in text
        assert _FAKE_VALUES["declared_dataset_ref"] in text

    def test_unrendered_template_refuses_to_run(self):
        module = _load_rendered_module(_TEMPLATE_PATH)
        with pytest.raises(SystemExit, match="never substituted"):
            module._require_rendered(module.CONFIG)


class TestDiagnosticExecutionAgainstFakeKaggleInput:
    """Runs the rendered script's real logic against a temp directory
    standing in for /kaggle/input, proving the mount-detection logic
    itself (not just that it's mockable)."""

    @pytest.fixture
    def rendered_module(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        return _load_rendered_module(out)

    def test_reports_mounted_when_dataset_dir_present(self, rendered_module, tmp_path, capsys):
        fake_input = tmp_path / "kaggle_input"
        dataset_dir = fake_input / "solar-ai-yolo-smoke-001"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "train" / "images" / "a.png").write_bytes(b"x")

        # Monkeypatch the hardcoded "/kaggle/input" root by patching the
        # module-level Path("/kaggle/input") construction indirectly -
        # _inspect_kaggle_input() builds Path("/kaggle/input") itself, so we
        # patch pathlib.Path at the module level to redirect that one string.
        real_path_cls = rendered_module.Path

        def _redirect(*args, **kwargs):
            if args and args[0] == "/kaggle/input":
                return fake_input
            return real_path_cls(*args, **kwargs)

        with patch.object(rendered_module, "Path", side_effect=_redirect):
            rc = rendered_module.main()

        out = capsys.readouterr().out
        assert rc == 0
        assert "RESULT: MOUNTED, found at /kaggle/input/solar-ai-yolo-smoke-001" in out

    def test_reports_mounted_under_nested_datasets_path(self, rendered_module, tmp_path, capsys):
        """Kaggle actually mounts under /kaggle/input/datasets/<owner>/<slug>/,
        not the flat /kaggle/input/<slug>/ - confirmed via a real diagnostic
        kernel run (see build_yolo_smoke_package.py). This must be detected."""
        fake_input = tmp_path / "kaggle_input"
        nested_dir = fake_input / "datasets" / "edithstark" / "solar-ai-yolo-smoke-001"
        nested_dir.mkdir(parents=True)
        (nested_dir / "manifest.json").write_text("{}", encoding="utf-8")

        real_path_cls = rendered_module.Path

        def _redirect(*args, **kwargs):
            if args and args[0] == "/kaggle/input":
                return fake_input
            return real_path_cls(*args, **kwargs)

        with patch.object(rendered_module, "Path", side_effect=_redirect):
            rc = rendered_module.main()

        out = capsys.readouterr().out
        assert rc == 0
        assert "RESULT: MOUNTED, found at /kaggle/input/datasets/edithstark/solar-ai-yolo-smoke-001" in out

    def test_reports_not_mounted_when_absent(self, rendered_module, tmp_path, capsys):
        fake_input = tmp_path / "kaggle_input_empty"
        fake_input.mkdir()
        (fake_input / "some-other-dataset").mkdir()

        real_path_cls = rendered_module.Path

        def _redirect(*args, **kwargs):
            if args and args[0] == "/kaggle/input":
                return fake_input
            return real_path_cls(*args, **kwargs)

        with patch.object(rendered_module, "Path", side_effect=_redirect):
            rc = rendered_module.main()

        out = capsys.readouterr().out
        assert rc == 0
        assert "RESULT: NOT MOUNTED" in out
        assert "some-other-dataset" in out

    def test_slug_variants_checks_underscore_and_hyphen(self, rendered_module):
        variants = rendered_module._slug_variants("edithstark/solar-ai-yolo-smoke-001")
        assert "solar-ai-yolo-smoke-001" in variants
        assert "solar_ai_yolo_smoke_001" in variants

    def test_never_prints_secret_env_values(self, rendered_module, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("KAGGLE_SOME_TOKEN", "supersecretvalue12345")
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        fake_input = tmp_path / "kaggle_input"
        fake_input.mkdir()
        real_path_cls = rendered_module.Path

        def _redirect(*args, **kwargs):
            if args and args[0] == "/kaggle/input":
                return fake_input
            return real_path_cls(*args, **kwargs)

        with patch.object(rendered_module, "Path", side_effect=_redirect):
            rendered_module.main()

        out = capsys.readouterr().out
        assert "supersecretvalue12345" not in out
        assert "KAGGLE_SOME_TOKEN = <redacted" in out
        assert "KAGGLE_KERNEL_RUN_TYPE = 'Batch'" in out


class TestBuildDatasetMountDiagnosticPackage:
    def test_build_produces_no_gpu_no_internet_package(self, tmp_path):
        package_dir = tmp_path / "package"
        build(
            experiment_id="solar-yolo-smoke-001-dataset-mount-diagnostic",
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-yolo-smoke-001",
            registry_path=tmp_path / "registry.jsonl",
        )
        metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["enable_gpu"] is False
        assert metadata["enable_internet"] is False
        assert metadata["dataset_sources"] == ["edithstark/solar-ai-yolo-smoke-001"]
        assert metadata["title"] == metadata["id"].split("/", 1)[1]  # title == slug, avoids Kaggle rename
        assert (package_dir / "diagnostic.py").is_file()

    def test_build_output_has_no_unsubstituted_placeholders(self, tmp_path):
        package_dir = tmp_path / "package"
        build(
            experiment_id="solar-yolo-smoke-001-dataset-mount-diagnostic",
            package_dir=package_dir,
            kaggle_dataset_ref="edithstark/solar-ai-yolo-smoke-001",
            registry_path=tmp_path / "registry.jsonl",
        )
        module = _load_rendered_module(package_dir / "diagnostic.py")
        module._require_rendered(module.CONFIG)  # must not raise
        assert all(not v.startswith("__SOLAR_AI_") for v in module.CONFIG.values())
