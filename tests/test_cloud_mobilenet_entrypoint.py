"""Tests for training/cloud/kaggle/entrypoints/mobilenet_classification.py.

No test here performs a real git clone, pip install, or training run -
every subprocess boundary is mocked. render_entrypoint() is tested for
real (pure string substitution + file I/O, no shelling out).
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.cloud.kaggle.entrypoints.render import render_entrypoint

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "training" / "cloud" / "kaggle" / "entrypoints" / "mobilenet_classification.py"
)

_FAKE_VALUES = {
    "git_sha": "a" * 40,
    "data_root": "/kaggle/input/datasets/edithstark/solar-mobilenet-smoke",
    "output_path": "/kaggle/working/mobilenet_solar_candidate.pth",
    "classes": "Clean,Dusty,Hotspot",
    "epochs": "1",
    "batch_size": "4",
    "lr": "0.0003",
    "seed": "42",
}


def _load_rendered_module(path: Path):
    module_name = f"rendered_mobilenet_entrypoint_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestTemplateIsValidPython:
    def test_template_itself_parses_as_valid_python(self):
        ast.parse(_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def test_template_contains_no_training_logic_keywords(self):
        text = _TEMPLATE_PATH.read_text(encoding="utf-8")
        for forbidden in ("mobilenet_v2(", "nn.Linear(", "CrossEntropyLoss("):
            assert forbidden not in text, f"entrypoint template contains training logic: {forbidden!r}"

    def test_template_never_imports_torch_in_its_own_process(self):
        """Same rule as yolo_detection.py: the entrypoint's own process must
        never import torch directly - the string "import torch" legitimately
        appears inside the torch-cuda-check subprocess's `python -c` one-liner."""
        tree = ast.parse(_TEMPLATE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in ("torch", "torchvision"), (
                        f"entrypoint template directly imports {alias.name!r} in its own process"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in ("torch", "torchvision"), (
                    f"entrypoint template directly imports from {node.module!r} in its own process"
                )


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
        assert _FAKE_VALUES["data_root"] in text
        assert "Clean,Dusty,Hotspot" in text


class TestUnrenderedTemplateRefusesToRun:
    def test_raw_template_config_still_has_placeholders(self):
        module = _load_rendered_module(_TEMPLATE_PATH)
        with pytest.raises(SystemExit, match="never substituted"):
            module._require_rendered(module.CONFIG)


class TestPinnedDependencies:
    def test_torch_and_torchvision_pinned_exact(self):
        module = _load_rendered_module(_TEMPLATE_PATH)
        assert "torch==2.7.1" in module.PINNED_DEPENDENCIES
        assert "torchvision==0.22.1" in module.PINNED_DEPENDENCIES

    def test_no_ultralytics_or_pyyaml(self):
        """train_mobilenet.py needs neither - pinning them would be
        unnecessary packages."""
        module = _load_rendered_module(_TEMPLATE_PATH)
        joined = " ".join(module.PINNED_DEPENDENCIES).lower()
        assert "ultralytics" not in joined
        assert "pyyaml" not in joined

    def test_does_not_modify_train_mobilenet_py(self):
        repo_root = Path(__file__).resolve().parent.parent
        text = (repo_root / "training" / "classification" / "train_mobilenet.py").read_text(encoding="utf-8")
        assert "PINNED_DEPENDENCIES" not in text
        assert "torch==" not in text


class TestRenderedEntrypointExecutionFlow:
    @pytest.fixture
    def rendered_module(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        return _load_rendered_module(out)

    def test_happy_path_calls_steps_in_order_and_passes_classes(self, rendered_module):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(Path, "is_dir", return_value=True), \
             patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            rc = rendered_module.main()

        assert rc == 0
        assert calls[0][:2] == ["git", "clone"]
        assert calls[1][:2] == ["git", "checkout"]
        assert calls[2][:2] == ["git", "rev-parse"]
        assert "pip" in calls[3] or "install" in calls[3]
        assert "-c" in calls[4]  # torch-cuda-check
        train_call = calls[5]
        assert "train_mobilenet.py" in " ".join(train_call)
        assert "--classes" in train_call
        idx = train_call.index("--classes")
        assert train_call[idx + 1:idx + 4] == ["Clean", "Dusty", "Hotspot"]

    def test_git_clone_failure_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "clone"]:
                return MagicMock(returncode=128, stdout="", stderr="fatal: could not resolve host")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            with pytest.raises(SystemExit, match="git-clone"):
                rendered_module.main()

    def test_checkout_landing_on_wrong_sha_is_rejected(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout="b" * 40 + "\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            with pytest.raises(SystemExit, match="refusing to train"):
                rendered_module.main()

    def test_missing_dataset_path_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run), \
             patch.object(Path, "is_dir", return_value=False):
            with pytest.raises(SystemExit, match="dataset path does not exist"):
                rendered_module.main()

    def test_torch_cuda_check_failure_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            if "-c" in cmd and any("torch" in part for part in cmd):
                return MagicMock(returncode=1, stdout="", stderr="AssertionError")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run), \
             patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(SystemExit, match="torch-cuda-check"):
                rendered_module.main()

    def test_training_nonzero_exit_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            if "train_mobilenet.py" in " ".join(cmd):
                return MagicMock(returncode=1, stdout="", stderr="CUDA out of memory")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run), \
             patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(SystemExit, match="train_mobilenet"):
                rendered_module.main()
