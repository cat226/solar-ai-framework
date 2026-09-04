"""Tests for training/cloud/kaggle/entrypoints/{render.py, yolo_detection.py}.

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
    / "training" / "cloud" / "kaggle" / "entrypoints" / "yolo_detection.py"
)

_FAKE_VALUES = {
    "git_sha": "a" * 40,
    "data_root": "/kaggle/input/solar-smoke/train_test",
    "output_path": "/kaggle/working/yolo_solar_candidate.pt",
    "epochs": "1",
    "batch": "8",
    "imgsz": "640",
    "seed": "42",
    "base_model": "yolov8n.pt",
}


def _load_rendered_module(path: Path):
    """Dynamically import a rendered entrypoint file as a fresh module, so
    each test gets its own isolated copy (module-level CONFIG mutation in
    one test must never leak into another). Registered in sys.modules under
    a unique name so mock.patch's string-based target resolution can find
    it - a module created via module_from_spec alone is not importable by
    name until it's registered there."""
    module_name = f"rendered_entrypoint_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestTemplateIsValidPython:
    def test_template_itself_parses_as_valid_python(self):
        ast.parse(_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def test_template_contains_no_training_logic_keywords(self):
        """Belt-and-braces check that this stays a thin wrapper - it must
        never grow YOLO/model-construction logic of its own."""
        text = _TEMPLATE_PATH.read_text(encoding="utf-8")
        for forbidden in ("YOLO(", "model.train(", "import torch", "import ultralytics"):
            assert forbidden not in text, f"entrypoint template contains training logic: {forbidden!r}"


class TestRenderEntrypoint:
    def test_render_produces_valid_python(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        ast.parse(out.read_text(encoding="utf-8"))  # must not raise

    def test_render_leaves_no_placeholder_tokens_as_config_values(self, tmp_path):
        # The literal string "__SOLAR_AI_" legitimately appears in the module's
        # own docstring/comments explaining the mechanism - the real check is
        # that _require_rendered() (the runtime guard) is satisfied, i.e. no
        # CONFIG *value* is still a placeholder.
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        module = _load_rendered_module(out)
        module._require_rendered(module.CONFIG)  # must not raise
        assert all(not v.startswith("__SOLAR_AI_") for v in module.CONFIG.values())

    def test_render_embeds_actual_values(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        text = out.read_text(encoding="utf-8")
        assert _FAKE_VALUES["git_sha"] in text
        assert _FAKE_VALUES["data_root"] in text

    def test_render_missing_required_key_raises(self, tmp_path):
        incomplete = dict(_FAKE_VALUES)
        del incomplete["git_sha"]
        out = tmp_path / "rendered.py"
        with pytest.raises(ValueError, match="unsubstituted placeholder"):
            render_entrypoint(_TEMPLATE_PATH, incomplete, out)

    def test_render_rejects_unknown_template_placeholder_mismatch(self, tmp_path):
        bad_template = tmp_path / "bad_template.py"
        bad_template.write_text("CONFIG = {'foo': '__SOLAR_AI_FOO__'}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no placeholder"):
            render_entrypoint(bad_template, {"bar": "x"}, tmp_path / "out.py")

    def test_paths_with_backslashes_render_safely(self, tmp_path):
        """Windows-style paths must not break the rendered file's syntax -
        repr() must correctly escape them."""
        values = dict(_FAKE_VALUES)
        values["data_root"] = r"E:\Solar AI Training Images\yolo_smoke_dataset"
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, values, out)
        ast.parse(out.read_text(encoding="utf-8"))  # must not raise despite backslashes


class TestUnrenderedTemplateRefusesToRun:
    def test_raw_template_config_still_has_placeholders(self):
        module = _load_rendered_module(_TEMPLATE_PATH)
        with pytest.raises(SystemExit, match="never substituted"):
            module._require_rendered(module.CONFIG)


class TestRenderedEntrypointExecutionFlow:
    """Every git/pip/train_yolo call is mocked - this proves the *sequence
    and failure handling* is correct without touching the network or
    running real training."""

    @pytest.fixture
    def rendered_module(self, tmp_path):
        out = tmp_path / "rendered.py"
        render_entrypoint(_TEMPLATE_PATH, _FAKE_VALUES, out)
        return _load_rendered_module(out)

    def test_happy_path_calls_steps_in_order(self, rendered_module, tmp_path):
        data_root = Path(_FAKE_VALUES["data_root"])
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("os.path.isdir", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            rc = rendered_module.main()

        assert rc == 0
        step_names = [c[0] for c in calls if c[0] in ("git", "python", sys.executable)]
        assert calls[0][:2] == ["git", "clone"]
        assert calls[1][:2] == ["git", "checkout"]
        assert calls[2][:2] == ["git", "rev-parse"]
        assert "pip" in calls[3] or "install" in calls[3]
        assert str(rendered_module.REPO_DIR / "training" / "detection" / "train_yolo.py") in calls[4]

    def test_git_clone_failure_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "clone"]:
                return MagicMock(returncode=128, stdout="", stderr="fatal: could not resolve host")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            with pytest.raises(SystemExit, match="git-clone"):
                rendered_module.main()

    def test_checkout_of_unresolvable_commit_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "checkout"]:
                return MagicMock(returncode=1, stdout="", stderr="fatal: reference is not a tree")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run):
            with pytest.raises(SystemExit, match="git-checkout"):
                rendered_module.main()

    def test_checkout_landing_on_wrong_sha_is_rejected(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout="b" * 40 + "\n", stderr="")  # not the requested SHA
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

    def test_pip_install_failure_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            if "pip" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="no matching distribution")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run), \
             patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(SystemExit, match="pip-install"):
                rendered_module.main()

    def test_training_nonzero_exit_is_surfaced_clearly(self, rendered_module):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return MagicMock(returncode=0, stdout=_FAKE_VALUES["git_sha"] + "\n", stderr="")
            if "train_yolo.py" in " ".join(cmd):
                return MagicMock(returncode=1, stdout="", stderr="CUDA out of memory")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rendered_module.subprocess, "run", side_effect=fake_run), \
             patch.object(Path, "is_dir", return_value=True):
            with pytest.raises(SystemExit, match="train_yolo"):
                rendered_module.main()
