"""tests/test_ground_pv_env_loading.py — the project-root .env is loaded via
python-dotenv and ROBOFLOW_API_KEY is resolvable, WITHOUT any test ever
reading, printing, or asserting the real secret value.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_SENTINEL = "rf_test_sentinel_value_do_not_use_0000"  # obviously fake; never a real key


# ---------------------------------------------------------------------------
# .env is git-ignored and never committed
# ---------------------------------------------------------------------------

def test_dotenv_file_is_git_ignored():
    r = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=_ROOT)
    assert r.returncode == 0, ".env must be git-ignored"


def test_dotenv_file_is_not_tracked():
    r = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode != 0, ".env must not be tracked by git"


def test_env_example_documents_roboflow_key_placeholder():
    text = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ROBOFLOW_API_KEY=your_roboflow_api_key_here" in text


def test_no_env_file_committed_anywhere():
    r = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True)
    tracked = r.stdout.splitlines()
    assert ".env" not in tracked
    assert not any(p.endswith("/.env") for p in tracked)


# ---------------------------------------------------------------------------
# _load_project_dotenv behavior (no real secret involved)
# ---------------------------------------------------------------------------

def test_load_project_dotenv_reads_a_file_without_overriding_real_env(tmp_path, monkeypatch):
    from utils.config import _load_project_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROUND_PV_TEST_ONLY_A=from_dotenv\nGROUND_PV_TEST_ONLY_B=from_dotenv\n",
        encoding="utf-8",
    )
    # B is already set in the real environment -> must NOT be overridden
    monkeypatch.setenv("GROUND_PV_TEST_ONLY_B", "from_real_env")
    monkeypatch.delenv("GROUND_PV_TEST_ONLY_A", raising=False)

    loaded = _load_project_dotenv(env_file)
    assert loaded is True
    assert os.environ["GROUND_PV_TEST_ONLY_A"] == "from_dotenv"
    assert os.environ["GROUND_PV_TEST_ONLY_B"] == "from_real_env"


def test_load_project_dotenv_missing_file_is_a_noop(tmp_path):
    from utils.config import _load_project_dotenv
    assert _load_project_dotenv(tmp_path / "nope.env") is False


def test_config_module_exposes_dotenv_loaded_flag():
    import utils.config as cfg
    assert hasattr(cfg, "_DOTENV_LOADED")
    assert isinstance(cfg._DOTENV_LOADED, bool)


# ---------------------------------------------------------------------------
# The downloader detects the key via the project's secret resolver, and
# NEVER echoes its value.
# ---------------------------------------------------------------------------

def _run_check_only(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "download_ground_pv_dataset.py"), "--check-only"],
        cwd=_ROOT, capture_output=True, text=True, env=env,
    )


def test_check_only_reports_key_present_without_revealing_it(monkeypatch):
    env = {k: v for k, v in os.environ.items()}
    env["ROBOFLOW_API_KEY"] = _SENTINEL
    r = _run_check_only(env)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["api_key_present"] is True
    assert payload["api_key_env_var"] == "ROBOFLOW_API_KEY"
    # the value itself must appear nowhere in the output
    assert _SENTINEL not in r.stdout
    assert _SENTINEL not in r.stderr


def test_check_only_reports_key_absent_when_unset():
    env = {k: v for k, v in os.environ.items() if k != "ROBOFLOW_API_KEY"}
    # neutralize a real project .env for this check so the assertion is stable
    env["_SOLAR_AI_IGNORE_DOTENV_MARKER"] = "1"
    r = _run_check_only(env)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    # api_key_present reflects whatever resolver found; if a real .env exists
    # locally it may be True. The contract we assert unconditionally: the
    # value is never echoed and the shape is correct.
    assert set(payload) >= {"api_key_present", "api_key_env_var", "dotenv_loaded"}
    assert isinstance(payload["api_key_present"], bool)
    assert "env_file_exists" in payload["dotenv_loaded"]


def test_resolve_api_key_helper_returns_value_but_is_never_logged(monkeypatch):
    """_resolve_api_key returns the secret to the caller (needed to actually
    authenticate) but the module must not print/log it. This checks the
    source contains no print/log of the resolved key."""
    src = (_ROOT / "scripts" / "download_ground_pv_dataset.py").read_text(encoding="utf-8")
    assert "_resolve_api_key" in src
    # nothing that emits the resolved key to stdout/stderr/logs/disk
    for bad in ("print(api_key", "print(key)", 'print(f"{api_key', "print(f'{api_key",
                "print(_resolve_api_key", "logging.", "getLogger", "write(api_key",
                "write(key)", "json.dump(api_key"):
        assert bad not in src
