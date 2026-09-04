"""Kaggle kernel entrypoint for YOLO detection training.

THIS FILE CONTAINS NO YOLO TRAINING LOGIC. Its only job is to make an
exact, pinned commit of the Solar AI repository available inside the
Kaggle kernel environment and then invoke
training/detection/train_yolo.py completely unchanged, as a subprocess.
The scientific training configuration (model architecture, hyperparameters,
class order) lives in that one shared script, not here or duplicated here.

CONFIGURATION
-------------
Kaggle's kernel-metadata.json has no environment-variable mechanism
(confirmed by inspecting the real template `kaggle kernels init` writes -
its fields are id/title/code_file/language/kernel_type/is_private/
enable_gpu/enable_tpu/enable_internet/dataset_sources/competition_sources/
kernel_sources/model_sources - nothing else). So per-experiment
configuration cannot be read from the Kaggle environment the way a normal
CI job would; it must be baked into this script's source at package-build
time instead.

The CONFIG dict below holds placeholder tokens. Before this file is copied
into a kernel package, training.cloud.kaggle.entrypoints.render.render_entrypoint()
must replace every "__SOLAR_AI_*__" token with a real value. This script
refuses to run (fails loudly, does not silently proceed with placeholder
text) if any token was never substituted - see _require_rendered() below.

FAILURE MODES THIS SCRIPT MUST SURFACE CLEARLY (per design spec):
- git checkout of the requested commit fails
- dependency installation fails
- the requested commit cannot be resolved
- the dataset path is missing
- the training subprocess exits non-zero
Every one of those raises SystemExit with a specific message identifying
which step failed - never a bare non-zero exit with no explanation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/cat226/solar-ai-framework.git"
REPO_DIR = Path("/kaggle/working/solar-ai-framework")

# =============================================================================
# CONFIGURATION - every "__SOLAR_AI_*__" token must be substituted by
# render_entrypoint() before this file is packaged. See module docstring.
# =============================================================================
CONFIG: dict[str, str] = {
    "git_sha": "__SOLAR_AI_GIT_SHA__",
    "data_root": "__SOLAR_AI_DATA_ROOT__",
    "output_path": "__SOLAR_AI_OUTPUT_PATH__",
    "epochs": "__SOLAR_AI_EPOCHS__",
    "batch": "__SOLAR_AI_BATCH__",
    "imgsz": "__SOLAR_AI_IMGSZ__",
    "seed": "__SOLAR_AI_SEED__",
    "base_model": "__SOLAR_AI_BASE_MODEL__",
}


def _require_rendered(config: dict[str, str]) -> None:
    unrendered = [k for k, v in config.items() if isinstance(v, str) and v.startswith("__SOLAR_AI_")]
    if unrendered:
        raise SystemExit(
            f"CONFIG placeholder(s) were never substituted: {unrendered}. "
            "This file must be built via training.cloud.kaggle.entrypoints.render."
            "render_entrypoint() before being packaged - it must never be copied "
            "into a kernel package as-is."
        )


def _run_step(step_name: str, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[{step_name}] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"[{step_name}] FAILED (exit {result.returncode}): {' '.join(cmd)}")
    return result


def main() -> int:
    _require_rendered(CONFIG)

    git_sha = CONFIG["git_sha"]
    data_root = Path(CONFIG["data_root"])

    # --- 1-2. Obtain the repository and check out the exact requested commit. ---
    # Not "clone main" - the whole point of this smoke test is proving
    # experiment -> git SHA -> Kaggle environment -> the same train_yolo.py works.
    _run_step("git-clone", ["git", "clone", REPO_URL, str(REPO_DIR)])
    _run_step(
        "git-checkout",
        ["git", "checkout", git_sha],
        cwd=str(REPO_DIR),
    )
    # Confirm the checkout actually landed on the requested commit - `git checkout`
    # can exit 0 while leaving you somewhere unexpected in edge cases (e.g. a
    # local branch with the same name shadowing the intended commit). git_sha is
    # always a full 40-char SHA (TrainingJobSpec._git_sha() uses `git rev-parse
    # HEAD`), so this is a plain exact-match check, not a prefix heuristic.
    verify = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), capture_output=True, text=True,
    )
    actual_sha = verify.stdout.strip()
    if actual_sha != git_sha:
        raise SystemExit(
            f"git checkout landed on {actual_sha!r}, not the requested {git_sha!r} - refusing to train "
            "on an unverified commit."
        )

    # --- 3. Dataset path must actually exist before spending any more time. ---
    if not data_root.is_dir():
        raise SystemExit(f"dataset path does not exist: {data_root}")

    # --- 4. Install the minimal runtime dependencies train_yolo.py actually needs. ---
    # (scipy/pyarrow are prepare_dataset.py's dependencies, for converting raw
    # parquet - not needed here, since the dataset arrives already converted.)
    _run_step(
        "pip-install",
        [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics>=8.2.0", "PyYAML>=6.0.1"],
    )

    # --- 5. Invoke the real, unmodified training script. ---
    train_script = REPO_DIR / "training" / "detection" / "train_yolo.py"
    _run_step(
        "train_yolo",
        [
            sys.executable, str(train_script),
            "--data-root", str(data_root),
            "--output", CONFIG["output_path"],
            "--base-model", CONFIG["base_model"],
            "--epochs", CONFIG["epochs"],
            "--batch", CONFIG["batch"],
            "--imgsz", CONFIG["imgsz"],
            "--seed", CONFIG["seed"],
        ],
    )

    print(f"training complete, checkpoint at {CONFIG['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
