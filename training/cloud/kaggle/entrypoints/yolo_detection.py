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
# PINNED TRAINING DEPENDENCIES - deterministic, GPU-generation-aware.
#
# Kaggle's free-tier GPU pool can hand out Pascal-generation hardware (Tesla
# P100, CUDA compute capability 6.0 / sm_60) - confirmed for real on
# solar-yolo-smoke-001-retry2 (2026-09-04). Kaggle's own preinstalled torch in
# that run was 2.10.0+cu128, which failed at model.to(device) with
# "CUDA error: no kernel image is available for execution on the device".
#
# Root cause, per PyTorch's own release notes (pytorch/pytorch GitHub
# releases, 2.8.0 notes): "Removed support for Maxwell and Pascal
# architectures with CUDA 12.8 and 12.9 builds... If you need support for
# these architectures, please utilize CUDA 12.6 instead." PyTorch 2.7.x is
# the last stable minor line documented to retain sm_50-sm_60 support.
#
# torch==2.7.1's default Linux/cp3xx PyPI wheel is itself CUDA-12.6-based
# (its own metadata pins nvidia-cuda-nvrtc-cu12==12.6.77, nvidia-cublas-
# cu12==12.6.4.1, etc. - verified via https://pypi.org/pypi/torch/2.7.1/json)
# - so no special --index-url is needed, plain PyPI already serves the
# correct build for this pin.
#
# torchvision==0.22.1 is the exact companion release: its own PyPI metadata
# requires "torch==2.7.1" precisely (verified via
# https://pypi.org/pypi/torchvision/0.22.1/json) - not a guessed pairing.
#
# torchaudio is intentionally NOT installed - train_yolo.py/Ultralytics have
# no audio dependency, and installing it would be an unnecessary package.
#
# ultralytics is pinned to the exact version that was already proven, on
# solar-yolo-smoke-001-retry2, to build the YOLOv8n architecture and
# transfer pretrained weights correctly (only the CUDA-kernel step failed) -
# keeping it fixed rather than letting a future run silently pick up a
# newer, untested release. Its own metadata (torch>=1.8.0, torchvision>=
# 0.9.0 - verified via https://pypi.org/pypi/ultralytics/8.4.138/json) is
# satisfied by the torch/torchvision pins above.
#
# These pins force pip to genuinely reinstall over whatever Kaggle's base
# image ships (an unpinned "pip install ultralytics" saw the preinstalled
# 2.10.0+cu128 already satisfied ultralytics's loose torch>=1.8.0 constraint
# and skipped reinstalling it entirely - which is how the incompatible
# version got used in the first place).
# =============================================================================
PINNED_DEPENDENCIES = [
    "torch==2.7.1",
    "torchvision==0.22.1",
    "ultralytics==8.4.138",
    "PyYAML>=6.0.1",
]

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
    # See PINNED_DEPENDENCIES above for why torch/torchvision are pinned to
    # exact, GPU-generation-compatible versions rather than left unconstrained.
    _run_step(
        "pip-install",
        [sys.executable, "-m", "pip", "install", "--quiet", *PINNED_DEPENDENCIES],
    )

    # --- 4b. Verify the installed torch build can actually see and use the ---
    # assigned GPU with its real compute capability, before spending any more
    # time - fail fast and clearly rather than let train_yolo.py hit an
    # opaque CUDA error deep inside Ultralytics.
    _run_step(
        "torch-cuda-check",
        [
            sys.executable, "-c",
            "import torch; "
            "print('torch.__version__ =', torch.__version__); "
            "print('torch.version.cuda =', torch.version.cuda); "
            "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; "
            "print('device name =', torch.cuda.get_device_name(0)); "
            "print('device capability =', torch.cuda.get_device_capability(0))",
        ],
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
