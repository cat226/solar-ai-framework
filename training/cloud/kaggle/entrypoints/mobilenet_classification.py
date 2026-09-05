"""Kaggle kernel entrypoint for MobileNetV2 fault classification training.

THIS FILE CONTAINS NO MOBILENET TRAINING LOGIC. Its only job is to make an
exact, pinned commit of the Solar AI repository available inside the
Kaggle kernel environment and then invoke
training/classification/train_mobilenet.py completely unchanged, as a
subprocess. The scientific training configuration (model architecture,
hyperparameters, class order/remapping) lives in that one shared script,
not here or duplicated here.

Mirrors training/cloud/kaggle/entrypoints/yolo_detection.py's design and
its P100/CUDA-compatibility dependency pins (see PINNED_DEPENDENCIES below)
- the same torch build proven, on the real YOLO Kaggle runs, to support the
Pascal-generation P100 Kaggle's free tier can assign.

CONFIGURATION
-------------
Same mechanism as yolo_detection.py: Kaggle's kernel-metadata.json has no
environment-variable field, so per-experiment configuration is baked into
this script's source at package-build time by
training.cloud.kaggle.entrypoints.render.render_entrypoint(). Refuses to
run on unrendered placeholder text - see _require_rendered() below.

FAILURE MODES THIS SCRIPT MUST SURFACE CLEARLY (per design spec, same as
yolo_detection.py):
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
# PINNED TRAINING DEPENDENCIES - see yolo_detection.py's own PINNED_DEPENDENCIES
# comment for the full derivation (PyTorch's own release notes + each
# package's real PyPI metadata). torch==2.7.1/torchvision==0.22.1 is the
# same exact pair already proven, on the real YOLO Kaggle runs, to support
# CUDA capability (6,0) - a Kaggle-assigned Tesla P100. train_mobilenet.py
# needs nothing beyond torch/torchvision (stdlib argparse/pathlib for the
# rest) - no ultralytics, no PyYAML.
# =============================================================================
PINNED_DEPENDENCIES = [
    "torch==2.7.1",
    "torchvision==0.22.1",
]

# =============================================================================
# CONFIGURATION - every "__SOLAR_AI_*__" token must be substituted by
# render_entrypoint() before this file is packaged. See module docstring.
# =============================================================================
CONFIG: dict[str, str] = {
    "git_sha": "__SOLAR_AI_GIT_SHA__",
    "data_root": "__SOLAR_AI_DATA_ROOT__",
    "output_path": "__SOLAR_AI_OUTPUT_PATH__",
    "classes": "__SOLAR_AI_CLASSES__",  # comma-joined, e.g. "Clean,Dusty,Hotspot"
    "epochs": "__SOLAR_AI_EPOCHS__",
    "batch_size": "__SOLAR_AI_BATCH_SIZE__",
    "lr": "__SOLAR_AI_LR__",
    "seed": "__SOLAR_AI_SEED__",
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
    classes = CONFIG["classes"].split(",")

    # --- 1-2. Obtain the repository and check out the exact requested commit. ---
    _run_step("git-clone", ["git", "clone", REPO_URL, str(REPO_DIR)])
    _run_step("git-checkout", ["git", "checkout", git_sha], cwd=str(REPO_DIR))
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

    # --- 4. Install the pinned, GPU-generation-compatible dependencies. ---
    _run_step(
        "pip-install",
        [sys.executable, "-m", "pip", "install", "--quiet", *PINNED_DEPENDENCIES],
    )

    # --- 4b. Verify the installed torch build can actually see and use the ---
    # assigned GPU with its real compute capability before spending any more
    # time - fail fast and clearly rather than let train_mobilenet.py hit an
    # opaque CUDA error mid-training.
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
    train_script = REPO_DIR / "training" / "classification" / "train_mobilenet.py"
    _run_step(
        "train_mobilenet",
        [
            sys.executable, str(train_script),
            "--data-root", str(data_root),
            "--output", CONFIG["output_path"],
            "--epochs", CONFIG["epochs"],
            "--batch-size", CONFIG["batch_size"],
            "--lr", CONFIG["lr"],
            "--seed", CONFIG["seed"],
            "--classes", *classes,
        ],
    )

    print(f"training complete, checkpoint at {CONFIG['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
