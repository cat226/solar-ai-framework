"""Kaggle kernel entrypoint for diagnosing a dataset-mount failure.

THIS FILE CONTAINS NO YOLO TRAINING LOGIC, NO TORCH/ULTRALYTICS IMPORTS, AND
REQUESTS NO GPU. It exists solely to answer one question from inside a real
Kaggle kernel: why does a dataset correctly declared in dataset_sources not
show up at its expected /kaggle/input/<slug> path? It is entirely
self-contained (stdlib only) - it does not clone the repository, so it has
no dependency on internet access at all, unlike the training entrypoint.

CONFIGURATION
-------------
Same mechanism as entrypoints/yolo_detection.py: Kaggle's kernel-metadata.json
has no environment-variable field, so per-run values (the git SHA this
diagnostic corresponds to, and the dataset ref we expect to see mounted) are
baked into the CONFIG dict below at package-build time by
training.cloud.kaggle.entrypoints.render.render_entrypoint(). This script
refuses to run on unrendered placeholder text - see _require_rendered().

WHAT THIS SCRIPT REPORTS
-------------------------
1. The CONFIG values themselves (git SHA + the dataset ref we expect).
2. Every KAGGLE_*-prefixed environment variable, with any variable whose
   name looks secret-shaped (TOKEN/SECRET/KEY/PASSWORD/CREDENTIAL/BEARER)
   printed as "<redacted, N chars>" rather than its real value - never a
   real secret value, ever.
3. A full listing of /kaggle/input/: every immediate child directory, a
   shallow recursive walk of each with a file count, and highlighted
   checks against the specific candidate paths this diagnosis needs
   (the exact expected slug, an underscore variant, and anything else
   found).
4. An explicit PASS/FAIL verdict: was the declared dataset ref's directory
   actually found under /kaggle/input/ or not, and under what name.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - every "__SOLAR_AI_*__" token must be substituted by
# render_entrypoint() before this file is packaged. See module docstring.
# =============================================================================
CONFIG: dict[str, str] = {
    "git_sha": "__SOLAR_AI_GIT_SHA__",
    "declared_dataset_ref": "__SOLAR_AI_DECLARED_DATASET_REF__",
}

_SECRET_NAME_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "BEARER")


def _require_rendered(config: dict[str, str]) -> None:
    unrendered = [k for k, v in config.items() if isinstance(v, str) and v.startswith("__SOLAR_AI_")]
    if unrendered:
        raise SystemExit(
            f"CONFIG placeholder(s) were never substituted: {unrendered}. "
            "This file must be built via training.cloud.kaggle.entrypoints.render."
            "render_entrypoint() before being packaged - it must never be copied "
            "into a kernel package as-is."
        )


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _print_kaggle_env() -> None:
    _print_header("KAGGLE-RELATED ENVIRONMENT VARIABLES")
    kaggle_vars = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KAGGLE")}
    if not kaggle_vars:
        print("(no KAGGLE_* environment variables found)")
        return
    for key, value in kaggle_vars.items():
        if any(marker in key.upper() for marker in _SECRET_NAME_MARKERS):
            print(f"{key} = <redacted, {len(value)} chars>")
        else:
            print(f"{key} = {value!r}")


def _walk_shallow(path: Path, max_files: int = 25) -> tuple[int, list[str]]:
    """Returns (total_file_count, sample_of_relative_paths) for a shallow
    recursive walk - capped, since a real dataset can hold thousands of
    files and this is a diagnostic printout, not a full inventory."""
    count = 0
    sample: list[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            count += 1
            if len(sample) < max_files:
                rel = str((Path(root) / name).relative_to(path))
                sample.append(rel)
    return count, sample


def _inspect_kaggle_input() -> dict[str, object]:
    _print_header("/kaggle/input/ CONTENTS")
    input_root = Path("/kaggle/input")
    result: dict[str, object] = {"input_root_exists": input_root.is_dir(), "children": {}}
    if not input_root.is_dir():
        print(f"{input_root} does not exist or is not a directory.")
        return result

    children = sorted(input_root.iterdir())
    print(f"{len(children)} entr{'y' if len(children) == 1 else 'ies'} directly under {input_root}:")
    for child in children:
        is_dir = child.is_dir()
        entry: dict[str, object] = {"is_dir": is_dir}
        print(f"  - {child.name}  (is_dir={is_dir})")
        if is_dir:
            count, sample = _walk_shallow(child)
            entry["file_count"] = count
            entry["sample_files"] = sample
            print(f"      file_count={count}")
            for rel in sample[:10]:
                print(f"      · {rel}")
            if count > len(sample):
                print(f"      ... and {count - len(sample)} more (sample capped)")
        result["children"][child.name] = entry  # type: ignore[index]
    return result


def _slug_variants(dataset_ref: str) -> list[str]:
    slug = dataset_ref.split("/", 1)[-1]
    return sorted({slug, slug.replace("-", "_"), slug.replace("_", "-")})


def main() -> int:
    _require_rendered(CONFIG)

    _print_header("DIAGNOSTIC CONFIG (baked in at package-build time)")
    print(json.dumps(CONFIG, indent=2))

    _print_kaggle_env()
    kaggle_input_report = _inspect_kaggle_input()

    dataset_ref = CONFIG["declared_dataset_ref"]
    # Flat candidates (/kaggle/input/<slug>) plus the nested form Kaggle was
    # confirmed to actually use (/kaggle/input/datasets/<owner>/<slug>) -
    # discovered empirically by this very script's first real run
    # (solar-yolo-smoke-001-dataset-mount-diagnostic, 2026-09-04).
    candidates = _slug_variants(dataset_ref) + [f"datasets/{dataset_ref}"]

    _print_header("VERDICT: was the declared dataset found under /kaggle/input/?")
    print(f"declared dataset_sources entry: {dataset_ref!r}")
    print(f"candidate mount-path names checked: {candidates}")

    found_as = None
    for candidate in candidates:
        candidate_path = Path("/kaggle/input") / candidate
        if candidate_path.is_dir():
            found_as = candidate
            break

    if found_as is not None:
        print(f"RESULT: MOUNTED, found at /kaggle/input/{found_as}")
    else:
        actual_children = sorted((kaggle_input_report.get("children") or {}).keys())  # type: ignore[union-attr]
        print("RESULT: NOT MOUNTED under any expected candidate name.")
        print(f"actual /kaggle/input/ children: {actual_children}")

    print("\nDiagnostic complete. Exiting 0 (this script never trains anything).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
