#!/usr/bin/env python3
"""scripts/compare_condition_resolution_experiment.py — Phase 15D-A:
compares the v2 (imgsz=640, 100ep) baseline against the v3_960
(imgsz=960, 100ep) resolution ablation and writes a machine-readable
comparison JSON + prints a concise report.

Inputs (all already produced by prior steps, never recomputed here):
  - two training_manifest.json files (official Ultralytics metrics)
  - two error_analysis_summary.json files, produced by
    scripts/analyze_condition_4class_errors.py run against EACH
    checkpoint at ITS OWN training imgsz (this script does not run
    inference itself)

Example
-------
    python scripts/compare_condition_resolution_experiment.py \
        --baseline-manifest training/cloud/runs/condition_yolo11n_4class_baseline_v1/output/training_manifest.json \
        --baseline-error-summary research_runs/ground_pv/condition_error_analysis_v2/error_analysis_summary.json \
        --candidate-manifest training/cloud/runs/condition_yolo11n_4class_960_v1/output/training_manifest.json \
        --candidate-error-summary research_runs/ground_pv/condition_error_analysis_960/error_analysis_summary.json \
        --output datasets/ground_pv_research/condition_4class_v1/reports/condition_yolo11n_4class_960_v1/resolution_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.condition_resolution_comparison import (  # noqa: E402
    official_overall_delta,
    official_per_class_delta,
    size_bin_recall_delta,
    small_object_recall_improved,
)


def build_comparison(baseline_manifest: dict[str, Any], candidate_manifest: dict[str, Any],
                      baseline_error_summary: dict[str, Any], candidate_error_summary: dict[str, Any]) -> dict[str, Any]:
    overall = official_overall_delta(baseline_manifest, candidate_manifest)
    per_class = official_per_class_delta(baseline_manifest, candidate_manifest)
    size_delta = size_bin_recall_delta(baseline_error_summary, candidate_error_summary)
    small_object_verdict = small_object_recall_improved(size_delta)

    return {
        "schema_version": "1.0",
        "phase": "15D-A — resolution ablation (imgsz 640 vs 960)",
        "baseline": {
            "experiment_name": baseline_manifest.get("experiment_name"),
            "imgsz": baseline_manifest.get("imgsz"),
            "epochs": baseline_manifest.get("epochs"),
            "best_epoch": baseline_manifest.get("best_epoch"),
            "train_seconds": baseline_manifest.get("train_seconds"),
            "checkpoint_sha256": baseline_manifest.get("checkpoint_sha256"),
        },
        "candidate": {
            "experiment_name": candidate_manifest.get("experiment_name"),
            "imgsz": candidate_manifest.get("imgsz"),
            "epochs": candidate_manifest.get("epochs"),
            "best_epoch": candidate_manifest.get("best_epoch"),
            "train_seconds": candidate_manifest.get("train_seconds"),
            "checkpoint_sha256": candidate_manifest.get("checkpoint_sha256"),
        },
        "environment_match": baseline_manifest.get("env") == candidate_manifest.get("env"),
        "dataset_hashes_match": baseline_manifest.get("dataset_hashes") == candidate_manifest.get("dataset_hashes"),
        "official_overall_delta_test_default": overall,
        "official_per_class_delta_test_default": per_class,
        "size_bin_recall_delta": size_delta,
        "small_object_recall_verdict": small_object_verdict,
        "note_on_matching_methodology": (
            "size_bin_recall_delta comes from this project's own greedy IoU "
            "matcher (training/detection/ground_pv/condition_error_analysis.py), "
            "run independently against BOTH checkpoints at each checkpoint's OWN "
            "training imgsz, at a fixed conf>=0.20/IoU>=0.5 matching threshold — "
            "the same methodology Phase 15C used for v3. This is why matching-based "
            "overall recall can differ from the official Ultralytics test recall "
            "(which reports each class's best-F1 point on the full PR curve, not "
            "a literal count at conf=0.20) — both are reported, never conflated."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-error-summary", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-error-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    for p in (args.baseline_manifest, args.baseline_error_summary, args.candidate_manifest, args.candidate_error_summary):
        if not p.is_file():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    baseline_error_summary = json.loads(args.baseline_error_summary.read_text(encoding="utf-8"))
    candidate_error_summary = json.loads(args.candidate_error_summary.read_text(encoding="utf-8"))

    comparison = build_comparison(baseline_manifest, candidate_manifest, baseline_error_summary, candidate_error_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    print(f"environment_match: {comparison['environment_match']}")
    print(f"dataset_hashes_match: {comparison['dataset_hashes_match']}")
    print()
    print("=== official overall delta (test_default, candidate - baseline) ===")
    for k, row in comparison["official_overall_delta_test_default"].items():
        print(f"  {k:10s} baseline={row['baseline']:.4f}  candidate={row['candidate']:.4f}  delta={row['delta']:+.4f}")
    print()
    print("=== small-object recall verdict (tiny+small bins, avg delta) ===")
    v = comparison["small_object_recall_verdict"]
    for cls, row in v["per_class"].items():
        avg = row["avg_small_bin_delta"]
        avg_s = f"{avg:+.4f}" if avg is not None else "n/a"
        print(f"  {cls:16s} avg_delta={avg_s}  improved={row['improved']}")
    print(f"  classes_improved={v['classes_improved']}/{v['classes_total']}  material_improvement={v['material_improvement']}")
    print()
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
