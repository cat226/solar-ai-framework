"""training/detection/ground_pv/condition_resolution_comparison.py — Phase
15D-A deterministic comparison logic for the imgsz=640 (v2) vs imgsz=960
(v3_960) resolution ablation of the 4-class condition model.

Pure, dependency-light logic (no torch/ultralytics, no disk I/O) so it can
be unit tested with synthetic manifest/summary dicts. The orchestration
that loads the real training_manifest.json / error_analysis_summary.json
files and writes the comparison report lives in
`scripts/compare_condition_resolution_experiment.py`.
"""
from __future__ import annotations

from typing import Any

OVERALL_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "mAP50", "mAP50_95")
PER_CLASS_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "AP50", "AP50_95")
ACTIVE_CLASSES: tuple[str, ...] = ("CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE")
SIZE_BINS: tuple[str, ...] = ("tiny", "small", "medium", "large")


def official_overall_delta(baseline_manifest: dict[str, Any], candidate_manifest: dict[str, Any],
                            eval_key: str = "test_default_metrics") -> dict[str, dict[str, float]]:
    """Delta (candidate - baseline) for each official overall metric, read
    straight from each training_manifest.json's `eval_key` block — never
    recomputed, so this can't silently disagree with the officially
    reported numbers."""
    base = baseline_manifest[eval_key]
    cand = candidate_manifest[eval_key]
    out = {}
    for k in OVERALL_METRIC_KEYS:
        out[k] = {"baseline": base[k], "candidate": cand[k], "delta": cand[k] - base[k]}
    return out


def official_per_class_delta(baseline_manifest: dict[str, Any], candidate_manifest: dict[str, Any],
                              eval_key: str = "test_default") -> dict[str, dict[str, dict[str, float]]]:
    """Same as `official_overall_delta` but per active class, reading each
    manifest's `per_class[eval_key]` block."""
    base = baseline_manifest["per_class"][eval_key]
    cand = candidate_manifest["per_class"][eval_key]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cls in ACTIVE_CLASSES:
        out[cls] = {}
        for k in PER_CLASS_METRIC_KEYS:
            b, c = base[cls][k], cand[cls][k]
            out[cls][k] = {"baseline": b, "candidate": c, "delta": c - b}
    return out


def size_bin_recall(size_bin_recall_counts: dict[str, dict[str, int]]) -> dict[str, float | None]:
    """Converts a `{bin: {tp, fn}}` count dict (as produced by
    `condition_error_analysis.py`'s `size_bin_recall` field) into
    `{bin: recall}`, with None where there is no data for that bin."""
    out: dict[str, float | None] = {}
    for b in SIZE_BINS:
        counts = size_bin_recall_counts.get(b)
        if not counts:
            out[b] = None
            continue
        tp, fn = counts["tp"], counts["fn"]
        out[b] = (tp / (tp + fn)) if (tp + fn) > 0 else None
    return out


def size_bin_recall_delta(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> dict[str, dict[str, dict[str, float | None]]]:
    """Per-class, per-size-bin recall delta (candidate - baseline) from two
    error_analysis_summary.json dicts (the same schema
    `scripts/analyze_condition_4class_errors.py` produces)."""
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for cls in ACTIVE_CLASSES:
        base_bins = size_bin_recall(baseline_summary["per_class"][cls]["size_bin_recall"] or {})
        cand_bins = size_bin_recall(candidate_summary["per_class"][cls]["size_bin_recall"] or {})
        out[cls] = {}
        for b in SIZE_BINS:
            br, cr = base_bins[b], cand_bins[b]
            delta = (cr - br) if (br is not None and cr is not None) else None
            out[cls][b] = {"baseline": br, "candidate": cr, "delta": delta}
    return out


def small_object_recall_improved(size_delta: dict[str, dict[str, dict[str, float | None]]],
                                  bins: tuple[str, ...] = ("tiny", "small"),
                                  min_improving_classes: int = 3,
                                  min_delta: float = 0.02) -> dict[str, Any]:
    """Applies the task's decision rule mechanically: for each active
    class, does its small-object recall (averaged over `bins`, skipping
    None entries) improve by at least `min_delta`? Returns a per-class
    verdict plus an overall boolean (True only if at least
    `min_improving_classes` of the 4 classes show a real improvement) —
    the function does not itself decide the final recommendation text,
    only computes the mechanical, reproducible signal that a human-written
    recommendation should cite.
    """
    per_class: dict[str, dict[str, Any]] = {}
    improving = 0
    for cls, bin_deltas in size_delta.items():
        deltas = [bin_deltas[b]["delta"] for b in bins if bin_deltas.get(b) and bin_deltas[b]["delta"] is not None]
        avg_delta = (sum(deltas) / len(deltas)) if deltas else None
        improved = avg_delta is not None and avg_delta >= min_delta
        per_class[cls] = {"avg_small_bin_delta": avg_delta, "improved": improved}
        if improved:
            improving += 1
    return {
        "per_class": per_class,
        "classes_improved": improving,
        "classes_total": len(size_delta),
        "material_improvement": improving >= min_improving_classes,
    }
