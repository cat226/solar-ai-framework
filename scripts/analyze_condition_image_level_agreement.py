#!/usr/bin/env python3
"""scripts/analyze_condition_image_level_agreement.py — Phase 16-A.1
zero-training, zero-new-inference analysis.

Question (from docs/PHASE_16_ARCHITECTURE_REVIEW.md §8/§14): is the active
4-class condition model's weakness (test mAP50 ~= 0.444, DUSTY worst
despite being the most common class) a LOCALIZATION problem — the model
usually gets the right class somewhere in the image, just not a tight
enough box — or a SEMANTIC one, where it doesn't reliably recognize the
class at all? The answer decides whether a crop-level classifier
reformulation (Experiment 16-B) is worth building.

Method: reuse the box-level TP/FP/FN records ALREADY COMPUTED for
`weights/yolo_solar_condition_4class_research_v2.pt` by the Phase 15C error
analysis (research_runs/ground_pv/condition_error_analysis_v2/records.json)
— no new model inference, no new training, no new data. For every test
image:
  - dominant_gt_class  = the class with the most ground-truth instances in
    that image (ties broken by larger total GT area, then alphabetically,
    for a fully deterministic rule).
  - top_prediction_class = the class of the single highest-confidence
    prediction the model made anywhere in that image (TP or FP — i.e. ANY
    box it output, not just correctly-localized ones), or "no_prediction"
    if the model produced nothing for that image.
  - agreement = whether top_prediction_class == dominant_gt_class.

This is a strictly weaker requirement than box-level TP matching (no IoU
threshold, no per-instance localization needed) — comparing this
image-level accuracy against the already-documented box-level recall/AP50
per class tells us whether localization is the bottleneck.

Nothing here retrains, modifies, or re-runs any checkpoint. Read-only
analysis of an existing, already-committed-adjacent artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

_DEFAULT_RECORDS = Path("research_runs/ground_pv/condition_error_analysis_v2/records.json")
_DEFAULT_SUMMARY = Path("datasets/ground_pv_research/condition_4class_v1/reports/condition_error_analysis_v2/error_analysis_summary.json")


def _box_area(b: list[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _dominant_class(gt_instances: list[tuple[str, float]]) -> str:
    """gt_instances: list of (class_name, area). Majority by COUNT; ties
    broken by larger total AREA, then alphabetically — deterministic."""
    count: dict[str, int] = defaultdict(int)
    area: dict[str, float] = defaultdict(float)
    for cls, a in gt_instances:
        count[cls] += 1
        area[cls] += a
    return max(count.keys(), key=lambda c: (count[c], area[c], [-ord(ch) for ch in c]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=_DEFAULT_RECORDS)
    ap.add_argument("--official-summary", type=Path, default=_DEFAULT_SUMMARY,
                     help="Phase 15C error_analysis_summary.json, for the box-level comparison numbers.")
    ap.add_argument("--output", type=Path,
                     default=Path("research_runs/ground_pv/condition_image_level_agreement_v2/image_level_agreement.json"))
    args = ap.parse_args(argv)

    if not args.records.is_file():
        raise SystemExit(f"records file not found: {args.records} — run scripts/analyze_condition_4class_errors.py "
                          "--weights weights/yolo_solar_condition_4class_research_v2.pt first.")
    data = json.loads(args.records.read_text(encoding="utf-8"))
    tp_records, fp_records, fn_records = data["tp_records"], data["fp_records"], data["fn_records"]

    # Per-image ground truth: TP class == matched GT class; FN class == missed GT class.
    gt_by_image: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in tp_records:
        gt_by_image[r["image"]].append((r["class"], _box_area(r["bbox"])))
    for r in fn_records:
        gt_by_image[r["image"]].append((r["class"], _box_area(r["bbox"])))

    # Per-image predictions: every box the model output, TP or FP alike.
    preds_by_image: dict[str, list[tuple[str, float]]] = defaultdict(list)  # (class, confidence)
    for r in tp_records:
        preds_by_image[r["image"]].append((r["class"], r["confidence"]))
    for r in fp_records:
        preds_by_image[r["image"]].append((r["class"], r["confidence"]))

    per_image: list[dict[str, Any]] = []
    for image, gt_instances in gt_by_image.items():
        dominant = _dominant_class(gt_instances)
        preds = preds_by_image.get(image, [])
        if preds:
            top_class, top_conf = max(preds, key=lambda p: p[1])
        else:
            top_class, top_conf = "no_prediction", None
        per_image.append({
            "image": image,
            "dominant_gt_class": dominant,
            "gt_class_counts": {c: sum(1 for cc, _ in gt_instances if cc == c) for c in CONDITION_4CLASS_ORDER},
            "top_prediction_class": top_class,
            "top_prediction_confidence": top_conf,
            "n_predictions_this_image": len(preds),
            "agreement": (top_class == dominant),
        })

    n = len(per_image)
    n_agree = sum(1 for r in per_image if r["agreement"])
    overall_image_level_accuracy = n_agree / n if n else None

    per_class: dict[str, dict[str, Any]] = {}
    for cls in CONDITION_4CLASS_ORDER:
        subset = [r for r in per_image if r["dominant_gt_class"] == cls]
        agree = sum(1 for r in subset if r["agreement"])
        no_pred = sum(1 for r in subset if r["top_prediction_class"] == "no_prediction")
        confused_with: dict[str, int] = defaultdict(int)
        for r in subset:
            if not r["agreement"] and r["top_prediction_class"] != "no_prediction":
                confused_with[r["top_prediction_class"]] += 1
        per_class[cls] = {
            "n_images_dominant_this_class": len(subset),
            "image_level_agreement": (agree / len(subset)) if subset else None,
            "no_prediction_count": no_pred,
            "confused_with_other_active_class_counts": dict(confused_with),
        }

    # Box-level comparison numbers, pulled from the already-committed Phase 15C summary
    # (never re-derived here — quoting the existing, already-verified numbers).
    box_level_comparison: dict[str, Any] | None = None
    if args.official_summary.is_file():
        official = json.loads(args.official_summary.read_text(encoding="utf-8"))
        box_level_comparison = {
            cls: {
                "recall_at_match_threshold": official["per_class"].get(cls, {}).get("recall_at_match_threshold"),
                "precision_at_match_threshold": official["per_class"].get(cls, {}).get("precision_at_match_threshold"),
            }
            for cls in CONDITION_4CLASS_ORDER
            if cls in official.get("per_class", {})
        }

    result = {
        "schema_version": "1.0",
        "phase": "16-A.1 — image-level agreement analysis (zero new training/inference)",
        "method": (
            "Reuses TP/FP/FN records already computed for "
            "weights/yolo_solar_condition_4class_research_v2.pt by the Phase 15C error "
            "analysis. No new model inference was run to produce this report."
        ),
        "source_records": str(args.records),
        "num_test_images_with_gt": n,
        "overall_image_level_accuracy": overall_image_level_accuracy,
        "per_class_image_level": per_class,
        "box_level_comparison_same_checkpoint": box_level_comparison,
        "interpretation_note": (
            "image_level_agreement is a STRICTLY WEAKER criterion than box-level recall "
            "(no IoU/localization requirement, any confidence). If image-level agreement "
            "is materially higher than box-level recall for a class, the gap is evidence "
            "the model often 'knows' the right class somewhere in the image but fails to "
            "localize it tightly — supporting a crop-level classifier reformulation "
            "(Experiment 16-B). If image-level agreement is comparably weak, the "
            "bottleneck is more likely semantic (the model doesn't reliably recognize the "
            "class at all), and a classifier reformulation should not be assumed to fix it."
        ),
        "per_image_records": per_image,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"images analyzed: {n}")
    print(f"overall image-level accuracy: {overall_image_level_accuracy:.3f}" if overall_image_level_accuracy is not None else "n/a")
    print()
    print(f"{'class':<16} {'img-level agree':>16} {'box-level recall':>18} {'gap':>8} {'no_pred':>8}")
    for cls in CONDITION_4CLASS_ORDER:
        pc = per_class[cls]
        il = pc["image_level_agreement"]
        bl = (box_level_comparison or {}).get(cls, {}).get("recall_at_match_threshold")
        gap = (il - bl) if (il is not None and bl is not None) else None
        print(f"{cls:<16} {il if il is not None else float('nan'):>16.3f} "
              f"{bl if bl is not None else float('nan'):>18.3f} "
              f"{gap if gap is not None else float('nan'):>8.3f} {pc['no_prediction_count']:>8}")
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
