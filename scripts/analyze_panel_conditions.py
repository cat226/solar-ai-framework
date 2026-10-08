#!/usr/bin/env python3
"""scripts/analyze_panel_conditions.py — Phase 15A end-to-end CLI.

    full camera image
      -> dynamic panel detector (weights/yolo_solar_ground_research_v1.pt, unmodified)
      -> P1 .. PN
      -> crop each panel
      -> 4-class condition model
      -> CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE (or UNCERTAIN / NO_PREDICTION)

QUALITATIVE ONLY unless a real ground-truth-verified evaluation is
separately run — this script computes NO accuracy / precision / recall /
mAP. A prediction is a raw model output, not a fact. A "no_prediction" or
"uncertain" condition read is NEVER interpreted as "Clean".

Requires a condition checkpoint whose class taxonomy is EXACTLY the active
4-class set (CLEAN/DUSTY/BIRD_DROP/PHYSICAL_DAMAGE) — refuses (fails
closed) to run against the existing 5-class research checkpoint
(weights/yolo_solar_condition_research_v1.pt), which remains frozen and
untouched.

Example
-------
    python scripts/analyze_panel_conditions.py \
        --source "E:\\Solar AI Project Data\\first_party_photos\\solar_real_images_6_holdout" \
        --panel-weights weights/yolo_solar_ground_research_v1.pt \
        --condition-weights weights/yolo_solar_condition_4class_research_v1.pt \
        --output-dir research_runs/ground_pv/panel_condition_4class
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.condition_taxonomy import assert_active_taxonomy  # noqa: E402
from training.detection.ground_pv.dynamic_panel_detection import DynamicPanelConfig  # noqa: E402
from training.detection.ground_pv.panel_condition_pipeline import (  # noqa: E402
    ConditionInferenceConfig,
    run_panel_condition_pipeline,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _IMAGE_EXT:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory (recursed).")
    parser.add_argument("--panel-weights", type=Path, required=True)
    parser.add_argument("--condition-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-conf", type=float, default=0.45,
                        help="Matches the documented real-world-holdout convention (docs/DYNAMIC_PANEL_DETECTION.md).")
    parser.add_argument("--condition-detection-floor", type=float, default=0.25)
    parser.add_argument("--condition-predicted-threshold", type=float, default=0.45)
    args = parser.parse_args(argv)

    for w in (args.panel_weights, args.condition_weights):
        if not w.is_file():
            raise SystemExit(f"weights not found: {w}")
    images = _iter_images(args.source)
    if not images:
        raise SystemExit(f"no images found under {args.source}")

    from ultralytics import YOLO
    from PIL import Image

    panel_model = YOLO(str(args.panel_weights))
    condition_model = YOLO(str(args.condition_weights))

    cond_names = condition_model.names if isinstance(condition_model.names, dict) else dict(enumerate(condition_model.names))
    try:
        assert_active_taxonomy(cond_names)
    except ValueError as exc:
        raise SystemExit(
            f"REFUSING to run: {args.condition_weights} is not the active 4-class condition "
            f"taxonomy. {exc}"
        ) from exc

    panel_config = DynamicPanelConfig(confidence_threshold=args.panel_conf)
    condition_config = ConditionInferenceConfig(
        detection_floor=args.condition_detection_floor,
        predicted_threshold=args.condition_predicted_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    zero_panel_images: list[str] = []
    total_panels = 0
    status_counts = {"predicted": 0, "uncertain": 0, "no_prediction": 0}

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        report = run_panel_condition_pipeline(
            panel_model, condition_model, image, image_name=img_path.name,
            panel_model_name=args.panel_weights.name, condition_model_name=args.condition_weights.name,
            panel_config=panel_config, condition_config=condition_config,
        )
        report["image_path"] = str(img_path)
        report["image_sha256"] = _sha256(img_path)

        if report["panel_count"] == 0:
            zero_panel_images.append(img_path.name)
        else:
            total_panels += report["panel_count"]
            for p in report["panels"]:
                status_counts[p["condition"]["status"]] += 1

        records.append(report)
        (args.output_dir / f"{img_path.stem}_condition.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )

    summary = {
        "schema_version": "1.0",
        "phase": "15A — four-class panel condition pipeline",
        "panel_weights": str(args.panel_weights),
        "panel_weights_sha256": _sha256(args.panel_weights),
        "condition_weights": str(args.condition_weights),
        "condition_weights_sha256": _sha256(args.condition_weights),
        "condition_taxonomy": ["CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE"],
        "num_images": len(images),
        "num_images_zero_panels": len(zero_panel_images),
        "images_zero_panels": zero_panel_images,
        "total_panels_detected": total_panels,
        "condition_status_counts": status_counts,
        "records": records,
        "disclaimer": (
            "QUALITATIVE ONLY. No ground truth. NO accuracy/precision/recall/F1/mAP computed. "
            "A condition prediction is a raw model output, NOT a fact. 'uncertain' and "
            "'no_prediction' are NEVER interpreted as CLEAN. ELECTRICAL_DAMAGE is not part of "
            "the active taxonomy. HOTSPOT requires thermal/IR data and is never inferred from RGB."
        ),
    }
    (args.output_dir / "panel_condition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"images: {len(images)}   zero-panel images: {len(zero_panel_images)}   "
          f"total panels: {total_panels}")
    print(f"condition status counts: {status_counts}")
    print(f"summary: {args.output_dir / 'panel_condition_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
