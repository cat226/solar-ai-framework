#!/usr/bin/env python3
"""training/evaluation/validate_real_images.py — Real-image validation gate
for the frozen Solar AI v1 ML pipeline.

Runs the actual production pipeline (models.model_manager -> models.detector
/ models.classifier via services.pipeline.run_pipeline, exactly as app.py
does) against a directory of genuine sample images. This is a
validation/observation tool, not an accuracy benchmark: it does NOT assume
any folder/filename encodes verified ground truth, does not compute
precision/recall/accuracy, and never fabricates a result for a case the
pipeline itself could not produce one for (e.g. XGBoost).

Never modifies the sample directory. Never modifies model weights,
configuration, or thresholds - it only reads the already-committed
production configuration and reports what it observes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from training.evaluation.common import sha256_file

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff"}


def _inventory(sample_dir: Path) -> list[dict]:
    records = []
    for path in sorted(sample_dir.iterdir()):
        if not path.is_file():
            continue
        entry = {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "readable": False,
            "width": None,
            "height": None,
            "declared_format": None,
        }
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                entry["width"], entry["height"] = im.size
                entry["declared_format"] = im.format
            entry["readable"] = True
        except Exception as exc:  # noqa: BLE001 - recording the failure itself is the point
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
        records.append(entry)
    return records


def _serialize_panel(panel) -> dict:
    return {
        "panel_index": panel.panel_index,
        "box": panel.box,
        "detection_confidence": panel.detection_confidence,
        "classification_label": panel.classification.label,
        "classification_confidence": panel.classification.confidence,
        "classification_probabilities": panel.classification.probabilities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Where to write the JSON results (should be outside the git repo or gitignored).")
    parser.add_argument("--city", type=str, default="Chennai")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from models.model_manager import model_manager
    from utils.config import CFG

    yolo_path = Path(CFG["models"]["yolo"]["weights"])
    mn_status = model_manager.mobilenet_status

    env = {
        "git_sha": None,  # filled by the caller/report from `git rev-parse HEAD`
        "config": {
            "yolo": dict(CFG["models"]["yolo"]),
            "mobilenet_v1_labels": mn_status["v1_labels"],
        },
        "yolo_weights_sha256": sha256_file(yolo_path),
        "classifier_source": None,
        "classifier_labels": None,
        "mobilenet_weights_sha256": None,
    }

    # Force-load once up front so classifier_source/labels/hash are known
    # before processing any image (fail fast if artifacts are unavailable).
    labels = model_manager.classifier_labels
    env["classifier_source"] = model_manager.classifier_source
    env["classifier_labels"] = labels
    active_mn_path = Path(mn_status["v1_path"]) if model_manager.classifier_source == "v1" else Path(mn_status["six_class_path"])
    env["mobilenet_weights_sha256"] = sha256_file(active_mn_path)
    env["mobilenet_weights_path"] = str(active_mn_path)

    print(f"YOLO sha256={env['yolo_weights_sha256']}")
    print(f"MobileNet source={env['classifier_source']} labels={labels} sha256={env['mobilenet_weights_sha256']}")

    inventory = _inventory(args.sample_dir)
    print(f"\nInventory: {len(inventory)} files found in {args.sample_dir}")
    for rec in inventory:
        print(f"  {rec['filename']}: readable={rec['readable']} dim={rec['width']}x{rec['height']} format={rec['declared_format']}")

    from services.pipeline import run_pipeline

    results = []
    for rec in inventory:
        path = args.sample_dir / rec["filename"]
        result_entry = {"filename": rec["filename"]}
        if not rec["readable"]:
            result_entry["status"] = "UNREADABLE"
            result_entry["error"] = rec.get("read_error")
            results.append(result_entry)
            continue

        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            result_entry["status"] = "DECODE_ERROR"
            result_entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result_entry)
            continue

        t0 = time.time()
        try:
            pipeline_result = run_pipeline(image=image, city=args.city)
        except Exception as exc:  # noqa: BLE001 - record, never hide
            result_entry["status"] = "PIPELINE_EXCEPTION"
            result_entry["error"] = f"{type(exc).__name__}: {exc}"
            result_entry["wall_seconds"] = time.time() - t0
            results.append(result_entry)
            continue
        wall = time.time() - t0

        result_entry["status"] = pipeline_result.status
        result_entry["wall_seconds"] = wall
        result_entry["xgboost_available"] = pipeline_result.xgboost_available
        result_entry["classifier_source"] = pipeline_result.classifier_source

        if pipeline_result.status == "ERROR":
            result_entry["error_type"] = pipeline_result.error_type
            result_entry["error_message"] = pipeline_result.error_message
            results.append(result_entry)
            continue

        det = pipeline_result.detection_result
        result_entry["detection"] = {
            "panel_count": det.panel_count,
            "best_confidence": det.best_confidence,
            "boxes": det.boxes,
            "confidences": det.confidences,
            "detection_successful": det.detection_successful,
        }
        result_entry["whole_image_classification"] = {
            "label": pipeline_result.classification_result.label,
            "confidence": pipeline_result.classification_result.confidence,
            "probabilities": pipeline_result.classification_result.probabilities,
        }
        result_entry["panels"] = [_serialize_panel(p) for p in pipeline_result.panels]
        result_entry["site_summary"] = {
            "total_panels": pipeline_result.site_summary.total_panels,
            "class_counts": pipeline_result.site_summary.class_counts,
            "clean_pct": pipeline_result.site_summary.clean_pct,
        }
        results.append(result_entry)
        print(f"  [{result_entry['status']}] {rec['filename']}: panels={det.panel_count} "
              f"whole_image={result_entry['whole_image_classification']['label']} "
              f"({result_entry['whole_image_classification']['confidence']:.3f}) "
              f"[{wall:.2f}s]")

    out_path = args.output_dir / "real_image_validation_results.json"
    out_path.write_text(json.dumps({"environment": env, "inventory": inventory, "results": results}, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
