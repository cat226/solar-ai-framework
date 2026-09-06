#!/usr/bin/env python3
"""training/evaluation/compare_yolo_checkpoints_real_images.py

Runs the REAL production pipeline (services.pipeline.run_pipeline, the
same entry point app.py uses) twice per real validation image - once
with the production YOLO checkpoint, once with an experimental
checkpoint - so the two runs are comparable with only the detector
weights differing.

Every other stage is byte-identical between the two runs:
- preprocessing (utils.image_utils.resize_for_yolo, same crop_panel)
- confidence/IoU thresholds and image size (models/detector.py module-
  level constants, read once from configs/settings.yaml - never
  modified by this script)
- EXIF-orientation handling (PIL.ImageOps.exif_transpose, matching
  app.py's own loading path exactly)
- the same MobileNet classifier instance/weights for both runs
- the same device (whatever torch resolves by default - this machine
  has no CUDA, so both runs are CPU)

This is achieved via services.pipeline.run_pipeline's optional
`detector` parameter: the production run passes a SolarPanelDetector
built from models.model_manager.model_manager.get_detector() (the
normal singleton, unchanged); the experimental run passes a second
SolarPanelDetector manually loaded from --experimental-weights.
Neither run touches configs/settings.yaml, weights/yolo_solar.pt, or
weights/manifest.json.

Never fabricates a detection, a classification, ground truth, or an
XGBoost prediction - if the XGBoost artifact is unavailable, that is
reported as such (xgboost_available=False) and the run continues.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image, ImageOps
from ultralytics import YOLO

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
        except Exception as exc:  # noqa: BLE001
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
    }


def _run_one(image: Image.Image, city: str, detector) -> dict:
    from services.pipeline import run_pipeline

    entry: dict = {}
    t0 = time.time()
    try:
        pipeline_result = run_pipeline(image=image, city=city, detector=detector)
    except Exception as exc:  # noqa: BLE001 - record, never hide
        entry["status"] = "PIPELINE_EXCEPTION"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["wall_seconds"] = time.time() - t0
        return entry
    entry["wall_seconds"] = time.time() - t0
    entry["status"] = pipeline_result.status
    entry["xgboost_available"] = pipeline_result.xgboost_available

    if pipeline_result.status == "ERROR":
        entry["error_type"] = pipeline_result.error_type
        entry["error_message"] = pipeline_result.error_message
        return entry

    det = pipeline_result.detection_result
    entry["detection_count"] = det.panel_count
    entry["boxes"] = det.boxes
    entry["confidences"] = det.confidences
    entry["mean_confidence"] = (sum(det.confidences) / len(det.confidences)) if det.confidences else None
    entry["inference_success"] = True
    entry["crop_count"] = len(pipeline_result.panels)
    entry["whole_image_classification"] = {
        "label": pipeline_result.classification_result.label,
        "confidence": pipeline_result.classification_result.confidence,
    }
    entry["panels"] = [_serialize_panel(p) for p in pipeline_result.panels]
    entry["pipeline_status"] = pipeline_result.status
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--experimental-weights", type=Path, required=True)
    parser.add_argument("--expected-production-sha256", type=str, required=True,
                         help="Refuses to run if weights/yolo_solar.pt does not match this hash.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", type=str, default="Chennai")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from models.detector import SolarPanelDetector
    from models.model_manager import model_manager
    from utils.config import CFG

    production_weights = Path(CFG["models"]["yolo"]["weights"])
    production_sha256 = sha256_file(production_weights)
    if production_sha256 != args.expected_production_sha256:
        raise SystemExit(
            f"REFUSING TO RUN: weights/yolo_solar.pt SHA-256 is {production_sha256}, "
            f"expected {args.expected_production_sha256}. Production weights may have "
            "changed - this script never proceeds against an unverified baseline."
        )
    experimental_sha256 = sha256_file(args.experimental_weights)

    print(f"Production YOLO ({production_weights}): sha256={production_sha256} [VERIFIED]")
    print(f"Experimental YOLO ({args.experimental_weights}): sha256={experimental_sha256}")

    # Force classifier load once up front (same instance/weights used for
    # both detector runs - only the detector varies).
    labels = model_manager.classifier_labels
    classifier_source = model_manager.classifier_source
    print(f"MobileNet source={classifier_source} labels={labels}")

    baseline_detector = SolarPanelDetector()
    baseline_detector.set_model(model_manager.get_detector())

    experimental_model = YOLO(str(args.experimental_weights))
    experimental_detector = SolarPanelDetector()
    experimental_detector.set_model(experimental_model)

    inventory = _inventory(args.sample_dir)
    print(f"\nInventory: {len(inventory)} files found in {args.sample_dir}")

    results = []
    for rec in inventory:
        path = args.sample_dir / rec["filename"]
        result_entry: dict = {"filename": rec["filename"]}
        if not rec["readable"]:
            result_entry["status"] = "UNREADABLE"
            result_entry["error"] = rec.get("read_error")
            results.append(result_entry)
            continue

        try:
            with Image.open(path) as im:
                # Match app.py's real production loading path exactly.
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            result_entry["status"] = "DECODE_ERROR"
            result_entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result_entry)
            continue

        result_entry["production"] = _run_one(image, args.city, baseline_detector)
        result_entry["experimental"] = _run_one(image, args.city, experimental_detector)
        results.append(result_entry)

        p = result_entry["production"]
        e = result_entry["experimental"]
        print(
            f"  {rec['filename']}: prod_det={p.get('detection_count','-')} "
            f"exp_det={e.get('detection_count','-')} "
            f"prod_status={p.get('status')} exp_status={e.get('status')}"
        )

    out = {
        "production_weights_path": str(production_weights),
        "production_weights_sha256": production_sha256,
        "experimental_weights_path": str(args.experimental_weights),
        "experimental_weights_sha256": experimental_sha256,
        "classifier_source": classifier_source,
        "classifier_labels": labels,
        "inventory": inventory,
        "results": results,
    }
    out_path = args.output_dir / "yolo_checkpoint_comparison_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
