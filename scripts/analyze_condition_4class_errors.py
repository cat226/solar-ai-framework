#!/usr/bin/env python3
"""scripts/analyze_condition_4class_errors.py — Phase 15C error-analysis
orchestration for the active 4-class RGB condition model.

Runs a frozen checkpoint (default: `weights/yolo_solar_condition_4class_research_v3.pt`)
over the TEST split of `datasets/ground_pv_research/condition_4class_v1/`,
matches predictions to ground truth with a greedy IoU matcher
(`training.detection.ground_pv.condition_error_analysis`), and writes:

  - `error_analysis_summary.json` — per-class / per-family / per-size-bin
    aggregates, FP-cause and FN-signal category counts, data-quality
    flags. Small, deterministic, safe to commit.
  - `records.json` — every TP/FP/FN instance (image, class, confidence,
    bbox, category). Larger, kept out of git (research_runs/ pattern).
  - `gallery/<class>_<fp|fn>/*.jpg` — annotated crops for visual review,
    capped per category. Kept out of git.

THIS SCRIPT DOES NOT RETRAIN, DOES NOT MODIFY THE DATASET, AND DOES NOT
TOUCH ANY CHECKPOINT. Read-only analysis only. See
docs/CONDITION_MODEL_4CLASS_ERROR_ANALYSIS.md for the human-authored
report built from this script's output.

Example
-------
    python scripts/analyze_condition_4class_errors.py \
        --weights weights/yolo_solar_condition_4class_research_v3.pt \
        --dataset-root datasets/ground_pv_research/condition_4class_v1 \
        --output-dir research_runs/ground_pv/condition_error_analysis_v3 \
        --committed-summary-dir datasets/ground_pv_research/condition_4class_v1/reports/condition_error_analysis_v3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.condition_error_analysis import (  # noqa: E402
    box_area_fraction,
    categorize_false_positive,
    find_overlapping_gt_pairs,
    match_class_instances,
    nearest_same_class_confidence,
    parse_yolo_label_text,
    quartile_cuts,
    size_bin_label,
    summarize_confidences,
    touches_border,
)
from training.detection.ground_pv.condition_taxonomy import (  # noqa: E402
    CONDITION_4CLASS_IDS,
    CONDITION_4CLASS_NAMES,
    assert_active_taxonomy,
)

_DENSE_ANNOTATION_PERCENTILE = 99  # per-class per-image box count flagged above this percentile
_TINY_GT_AREA_FRACTION_ABS = 0.001  # 0.1% of image area — absolute "possibly noise" flag
_GT_OVERLAP_IOU_FLAG = 0.8
_MIN_GT_FOR_FAMILY_RECALL = 3  # avoid ranking single-instance families as "worst"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_family_lookup(splits_csv: Path, split: str) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with splits_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["derived_split"] != split:
                continue
            lookup[row["derived_filename"]] = {
                "family_id": row["family_id"],
                "source": row["source"],
                "source_dataset": row["source_dataset"],
            }
    return lookup


def _run_inference(weights_path: Path, image_paths: list[Path], conf_floor: float, imgsz: int = 640,
                    infer_batch: int = 4) -> dict[str, list[tuple[int, tuple[float, float, float, float], float]]]:
    """Returns {image_filename: [(class_id, xyxy_box, confidence), ...]}.

    `imgsz` MUST match the checkpoint's own training resolution — evaluating
    at a mismatched resolution confounds any cross-checkpoint comparison
    (e.g. the Phase 15D-A imgsz=960 experiment must be evaluated at 960,
    not the imgsz=640 default used for v2/v3)."""
    from ultralytics import YOLO  # local import — keeps the analysis module torch-free

    model = YOLO(str(weights_path))
    assert_active_taxonomy(model.names)

    out: dict[str, list[tuple[int, tuple[float, float, float, float], float]]] = {}
    # CPU-only local inference on this machine (8GB available RAM): a
    # single model.predict() call streaming over the FULL 242-image list
    # was observed to reliably CPU-OOM on one specific conv2d allocation
    # (~3.6GB) at imgsz=960 — identical failure regardless of infer_batch
    # (4, 2, or 1) or rect=True/False, and never on a small subset (10
    # images). All 242 test images are uniformly 640x640 (checked directly,
    # no oversized outlier), so this is memory fragmentation building up
    # across many sequential predict() calls inside one long-lived
    # generator/process, not a genuine per-image memory requirement.
    # Fix: predict in small CHUNKS via separate predict() calls, running gc
    # between chunks — a runtime-only workaround, no effect on outputs.
    import gc
    chunk_size = infer_batch if imgsz <= 640 else max(1, min(infer_batch, 8))
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start:start + chunk_size]
        results = model.predict(
            source=[str(p) for p in chunk], conf=conf_floor, imgsz=imgsz, verbose=False,
            batch=infer_batch, rect=False,
        )
        for path, res in zip(chunk, results):
            preds = []
            boxes = res.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = tuple(float(v) for v in boxes.xyxy[i].tolist())
                    preds.append((cls_id, xyxy, conf))
            out[path.name] = preds
        del results
        gc.collect()
    return out, str(model.ckpt_path) if hasattr(model, "ckpt_path") else str(weights_path)


def _draw_gallery_image(image_path: Path, out_path: Path, gt_boxes: list[tuple[int, tuple]], pred_box: tuple | None,
                         pred_label: str, class_names: dict[int, str]) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cls_id, box in gt_boxes:
        draw.rectangle(box, outline=(0, 200, 0), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"GT:{class_names.get(cls_id, cls_id)}", fill=(0, 200, 0))
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(220, 30, 30), width=2)
        draw.text((pred_box[0] + 2, min(img.height - 12, pred_box[3] + 2)), pred_label, fill=(220, 30, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


def analyze(
    weights_path: Path,
    dataset_root: Path,
    output_dir: Path,
    committed_summary_dir: Path,
    match_conf: float,
    match_iou: float,
    conf_floor: float,
    gallery_limit: int,
    limit: int | None = None,
    imgsz: int = 640,
    infer_batch: int = 4,
) -> dict[str, Any]:
    images_dir = dataset_root / "images" / "test" / "images"
    labels_dir = dataset_root / "images" / "test" / "labels"
    splits_csv = dataset_root / "splits_4class_manifest.csv"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert image_paths, f"no test images found under {images_dir}"
    if limit is not None:
        image_paths = image_paths[:limit]

    family_lookup = _load_family_lookup(splits_csv, split="test")

    checkpoint_sha256 = _sha256_file(weights_path)

    preds_by_image, ckpt_path_used = _run_inference(weights_path, image_paths, conf_floor=conf_floor, imgsz=imgsz, infer_batch=infer_batch)

    from PIL import Image as _PILImage

    # ---- Per-image GT + prediction bookkeeping -----------------------------------
    per_class_gt_areas: dict[int, list[float]] = defaultdict(list)
    gt_by_image: dict[str, list[tuple[int, tuple]]] = {}
    img_dims: dict[str, tuple[int, int]] = {}
    per_image_per_class_box_count: dict[str, dict[int, int]] = defaultdict(dict)

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        with _PILImage.open(img_path) as im:
            w, h = im.size
        img_dims[img_path.name] = (w, h)
        text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
        gt_list = parse_yolo_label_text(text, w, h)
        gt_by_image[img_path.name] = gt_list
        counts: dict[int, int] = defaultdict(int)
        for cls_id, box in gt_list:
            per_class_gt_areas[cls_id].append(box_area_fraction(box, w, h))
            counts[cls_id] += 1
        per_image_per_class_box_count[img_path.name] = dict(counts)

    size_cuts_by_class: dict[int, tuple[float, float, float]] = {}
    tiny_area_by_class: dict[int, float] = {}
    for cls_id, areas in per_class_gt_areas.items():
        if len(areas) >= 4:
            cuts = quartile_cuts(areas)
            size_cuts_by_class[cls_id] = cuts
            tiny_area_by_class[cls_id] = cuts[0]  # Q1 of this class's own GT area — data-driven "tiny" reference
        else:
            tiny_area_by_class[cls_id] = _TINY_GT_AREA_FRACTION_ABS

    # ---- Matching per image, per class --------------------------------------------
    class_counts: dict[int, dict[str, int]] = {c: {"gt_count": 0, "tp": 0, "fp": 0, "fn": 0} for c in CONDITION_4CLASS_NAMES}
    tp_confidences: dict[int, list[float]] = defaultdict(list)
    fp_confidences: dict[int, list[float]] = defaultdict(list)
    fn_nearby_confidences: dict[int, list[float | None]] = defaultdict(list)

    fp_records: list[dict[str, Any]] = []
    fn_records: list[dict[str, Any]] = []
    tp_records: list[dict[str, Any]] = []
    fp_category_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    family_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"gt_count": 0, "tp": 0, "fp": 0, "fn": 0})
    data_quality_flags: list[dict[str, Any]] = []

    for img_name in sorted(gt_by_image):
        w, h = img_dims[img_name]
        gt_list = gt_by_image[img_name]
        meta = family_lookup.get(img_name, {"family_id": "UNKNOWN", "source": "UNKNOWN", "source_dataset": "UNKNOWN"})
        low_conf_preds = preds_by_image.get(img_name, [])
        matching_preds = [(cid, box, conf) for cid, box, conf in low_conf_preds if conf >= match_conf]

        # data-quality: GT/GT overlap flags (never edits, only flags)
        overlap_pairs = find_overlapping_gt_pairs(gt_list, iou_threshold=_GT_OVERLAP_IOU_FLAG)
        for i, j, iou in overlap_pairs:
            ci, _ = gt_list[i]
            cj, _ = gt_list[j]
            kind = "near-duplicate GT annotation (same class)" if ci == cj else "overlapping GT boxes of DIFFERENT classes — ambiguous class semantics candidate"
            data_quality_flags.append({
                "image": img_name, "flag": kind, "iou": round(iou, 3),
                "class_a": CONDITION_4CLASS_NAMES[ci], "class_b": CONDITION_4CLASS_NAMES[cj],
                "family_id": meta["family_id"],
            })
        # data-quality: extremely small GT boxes (possible noise/mislabel candidate)
        for cls_id, box in gt_list:
            frac = box_area_fraction(box, w, h)
            if frac <= _TINY_GT_AREA_FRACTION_ABS:
                data_quality_flags.append({
                    "image": img_name, "flag": "possibly mislabeled/noise (extremely small GT box)",
                    "class": CONDITION_4CLASS_NAMES[cls_id], "area_fraction": round(frac, 6),
                    "family_id": meta["family_id"],
                })

        for cls_id in CONDITION_4CLASS_NAMES:
            gt_boxes_cls = [box for c, box in gt_list if c == cls_id]
            preds_cls_matching = [(box, conf) for c, box, conf in matching_preds if c == cls_id]
            preds_cls_lowconf = [(box, conf) for c, box, conf in low_conf_preds if c == cls_id]

            tp_idx, fp_idx, fn_idx = match_class_instances(gt_boxes_cls, preds_cls_matching, iou_threshold=match_iou)

            class_counts[cls_id]["gt_count"] += len(gt_boxes_cls)
            class_counts[cls_id]["tp"] += len(tp_idx)
            class_counts[cls_id]["fp"] += len(fp_idx)
            class_counts[cls_id]["fn"] += len(fn_idx)

            fam_key = (meta["family_id"], cls_id)
            family_counts[fam_key]["gt_count"] += len(gt_boxes_cls)
            family_counts[fam_key]["tp"] += len(tp_idx)
            family_counts[fam_key]["fp"] += len(fp_idx)
            family_counts[fam_key]["fn"] += len(fn_idx)

            accepted_boxes = [preds_cls_matching[i][0] for i in tp_idx]
            same_class_gt_boxes = gt_boxes_cls
            other_class_gt_boxes = [box for c, box in gt_list if c != cls_id]

            for i in tp_idx:
                box, conf = preds_cls_matching[i]
                tp_confidences[cls_id].append(conf)
                tp_records.append({
                    "image": img_name, "class": CONDITION_4CLASS_NAMES[cls_id], "confidence": conf,
                    "bbox": box, "family_id": meta["family_id"], "source": meta["source"],
                })

            for i in fp_idx:
                box, conf = preds_cls_matching[i]
                fp_confidences[cls_id].append(conf)
                category = categorize_false_positive(
                    box, accepted_boxes, same_class_gt_boxes, other_class_gt_boxes, w, h,
                    tiny_area_fraction=tiny_area_by_class[cls_id],
                )
                fp_category_counts[cls_id][category] += 1
                fp_records.append({
                    "image": img_name, "class": CONDITION_4CLASS_NAMES[cls_id], "confidence": conf,
                    "bbox": list(box), "category": category,
                    "family_id": meta["family_id"], "source": meta["source"],
                })

            for gi in fn_idx:
                box = gt_boxes_cls[gi]
                nearby = nearest_same_class_confidence(box, preds_cls_lowconf)
                fn_nearby_confidences[cls_id].append(nearby)
                size_bin = size_bin_label(box_area_fraction(box, w, h), size_cuts_by_class[cls_id]) if cls_id in size_cuts_by_class else "n/a"
                fn_records.append({
                    "image": img_name, "class": CONDITION_4CLASS_NAMES[cls_id], "bbox": list(box),
                    "size_bin": size_bin, "touches_border": touches_border(box, w, h),
                    "nearby_same_class_confidence": nearby,
                    "family_id": meta["family_id"], "source": meta["source"],
                })

    # ---- Object-size / recall-by-bin ------------------------------------------------
    size_bin_recall: dict[int, dict[str, dict[str, int]]] = {}
    for cls_id in CONDITION_4CLASS_NAMES:
        if cls_id not in size_cuts_by_class:
            continue
        bins: dict[str, dict[str, int]] = {b: {"tp": 0, "fn": 0} for b in ("tiny", "small", "medium", "large")}
        # TP boxes binned by GT area (use the matched GT box's own area, i.e. re-derive from fn/tp — easiest: re-walk tp_records+fn_records)
        for rec in fn_records:
            if rec["class"] == CONDITION_4CLASS_NAMES[cls_id] and rec["size_bin"] != "n/a":
                bins[rec["size_bin"]]["fn"] += 1
        size_bin_recall[cls_id] = bins
    # TPs need their GT box's size bin too — recompute by re-matching size bin from GT area fraction per TP (approx via bbox area, since TP pred box ~ GT box at IoU>=match_iou)
    for rec in tp_records:
        cls_id = CONDITION_4CLASS_IDS[rec["class"]]
        if cls_id not in size_cuts_by_class:
            continue
        # find image dims
        w, h = img_dims[rec["image"]]
        frac = box_area_fraction(tuple(rec["bbox"]), w, h)
        b = size_bin_label(frac, size_cuts_by_class[cls_id])
        size_bin_recall[cls_id][b]["tp"] += 1

    # ---- Family worst-performer summary --------------------------------------------
    family_summary: list[dict[str, Any]] = []
    for (fam_id, cls_id), c in family_counts.items():
        if c["gt_count"] < _MIN_GT_FOR_FAMILY_RECALL:
            continue
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else None
        family_summary.append({
            "family_id": fam_id, "class": CONDITION_4CLASS_NAMES[cls_id],
            "gt_count": c["gt_count"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "recall": recall,
        })
    family_summary.sort(key=lambda r: (r["recall"] if r["recall"] is not None else 1.0))

    # ---- Dense-annotation data-quality flag (percentile-based, computed once) -----
    per_class_box_counts: dict[int, list[int]] = defaultdict(list)
    for counts in per_image_per_class_box_count.values():
        for cls_id, n in counts.items():
            per_class_box_counts[cls_id].append(n)
    dense_thresholds: dict[int, float] = {}
    for cls_id, counts in per_class_box_counts.items():
        if len(counts) >= 4:
            sorted_counts = sorted(counts)
            idx = min(len(sorted_counts) - 1, int(len(sorted_counts) * _DENSE_ANNOTATION_PERCENTILE / 100))
            dense_thresholds[cls_id] = sorted_counts[idx]
    for img_name, counts in per_image_per_class_box_count.items():
        for cls_id, n in counts.items():
            thresh = dense_thresholds.get(cls_id)
            if thresh is not None and n > thresh:
                data_quality_flags.append({
                    "image": img_name, "flag": "dense/many-box annotation (above P99 for this class) — review for annotation-quality risk",
                    "class": CONDITION_4CLASS_NAMES[cls_id], "box_count": n, "p99_threshold": thresh,
                    "family_id": family_lookup.get(img_name, {}).get("family_id", "UNKNOWN"),
                })

    # ---- Assemble summary -----------------------------------------------------------
    per_class_summary = {}
    for cls_id, name in CONDITION_4CLASS_NAMES.items():
        c = class_counts[cls_id]
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else None
        per_class_summary[name] = {
            **c,
            "precision_at_match_threshold": precision,
            "recall_at_match_threshold": recall,
            "tp_confidence_distribution": summarize_confidences(tp_confidences[cls_id]),
            "fp_confidence_distribution": summarize_confidences(fp_confidences[cls_id]),
            "fn_nearby_confidence_distribution": summarize_confidences([v for v in fn_nearby_confidences[cls_id] if v is not None]),
            "fn_total_miss_count": sum(1 for v in fn_nearby_confidences[cls_id] if v is None),
            "fp_category_counts": dict(fp_category_counts[cls_id]),
            "size_quartile_cuts_area_fraction": size_cuts_by_class.get(cls_id),
            "size_bin_recall": size_bin_recall.get(cls_id),
        }

    summary = {
        "schema_version": "1.0",
        "phase": "15C — four-class condition model error analysis",
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_root": str(dataset_root),
        "match_confidence_threshold": match_conf,
        "match_iou_threshold": match_iou,
        "inference_conf_floor": conf_floor,
        "inference_imgsz": imgsz,
        "note_on_thresholds": (
            "match_confidence_threshold reuses the existing v3 operating threshold (0.20) "
            "purely as a fixed, already-documented point for INSTANCE-LEVEL TP/FP/FN counting "
            "in this diagnostic analysis. This phase does not redefine deployment policy — "
            "see docs/CONDITION_MODEL_4CLASS_ERROR_ANALYSIS.md section I."
        ),
        "num_test_images": len(image_paths),
        "environment": {
            "python": platform.python_version(),
            "ultralytics_local": _ultralytics_version(),
            "note": "local inference environment — may differ in patch version from the Kaggle training environment recorded in training_manifest.json; see report for the exact delta.",
        },
        "per_class": per_class_summary,
        "family_summary_sorted_by_recall_asc": family_summary,
        "data_quality_flags": data_quality_flags,
        "fp_record_count": len(fp_records),
        "fn_record_count": len(fn_records),
        "tp_record_count": len(tp_records),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(
        json.dumps({"fp_records": fp_records, "fn_records": fn_records, "tp_records": tp_records}, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    committed_summary_dir.mkdir(parents=True, exist_ok=True)
    (committed_summary_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _render_gallery(output_dir / "gallery", fp_records, fn_records, images_dir, gt_by_image, gallery_limit)

    return summary


def _ultralytics_version() -> str:
    try:
        import ultralytics
        return ultralytics.__version__
    except Exception:
        return "unknown"


def _render_gallery(gallery_dir: Path, fp_records: list[dict], fn_records: list[dict], images_dir: Path,
                     gt_by_image: dict[str, list[tuple[int, tuple]]], limit: int) -> None:
    priority = ("DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE", "CLEAN")

    for cls_name in priority:
        cls_fps = sorted([r for r in fp_records if r["class"] == cls_name], key=lambda r: -r["confidence"])[:limit]
        for i, rec in enumerate(cls_fps):
            img_path = images_dir / rec["image"]
            out_path = gallery_dir / f"{cls_name.lower()}_fp" / f"{i:02d}_{rec['image']}"
            gt_list = gt_by_image.get(rec["image"], [])
            _draw_gallery_image(img_path, out_path, gt_list, tuple(rec["bbox"]),
                                 f"FP {cls_name} {rec['confidence']:.2f} [{rec['category']}]",
                                 CONDITION_4CLASS_NAMES)

        cls_fns = sorted([r for r in fn_records if r["class"] == cls_name],
                          key=lambda r: (r["nearby_same_class_confidence"] is None, -(r["nearby_same_class_confidence"] or 0.0)))[:limit]
        for i, rec in enumerate(cls_fns):
            img_path = images_dir / rec["image"]
            out_path = gallery_dir / f"{cls_name.lower()}_fn" / f"{i:02d}_{rec['image']}"
            gt_list = gt_by_image.get(rec["image"], [])
            nearby = rec["nearby_same_class_confidence"]
            label = f"FN {cls_name} nearby_conf={nearby:.2f}" if nearby is not None else f"FN {cls_name} total_miss"
            _draw_gallery_image(img_path, out_path, gt_list, None, label, CONDITION_4CLASS_NAMES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolo_solar_condition_4class_research_v3.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/ground_pv_research/condition_4class_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("research_runs/ground_pv/condition_error_analysis_v3"))
    parser.add_argument("--committed-summary-dir", type=Path,
                         default=Path("datasets/ground_pv_research/condition_4class_v1/reports/condition_error_analysis_v3"))
    parser.add_argument("--match-conf", type=float, default=0.20, help="confidence threshold used for TP/FP/FN instance counting")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for the greedy matcher")
    parser.add_argument("--conf-floor", type=float, default=0.001, help="inference confidence floor (captures near-miss signal)")
    parser.add_argument("--gallery-limit", type=int, default=12, help="max gallery images per class/fp-or-fn category")
    parser.add_argument("--limit", type=int, default=None, help="debug: cap number of test images processed")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="inference resolution — MUST match the checkpoint's own training imgsz "
                              "(e.g. 960 for the Phase 15D-A resolution-ablation checkpoint)")
    parser.add_argument("--infer-batch", type=int, default=4,
                         help="CPU inference batch size — lower for higher imgsz to avoid OOM (e.g. 2 for imgsz=960)")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"ERROR: checkpoint not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {args.dataset_root}", file=sys.stderr)
        return 1

    summary = analyze(
        args.weights, args.dataset_root, args.output_dir, args.committed_summary_dir,
        match_conf=args.match_conf, match_iou=args.match_iou, conf_floor=args.conf_floor,
        gallery_limit=args.gallery_limit, limit=args.limit, imgsz=args.imgsz, infer_batch=args.infer_batch,
    )

    print(f"checkpoint sha256: {summary['checkpoint_sha256']}")
    print(f"test images: {summary['num_test_images']}")
    for cls_name, c in summary["per_class"].items():
        print(f"  {cls_name:16s} gt={c['gt_count']:4d} tp={c['tp']:4d} fp={c['fp']:4d} fn={c['fn']:4d} "
              f"P={c['precision_at_match_threshold']} R={c['recall_at_match_threshold']}")
    print(f"data quality flags: {len(summary['data_quality_flags'])}")
    print(f"summary written to: {args.output_dir / 'error_analysis_summary.json'}")
    print(f"committed copy: {args.committed_summary_dir / 'error_analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
