#!/usr/bin/env python3
"""scripts/evaluate_ground_pv.py — Evaluate the ground-level PV research
detector on the validation and/or test split and write per-class metrics.

Outputs:
    reports/ground_research_evaluation.md
    reports/ground_research_evaluation.json

Reports (mandatory): overall mAP50, mAP50-95, precision, recall, and
PER-CLASS mAP50 / mAP50-95 / precision / recall for panel, bird_drop,
cracked, dusty — plus, per split, the number of images, instances per
class, and images containing each class. BirdDrop and Dust are highlighted
because they are rare.

Never modifies weights, config, or ground truth.

Examples
--------
    python scripts/evaluate_ground_pv.py --weights weights/yolo_solar_ground_research_v1.pt \
        --data-yaml datasets/ground_pv_research/yolo/data.yaml --splits val test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

# Fallback only. The actual class names are read from the dataset's own
# data.yaml so this script works for either the 4-class research taxonomy
# or the 3-class first-experiment taxonomy.
RESEARCH_CLASS_ORDER = ("panel", "bird_drop", "cracked", "dusty")
RARE_CLASSES = ("bird_drop", "dusty")
DEFAULT_REPORTS_DIR = _REPO_ROOT / "datasets" / "ground_pv_research" / "reports"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()} or dict(enumerate(RESEARCH_CLASS_ORDER))


def _split_stats(data_yaml: Path, split_key: str) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path", data_yaml.parent))
    rel = data.get(split_key)
    if not rel:
        return {}
    names = _class_names(data_yaml)
    order = [names[i] for i in sorted(names)]
    images_dir = (root / rel) if not Path(rel).is_absolute() else Path(rel)
    labels_dir = Path(str(images_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    inst: Counter[str] = Counter()
    img_with: Counter[str] = Counter()
    for img in images:
        lp = labels_dir / f"{img.stem}.txt"
        if not lp.is_file():
            continue
        seen: set[str] = set()
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid in names:
                inst[names[cid]] += 1
                seen.add(names[cid])
        for name in seen:
            img_with[name] += 1
    return {
        "num_images": len(images),
        "instances_per_class": {c: inst.get(c, 0) for c in order},
        "images_containing_class": {c: img_with.get(c, 0) for c in order},
    }


def _evaluate_split(model, data_yaml: Path, split: str, imgsz: int, iou: float) -> dict:
    ul_split = {"val": "val", "valid": "val", "test": "test"}.get(split, split)
    metrics = model.val(data=str(data_yaml.resolve()), split=ul_split, imgsz=imgsz, iou=iou,
                        plots=True, save_json=False, verbose=False)
    box = metrics.box
    # class index order from ultralytics
    names = metrics.names if hasattr(metrics, "names") else dict(enumerate(RESEARCH_CLASS_ORDER))
    per_class = {}
    try:
        ap_class_index = list(box.ap_class_index)
    except Exception:  # noqa: BLE001
        ap_class_index = list(range(len(RESEARCH_CLASS_ORDER)))
    for pos, cidx in enumerate(ap_class_index):
        cname = names.get(int(cidx), str(cidx)) if isinstance(names, dict) else RESEARCH_CLASS_ORDER[int(cidx)]
        try:
            p, r, ap50, ap = box.class_result(pos)
        except Exception:  # noqa: BLE001
            p = r = ap50 = ap = None
        per_class[cname] = {
            "precision": _f(p), "recall": _f(r), "mAP50": _f(ap50), "mAP50_95": _f(ap),
        }
    return {
        "overall": {
            "precision": _f(box.mp), "recall": _f(box.mr),
            "mAP50": _f(box.map50), "mAP50_95": _f(box.map),
        },
        "per_class": per_class,
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights} (train on Kaggle and download best.pt first)")
    if not args.data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {args.data_yaml}")

    from ultralytics import YOLO
    import ultralytics

    print(f"Evaluating {args.weights} (sha256={_sha256(args.weights)[:16]}...)")
    model = YOLO(str(args.weights))

    results: dict = {
        "weights": str(args.weights),
        "weights_sha256": _sha256(args.weights),
        "data_yaml": str(args.data_yaml),
        "ultralytics_version": ultralytics.__version__,
        "config": {"imgsz": args.imgsz, "iou": args.iou},
        "splits": {},
    }
    for split in args.splits:
        stats = _split_stats(args.data_yaml, {"val": "val", "test": "test"}.get(split, split))
        metrics = _evaluate_split(model, args.data_yaml, split, args.imgsz, args.iou)
        results["splits"][split] = {"dataset_stats": stats, "metrics": metrics}
        o = metrics["overall"]
        print(f"  [{split}] mAP50={o['mAP50']} mAP50-95={o['mAP50_95']} P={o['precision']} R={o['recall']}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "ground_research_evaluation.json"
    # ground_research_evaluation.md is the curated narrative report (kept by
    # hand: metrics + overfitting + cross-domain + verdict). This script
    # writes the machine-generated metrics table to a sibling file so it
    # never clobbers the curated one.
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md_path = args.reports_dir / "ground_research_evaluation_metrics.md"
    md_path.write_text(_render_md(results), encoding="utf-8")
    print(f"\nWrote {md_path}\n      {json_path}")
    return 0


def _render_md(results: dict) -> str:
    lines = ["# Ground-level PV research detector — evaluation\n"]
    lines.append(f"Weights: `{results['weights']}`")
    lines.append(f"SHA-256: `{results['weights_sha256']}`")
    lines.append(f"Ultralytics: {results['ultralytics_version']}  |  imgsz={results['config']['imgsz']}  iou={results['config']['iou']}\n")
    for split, blob in results["splits"].items():
        st = blob["dataset_stats"]
        m = blob["metrics"]
        lines.append(f"## Split: {split}\n")
        lines.append(f"- images: {st.get('num_images')}")
        lines.append(f"- instances per class: {st.get('instances_per_class')}")
        lines.append(f"- images containing class: {st.get('images_containing_class')}\n")
        o = m["overall"]
        lines.append(f"**Overall**: mAP50={o['mAP50']}  mAP50-95={o['mAP50_95']}  P={o['precision']}  R={o['recall']}\n")
        lines.append("| class | precision | recall | mAP50 | mAP50-95 | rare? |")
        lines.append("|---|---|---|---|---|---|")
        for c, pc in m["per_class"].items():
            lines.append(f"| {c} | {pc.get('precision')} | {pc.get('recall')} | {pc.get('mAP50')} | "
                         f"{pc.get('mAP50_95')} | {'**YES**' if c in RARE_CLASSES else ''} |")
        lines.append("")
    lines.append("> `dusty` (and, in the 4-class taxonomy, `bird_drop`) are severely "
                 "underrepresented; their per-class metrics are high-variance and must "
                 "not be over-interpreted.\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
