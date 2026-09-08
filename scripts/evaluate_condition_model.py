#!/usr/bin/env python3
"""scripts/evaluate_condition_model.py — evaluate the EXPERIMENTAL ground-PV
condition/defect detector on the held-out leakage-clean test split (and,
optionally, validation).

Taxonomy: 0=CLEAN 1=DUSTY 2=BIRD_DROP 3=PHYSICAL_DAMAGE 4=ELECTRICAL_DAMAGE.

Reports overall + per-class precision / recall / AP50 / AP50-95, and per class
the test image count, test instance count and unique source-family count
(from the bundled split manifest). ELECTRICAL_DAMAGE is reported separately and
prominently with its HIGH semantic-risk warning (S1 electrical-damage boxes
visually correspond to cell cracks — a high score is NOT electrical-fault
diagnosis).

Never modifies weights, config, ground truth or the dataset.

    python scripts/evaluate_condition_model.py \
        --weights weights/yolo_solar_condition_research_v1.pt \
        --data-yaml datasets/ground_pv_research/condition/data.yaml \
        --splits test val
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

CONDITION_ORDER = ("CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE", "ELECTRICAL_DAMAGE")
HIGH_RISK_CLASS = "ELECTRICAL_DAMAGE"
DEFAULT_REPORTS = _REPO_ROOT / "datasets" / "ground_pv_research" / "condition" / "reports"
DEFAULT_SPLIT_MANIFEST = _REPO_ROOT / "datasets" / "ground_pv_research" / "condition" / "splits" / "condition_split_manifest.csv"
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f(v) -> float | None:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _split_dirs(data_yaml: Path, split_key: str) -> tuple[Path, Path] | None:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    rel = data.get(split_key)
    if not rel:
        return None
    path_val = str(data.get("path", "."))
    # the committed data.yaml `path:` is repo-relative and can be stale; try a
    # few candidate roots and use whichever actually contains the split dir.
    candidates = [
        data_yaml.parent,
        Path(path_val) if Path(path_val).is_absolute() else _REPO_ROOT / path_val,
        data_yaml.parent / path_val,
    ]
    images_dir = None
    for root in candidates:
        cand = (root / rel).resolve()
        if cand.is_dir():
            images_dir = cand
            break
    if images_dir is None:
        images_dir = (data_yaml.parent / rel).resolve()
    # condition dataset nests images/<split>/images + images/<split>/labels;
    # the label dir is the image dir's sibling "labels", else the flat
    # ".../labels/<split>" YOLO convention.
    if images_dir.name == "images":
        labels_dir = images_dir.parent / "labels"
    else:
        labels_dir = Path(str(images_dir).replace(f"{'/'}images", f"{'/'}labels")
                          .replace("\\images", "\\labels"))
    return images_dir, labels_dir


def _dataset_stats(data_yaml: Path, split_key: str, names: dict[int, str]) -> dict:
    dirs = _split_dirs(data_yaml, split_key)
    if not dirs:
        return {}
    images_dir, labels_dir = dirs
    order = [names[i] for i in sorted(names)]
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in _IMG_EXT]
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
        for n in seen:
            img_with[n] += 1
    return {
        "num_images": len(images),
        "instances_per_class": {c: inst.get(c, 0) for c in order},
        "images_containing_class": {c: img_with.get(c, 0) for c in order},
    }


def _family_counts(split_manifest: Path) -> dict[str, dict[str, int]]:
    """derived_split -> unified_class -> unique family count."""
    out: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    if not split_manifest.is_file():
        return {}
    with split_manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = row["derived_split"]
            for c in row["unified_classes"].split(";"):
                if c and c != "(none)":
                    out[s][c].add(row["family_id"])
    return {s: {c: len(f) for c, f in d.items()} for s, d in out.items()}


def _evaluate(model, data_yaml: Path, split: str, imgsz: int, iou: float, names: dict[int, str]) -> dict:
    ul = {"val": "val", "valid": "val", "test": "test"}.get(split, split)
    m = model.val(data=str(data_yaml.resolve()), split=ul, imgsz=imgsz, iou=iou,
                  plots=True, save_json=False, verbose=False)
    box = m.box
    per_class = {}
    try:
        idx = list(box.ap_class_index)
    except Exception:  # noqa: BLE001
        idx = list(range(len(names)))
    for pos, cidx in enumerate(idx):
        cname = names.get(int(cidx), str(cidx))
        try:
            p, r, ap50, ap = box.class_result(pos)
        except Exception:  # noqa: BLE001
            p = r = ap50 = ap = None
        per_class[cname] = {"precision": _f(p), "recall": _f(r), "AP50": _f(ap50), "AP50_95": _f(ap)}
    return {
        "overall": {"precision": _f(box.mp), "recall": _f(box.mr),
                    "mAP50": _f(box.map50), "mAP50_95": _f(box.map)},
        "per_class": per_class,
        "save_dir": str(getattr(m, "save_dir", "")),
    }


def _render_md(results: dict) -> str:
    L = ["# Ground-PV CONDITION/DEFECT detector — evaluation\n",
         "> EXPERIMENTAL research model. Second-stage condition detector (runs on panel",
         "> crops). NOT a panel detector, NOT production. Provenance grade C "
         "(`personal-experimental-unverified-provenance`).\n",
         f"Weights: `{results['weights']}`  ",
         f"SHA-256: `{results['weights_sha256']}`  ",
         f"Ultralytics {results['ultralytics_version']} · imgsz={results['config']['imgsz']} · iou={results['config']['iou']}\n"]
    for split, blob in results["splits"].items():
        st = blob["dataset_stats"]
        m = blob["metrics"]
        fam = blob.get("unique_families_per_class", {})
        o = m["overall"]
        L.append(f"## Split: {split}\n")
        L.append(f"- images: {st.get('num_images')}\n")
        L.append(f"**Overall** — precision {o['precision']} · recall {o['recall']} · "
                 f"mAP50 {o['mAP50']} · mAP50-95 {o['mAP50_95']}\n")
        L.append("| class | precision | recall | AP50 | AP50-95 | test imgs w/ class | test instances | unique families |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in CONDITION_ORDER:
            pc = m["per_class"].get(c, {})
            L.append(f"| {c} | {pc.get('precision')} | {pc.get('recall')} | {pc.get('AP50')} | "
                     f"{pc.get('AP50_95')} | {st.get('images_containing_class', {}).get(c)} | "
                     f"{st.get('instances_per_class', {}).get(c)} | {fam.get(c)} |")
        L.append("")
        if split in ("test", "val"):
            pc = m["per_class"].get(HIGH_RISK_CLASS, {})
            L.append(f"### {HIGH_RISK_CLASS} — ⚠️ HIGH SEMANTIC RISK ({split})\n")
            L.append("Every ELECTRICAL_DAMAGE label is S1's `electrical-damage` class. The provenance")
            L.append("audit found these boxes visually correspond to **cell cracks**, not confirmed")
            L.append("electrical failure. Labels were preserved unchanged (not relabelled). **A high")
            L.append("score here is NOT an electrical-fault diagnosis** — it may simply reflect the")
            L.append("source dataset labelling cell cracks as `electrical-damage`.\n")
            L.append(f"- precision {pc.get('precision')} · recall {pc.get('recall')} · "
                     f"AP50 {pc.get('AP50')} · AP50-95 {pc.get('AP50_95')}")
            L.append(f"- {split} images with class: {st.get('images_containing_class', {}).get(HIGH_RISK_CLASS)} · "
                     f"instances: {st.get('instances_per_class', {}).get(HIGH_RISK_CLASS)} · "
                     f"unique families: {blob.get('unique_families_per_class', {}).get(HIGH_RISK_CLASS)}\n")
    L.append("> CLEAN: trained only on explicit clean-region annotations. Absence of a defect")
    L.append("> detection is NOT proof of Clean. PHYSICAL_DAMAGE / ELECTRICAL_DAMAGE families are")
    L.append("> few — per-class metrics are high-variance; do not over-interpret.\n")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    ap.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    ap.add_argument("--copy-plots-to", type=Path, default=None,
                    help="Directory to copy the val()/test() confusion matrix + PR/F1 curves into.")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights} (download best.pt from Kaggle first)")
    if not args.data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {args.data_yaml}")

    from ultralytics import YOLO
    import ultralytics

    names = _class_names(args.data_yaml)
    fam_counts = _family_counts(args.split_manifest)
    model = YOLO(str(args.weights))
    print(f"Evaluating {args.weights} (sha256={_sha256(args.weights)[:16]}...)  classes={names}")

    results: dict = {
        "weights": str(args.weights),
        "weights_sha256": _sha256(args.weights),
        "data_yaml": str(args.data_yaml),
        "ultralytics_version": ultralytics.__version__,
        "config": {"imgsz": args.imgsz, "iou": args.iou},
        "taxonomy": names,
        "electrical_damage_semantic_risk": "HIGH",
        "splits": {},
    }
    for split in args.splits:
        sk = {"val": "val", "test": "test"}.get(split, split)
        stats = _dataset_stats(args.data_yaml, sk, names)
        metrics = _evaluate(model, args.data_yaml, split, args.imgsz, args.iou, names)
        fam = fam_counts.get({"val": "valid", "test": "test"}.get(split, split), {})
        results["splits"][split] = {
            "dataset_stats": stats, "metrics": metrics, "unique_families_per_class": fam,
        }
        o = metrics["overall"]
        print(f"  [{split}] mAP50={o['mAP50']} mAP50-95={o['mAP50_95']} P={o['precision']} R={o['recall']}")
        ed = metrics["per_class"].get(HIGH_RISK_CLASS, {})
        print(f"  [{split}] ELECTRICAL_DAMAGE (HIGH RISK): P={ed.get('precision')} R={ed.get('recall')} "
              f"AP50={ed.get('AP50')} AP50-95={ed.get('AP50_95')}")
        if args.copy_plots_to and metrics.get("save_dir") and Path(metrics["save_dir"]).is_dir():
            dst = args.copy_plots_to / f"eval_{split}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in Path(metrics["save_dir"]).glob("*"):
                if f.suffix.lower() in {".png", ".jpg", ".csv"}:
                    shutil.copy2(f, dst / f.name)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "condition_evaluation.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (args.reports_dir / "condition_evaluation.md").write_text(_render_md(results), encoding="utf-8")
    print(f"\nWrote {args.reports_dir / 'condition_evaluation.md'}\n      {args.reports_dir / 'condition_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
