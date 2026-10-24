#!/usr/bin/env python3
"""scripts/build_unified_condition_dataset.py — build ONE deduplicated,
family-split 4-class (CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE) image-level
condition dataset from the three public RGB sources the owner supplied in
Oct-2026:

  1. Faulty_solar_panel        (pythonafroz root set, 6 folders)
  2. Roboflow ahmed-tamer/solar-panels-dust-detection v2 (Clean/Dusty, 3x augmented)
  3. GitHub suhasmaddali/Solar-Panels-Dust-Detection (Clean/Dusty, LFS)

These re-ship the same photos, so independence is established by content
FAMILY (orientation-invariant dHash within 4 bits), and the train/valid/test
split is made per family. Electrical-damage / Snow-Covered are deliberately
excluded (rejected taxonomy classes). Cross-source label conflicts are
dropped. The single-panel macro photo session is tagged and excluded by
default (wrong domain for a rooftop camera).

Bulk images go to the data drive (default F:); small manifests/reports go in
the repo under datasets/ground_pv_research/unified_condition_v1/.
Research-only: every source has an unresolved licence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from training.detection.ground_pv import unified_condition_pool as ucp  # noqa: E402
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

_EXT = {".jpg", ".jpeg", ".png"}
_REPORT_DIR = _REPO / "datasets" / "ground_pv_research" / "unified_condition_v1"
_MAX_SIDE_OUT = 800


def _fingerprint(path_str: str):
    p = Path(path_str)
    try:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            im.load()
            return path_str, (sha, im.width, im.height, ucp.canon_hash(im)), None
    except Exception as exc:  # noqa: BLE001
        return path_str, None, repr(exc)[:120]


def scan_faulty(root: Path) -> list[ucp.Rec]:
    recs = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _EXT:
            folder = p.relative_to(root).parts[0]
            recs.append(ucp.Rec(str(p), "faulty", folder, ucp.faulty_label(folder), ucp.lineage_of("faulty", p.name)))
    return recs


def scan_github(root: Path) -> tuple[list[ucp.Rec], Counter]:
    """Originals only: top-level Clean/ Dusty/ (raw) and Train/ Test/ re-cuts.
    Augmented_images, *_Discarded, *_temporary, *_with_Labels are skipped."""
    recs, skipped = [], Counter()
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        parts = p.relative_to(root).parts
        top = parts[0]
        if "Augmented_images" in parts:
            skipped["github_augmented"] += 1
            continue
        if top.endswith(("_Discarded", "_temporary", "_with_Labels")):
            skipped["github_" + top.split("_", 1)[1].lower()] += 1
            continue
        if top in ("Train", "Test") and len(parts) >= 3:
            source, folder = "github_split", parts[1]
        elif top in ("Clean", "Dusty") and len(parts) == 2:
            source, folder = "github_raw", top
        else:
            skipped["github_unrecognized"] += 1
            continue
        label = ucp.binary_label(folder)
        if label is None:
            skipped["github_unrecognized"] += 1
            continue
        recs.append(ucp.Rec(str(p), source, folder, label, ucp.lineage_of(source, p.name)))
    return recs, skipped


def scan_roboflow(root: Path) -> list[ucp.Rec]:
    recs = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in _EXT):
            continue
        parts = p.relative_to(root).parts
        if len(parts) < 3 or parts[0] not in ("train", "valid", "test"):
            continue
        label = ucp.binary_label(parts[1])
        if label is None:
            continue
        source = "roboflow_train" if parts[0] == "train" else "roboflow_eval"
        recs.append(ucp.Rec(str(p), source, parts[1], label, ucp.lineage_of(source, p.name),
                            extra={"rf_split": parts[0], "rf_base": ucp.roboflow_base(p.name)}))
    return recs


def fingerprint_all(recs: list[ucp.Rec], cache_path: Path, workers: int) -> list[ucp.Rec]:
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:  # noqa: BLE001
            cache = {}
    todo = []
    for r in recs:
        st = Path(r.path).stat()
        key = (st.st_size, st.st_mtime_ns)
        hit = cache.get(r.path)
        if hit and hit[0] == key:
            r.sha256, r.w, r.h, r.chash = hit[1]
        else:
            todo.append(r)
    print(f"fingerprint: {len(recs) - len(todo)} cached, {len(todo)} to compute ({workers} workers)", flush=True)
    by_path = {r.path: r for r in todo}
    bad = []
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (path, fp, err) in enumerate(ex.map(_fingerprint, [r.path for r in todo], chunksize=16), 1):
                r = by_path[path]
                if fp is None:
                    r.exclude_reason = "undecodable"
                    bad.append((path, err))
                else:
                    r.sha256, r.w, r.h, r.chash = fp
                    st = Path(path).stat()
                    cache[path] = ((st.st_size, st.st_mtime_ns), fp)
                if i % 2000 == 0:
                    print(f"  ... {i}/{len(todo)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
    if bad:
        print(f"undecodable: {len(bad)} (e.g. {bad[:2]})")
    return [r for r in recs if not r.exclude_reason]


def write_image(src: str, dst: Path) -> None:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((_MAX_SIDE_OUT, _MAX_SIDE_OUT), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=92)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faulty", type=Path, default=Path("F:/Faulty_solar_panel"))
    ap.add_argument("--roboflow", type=Path, default=Path("F:/Roboflow_candidates/ahmed-tamer_solar-panels-dust-detection/v2/data"))
    ap.add_argument("--github", type=Path, default=Path("F:/Github_candidates/suhasmaddali_Solar-Panels-Dust-Detection/Detect_solar_dust"))
    ap.add_argument("--out-dir", type=Path, default=Path("F:/Solar AI Datasets/condition_unified_v1"))
    ap.add_argument("--report-dir", type=Path, default=_REPORT_DIR)
    ap.add_argument("--include-lineage", action="append", default=[],
                    help="re-include an excluded lineage (e.g. phone2021_macro)")
    ap.add_argument("--max-dist", type=int, default=ucp.DEFAULT_MAX_DIST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="compute manifest + report, do not write images")
    args = ap.parse_args(argv)

    for label, p in (("faulty", args.faulty), ("roboflow", args.roboflow), ("github", args.github)):
        if not p.exists():
            sys.exit(f"source missing ({label}): {p}")

    recs = scan_faulty(args.faulty)
    gh, gh_skipped = scan_github(args.github)
    recs += gh + scan_roboflow(args.roboflow)
    n_scanned = Counter(r.source for r in recs)
    print("scanned:", dict(n_scanned), "github skipped:", dict(gh_skipped))

    recs = fingerprint_all(recs, args.out_dir / "_fingerprint_cache.pkl", args.workers)
    fam = ucp.group_families([r.chash for r in recs], max_dist=args.max_dist)
    for r, f in zip(recs, fam):
        r.family = f
    print(f"{len(recs)} decodable images -> {len(set(fam))} content families")

    excluded_lineages = ucp.DEFAULT_EXCLUDED_LINEAGES - set(args.include_lineage)
    kept, dropped = ucp.resolve_families(recs, excluded_lineages=frozenset(excluded_lineages))
    splits = ucp.assign_splits({f: v["label"] for f, v in kept.items()}, seed=args.seed)

    rows, ledger = [], []
    for f, v in sorted(kept.items()):
        rep = v["rep"]
        srcs = sorted({m.source.split("_")[0] for m in v["members"]})
        image_id = f"{rep.source.split('_')[0]}_{rep.sha256[:12]}"
        rel = f"{splits[f]}/{v['label']}/{image_id}.jpg"
        rows.append({
            "image_id": image_id, "family": f, "split": splits[f], "label": v["label"], "lineage": v["lineage"],
            "rep_source": rep.source, "rep_path": rep.path, "sources_in_family": "+".join(srcs),
            "n_members": len(v["members"]), "w": rep.w, "h": rep.h, "sha256": rep.sha256, "out_rel": rel,
        })
    ucp.assert_no_family_straddle(rows)
    ucp.assert_active_labels(r["label"] for r in rows)
    for r in recs:
        reason = dropped.get(r.family, "")
        in_kept = r.family in kept
        ledger.append({"path": r.path, "source": r.source, "source_class": r.source_class, "family": r.family,
                       "sha256": r.sha256, "w": r.w, "h": r.h, "lineage": r.lineage,
                       "status": "kept_family" if in_kept else "dropped_family", "reason": reason})

    # ---- report numbers
    def cnt(key, filt=lambda x: True):
        return dict(Counter(r[key] for r in rows if filt(r)))
    per_split_class = {s: {c: sum(1 for r in rows if r["split"] == s and r["label"] == c) for c in CONDITION_4CLASS_ORDER}
                       for s in ("train", "valid", "test")}
    drop_images = Counter()
    drop_fams = Counter(dropped.values())
    for r in recs:
        if r.family in dropped:
            drop_images[dropped[r.family]] += 1
    multi_src = Counter(r["sources_in_family"] for r in rows)
    lineage_by_class = {c: dict(Counter(r["lineage"] for r in rows if r["label"] == c)) for c in CONDITION_4CLASS_ORDER}
    report = {
        "schema_version": ucp.SCHEMA_VERSION, "seed": args.seed, "max_hamming_bits": args.max_dist,
        "min_side_px": ucp.MIN_SIDE_PX, "excluded_lineages": sorted(excluded_lineages),
        "scanned_by_source": dict(n_scanned), "github_skipped_non_original": dict(gh_skipped),
        "decodable_images": len(recs), "families_total": len(set(fam)),
        "families_kept": len(kept), "coarse_to_fine_overrides": dict(Counter(
            v["label"] for v in kept.values() if v["coarse_to_fine_override"])),
        "families_dropped": dict(drop_fams), "images_in_dropped_families": dict(drop_images),
        "kept_by_class": cnt("label"), "kept_by_split": cnt("split"), "kept_per_split_class": per_split_class,
        "kept_by_lineage": cnt("lineage"), "lineage_by_class": lineage_by_class,
        "families_by_contributing_sources": dict(multi_src),
        "licence_status": "UNRESOLVED for all three sources (Kaggle pythonafroz 'Unknown'; Roboflow CC BY 4.0 is a "
                          "re-declaration; GitHub repo has no licence) - research only, do not redistribute",
        "out_dir": str(args.out_dir),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "manifest_representatives.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.report_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "ledger_all_files.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ledger[0]))
            w.writeheader()
            w.writerows(ledger)
        for i, r in enumerate(rows, 1):
            write_image(r["rep_path"], args.out_dir / r["out_rel"])
            if i % 1000 == 0:
                print(f"  wrote {i}/{len(rows)}", flush=True)
        (args.out_dir / "CLASS_ORDER.json").write_text(json.dumps({
            "canonical_condition_order": list(CONDITION_4CLASS_ORDER),
            "imagefolder_alphabetical_order": sorted(CONDITION_4CLASS_ORDER),
            "note": "torchvision ImageFolder indexes classes alphabetically; map to canonical order downstream."}, indent=2))

    print(json.dumps({k: report[k] for k in ("families_total", "families_kept", "coarse_to_fine_overrides", "kept_by_class",
                                               "kept_per_split_class", "families_dropped", "kept_by_lineage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
