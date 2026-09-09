#!/usr/bin/env python3
"""scripts/audit_phase12_candidates.py — Phase 12 dataset-acquisition audit.

DISCOVERY / AUDIT ONLY. Does not train anything, does not modify production.

Given locally-sampled candidate datasets (small YOLO-format trees), compute the
domain + annotation + overlap signals used to rank them for representative
ground-level PV panel detection:

  * native resolution + aspect-ratio distribution (portrait / landscape / square)
  * panel-box count per image, box-area distribution, boundary-touching fraction
  * internal perceptual self-redundancy (independent-physical-scene proxy)
  * exact + perceptual overlap vs an existing dataset

Samples are expected under
`datasets/ground_pv_research/phase12_audit/samples/<name>/` (gitignored). The
Kaggle/Roboflow acquisition of those samples is a separate manual step recorded
in `datasets/ground_pv_research/reports/phase12_download_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from training.detection.ground_pv.image_fingerprint import compare, fingerprint_tree  # noqa: E402

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _internal_scene_diversity(fp: dict, *, hamming: int = 6) -> dict:
    """Union-find over exact-SHA + dHash near-dup (excluding identity) -> number
    of distinct near-duplicate clusters = an independent-physical-scene proxy."""
    recs = fp["records"]
    n = len(recs)
    if n == 0:
        return {"images": 0, "distinct_scene_clusters": 0, "redundancy_fraction": 0.0}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_sha: dict[str, int] = {}
    for i, r in enumerate(recs):
        if r["sha256"] in by_sha:
            union(i, by_sha[r["sha256"]])
        else:
            by_sha[r["sha256"]] = i
    H = np.stack([np.frombuffer(bytes.fromhex(r["dhash16"]), dtype=np.uint8) for r in recs])
    for i in range(n):
        d = _POPCOUNT[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
        for off in np.where(d <= hamming)[0]:
            union(i, i + 1 + int(off))
    clusters = len({find(i) for i in range(n)})
    return {
        "images": n,
        "distinct_scene_clusters": clusters,
        "redundancy_fraction": round(1 - clusters / n, 4),
    }


_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EDGE_EPS = 0.01


def _aspect_bucket(w: int, h: int) -> str:
    ar = w / h if h else 1.0
    if ar < 0.9:
        return "portrait"
    if ar > 1.1:
        return "landscape"
    return "square"


def _iter_pairs(root: Path):
    """Yield (image_path, label_path|None) for a flat or split YOLO tree."""
    img_dirs = [p for p in root.rglob("images") if p.is_dir()]
    if not img_dirs and (root / "images").is_dir():
        img_dirs = [root / "images"]
    for idir in img_dirs:
        ldir = idir.parent / "labels"
        for img in sorted(idir.glob("*")):
            if img.suffix.lower() in _IMG_EXT:
                lp = ldir / f"{img.stem}.txt"
                yield img, (lp if lp.is_file() else None)


def domain_stats(root: Path, *, panel_class: int | None = 0) -> dict:
    aspects: Counter[str] = Counter()
    dims: Counter[tuple[int, int]] = Counter()
    ars: list[float] = []
    boxes_per: list[int] = []
    areas: list[float] = []
    edge = 0
    total_boxes = 0
    n_img = 0
    polygon_label_images = 0
    for img, lp in _iter_pairs(root):
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            continue
        n_img += 1
        dims[(w, h)] += 1
        ars.append(round(w / h, 3) if h else 1.0)
        aspects[_aspect_bucket(w, h)] += 1
        nb = 0
        polygon_lines = 0
        if lp is not None:
            for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    cid = int(p[0])
                    vals = [float(v) for v in p[1:]]
                except ValueError:
                    continue
                if panel_class is not None and cid != panel_class:
                    continue
                if len(vals) == 4:
                    cx, cy, bw, bh = vals
                elif len(vals) >= 6 and len(vals) % 2 == 0:  # YOLO-seg polygon -> bbox
                    xs, ys = vals[0::2], vals[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    polygon_lines += 1
                else:
                    continue
                nb += 1
                total_boxes += 1
                areas.append(max(0.0, bw) * max(0.0, bh))
                if (cx - bw / 2 <= EDGE_EPS or cy - bh / 2 <= EDGE_EPS
                        or cx + bw / 2 >= 1 - EDGE_EPS or cy + bh / 2 >= 1 - EDGE_EPS):
                    edge += 1
        boxes_per.append(nb)
        if polygon_lines:
            polygon_label_images += 1
    return {
        "images": n_img,
        "distinct_dimensions": len(dims),
        "top_dimensions": dims.most_common(5),
        "aspect": dict(aspects),
        "polygon_label_images": polygon_label_images,
        "annotation_style": ("segmentation-polygon (convertible to bbox)"
                             if polygon_label_images > n_img * 0.3 else "bounding-box"),
        "aspect_ratio_range": [min(ars), max(ars)] if ars else None,
        "panel_boxes": total_boxes,
        "boxes_per_image": {
            "mean": round(st.mean(boxes_per), 2) if boxes_per else 0,
            "median": st.median(boxes_per) if boxes_per else 0,
            "max": max(boxes_per) if boxes_per else 0,
            "images_with_zero": sum(1 for b in boxes_per if b == 0),
        },
        "box_area_fraction": {
            "median": round(st.median(areas), 4) if areas else None,
            "gt_0_25": sum(1 for a in areas if a > 0.25),
            "gt_0_5": sum(1 for a in areas if a > 0.5),
            "n": len(areas),
        },
        "boundary_touching_boxes": edge,
        "boundary_touching_pct": round(100 * edge / max(1, total_boxes), 1),
    }


def audit(samples_dir: Path, existing_dir: Path | None) -> dict:
    existing_fp = fingerprint_tree(existing_dir, label="existing") if existing_dir and existing_dir.is_dir() else None
    out: dict = {"samples_dir": str(samples_dir), "existing_dir": str(existing_dir) if existing_dir else None, "candidates": {}}
    for d in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
        fp = fingerprint_tree(d, label=d.name)
        rec: dict = {"domain": domain_stats(d), "images_fingerprinted": fp["count"], "corrupt": len(fp["corrupt"])}
        div = _internal_scene_diversity(fp)
        rec["internal_redundancy_fraction"] = div["redundancy_fraction"]
        rec["independent_scene_proxy"] = div["distinct_scene_clusters"]
        if existing_fp is not None:
            c = compare(fp, existing_fp, near_dup_hamming=6)
            rec["overlap_vs_existing"] = {
                "exact": c["exact_a_images"],
                "shared_basename": c["shared_basename_a_images"],
                "perceptual_near_dup": c["perceptual_near_dup_a_images"],
                "any_signal": c["a_images_overlapping_by_any_signal"],
                "fraction": round(c["a_images_overlap_fraction"], 4),
            }
        out["candidates"][d.name] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "phase12_audit" / "samples")
    ap.add_argument("--existing-dir", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "yolo")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "datasets" / "ground_pv_research" / "reports" / "phase12_sample_domain_audit.json")
    args = ap.parse_args(argv)
    if not args.samples_dir.is_dir():
        raise SystemExit(f"no samples dir: {args.samples_dir} (sample candidates first)")
    res = audit(args.samples_dir, args.existing_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: {"images": v["domain"]["images"], "aspect": v["domain"]["aspect"],
                          "panel_boxes": v["domain"]["panel_boxes"],
                          "internal_redundancy": v["internal_redundancy_fraction"],
                          "independent_scene_proxy": v["independent_scene_proxy"],
                          "overlap_vs_existing": v.get("overlap_vs_existing", {}).get("fraction")}
                      for k, v in res["candidates"].items()}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
