#!/usr/bin/env python3
"""scripts/audit_candidate_ubwtz.py — Phase 13B: audit the downloaded Roboflow
candidate `solar-panel-detection-bwpkx/solar-panel-detection-ubwtz` (v2, native
resolution export) BEFORE any use in training.

Produces two committed artifacts:
  * datasets/ground_pv_research/candidate_acquisition/
        solar_panel_detection_ubwtz_download_manifest.json
  * datasets/ground_pv_research/candidate_acquisition/
        solar_panel_detection_ubwtz_audit.json

DISCOVERY / AUDIT ONLY. Does not modify the raw download, does not train, does
not touch production. The raw download stays immutable.

Provenance: the Roboflow re-uploader displays "CC BY 4.0" but states no original
image source. Per the Phase 7/12 rule that does NOT establish image-level rights
-> tag `personal-experimental-unverified-provenance`, research-only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.image_fingerprint import fingerprint_tree  # noqa: E402

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _sha256(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _capture_key(name: str) -> str:
    """Group frames by their capture event: a video sequence id or a photo
    timestamp. Roboflow appends `_jpg.rf.<hash>` which is stripped first."""
    stem = re.split(r"_jpe?g\.rf\.|\.rf\.", name)[0]
    m = re.match(r"(\d{8})[_-](\d{6})(?:[_-]([A-Za-z0-9]+?)[-_](\d+))?", stem)
    if m:
        date, tm, vid, _frame = m.groups()
        return f"{date}_{tm}_{vid}" if vid else f"{date}_{tm}"
    return stem


def _iter(root: Path):
    for split in ("train", "valid", "test"):
        idir = root / split / "images"
        ldir = root / split / "labels"
        if not idir.is_dir():
            continue
        for img in sorted(idir.glob("*")):
            if img.suffix.lower() in _IMG_EXT:
                yield split, img, ldir / f"{img.stem}.txt"


def audit(root: Path) -> dict:
    import yaml
    meta = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    rf = meta.get("roboflow", {})

    dims = collections.Counter()
    ars: list[float] = []
    per_split = collections.Counter()
    capture_events = collections.Counter()
    boxes_per_img: list[int] = []
    areas: list[float] = []
    edge_boxes = 0
    n_polys = n_valid = n_zero = n_scene_span = n_malformed = n_odd = 0
    empty_imgs = 0
    total_bytes = 0
    n_img = 0

    for split, img, lp in _iter(root):
        n_img += 1
        per_split[split] += 1
        total_bytes += img.stat().st_size
        with Image.open(img) as im:
            w, h = im.size
        dims[(w, h)] += 1
        ars.append(round(w / h, 3))
        capture_events[_capture_key(img.name)] += 1
        nb = 0
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()] if lp.is_file() else []
        if not lines:
            empty_imgs += 1
        for ln in lines:
            parts = ln.split()
            try:
                _cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                n_malformed += 1
                continue
            if len(coords) == 4:  # already a bbox
                cx, cy, bw, bh = coords
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                n_polys += 1
                xs, ys = coords[0::2], coords[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            else:
                n_odd += 1
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw * bh <= 1e-6:
                n_zero += 1
                continue
            if bw > 0.92 and bh > 0.92:
                n_scene_span += 1
            n_valid += 1
            nb += 1
            areas.append(bw * bh)
            if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
                edge_boxes += 1
        boxes_per_img.append(nb)

    # perceptual self-redundancy (strict + loose) + capture-event grouping
    fp = fingerprint_tree(root, label="candidate")
    recs = fp["records"]
    n = len(recs)
    H = np.stack([np.frombuffer(bytes.fromhex(r["dhash16"]), dtype=np.uint8) for r in recs])
    by_sha = {}
    dup_exact = 0
    for i, r in enumerate(recs):
        if r["sha256"] in by_sha:
            dup_exact += 1
        else:
            by_sha[r["sha256"]] = i

    def _clusters(ham: int) -> int:
        p = list(range(n))

        def f(x):
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        for i in range(n):
            d = _POP[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
            for off in np.where(d <= ham)[0]:
                a, b = f(i), f(i + 1 + int(off))
                if a != b:
                    p[a] = b
        return len({f(i) for i in range(n)})

    video_events = {k: v for k, v in capture_events.items() if re.search(r"_(mp4|MP4|mov|MOV)$", k)}
    photo_events = {k: v for k, v in capture_events.items() if k not in video_events}
    dates = collections.Counter(k[:8] for k in capture_events)

    port = sum(1 for a in ars if a < 0.9)
    land = sum(1 for a in ars if a > 1.1)

    return {
        "source": {
            "roboflow_workspace": rf.get("workspace"),
            "roboflow_project": rf.get("project"),
            "roboflow_version": rf.get("version"),
            "roboflow_url": rf.get("url"),
            "license_shown": rf.get("license"),
            "license_source": "Roboflow re-uploader dataset-level claim - NO original image source stated",
            "export_format": "yolov8 (segmentation polygons)",
            "export_resize": "none (native resolution preserved)",
        },
        "provenance_status": "UNRESOLVED - personal-experimental-unverified-provenance, research-only. "
                             "NOT publication/redistribution/commercial/production cleared.",
        "counts": {
            "images": n_img,
            "per_split": dict(per_split),
            "empty_label_images": empty_imgs,
            "polygon_annotations": n_polys,
            "valid_annotations_to_bbox": n_valid,
            "zero_area": n_zero,
            "scene_spanning_gt92pct": n_scene_span,
            "malformed": n_malformed,
            "odd_coord_count": n_odd,
            "exact_duplicate_images": dup_exact,
            "total_image_bytes": total_bytes,
        },
        "geometry": {
            "distinct_dimensions": len(dims),
            "top_dimensions": [[list(k), v] for k, v in dims.most_common(8)],
            "aspect_ratio_range": [min(ars), max(ars)],
            "portrait_images": port,
            "landscape_images": land,
            "square_images": n_img - port - land,
            "native_aspect_preserved": True,
            "boxes_per_image": {"mean": round(st.mean(boxes_per_img), 2),
                                "median": st.median(boxes_per_img), "max": max(boxes_per_img)},
            "bbox_area_fraction": {
                "median": round(st.median(areas), 3),
                "p10": round(float(np.percentile(areas, 10)), 3),
                "p90": round(float(np.percentile(areas, 90)), 3),
                "gt_0_25": sum(1 for a in areas if a > 0.25),
                "gt_0_5": sum(1 for a in areas if a > 0.5),
            },
            "boundary_touching_boxes": edge_boxes,
        },
        "scene_analysis": {
            "capture_dates": dict(dates),
            "distinct_capture_events": len(capture_events),
            "video_sequences": {"count": len(video_events), "frames": sum(video_events.values()),
                                "sequences": video_events},
            "photo_events": {"count": len(photo_events), "images": sum(photo_events.values())},
            "largest_capture_event": capture_events.most_common(1)[0],
            "perceptual_clusters_dhash16": {f"hamming_le_{h}": _clusters(h) for h in (4, 6, 8, 12)},
            "verdict": "ONE physical installation, ONE capture session (all frames dated "
                       f"{list(dates)[0]}, timestamps within a few minutes). {len(video_events)} video "
                       f"sequences ({sum(video_events.values())} frames) + burst photos. Strict dHash "
                       "counts every frame distinct, but INDEPENDENT PHYSICAL SCENES ~= 1. "
                       "Frames must NOT be treated as independent examples; the whole candidate belongs "
                       "in ONE split (train only).",
        },
        "domain_fit_vs_owner_photos": {
            "owner_photos": "6x phone (5 portrait 573x1280 ar 0.448, 1 landscape 1280x573 ar 2.234)",
            "candidate": f"{port} portrait / {land} landscape, ar {min(ars)}-{max(ars)}, "
                         "native (non-square) - the FIRST ground-level non-square PV panel data we have; "
                         "but a single installation, so it broadens aspect-ratio coverage, not scene coverage",
        },
    }


def build_download_manifest(root: Path) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {
        "artifact": "solar-panel-detection-bwpkx/solar-panel-detection-ubwtz v2 (native-resolution yolov8 export)",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "download_root": str(root.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "roboflow_url": "https://universe.roboflow.com/solar-panel-detection-bwpkx/solar-panel-detection-ubwtz/dataset/2",
        "immutable": True,
        "provenance_tag": "personal-experimental-unverified-provenance",
        "provenance_note": "Roboflow displays 'CC BY 4.0' but states no original image source. "
                           "Research-only. NOT redistribution/publication/commercial/production cleared.",
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "sha256": {str(p.relative_to(root)).replace("\\", "/"): _sha256(p)
                   for p in files if p.suffix.lower() in _IMG_EXT or p.name == "data.yaml"
                   or p.suffix == ".txt"},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = _REPO_ROOT / "datasets" / "ground_pv_research" / "candidate_acquisition"
    ap.add_argument("--root", type=Path, default=base / "solar_panel_detection_ubwtz" / "v2_native_yolov8")
    ap.add_argument("--out-audit", type=Path, default=base / "solar_panel_detection_ubwtz_audit.json")
    ap.add_argument("--out-manifest", type=Path, default=base / "solar_panel_detection_ubwtz_download_manifest.json")
    args = ap.parse_args(argv)
    if not (args.root / "data.yaml").is_file():
        raise SystemExit(f"candidate not found at {args.root} - download it first")

    args.out_manifest.write_text(json.dumps(build_download_manifest(args.root), indent=2), encoding="utf-8")
    res = audit(args.root)
    args.out_audit.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({
        "images": res["counts"]["images"],
        "per_split": res["counts"]["per_split"],
        "polygons_valid": f'{res["counts"]["valid_annotations_to_bbox"]}/{res["counts"]["polygon_annotations"]}',
        "rejected": {k: res["counts"][k] for k in ("zero_area", "scene_spanning_gt92pct", "malformed", "odd_coord_count")},
        "portrait/landscape": f'{res["geometry"]["portrait_images"]}/{res["geometry"]["landscape_images"]}',
        "distinct_capture_events": res["scene_analysis"]["distinct_capture_events"],
        "independent_scenes": "~1 installation, 1 session",
    }, indent=2))
    print(f"\nwrote {args.out_manifest}\n      {args.out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
