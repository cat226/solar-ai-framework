#!/usr/bin/env python3
"""scripts/build_rooftop_panel_dataset.py - build the Phase 19 rooftop PANEL dataset
(1 class: panel) from the first-party representative manifest + label files.

Split is SCENE-disjoint (whole scenes go to one split), so near-identical frames of
one physical spot can never straddle train/test. Images are resized to a 1280 px long
side (aspect preserved; normalised labels are unchanged) and written as JPEG.

Only images whose annotation_status is annotated_ai_assisted_unverified or
annotated_human_verified are used; excluded/pending images are skipped and reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.representative_dataset import load_source_manifest  # noqa: E402

_REP = _REPO_ROOT / "datasets/ground_pv_research/representative_panel_v1"
USABLE = {"annotated_ai_assisted_unverified", "annotated_human_verified"}
# Scene -> split. Chosen so the wide-view scenes and the bird-drop scene stay in train,
# and the two held-out test scenes differ in camera (video frames vs. video frames at
# a different rooftop position/lighting).
SPLIT_BY_SCENE = {
    "scene0007": "test", "scene0002": "test",
    "scene0008": "valid", "scene0006": "valid",
}
BATCH = "batch_002"
MAX_SIDE = 1280


def split_of(scene_id: str) -> str:
    return SPLIT_BY_SCENE.get(scene_id, "train")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets/ground_pv_research/rooftop_panel_v1")
    args = ap.parse_args(argv)

    entries = [e for e in load_source_manifest(_REP / "manifests/source_manifest.json")
               if (e.capture_batch_id or "").startswith(BATCH)]
    rows, skipped = [], []
    for e in entries:
        lab = _REP / "annotations" / f"{Path(e.original_filename).stem}.txt"
        if e.annotation_status not in USABLE or not lab.is_file() or not lab.read_text().strip():
            skipped.append((e.original_filename, e.annotation_status))
            continue
        split = split_of(e.scene_id)
        src = next((_REP / "raw").rglob(e.original_filename))
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        stem = f"rooftop_{e.image_id}"
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(args.out / "images" / split / f"{stem}.jpg"))
        (args.out / "labels" / split / f"{stem}.txt").write_text(lab.read_text())
        n = sum(1 for l in lab.read_text().splitlines() if l.strip())
        rows.append({"file": f"{stem}.jpg", "original": e.original_filename, "scene_id": e.scene_id,
                     "split": split, "n_boxes": n, "annotation_status": e.annotation_status})

    with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scenes_by_split: dict[str, set] = {}
    for r in rows:
        scenes_by_split.setdefault(r["split"], set()).add(r["scene_id"])
    all_sets = list(scenes_by_split.values())
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            assert not (all_sets[i] & all_sets[j]), "scene leaked across splits"
    (args.out / "data.yaml").write_text(
        f"path: {args.out.as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: panel\n")
    summary = {sp: {"images": sum(r["split"] == sp for r in rows),
                    "boxes": sum(r["n_boxes"] for r in rows if r["split"] == sp),
                    "scenes": sorted(scenes_by_split.get(sp, []))} for sp in ("train", "valid", "test")}
    summary["skipped"] = skipped
    summary["label_provenance"] = "annotated_ai_assisted_unverified (reviewer-corrected SAM proposals); NOT human-verified"
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
