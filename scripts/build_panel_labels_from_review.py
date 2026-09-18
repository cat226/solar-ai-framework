#!/usr/bin/env python3
"""scripts/build_panel_labels_from_review.py - turn recorded review decisions into
YOLO panel labels + manifest status.

Input: a decisions JSON (see docs/PANEL_REVIEW_DECISIONS_FORMAT below) and the
`<stem>.filtered.json` files from scripts/render_panel_proposal_review.py.

    {
      "<image stem>": {
        "keep": [0, 2, 3],                       # accepted SAM candidate ids
        "add":  [[x1, y1, x2, y2], ...],         # hand-entered boxes, normalised 0-1
        "note": "free text"
      }, ...
    }

Output: `<annotations-dir>/<stem>.txt` (class 0 = panel, one box per physical
panel) and, in the source manifest, annotation_status =
`annotated_ai_assisted_unverified` for every image with a decision.

These labels are reviewer-corrected proposals, NOT human-verified ground truth,
and are recorded as such. They never touch `annotated_human_verified`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv.representative_dataset import (  # noqa: E402
    load_source_manifest,
    save_source_manifest,
)

STATUS = "annotated_ai_assisted_unverified"
REFINE_MIN_IOU = 0.55  # a SAM-refined box must stay this close to the reviewer's rough box


def _clip(b: list[float]) -> list[float]:
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in b)
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def refine_with_sam(sam_model, image_path: Path, rough: list[list[float]]) -> tuple[list[list[float]], list[str]]:
    """Snap reviewer-drawn rough boxes to SAM mask extents. A refined box is used only
    if it overlaps the rough box (IoU >= REFINE_MIN_IOU); otherwise the rough box stands."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = 1600 / max(h, w) if max(h, w) > 1600 else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    sh, sw = img.shape[:2]
    prompts = [[b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh] for b in rough]
    res = sam_model(img, bboxes=prompts, verbose=False)[0]
    out, log = [], []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else []
    for i, b in enumerate(rough):
        if i >= len(masks):
            out.append(b); log.append("no_mask_kept_rough"); continue
        ys, xs = np.nonzero(masks[i])
        if len(xs) == 0:
            out.append(b); log.append("empty_mask_kept_rough"); continue
        mh, mw = masks[i].shape
        ref = [xs.min() / mw, ys.min() / mh, (xs.max() + 1) / mw, (ys.max() + 1) / mh]
        iou = _iou(b, ref)
        if iou >= REFINE_MIN_IOU:
            out.append(_clip(ref)); log.append(f"refined_iou_{iou:.2f}")
        else:
            out.append(b); log.append(f"kept_rough_refined_iou_only_{iou:.2f}")
    return out, log


def build(decisions: dict, filtered_dir: Path, annotations_dir: Path,
          sam_model=None, images_dir: Path | None = None) -> tuple[dict[str, int], dict[str, list[str]]]:
    annotations_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    refine_log: dict[str, list[str]] = {}
    by_stem = {p.stem: p for p in images_dir.rglob("*")} if images_dir else {}
    for stem, d in decisions.items():
        if stem.startswith("_"):
            continue
        fpath = filtered_dir / f"{stem}.filtered.json"
        by_id = {}
        if fpath.is_file():
            by_id = {c["id"]: c for c in json.loads(fpath.read_text(encoding="utf-8"))["candidates"]}
        boxes = []
        for i in d.get("keep", []):
            if i not in by_id:
                raise SystemExit(f"{stem}: kept candidate id {i} does not exist in {fpath.name}")
            boxes.append(_clip(by_id[i]["bbox_norm"]))
        added = [_clip(b) for b in d.get("add", [])]
        if added and sam_model is not None and stem in by_stem:
            added, refine_log[stem] = refine_with_sam(sam_model, by_stem[stem], added)
        boxes += added
        lines = []
        for x1, y1, x2, y2 in boxes:
            if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                raise SystemExit(f"{stem}: degenerate box {x1, y1, x2, y2}")
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        (annotations_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[stem] = len(lines)
    return counts, refine_log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--filtered-dir", type=Path, required=True)
    ap.add_argument("--annotations-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--refine-sam-weights", type=Path, default=None,
                    help="if given, hand-drawn 'add' boxes are snapped to SAM masks (needs --images-dir)")
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--refine-log", type=Path, default=None)
    args = ap.parse_args(argv)

    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    sam = None
    if args.refine_sam_weights:
        if not args.images_dir:
            raise SystemExit("--refine-sam-weights requires --images-dir")
        from ultralytics import SAM
        sam = SAM(str(args.refine_sam_weights))
    counts, refine_log = build(decisions, args.filtered_dir, args.annotations_dir, sam, args.images_dir)
    if args.refine_log:
        args.refine_log.write_text(json.dumps(refine_log, indent=1), encoding="utf-8")

    entries = load_source_manifest(args.manifest)
    n = 0
    for e in entries:
        stem = Path(e.original_filename).stem
        if stem in counts and e.annotation_status != "annotated_human_verified":
            e.annotation_status = STATUS
            n += 1
    save_source_manifest(entries, args.manifest)
    print(f"wrote {len(counts)} label files ({sum(counts.values())} panel boxes); "
          f"{n} manifest entries set to {STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
