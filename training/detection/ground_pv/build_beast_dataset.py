"""training/detection/ground_pv/build_beast_dataset.py — Phase 13B.

Build the research-only derived dataset `datasets/ground_pv_research/yolo_beast_v1/`
from two IMMUTABLE sources (neither is modified):

  A. `datasets/ground_pv_research/yolo/`  — the audited + remediated leakage-clean
     Susan-derived 3-class panel dataset (0=panel 1=cracked 2=dusty),
     train 4,546 / valid 1,297 / test 649.
  B. `datasets/ground_pv_research/candidate_acquisition/solar_panel_detection_ubwtz/
     v2_native_yolov8/` — the Roboflow candidate (ONE ground-mounted installation,
     ONE capture session, native-resolution portrait/landscape frames, polygon
     labels, class `solarpanel` -> our class 0 `panel`).
     Provenance: personal-experimental-unverified-provenance, research-only.

What this does to the candidate (deterministic):
  * drop the 58 empty-label images (panel-filled frames the annotator left
    unlabelled -> harmful as negatives)
  * scene-balanced subsample so a single walk-around does not dominate:
      - each video sequence: keep every `--video-stride`-th frame
      - burst photos: keep 1 image per distinct capture-second
      - hard cap at `--max-candidate` frames total
  * polygon -> TIGHT axis-aligned bbox, class 0, clamped to [0,1];
    reject zero-area / malformed / >92%x92% scene-spanning boxes
  * the ENTIRE candidate goes into the TRAIN split only (it is ~1 physical
    scene; putting it in valid/test would measure memorisation, not skill)

Outputs (small text committed; image/label trees gitignored + regenerable):
  yolo_beast_v1/
    candidate/train/{images,labels}/      materialised candidate frames + bboxes
    data_A.yaml                            Susan only  (Experiment A baseline)
    data_BC.yaml                           Susan train + candidate train, Susan valid/test
    beast_dataset_manifest.json            per-frame provenance + conversion metadata
    candidate_conversion.csv               every polygon->bbox conversion
    SOURCES.md                             immutable-source references + SHAs
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CANDIDATE_CLASS = 0  # solarpanel -> panel


def _sha256(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _capture_key(name: str) -> tuple[str, str, int]:
    """(event_key, kind, frame_index). event_key groups a video sequence or a
    photo capture-second; kind in {'video','photo'}."""
    stem = re.split(r"_jpe?g\.rf\.|\.rf\.", name)[0]
    m = re.match(r"(\d{8})[_-](\d{6})(?:[_-]([A-Za-z0-9]+?)[-_](\d+))?", stem)
    if not m:
        return (stem, "photo", 0)
    date, tm, vid, frame = m.groups()
    if vid:
        return (f"{date}_{tm}_{vid}", "video", int(frame))
    return (f"{date}_{tm}", "photo", 0)


def _poly_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) == 4:
        cx, cy, w, h = coords
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    else:
        return None
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    w, h = x2 - x1, y2 - y1
    if w <= 1e-4 or h <= 1e-4:
        return None
    if w > 0.92 and h > 0.92:  # scene-spanning -> reject
        return None
    return (round(x1 + w / 2, 6), round(y1 + h / 2, 6), round(w, 6), round(h, 6))


@dataclass
class BeastReport:
    candidate_raw_images: int = 0
    candidate_empty_dropped: int = 0
    candidate_annotated: int = 0
    candidate_selected: int = 0
    candidate_video_frames_selected: int = 0
    candidate_photo_frames_selected: int = 0
    polygons_converted: int = 0
    polygons_rejected: int = 0
    capture_events: int = 0
    susan_train: int = 0
    susan_valid: int = 0
    susan_test: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def build(
    susan_dir: Path,
    candidate_dir: Path,
    out_dir: Path,
    *,
    video_stride: int = 12,
    max_candidate: int = 130,
    overwrite: bool = False,
) -> BeastReport:
    susan_dir, candidate_dir, out_dir = Path(susan_dir), Path(candidate_dir), Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (pass overwrite=True)")
        shutil.rmtree(out_dir)
    (out_dir / "candidate" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate" / "train" / "labels").mkdir(parents=True, exist_ok=True)

    rep = BeastReport()
    for s in ("train", "valid", "test"):
        n = len(list((susan_dir / s / "images").glob("*")))
        setattr(rep, f"susan_{s}", n)

    # --- collect candidate frames from all splits (we re-split: all -> train) ---
    cand_imgs: list[Path] = []
    for s in ("train", "valid", "test"):
        cand_imgs += sorted((candidate_dir / s / "images").glob("*"))
    rep.candidate_raw_images = len(cand_imgs)

    events: dict[str, list[tuple[int, Path, Path]]] = defaultdict(list)
    for img in cand_imgs:
        if img.suffix.lower() not in _IMG_EXT:
            continue
        lp = img.parent.parent / "labels" / f"{img.stem}.txt"
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()] if lp.is_file() else []
        if not lines:
            rep.candidate_empty_dropped += 1
            continue
        rep.candidate_annotated += 1
        key, kind, frame = _capture_key(img.name)
        events[key].append((frame, img, lp))
    rep.capture_events = len(events)

    # --- scene-balanced selection ---
    selected: list[tuple[str, str, Path, Path]] = []  # (event, kind, img, label)
    for key, items in sorted(events.items()):
        items.sort()
        kind = "video" if re.search(r"_(mp4|MP4|mov|MOV)$", key) else "photo"
        if kind == "video":
            picks = items[::video_stride] or items[:1]
        else:
            picks = items[:1]  # 1 per capture-second (dedupe bursts)
        for _frame, img, lp in picks:
            selected.append((key, kind, img, lp))
    # deterministic cap: spread evenly across the ordered selection
    if len(selected) > max_candidate:
        step = len(selected) / max_candidate
        selected = [selected[int(i * step)] for i in range(max_candidate)]

    conv_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for (event, kind, img, lp) in selected:
        out_name = f"ubwtz__{img.name}"
        shutil.copy2(img, out_dir / "candidate" / "train" / "images" / out_name)
        new_lines: list[str] = []
        kept = rejected = 0
        for i, ln in enumerate(lp.read_text(encoding="utf-8").splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                _cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                rejected += 1
                continue
            bb = _poly_to_bbox(coords)
            if bb is None:
                rejected += 1
                conv_rows.append({"file": out_name, "poly_index": i, "result": "rejected",
                                  "n_coords": len(coords)})
                continue
            cx, cy, w, h = bb
            new_lines.append(f"{CANDIDATE_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            kept += 1
            conv_rows.append({"file": out_name, "poly_index": i, "result": "polygon->bbox",
                              "n_coords": len(coords), "bbox_cx_cy_w_h": f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"})
        (out_dir / "candidate" / "train" / "labels" / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        rep.polygons_converted += kept
        rep.polygons_rejected += rejected
        rep.candidate_selected += 1
        if kind == "video":
            rep.candidate_video_frames_selected += 1
        else:
            rep.candidate_photo_frames_selected += 1
        manifest_rows.append({
            "derived_filename": out_name,
            "split": "train",
            "source": "candidate_ubwtz",
            "source_filename": img.name,
            "source_sha256": _sha256(img),
            "capture_event": event,
            "capture_kind": kind,
            "family_id": "cand_ubwtz_scene1",  # ONE physical scene
            "boxes": kept,
            "polygons_rejected": rejected,
            "provenance_grade": "C",
            "provenance_tag": "personal-experimental-unverified-provenance",
        })

    # --- data.yaml files ---
    import yaml
    names = {0: "panel", 1: "cracked", 2: "dusty"}
    susan_abs = str(susan_dir.resolve()).replace("\\", "/")
    cand_train_abs = str((out_dir / "candidate" / "train" / "images").resolve()).replace("\\", "/")
    (out_dir / "data_A.yaml").write_text(yaml.safe_dump({
        "path": susan_abs, "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": names,
        "_experiment": "A - Susan-only baseline reproduction (no candidate)",
    }, sort_keys=False), encoding="utf-8")
    (out_dir / "data_BC.yaml").write_text(yaml.safe_dump({
        "train": [f"{susan_abs}/train/images", cand_train_abs],
        "val": f"{susan_abs}/valid/images",
        "test": f"{susan_abs}/test/images",
        "names": names,
        "_experiment": "B/C - Susan train + scene-capped candidate (train only); Susan valid/test unchanged",
        "_candidate": "personal-experimental-unverified-provenance, research-only, ~1 physical scene",
    }, sort_keys=False), encoding="utf-8")

    # --- manifests ---
    (out_dir / "beast_dataset_manifest.json").write_text(json.dumps({
        "derived_dataset": "yolo_beast_v1",
        "purpose": "Phase 13B research-only ground PANEL detector training (0=panel 1=cracked 2=dusty)",
        "sources": {
            "susan": {"dir": str(susan_dir).replace("\\", "/"), "immutable": True,
                      "splits": {"train": rep.susan_train, "valid": rep.susan_valid, "test": rep.susan_test},
                      "note": "audited + remediated leakage-clean; NOT modified"},
            "candidate": {"dir": str(candidate_dir).replace("\\", "/"), "immutable": True,
                          "roboflow": "solar-panel-detection-bwpkx/solar-panel-detection-ubwtz v2",
                          "provenance_tag": "personal-experimental-unverified-provenance",
                          "note": "ONE ground-mounted installation, ONE capture session; research-only; "
                                  "NOT redistribution/publication/commercial/production cleared"},
        },
        "candidate_selection": {
            "video_stride": video_stride, "max_candidate": max_candidate,
            "empty_label_images_dropped": rep.candidate_empty_dropped,
            "capture_events": rep.capture_events,
            "selected": rep.candidate_selected,
            "video_frames": rep.candidate_video_frames_selected,
            "photo_frames": rep.candidate_photo_frames_selected,
            "polygons_converted_to_bbox": rep.polygons_converted,
            "polygons_rejected": rep.polygons_rejected,
            "all_candidate_frames_in_split": "train (ONE physical scene -> family cand_ubwtz_scene1)",
        },
        "leakage_controls": {
            "susan_split": "existing leakage-clean family split preserved unchanged",
            "candidate_split": "train only - never valid/test (single scene would leak / measure memorisation)",
            "valid_test": "Susan valid/test ONLY for all experiments -> A vs B vs C compared on the identical held-out set",
            "owner_photos": "NOT included, NOT pseudo-labelled, NOT used for validation or threshold tuning",
        },
        "report": rep.as_dict(),
        "manifest_rows": manifest_rows,
    }, indent=2), encoding="utf-8")

    with (out_dir / "candidate_conversion.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "poly_index", "result", "n_coords", "bbox_cx_cy_w_h"])
        w.writeheader()
        for r in conv_rows:
            w.writerow({**{k: "" for k in w.fieldnames}, **r})

    (out_dir / "SOURCES.md").write_text(
        "# yolo_beast_v1 — immutable source references (Phase 13B)\n\n"
        f"- **Susan 3-class** `{susan_dir}` — audited + remediated leakage-clean, NOT modified.\n"
        f"- **Candidate** `{candidate_dir}` — Roboflow "
        "`solar-panel-detection-bwpkx/solar-panel-detection-ubwtz` v2 (native-resolution yolov8 export). "
        "**personal-experimental-unverified-provenance, research-only.** ONE ground-mounted installation, "
        "ONE capture session (2024-04-30). Polygons converted to tight bboxes deterministically; the "
        "whole candidate is a single family in the TRAIN split. See "
        "`../candidate_acquisition/solar_panel_detection_ubwtz_audit.json`.\n\n"
        "The six owner photos are NOT part of this dataset.\n",
        encoding="utf-8")
    return rep


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3] / "datasets" / "ground_pv_research"
    ap.add_argument("--susan", type=Path, default=root / "yolo")
    ap.add_argument("--candidate", type=Path,
                    default=root / "candidate_acquisition" / "solar_panel_detection_ubwtz" / "v2_native_yolov8")
    ap.add_argument("--out", type=Path, default=root / "yolo_beast_v1")
    ap.add_argument("--video-stride", type=int, default=12)
    ap.add_argument("--max-candidate", type=int, default=130)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    rep = build(a.susan, a.candidate, a.out, video_stride=a.video_stride,
                max_candidate=a.max_candidate, overwrite=a.overwrite)
    print(json.dumps(rep.as_dict(), indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
