"""training/detection/ground_pv/owner_condition_labels.py — pure helpers for the
owner's per-panel condition labeling tool (scripts/label_panel_conditions.py).

The owner is the only legitimate source of ground truth for this rooftop. The
tool shows each detected panel and records the owner's label verbatim. Panel
boxes are the current detector's PROPOSALS (recorded as such); the CONDITION
label is always the owner's, never a model's. No UI imports here, so this is
unit-testable without Streamlit or model weights.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER

# SKIP = the owner cannot tell, or the box is not a panel. Never trained on.
LABELS: tuple[str, ...] = (*CONDITION_4CLASS_ORDER, "SKIP")
LABEL_SOURCE = "owner"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CSV_FIELDS: tuple[str, ...] = (
    "image", "image_sha256", "panel_index", "x1", "y1", "x2", "y2", "box_source", "detector_conf",
    "label", "label_source", "simulated", "missed_panels_in_image", "notes", "labeled_at",
)


def list_images(folder: Path) -> list[Path]:
    """Images directly inside `folder`, sorted by name (stable labeling order)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labeled_image_keys(csv_path: Path) -> set[tuple[str, str]]:
    """(image name, sha256) pairs already labeled. Keyed on the hash too, so a file
    that is later replaced or edited under the same name is NOT treated as done."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {(r["image"], r["image_sha256"]) for r in csv.DictReader(fh)}


def validate_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"label {label!r} is not one of {LABELS}")
    return label


def build_rows(
    image_name: str,
    image_sha256: str,
    panels: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    missed_panels: int = 0,
    simulated: bool = False,
    notes: str = "",
    box_source: str = "detector_proposal",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """One row per panel. Refuses to produce rows unless EVERY panel has a valid label
    (a silently-unlabeled panel would otherwise read as 'no defect')."""
    if len(panels) != len(labels):
        raise ValueError(f"{len(panels)} panels but {len(labels)} labels")
    if missed_panels < 0:
        raise ValueError("missed_panels must be >= 0")
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    rows = []
    for i, (p, lab) in enumerate(zip(panels, labels)):
        b = p["bbox"]
        rows.append({
            "image": image_name, "image_sha256": image_sha256, "panel_index": i,
            "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
            "box_source": box_source, "detector_conf": p.get("confidence", ""),
            "label": validate_label(lab), "label_source": LABEL_SOURCE,
            "simulated": int(bool(simulated)), "missed_panels_in_image": int(missed_panels),
            "notes": notes.replace("\n", " ").strip(), "labeled_at": stamp,
        })
    return rows


def append_rows(csv_path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows (header written once). Returns number of rows written."""
    rows = list(rows)
    if not rows:
        return 0
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize(csv_path: Path) -> dict[str, Any]:
    """Counts the owner can watch grow: panels per label, simulated vs real, images done."""
    csv_path = Path(csv_path)
    out: dict[str, Any] = {"images": 0, "panels": 0, "by_label": {k: 0 for k in LABELS}, "simulated_panels": 0}
    if not csv_path.is_file():
        return out
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            seen.add((r["image"], r["image_sha256"]))
            out["panels"] += 1
            out["by_label"][r["label"]] = out["by_label"].get(r["label"], 0) + 1
            out["simulated_panels"] += int(r["simulated"] == "1")
    out["images"] = len(seen)
    return out


def propose_panels(model: Any, image: Any, *, image_name: str, conf: float = 0.25) -> list[dict[str, Any]]:
    """Panel boxes from the active detector (spatially ordered). PROPOSALS only."""
    from training.detection.ground_pv.dynamic_panel_detection import DynamicPanelConfig, detect_dynamic_panels

    report = detect_dynamic_panels(
        model, image, image_name=image_name, model_name="panel_detector",
        config=DynamicPanelConfig(confidence_threshold=conf),
    )
    return report["panels"]
