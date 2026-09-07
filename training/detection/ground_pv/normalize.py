"""training/detection/ground_pv/normalize.py — Materialize a class-normalized,
Ultralytics-ready copy of the raw Roboflow dataset at
`datasets/ground_pv_research/yolo/`.

The raw download stays immutable. This produces a SEPARATE tree:

    <out_dir>/
        data.yaml            # research taxonomy: 0 panel / 1 bird_drop / 2 cracked / 3 dusty
        train/{images,labels}
        valid/{images,labels}
        test/{images,labels}
        NORMALIZATION.json    # what was remapped, source hashes, counts

Images are copied byte-for-byte. Label files are rewritten ONLY to apply
the integer class-id remap (source_id -> research_id) computed from the
real source data.yaml; box coordinates are never touched. If the remap is
the identity, labels are still copied verbatim (no rewrite).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from training.detection.ground_pv.class_map import (
    RESEARCH_CLASS_NAMES,
    RESEARCH_CLASS_ORDER,
    build_class_id_remap,
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_names(raw_dir: Path) -> dict[int, str]:
    import yaml
    data = yaml.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("could not read names: from source data.yaml")


def _remap_label_text(text: str, remap: dict[int, int]) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            out_lines.append(line)  # leave malformed lines for the audit to catch
            continue
        try:
            src = int(parts[0])
        except ValueError:
            out_lines.append(line)
            continue
        parts[0] = str(remap.get(src, src))
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def normalize_dataset(raw_dir: Path, out_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if not (raw_dir / "data.yaml").is_file():
        raise FileNotFoundError(f"no data.yaml under {raw_dir}")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (pass overwrite=True to rebuild)")
        shutil.rmtree(out_dir)

    source_names = _read_source_names(raw_dir)
    remap = build_class_id_remap(source_names)
    is_identity = all(k == v for k, v in remap.items())

    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, dict[str, int]] = {}
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir() or entry.name not in _SPLIT_ALIASES:
            continue
        canon_split = _SPLIT_ALIASES[entry.name]
        src_images = entry / "images"
        src_labels = entry / "labels"
        dst_images = out_dir / canon_split / "images"
        dst_labels = out_dir / canon_split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        n_img = n_lbl = 0
        if src_images.is_dir():
            for img in sorted(src_images.glob("*")):
                if img.suffix.lower() in _IMAGE_EXT:
                    shutil.copy2(img, dst_images / img.name)
                    n_img += 1
        if src_labels.is_dir():
            for lbl in sorted(src_labels.glob("*.txt")):
                text = lbl.read_text(encoding="utf-8", errors="replace")
                if is_identity:
                    (dst_labels / lbl.name).write_text(text, encoding="utf-8")
                else:
                    (dst_labels / lbl.name).write_text(_remap_label_text(text, remap), encoding="utf-8")
                n_lbl += 1
        split_counts[canon_split] = {"images": n_img, "labels": n_lbl}

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: RESEARCH_CLASS_NAMES[i] for i in range(len(RESEARCH_CLASS_ORDER))},
    }
    import yaml
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    meta = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "source_names": source_names,
        "class_id_remap_source_to_research": remap,
        "remap_is_identity": is_identity,
        "research_names": data_yaml["names"],
        "split_counts": split_counts,
        "source_data_yaml_sha256": _sha256_file(raw_dir / "data.yaml"),
    }
    (out_dir / "NORMALIZATION.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
