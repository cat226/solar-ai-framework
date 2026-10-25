"""training/detection/ground_pv/image_fingerprint.py — Fingerprint an image
tree for the candidate-dataset overlap / provenance audit.

For each image: relative path, SHA-256 (exact-dup signal), 16x16 difference
hash (256-bit perceptual near-dup signal), pixel dimensions, and a
"source basename" with common Roboflow/aug export suffixes stripped
(shared-source-family signal).

Read-only. No third-party dependency beyond PIL + numpy (already project deps).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Roboflow / common export suffixes (longest first).
_RF_MARKERS = tuple(sorted((
    "_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_png_jpg.rf.", "_jpeg_jpg.rf.",
    "_JPG_jpg.rf.", "_JPEG_jpg.rf.", "_PNG_jpg.rf.", ".rf.",
    "_jpg.", "_png.", "_jpeg.",
), key=len, reverse=True))

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def source_basename(name: str) -> str:
    stem = Path(name).name
    for marker in _RF_MARKERS:
        if marker in stem:
            stem = stem.split(marker, 1)[0]
            break
    # strip a trailing "-<n>" / "_<n>" augmentation-copy counter and jpg/png tails
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def dhash16_bytes(image: Image.Image) -> bytes:
    g = image.convert("L").resize((17, 16), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten().astype(np.uint8)
    return np.packbits(bits).tobytes()


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


def fingerprint_tree(root: Path, *, label: str = "") -> dict[str, Any]:
    root = Path(root)
    records: list[dict[str, Any]] = []
    corrupt: list[str] = []
    for p in iter_images(root):
        rel = p.relative_to(root).as_posix()
        try:
            with Image.open(p) as im:
                w, h = im.size
                dh = dhash16_bytes(im.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            corrupt.append(rel)
            continue
        records.append({
            "rel": rel, "sha256": _sha256(p), "dhash16": dh.hex(),
            "w": w, "h": h, "basename": source_basename(p.name),
        })
    return {
        "label": label or root.name,
        "root": str(root),
        "count": len(records),
        "corrupt": corrupt,
        "records": records,
    }


def save_fingerprint(fp: dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fp, indent=1), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between a (N,32) uint8 and b (M,32) uint8
    -> (N, M) uint16.  Blocked to keep memory sane."""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.uint16)
    for i in range(a.shape[0]):
        out[i] = _POPCOUNT[np.bitwise_xor(b, a[i])].sum(axis=1)
    return out


def compare(fp_a: dict[str, Any], fp_b: dict[str, Any], *, near_dup_hamming: int = 6) -> dict[str, Any]:
    """Overlap of A against B. A is the candidate; B is existing/other data."""
    ra, rb = fp_a["records"], fp_b["records"]
    sha_b = {}
    for r in rb:
        sha_b.setdefault(r["sha256"], []).append(r["rel"])
    base_b: dict[str, list[str]] = {}
    for r in rb:
        base_b.setdefault(r["basename"], []).append(r["rel"])

    exact = []
    shared_basename = []
    for r in ra:
        if r["sha256"] in sha_b:
            exact.append({"a": r["rel"], "b": sha_b[r["sha256"]][:5], "sha256": r["sha256"]})
        if r["basename"] in base_b and len(r["basename"]) >= 4:
            shared_basename.append({"a": r["rel"], "basename": r["basename"],
                                    "b": base_b[r["basename"]][:5]})

    # perceptual near-dup (16x16 hamming <= threshold)
    def _stack(recs: list[dict[str, Any]]) -> np.ndarray:
        if not recs:
            return np.zeros((0, 32), np.uint8)
        return np.stack([np.frombuffer(bytes.fromhex(r["dhash16"]), dtype=np.uint8) for r in recs])
    ha, hb = _stack(ra), _stack(rb)
    near = []
    if len(ra) and len(rb):
        dist = _hamming(ha, hb)
        for i in range(len(ra)):
            js = np.where(dist[i] <= near_dup_hamming)[0]
            if js.size:
                near.append({
                    "a": ra[i]["rel"],
                    "b": [rb[int(j)]["rel"] for j in js[:5]],
                    "min_hamming": int(dist[i][js].min()),
                    "n_matches": int(js.size),
                })
    near_a_images = {n["a"] for n in near}
    exact_a_images = {e["a"] for e in exact}
    base_a_images = {s["a"] for s in shared_basename}
    overlap_any = exact_a_images | near_a_images | base_a_images
    return {
        "a_label": fp_a["label"], "b_label": fp_b["label"],
        "a_count": len(ra), "b_count": len(rb),
        "near_dup_hamming_threshold": near_dup_hamming,
        "exact_sha_matches": len(exact),
        "exact_a_images": len(exact_a_images),
        "perceptual_near_dup_a_images": len(near_a_images),
        "shared_basename_a_images": len(base_a_images),
        "a_images_overlapping_by_any_signal": len(overlap_any),
        "a_images_overlap_fraction": round(len(overlap_any) / max(1, len(ra)), 4),
        "samples": {
            "exact": exact[:15],
            "perceptual": near[:15],
            "shared_basename": shared_basename[:15],
        },
    }
