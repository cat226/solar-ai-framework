"""training/detection/ground_pv/tiled_inference.py — overlapping-tile inference
for the ground-PV panel detector (RESEARCH EXPERIMENT, Phase 10).

Motivation: the full-frame ground research detector
(`weights/yolo_solar_ground_research_v1.pt`) misses large foreground panels and
emits group boxes on wide array shots. Tiling the image and running the SAME
checkpoint per tile, then remapping + globally de-duplicating, is tested as a
first-stage improvement.

Nothing here retrains or modifies a checkpoint. Pure inference orchestration.

Pipeline per image:
  1. `tile_windows(W, H, tile, overlap)` -> list of (x0, y0, x1, y1) windows that
     tile the image with `overlap` fractional overlap. The last row/column is
     clamped to the image edge (never past it).
  2. For each window: crop EXACTLY that region, PAD (not resize) it up to
     tile x tile on a constant-fill canvas anchored at the top-left, run the
     detector on the canvas.
  3. Map each detection back: clip to the real (un-padded) crop extent, then add
     the window origin -> original-image pixel coordinates.
  4. `merge_detections(...)` -> per-class global NMS (IoU threshold) plus
     "mostly-contained" suppression (a lower-confidence box that is >=
     `contain_frac` inside a higher-confidence same-class box is dropped) to
     clean up the partial boxes tiling produces on panels that span tiles.

The image is NEVER globally resized before tiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

_DEFAULT_TILE = 640
_DEFAULT_OVERLAP = 0.25
_DEFAULT_CONF = 0.45
_DEFAULT_NMS_IOU = 0.50
_DEFAULT_CONTAIN_FRAC = 0.90
_PAD_FILL = 114  # Ultralytics letterbox grey


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in ORIGINAL-image pixels
    source_tiles: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "source_tiles": sorted(set(self.source_tiles)),
        }


def tile_windows(
    width: int,
    height: int,
    *,
    tile: int = _DEFAULT_TILE,
    overlap: float = _DEFAULT_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows covering [0,width) x [0,height).

    Every window is <= tile in each dimension and lies fully inside the image
    (the final window in a row/column is shifted back to end exactly at the
    edge when the image is larger than one tile; when the image is smaller than
    a tile in a dimension, that dimension yields a single 0..size window).
    Guarantees full coverage: every pixel lies in >= 1 window.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1): {overlap}")
    if tile <= 0:
        raise ValueError("tile must be positive")
    step = max(1, int(round(tile * (1.0 - overlap))))

    def _starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, step))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    xs = _starts(width)
    ys = _starts(height)
    windows: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile, width)
            y1 = min(y0 + tile, height)
            windows.append((x0, y0, x1, y1))
    return windows


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contained_fraction(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of `inner`'s area that lies within `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def merge_detections(
    dets: Sequence[Detection],
    *,
    nms_iou: float = _DEFAULT_NMS_IOU,
    contain_frac: float = _DEFAULT_CONTAIN_FRAC,
) -> list[Detection]:
    """Per-class global NMS + mostly-contained suppression. Deterministic:
    ties broken by (‑confidence, x1, y1, x2, y2)."""
    kept: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for d in dets:
        by_class.setdefault(d.class_id, []).append(d)
    for cid, group in by_class.items():
        ordered = sorted(group, key=lambda d: (-d.confidence, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]))
        survivors: list[Detection] = []
        for cand in ordered:
            drop = False
            for s in survivors:
                if _iou(cand.bbox, s.bbox) >= nms_iou or _contained_fraction(cand.bbox, s.bbox) >= contain_frac:
                    s.source_tiles.extend(cand.source_tiles)
                    drop = True
                    break
            if not drop:
                survivors.append(cand)
        kept.extend(survivors)
    kept.sort(key=lambda d: (-d.confidence, d.class_id))
    return kept


def _predict_boxes(model, canvas, *, conf: float, iou: float, imgsz: int) -> list[tuple[int, float, list[float]]]:
    """Run the detector on a single tile canvas. Returns
    [(class_id, confidence, [x1,y1,x2,y2] in canvas px), ...]."""
    res = model.predict(source=canvas, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    out: list[tuple[int, float, list[float]]] = []
    for b in res.boxes:
        out.append((int(b.cls.item()), float(b.conf.item()), [float(v) for v in b.xyxy[0].tolist()]))
    return out


def tiled_detect(
    model,
    image,
    *,
    tile: int = _DEFAULT_TILE,
    overlap: float = _DEFAULT_OVERLAP,
    conf: float = _DEFAULT_CONF,
    nms_iou: float = _DEFAULT_NMS_IOU,
    tile_nms_iou: float = _DEFAULT_NMS_IOU,
    contain_frac: float = _DEFAULT_CONTAIN_FRAC,
    imgsz: int = _DEFAULT_TILE,
    class_filter: int | None = None,
) -> dict[str, Any]:
    """Tiled inference over a PIL image. Returns a dict with the tile config,
    per-tile raw detection counts, and the merged detections in original-image
    coordinates.

    `model` only needs a `.predict(source=, conf=, iou=, imgsz=, verbose=)`
    method and a `.names` mapping (Ultralytics YOLO satisfies this).
    """
    from PIL import Image

    names = model.names if isinstance(getattr(model, "names", None), dict) else dict(enumerate(getattr(model, "names", [])))
    W, H = image.size
    windows = tile_windows(W, H, tile=tile, overlap=overlap)

    raw: list[Detection] = []
    per_tile: list[dict[str, Any]] = []
    for ti, (x0, y0, x1, y1) in enumerate(windows):
        crop = image.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        if (cw, ch) != (tile, tile):
            canvas = Image.new("RGB", (tile, tile), (_PAD_FILL, _PAD_FILL, _PAD_FILL))
            canvas.paste(crop, (0, 0))
        else:
            canvas = crop
        boxes = _predict_boxes(model, canvas, conf=conf, iou=tile_nms_iou, imgsz=imgsz)
        kept_here = 0
        for (cid, cf, (bx1, by1, bx2, by2)) in boxes:
            if class_filter is not None and cid != class_filter:
                continue
            # clip to the real (un-padded) crop extent, then offset to original
            bx1 = min(max(bx1, 0.0), cw)
            bx2 = min(max(bx2, 0.0), cw)
            by1 = min(max(by1, 0.0), ch)
            by2 = min(max(by2, 0.0), ch)
            if bx2 - bx1 < 1.0 or by2 - by1 < 1.0:
                continue
            raw.append(Detection(
                class_id=cid, class_name=str(names.get(cid, cid)), confidence=cf,
                bbox=[bx1 + x0, by1 + y0, bx2 + x0, by2 + y0], source_tiles=[ti],
            ))
            kept_here += 1
        per_tile.append({"tile_index": ti, "window_xyxy": [x0, y0, x1, y1],
                         "padded": (cw, ch) != (tile, tile), "detections": kept_here})

    merged = merge_detections(raw, nms_iou=nms_iou, contain_frac=contain_frac)
    return {
        "image_size_wh": [W, H],
        "tile_config": {
            "tile": tile, "overlap": overlap, "step_px": max(1, int(round(tile * (1.0 - overlap)))),
            "conf": conf, "nms_iou": nms_iou, "tile_nms_iou": tile_nms_iou,
            "contain_frac": contain_frac, "imgsz": imgsz, "pad_fill": _PAD_FILL,
            "n_tiles": len(windows),
        },
        "tiles": per_tile,
        "raw_detection_count": len(raw),
        "merged_detection_count": len(merged),
        "detections": [d.as_dict() for d in merged],
    }
