"""training/detection/ground_pv/owner_mark_boxes.py — recover the owner's
burned-in defect boxes by diffing a MARKED photo against its PRISTINE original.

The owner drew labelled rectangles (black = BIRD DROP, red = PHYSICAL DAMAGE)
directly into 6 of the 48 rooftop photos. The pristine originals are kept, so
the boxes are exactly the pixels that differ AND are pen-coloured. This yields
machine-readable ground-truth boxes without touching either image.

PIL + numpy only (no torch / scipy).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_BLACK_MAX = 70  # all channels below this = black pen


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected components of a boolean mask -> (x1, y1, x2, y2, n_pixels)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        x1 = x2 = x0
        y1 = y2 = y0
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            x1, x2, y1, y2 = min(x1, x), max(x2, x), min(y1, y), max(y2, y)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append((x1, y1, x2, y2, n))
    return out


def _merge_touching(boxes: list[list[int]], tol: int) -> list[list[int]]:
    """Union boxes [x1,y1,x2,y2,n] whose rectangles overlap or lie within `tol`
    pixels, repeatedly, so a frame broken by dark panel gaps becomes one box."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - tol <= b[2] and b[0] - tol <= a[2] and a[1] - tol <= b[3] and b[1] - tol <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def extract_boxes(pristine: Image.Image, marked: Image.Image, *, diff_thresh: int = 90,
                  min_box_side: int = 40, merge_tol: int = 15, max_fill: float = 0.5,
                  text_fill: float = 0.2) -> list[dict]:
    """Rectangle outlines drawn on `marked` that are absent in `pristine`, in
    pixel coordinates. Ink = strongly different pixels that are black or
    (dark-)red in the marked copy. Fragments are merged; blobs thinner than
    `min_box_side` (letters of the label text) and solid blobs (changed-pixel
    fraction >= `max_fill`) are dropped. `ink` is 'black' or 'red' by the
    dominant ink colour inside the final box."""
    if pristine.size != marked.size:
        marked = marked.resize(pristine.size, Image.BILINEAR)
    a = np.asarray(pristine.convert("RGB"), dtype=np.int16)
    b = np.asarray(marked.convert("RGB"), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2) > diff_thresh
    black = (b < _BLACK_MAX).all(axis=2)
    red = (b[..., 0] - b[..., 1] > 50) & (b[..., 0] - b[..., 2] > 30)
    ink_black, ink_red = diff & black, diff & red
    # text labels ("BIRD DROP") are dense stroke blobs; a frame fragment is a thin
    # line inside a large bbox. Drop dense blobs BEFORE merging so a label sitting
    # on a box edge can't stretch it.
    comps = [c for c in _components(ink_black | ink_red)
             if c[4] / float((c[2] - c[0] + 1) * (c[3] - c[1] + 1)) < text_fill]
    boxes = []
    for x1, y1, x2, y2, n in _merge_touching([list(c) for c in comps], merge_tol):
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if bw < min_box_side or bh < min_box_side:
            continue
        nb = int(ink_black[y1:y2 + 1, x1:x2 + 1].sum())
        nr = int(ink_red[y1:y2 + 1, x1:x2 + 1].sum())
        if (nb + nr) / float(bw * bh) >= max_fill:
            continue
        boxes.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                      "ink": "red" if nr > nb else "black", "ink_pixels": nb + nr})
    return sorted(boxes, key=lambda d: (d["y1"], d["x1"]))
