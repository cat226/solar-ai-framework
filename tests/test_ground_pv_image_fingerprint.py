"""tests/test_ground_pv_image_fingerprint.py — the image-fingerprint helpers
used for the candidate-dataset overlap / provenance audit."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from training.detection.ground_pv.image_fingerprint import (
    compare,
    dhash16_bytes,
    fingerprint_tree,
    source_basename,
)


def test_source_basename_strips_roboflow_export_suffix():
    assert source_basename("Bird_drop_25_jpg.rf.d0fbe8275943ed5ec9afb0017b0b98a6.jpg") == "Bird_drop_25"
    assert source_basename("Physical-Damge_100_jpg.rf.fe246fc5.jpg") == "Physical-Damge_100"
    assert source_basename("Imgdirty_1012_1_jpg.rf.d33cf80c.jpg") == "Imgdirty_1012_1"
    assert source_basename("plain_name.jpg") == "plain_name"


def _gradient(size=(200, 150), horiz=True):
    im = Image.new("RGB", size)
    w, h = size
    for x in range(w):
        for y in range(h):
            v = int(255 * (x / w if horiz else y / h))
            im.putpixel((x, y), (v, v, v))
    return im


def test_dhash16_is_256_bit_and_stable():
    im = _gradient(horiz=True)
    h = dhash16_bytes(im)
    assert len(h) == 32                       # 256 bits
    assert dhash16_bytes(im) == h             # deterministic
    # a vertical gradient has very different left/right neighbour structure
    assert dhash16_bytes(_gradient(horiz=False)) != h
    # a flat image legitimately hashes to all-zero (no gradient) - that is the
    # known dHash limitation the real audit compensates for with SHA + basename
    assert dhash16_bytes(Image.new("RGB", (200, 150), (128, 128, 128))) == b"\x00" * 32


def _make_tree(root: Path, spec: dict[str, tuple[int, int, int]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, rgb in spec.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        # a mild gradient so dHash is non-trivial
        im = Image.new("RGB", (64, 64), rgb)
        for x in range(64):
            for y in range(0, 64, 8):
                im.putpixel((x, y), (rgb[0], (rgb[1] + x) % 256, rgb[2]))
        im.save(p)
    return root


def test_fingerprint_tree_records_every_image(tmp_path):
    root = _make_tree(tmp_path / "a", {
        "train/x_jpg.rf.aaa.jpg": (10, 20, 30),
        "train/y_jpg.rf.bbb.jpg": (200, 100, 50),
        "test/z.jpg": (0, 0, 0),
    })
    fp = fingerprint_tree(root, label="a")
    assert fp["count"] == 3
    assert {r["rel"] for r in fp["records"]} == {
        "train/x_jpg.rf.aaa.jpg", "train/y_jpg.rf.bbb.jpg", "test/z.jpg"}
    for r in fp["records"]:
        assert len(r["sha256"]) == 64
        assert len(bytes.fromhex(r["dhash16"])) == 32
        assert r["w"] == 64 and r["h"] == 64
    assert {r["basename"] for r in fp["records"]} == {"x", "y", "z"}


def test_compare_flags_exact_basename_and_perceptual_overlap(tmp_path):
    a_root = _make_tree(tmp_path / "a", {
        "img/shared_jpg.rf.a1.jpg": (12, 34, 56),
        "img/only_in_a_jpg.rf.a2.jpg": (222, 111, 9),
    })
    # b has: a byte-identical copy of shared (diff name), a same-basename file,
    # and an unrelated image
    a_shared = (a_root / "img" / "shared_jpg.rf.a1.jpg").read_bytes()
    b_root = tmp_path / "b"
    (b_root / "img").mkdir(parents=True)
    (b_root / "img" / "shared_jpg.rf.b1.jpg").write_bytes(a_shared)          # exact + basename
    _make_tree(b_root, {"img/unrelated_jpg.rf.b2.jpg": (1, 2, 3)})

    fa = fingerprint_tree(a_root, label="a")
    fb = fingerprint_tree(b_root, label="b")
    r = compare(fa, fb, near_dup_hamming=6)

    assert r["exact_a_images"] == 1
    assert r["shared_basename_a_images"] == 1
    assert r["perceptual_near_dup_a_images"] >= 1   # exact copy also perceptually matches
    assert r["a_images_overlapping_by_any_signal"] == 1
    assert 0.0 < r["a_images_overlap_fraction"] <= 0.5


def test_compare_reports_zero_overlap_for_disjoint_trees(tmp_path):
    a = fingerprint_tree(_make_tree(tmp_path / "a", {"p_jpg.rf.a.jpg": (5, 90, 200)}), label="a")
    b = fingerprint_tree(_make_tree(tmp_path / "b", {"q_jpg.rf.b.jpg": (250, 40, 10)}), label="b")
    r = compare(a, b, near_dup_hamming=2)
    assert r["a_images_overlapping_by_any_signal"] == 0
    assert r["exact_a_images"] == 0
    assert r["shared_basename_a_images"] == 0
