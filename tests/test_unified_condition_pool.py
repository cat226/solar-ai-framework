"""Tests for the unified condition-pool helpers (family hashing, label policy,
family-level splitting). Synthetic images only - no dataset needed."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from training.detection.ground_pv import unified_condition_pool as ucp


def _img(seed: int, size=(160, 120)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.BICUBIC)


def test_canon_hash_is_invariant_to_flip_and_rot90():
    im = _img(1)
    h = ucp.canon_hash(im)
    for variant in (im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
                    im.transpose(Image.ROTATE_90), im.transpose(Image.ROTATE_180), im.transpose(Image.ROTATE_270)):
        assert ucp.hamming(h, ucp.canon_hash(variant)) <= ucp.DEFAULT_MAX_DIST
    assert ucp.hamming(h, ucp.canon_hash(_img(2))) > ucp.DEFAULT_MAX_DIST


def test_group_families_unions_close_hashes_and_keeps_far_ones_apart():
    a = 0b1010101010101010101010101010101010101010101010101010101010101010
    near = a ^ 0b11          # 2 bits away
    far = ~a & ((1 << 64) - 1)
    fam = ucp.group_families([a, far, near, a])
    assert fam[0] == fam[2] == fam[3] != fam[1]
    assert ucp.group_families([]) == []


def test_group_families_is_transitive():
    a = 0
    b = 0b1111          # 4 from a
    c = 0b11111111      # 4 from b, 8 from a
    fam = ucp.group_families([a, b, c], max_dist=4)
    assert fam[0] == fam[1] == fam[2]


def test_faulty_labels_follow_the_active_taxonomy():
    assert ucp.faulty_label("Bird-drop") == "BIRD_DROP"
    assert ucp.faulty_label("Physical-Damage") == "PHYSICAL_DAMAGE"
    assert ucp.faulty_label("Electrical-damage") is None
    assert ucp.faulty_label("Snow-Covered") is None
    with pytest.raises(KeyError):
        ucp.faulty_label("Hotspot")
    ucp.assert_active_labels(["CLEAN", "DUSTY"])
    with pytest.raises(ValueError):
        ucp.assert_active_labels(["CLEAN", "ELECTRICAL_DAMAGE"])


def test_lineage_and_roboflow_base():
    assert ucp.lineage_of("faulty", "Bird (3).jpg") == ucp.LINEAGE_FAULTY
    assert ucp.lineage_of("github_raw", "Imgclean_12_0.jpg") == ucp.LINEAGE_STOCK_WEB
    assert ucp.lineage_of("roboflow_train", "20210917_151349_jpg.rf.abc.jpg") == ucp.LINEAGE_PHONE_MACRO
    assert ucp.roboflow_base("Imgdirty_5_1_jpg.rf.0123abcd.jpg") == "Imgdirty_5_1"


def _rec(path, label, family, source="faulty", w=300, h=300, lineage=ucp.LINEAGE_FAULTY):
    return ucp.Rec(path=path, source=source, source_class="x", label=label, lineage=lineage, w=w, h=h, family=family)


def test_resolve_families_drops_conflicts_excluded_small_and_excluded_lineage():
    recs = [
        _rec("a1", "CLEAN", 0), _rec("a2", "CLEAN", 0, source="roboflow_train"),     # kept, faulty is rep
        _rec("b1", "BIRD_DROP", 1), _rec("b2", "PHYSICAL_DAMAGE", 1),               # label_conflict
        _rec("c1", "DUSTY", 2), _rec("c2", None, 2),                                # conflict_with_excluded_class
        _rec("d1", None, 3),                                                        # excluded_class
        _rec("e1", "DUSTY", 4, w=100, h=400),                                       # too_small
        _rec("f1", "CLEAN", 5, source="roboflow_train", lineage=ucp.LINEAGE_PHONE_MACRO),  # excluded lineage
    ]
    kept, dropped = ucp.resolve_families(recs)
    assert set(kept) == {0}
    assert kept[0]["rep"].path == "a1"
    assert dropped[1] == "label_conflict"
    assert dropped[2] == "conflict_with_excluded_class"
    assert dropped[3] == "excluded_class"
    assert dropped[4] == "too_small"
    assert dropped[5].startswith("excluded_lineage:")
    kept2, _ = ucp.resolve_families(recs, excluded_lineages=frozenset())
    assert 5 in kept2


def test_coarse_to_fine_override_is_narrow():
    # binary-source DUSTY vs Faulty BIRD_DROP -> finer Faulty label wins
    ok = [_rec("f", "BIRD_DROP", 0), _rec("r", "DUSTY", 0, source="roboflow_train", lineage=ucp.LINEAGE_STOCK_WEB)]
    kept, dropped = ucp.resolve_families(ok)
    assert kept[0]["label"] == "BIRD_DROP" and kept[0]["coarse_to_fine_override"] and not dropped
    # same for PHYSICAL_DAMAGE
    ok2 = [_rec("f", "PHYSICAL_DAMAGE", 0), _rec("g", "DUSTY", 0, source="github_raw", lineage=ucp.LINEAGE_STOCK_WEB)]
    assert ucp.resolve_families(ok2)[0][0]["label"] == "PHYSICAL_DAMAGE"
    # Faulty itself says DUSTY -> real conflict, no override
    bad1 = [_rec("f1", "BIRD_DROP", 0), _rec("f2", "DUSTY", 0)]
    assert ucp.resolve_families(bad1)[1][0] == "label_conflict"
    # CLEAN is never overridden
    bad2 = [_rec("f", "BIRD_DROP", 0), _rec("r", "CLEAN", 0, source="roboflow_train", lineage=ucp.LINEAGE_STOCK_WEB)]
    assert ucp.resolve_families(bad2)[1][0] == "label_conflict"
    # a third label blocks the override
    bad3 = ok + [_rec("c", "CLEAN", 0, source="github_raw", lineage=ucp.LINEAGE_STOCK_WEB)]
    assert ucp.resolve_families(bad3)[1][0] == "label_conflict"


def test_assign_splits_is_family_level_stratified_and_deterministic():
    fam_labels = {i: ("CLEAN" if i < 60 else "DUSTY" if i < 100 else "BIRD_DROP") for i in range(110)}
    s1 = ucp.assign_splits(fam_labels, seed=7)
    assert s1 == ucp.assign_splits(fam_labels, seed=7)
    assert s1 != ucp.assign_splits(fam_labels, seed=8)
    for cls in ("CLEAN", "DUSTY", "BIRD_DROP"):
        got = {s1[f] for f, c in fam_labels.items() if c == cls}
        assert got == {"train", "valid", "test"}
    n_train = sum(1 for s in s1.values() if s == "train")
    assert 0.6 < n_train / len(s1) < 0.8


def test_tiny_classes_are_never_silently_dropped():
    s = ucp.assign_splits({0: "PHYSICAL_DAMAGE", 1: "PHYSICAL_DAMAGE"}, seed=1)
    assert set(s) == {0, 1}


def test_assert_no_family_straddle():
    ucp.assert_no_family_straddle([{"family": 1, "split": "train"}, {"family": 1, "split": "train"}, {"family": 2, "split": "test"}])
    with pytest.raises(AssertionError):
        ucp.assert_no_family_straddle([{"family": 1, "split": "train"}, {"family": 1, "split": "test"}])
