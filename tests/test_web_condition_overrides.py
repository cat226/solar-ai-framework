"""apply_overrides fails closed and keeps folds/families intact (pure data, no images)."""
import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parent.parent / "scripts" / "build_web_condition_clean_package.py"
_spec = importlib.util.spec_from_file_location("bwc", _p)
bwc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bwc)

ROWS = [{"image_id": f"i{k}", "label": l, "fold": str(k % 5), "family": str(k)} for k, l in
        enumerate(["CLEAN", "DUSTY", "DUSTY", "BIRD_DROP"])]


def _o(i, action, orig, new=""):
    return {"image_id": i, "action": action, "original_label": orig, "new_label": new}


def test_drop_and_relabel_keep_everything_else():
    out, s = bwc.apply_overrides(ROWS, [_o("i0", "drop", "CLEAN"), _o("i1", "relabel", "DUSTY", "BIRD_DROP")])
    assert [r["image_id"] for r in out] == ["i1", "i2", "i3"] and out[0]["label"] == "BIRD_DROP"
    assert out[0]["fold"] == "1" and s == {"dropped": 1, "relabelled": 1}
    assert ROWS[1]["label"] == "DUSTY"  # input untouched


@pytest.mark.parametrize("bad", [
    _o("nope", "drop", "CLEAN"), _o("i0", "drop", "DUSTY"), _o("i1", "relabel", "DUSTY", "ELECTRICAL_DAMAGE"),
    _o("i1", "relabel", "DUSTY", ""), _o("i1", "merge", "DUSTY")])
def test_bad_overrides_fail_closed(bad):
    with pytest.raises((KeyError, ValueError)):
        bwc.apply_overrides(ROWS, [bad])
