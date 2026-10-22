"""Tests for the owner condition-labeling helpers and a Streamlit AppTest smoke of
scripts/label_panel_conditions.py (detector stubbed; no weights needed)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from training.detection.ground_pv import owner_condition_labels as ocl
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = str(_REPO_ROOT / "scripts" / "label_panel_conditions.py")


def _panel(x1=10, y1=10, x2=60, y2=90, conf=0.9):
    return {"bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "confidence": conf}


def _photo(folder: Path, name="a.jpg", color=(40, 80, 160)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    Image.new("RGB", (200, 120), color).save(p)
    return p


def test_labels_are_the_active_taxonomy_plus_skip():
    assert ocl.LABELS[:4] == CONDITION_4CLASS_ORDER and ocl.LABELS[4] == "SKIP"


def test_build_rows_requires_one_valid_label_per_panel():
    panels = [_panel(), _panel(x1=70, x2=120)]
    with pytest.raises(ValueError):
        ocl.build_rows("a.jpg", "h", panels, ["CLEAN"])
    with pytest.raises(ValueError):
        ocl.build_rows("a.jpg", "h", panels, ["CLEAN", "ELECTRICAL_DAMAGE"])
    rows = ocl.build_rows("a.jpg", "h", panels, ["CLEAN", "BIRD_DROP"], simulated=True, missed_panels=1, notes="x\ny")
    assert [r["label"] for r in rows] == ["CLEAN", "BIRD_DROP"]
    assert all(r["label_source"] == "owner" and r["simulated"] == 1 and r["missed_panels_in_image"] == 1 for r in rows)
    assert rows[0]["notes"] == "x y"


def test_append_summarize_and_resume_key_on_hash(tmp_path):
    img = _photo(tmp_path / "imgs")
    out = tmp_path / "o" / "labels.csv"
    sha = ocl.sha256_file(img)
    assert ocl.append_rows(out, ocl.build_rows(img.name, sha, [_panel(), _panel()], ["CLEAN", "DUSTY"])) == 2
    assert ocl.append_rows(out, []) == 0
    s = ocl.summarize(out)
    assert s["images"] == 1 and s["panels"] == 2 and s["by_label"]["DUSTY"] == 1
    assert (img.name, sha) in ocl.labeled_image_keys(out)
    # same name, different content (e.g. an edited/replaced file) is NOT considered labeled
    Image.new("RGB", (200, 120), (1, 2, 3)).save(img)
    assert (img.name, ocl.sha256_file(img)) not in ocl.labeled_image_keys(out)
    with out.open(newline="", encoding="utf-8") as fh:
        assert tuple(csv.DictReader(fh).fieldnames) == ocl.CSV_FIELDS


def test_list_images_sorted_and_missing_folder(tmp_path):
    _photo(tmp_path, "b.jpg"); _photo(tmp_path, "a.png"); (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in ocl.list_images(tmp_path)] == ["a.png", "b.jpg"]
    with pytest.raises(FileNotFoundError):
        ocl.list_images(tmp_path / "nope")


def test_labeling_ui_gates_save_until_every_panel_labeled(tmp_path, monkeypatch):
    img = _photo(tmp_path / "imgs")
    out = tmp_path / "labels.csv"
    monkeypatch.setenv("LABEL_TOOL_IMAGES", str(img.parent))
    monkeypatch.setenv("LABEL_TOOL_OUT", str(out))
    monkeypatch.setattr(ocl, "propose_panels", lambda *a, **k: [_panel(), _panel(x1=70, x2=120)])
    monkeypatch.setattr("streamlit.cache_resource", lambda *a, **k: (lambda f: f), raising=False)
    import services.ground_pipeline  # noqa: F401  (PANEL_WEIGHTS import inside the script)

    at = AppTest.from_file(_SCRIPT, default_timeout=60)
    monkeypatch.setattr("ultralytics.YOLO", lambda *_: object())
    at.run()
    assert not at.exception, at.exception
    save = next(b for b in at.button if b.label == "Save and next")
    assert save.disabled and not out.exists()

    for r in at.radio:
        r.set_value("CLEAN" if r.key.endswith(":0") else "BIRD_DROP")
    at.run()
    next(b for b in at.button if b.label == "Save and next").click().run()
    assert not at.exception, at.exception
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["label"] for r in rows] == ["CLEAN", "BIRD_DROP"]
