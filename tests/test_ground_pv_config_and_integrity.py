"""tests/test_ground_pv_config_and_integrity.py — the research config is
well-formed and consistent with the code, the Kaggle notebook is valid, and
the ground-PV work does not endanger the frozen production artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from training.detection.ground_pv.class_map import RESEARCH_CLASS_NAMES

_ROOT = Path(__file__).resolve().parents[1]

PROD_YOLO_SHA = "0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea"
V1_TAG_COMMIT = "4b0287dc98ad68f4fda8710fb522d6d37b08ffad"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((_ROOT / "configs" / "ground_pv_research.yaml").read_text(encoding="utf-8"))


def test_config_names_match_code_taxonomy(cfg):
    assert {int(k): v for k, v in cfg["names"].items()} == RESEARCH_CLASS_NAMES


def test_config_never_targets_the_production_weights(cfg):
    assert cfg["research_model"]["weights"] == "weights/yolo_solar_ground_research_v1.pt"
    assert "yolo_solar.pt" not in cfg["research_model"]["weights"]
    assert "yolo_solar.pt" not in cfg["research_model"]["base_model"]
    assert "hotspot" not in yaml.safe_dump(cfg).lower()


def test_config_defect_to_condition_is_complete(cfg):
    assert cfg["defect_to_condition"] == {"bird_drop": "BirdDrop", "cracked": "Damage", "dusty": "Dust"}
    assert cfg["no_defect_condition"] == "Clean"


def test_augmentation_is_conservative(cfg):
    aug = cfg["augmentation"]
    assert aug["flipud"] == 0.0          # panels have a physical "up"
    assert aug["mixup"] == 0.0           # no fabricated defect blends
    assert aug["copy_paste"] == 0.0
    assert aug["degrees"] <= 15.0
    assert aug["perspective"] == 0.0


def test_split_policy_preserves_official(cfg):
    assert cfg["splits"]["policy"] == "preserve_official"


def test_balanced_experiment_does_not_touch_val_or_test(cfg):
    strat = cfg["experiments"]["balanced"]["strategy"]
    assert "train_only" in strat


def test_scripts_expose_help():
    import subprocess
    import sys
    for name in ("download_ground_pv_dataset", "audit_ground_pv_dataset",
                 "train_ground_pv", "evaluate_ground_pv", "infer_ground_pv"):
        r = subprocess.run([sys.executable, str(_ROOT / "scripts" / f"{name}.py"), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)
        assert "usage" in r.stdout.lower()


def test_download_script_resolves_key_via_env_or_project_secret_resolver():
    src = (_ROOT / "scripts" / "download_ground_pv_dataset.py").read_text(encoding="utf-8")
    assert 'API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"' in src
    # resolved via the project's get_secret (env / .env / streamlit) or the raw env var
    assert "get_secret" in src
    assert "os.environ.get(API_KEY_ENV_VAR" in src
    # no hard-coded / literal key assignment
    assert 'api_key = "' not in src.replace('api_key="<key>"', "")


def test_kaggle_notebook_is_valid_and_gpu_configured():
    nb = json.loads((_ROOT / "kaggle" / "ground_pv_train.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"].get("accelerator") == "GPU"
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "torch.cuda.is_available()" in text
    assert "yolo11n.pt" in text
    assert "ground_yolo11n_baseline_v1" in text
    # never initializes / loads the aerial production model
    assert "YOLO('weights/yolo_solar.pt')" not in text
    assert 'YOLO("weights/yolo_solar.pt")' not in text
    assert "yolo_solar_ground_research" in text or "yolo_solar_ground" in text or "best.pt" in text


def test_train_script_refuses_production_base_model():
    src = (_ROOT / "scripts" / "train_ground_pv.py").read_text(encoding="utf-8")
    assert "refusing to initialize the research model" in src
    assert "yolo_solar" in src


def test_production_yolo_weights_unchanged_if_present():
    p = _ROOT / "weights" / "yolo_solar.pt"
    if not p.is_file():
        pytest.skip("weights/yolo_solar.pt not present (gitignored; absent on a fresh checkout)")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == PROD_YOLO_SHA


def test_production_manifest_untouched():
    manifest = json.loads((_ROOT / "weights" / "manifest.json").read_text(encoding="utf-8"))
    entry = next(a for a in manifest["artifacts"] if a["path"] == "yolo_solar.pt")
    assert entry["sha256"] == PROD_YOLO_SHA
    assert manifest["release"] == "v1.0.0"


def test_evaluate_ground_pv_reads_class_names_from_data_yaml(tmp_path):
    """The eval script's dataset-stats must use the dataset's own names,
    not a hardcoded 4-class order (which mislabelled the 3-class counts)."""
    import sys
    sys.path.insert(0, str(_ROOT))
    from scripts.evaluate_ground_pv import _class_names, _split_stats
    import yaml as _yaml
    from PIL import Image

    root = tmp_path / "ds"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n2 0.5 0.5 0.2 0.2\n")
    dy = root / "data.yaml"
    dy.write_text(_yaml.safe_dump({"path": str(root), "train": "train/images",
                                   "names": {0: "panel", 1: "cracked", 2: "dusty"}}))
    assert _class_names(dy) == {0: "panel", 1: "cracked", 2: "dusty"}
    stats = _split_stats(dy, "train")
    assert stats["instances_per_class"] == {"panel": 1, "cracked": 0, "dusty": 1}
    assert "bird_drop" not in stats["instances_per_class"]
