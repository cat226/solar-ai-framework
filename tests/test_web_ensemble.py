"""Ensemble inference plumbing with tiny randomly-initialised models (no real weights needed)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

_p = Path(__file__).resolve().parent.parent / "scripts" / "infer_web_ensemble.py"
_spec = importlib.util.spec_from_file_location("iwe", _p)
iwe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iwe)

CLASSES = ["CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE"]


@pytest.mark.parametrize("arch", ["convnext_tiny", "convnext_small", "convnext_base", "efficientnet_v2_s"])
def test_every_arch_builds_a_four_way_head(arch):
    m = iwe.build_model(arch, 4).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 64, 64)).shape == (1, 4)


def test_unknown_arch_fails_closed():
    with pytest.raises(KeyError):
        iwe.build_model("resnet18", 4)


def test_ensemble_averages_and_returns_a_distribution(tmp_path):
    members = []
    for i, arch in enumerate(["convnext_tiny", "efficientnet_v2_s"]):
        torch.manual_seed(i)
        m = iwe.build_model(arch, 4)
        torch.save(m.state_dict(), tmp_path / f"m{i}.pth")
        members.append({"arch": arch, "size": 64, "file": f"m{i}.pth"})
    (tmp_path / "ensemble.json").write_text(json.dumps({"classes_index_order": CLASSES, "members": members}))
    ens = iwe.WebEnsemble(tmp_path / "ensemble.json")
    p = ens.predict_proba(Image.new("RGB", (80, 50), (30, 60, 150)))
    assert len(p) == 4 and abs(sum(p) - 1) < 1e-4 and all(0 <= v <= 1 for v in p)
    assert ens.classes == CLASSES
