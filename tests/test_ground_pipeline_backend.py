"""Classifier-backend switch in services/ground_pipeline.py (no real weights required)."""
from __future__ import annotations

import pytest
from PIL import Image

from services import ground_pipeline as gp
from utils.exceptions import ModelLoadError


class _FakeEnsemble:
    classes = ["CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE"]  # canonical, NOT the MobileNet's alphabetical order

    def __init__(self, probs):
        self._p = probs

    def predict_proba(self, image):
        return self._p


def test_default_backend_is_the_rooftop_mobilenet(monkeypatch):
    monkeypatch.delenv(gp.CLASSIFIER_BACKEND_ENV, raising=False)
    assert gp.active_classifier_backend() == "rooftop_mix"


def test_env_selects_ensemble_and_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv(gp.CLASSIFIER_BACKEND_ENV, "web_ensemble")
    assert gp.active_classifier_backend() == "web_ensemble"
    assert gp.active_classifier_backend("rooftop_mix") == "rooftop_mix"


def test_unknown_backend_fails_closed(monkeypatch):
    monkeypatch.setenv(gp.CLASSIFIER_BACKEND_ENV, "magic")
    with pytest.raises(ModelLoadError):
        gp.active_classifier_backend()


def test_missing_ensemble_manifest_fails_closed_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(gp, "ENSEMBLE_MANIFEST", tmp_path / "nope.json")
    assert gp.ensemble_available() is False
    with pytest.raises(ModelLoadError, match="web ensemble manifest not found"):
        gp.load_condition_classifier("web_ensemble")


def test_classify_crop_uses_the_ensembles_own_class_order():
    # index 2 is BIRD_DROP in the canonical order but CLEAN... in the MobileNet's alphabetical order (index 1)
    out = gp.classify_crop(_FakeEnsemble([0.05, 0.05, 0.85, 0.05]), None, Image.new("RGB", (40, 40)))
    assert out["label"] == "BIRD_DROP" and out["status"] == "predicted"
    assert set(out["probabilities"]) == set(_FakeEnsemble.classes)
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-9


def test_low_confidence_ensemble_read_is_uncertain_not_confident():
    out = gp.classify_crop(_FakeEnsemble([0.30, 0.28, 0.22, 0.20]), None, Image.new("RGB", (40, 40)))
    assert out["status"] == "uncertain" and out["label"] == "CLEAN"
