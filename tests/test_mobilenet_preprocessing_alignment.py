"""tests/test_mobilenet_preprocessing_alignment.py — Locks down the exact
canonical v1 production MobileNet preprocessing, and guards against future
accidental divergence between production inference, training, and
evaluation.

Background (Phase 6A/6B finding): the original training-time evaluation
script (training/classification/train_mobilenet.py's `eval_tf`, reused by
training/classification/evaluate_mobilenet.py) preprocesses with
torchvision's Resize(256) -> CenterCrop(224) - the standard ImageNet
convention - while the real production inference path
(utils/image_utils.py::resize_for_mobilenet, used by models/classifier.py)
resizes the shortest side directly to 224 and center-crops, with no 256px
intermediate step. These two pipelines are demonstrably NOT equivalent
(see docs/ML_EVALUATION_v1.0.0.md) and can produce different predictions
for the same source image on the same weights.

This test suite does not change that mismatch (a code change is out of
scope for the evaluation/hardening phase that added it) - it makes the
canonical production contract explicit and machine-checked, so:
  1. training/evaluation/evaluate_mobilenet.py (the audit-corrected
     evaluation script) is verified to actually use the real production
     wrapper, not a hand-rolled duplicate transform that could silently
     drift from it.
  2. Any future change to resize_for_mobilenet or the normalization
     constants breaks a test immediately, rather than silently changing
     what "production preprocessing" means.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from models.classifier import _IMAGENET_MEAN, _IMAGENET_STD, _TRANSFORM
from utils.image_utils import resize_for_mobilenet


class TestCanonicalProductionTransform:
    """The exact transform documented as canonical in
    docs/ML_HARDENING_PHASE6B.md: RGB -> resize to 224 (no 256px
    intermediate) -> ImageNet normalization."""

    def test_imagenet_normalization_constants_are_the_documented_canonical_values(self):
        assert _IMAGENET_MEAN == [0.485, 0.456, 0.406]
        assert _IMAGENET_STD == [0.229, 0.224, 0.225]

    def test_resize_for_mobilenet_output_is_224x224(self):
        img = Image.new("RGB", (800, 600), (10, 20, 30))
        out = resize_for_mobilenet(img)
        assert out.size == (224, 224)

    def test_resize_for_mobilenet_output_is_224x224_for_portrait_input(self):
        img = Image.new("RGB", (600, 900), (10, 20, 30))
        out = resize_for_mobilenet(img)
        assert out.size == (224, 224)

    def test_resize_for_mobilenet_does_not_go_through_a_256px_intermediate(self):
        """The exact distinction from the training-eval pipeline: production
        resizes the shortest side directly to 224, never to 256 first. For an
        input already exactly 256 on its shortest side, a 256-intermediate
        pipeline would center-crop 224 out of an untouched 256x*
        image (i.e. resize_for_mobilenet must NOT match
        transforms.Compose([Resize(256), CenterCrop(224)]) pixel-for-pixel
        on non-trivial input)."""
        img = Image.new("RGB", (400, 256))
        # A real gradient, not a flat color, so resizing actually changes pixels
        px = img.load()
        for x in range(400):
            for y in range(256):
                px[x, y] = (x % 256, y % 256, (x + y) % 256)

        from torchvision import transforms
        training_eval_pipeline = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

        production_out = np.asarray(resize_for_mobilenet(img))
        training_eval_out = np.asarray(training_eval_pipeline(img))

        assert production_out.shape == training_eval_out.shape == (224, 224, 3)
        # They are documented to differ - assert that difference exists,
        # so this test fails loudly (not silently) if a future change makes
        # them accidentally identical without anyone noticing the mismatch
        # was "fixed" (which would need its own deliberate, documented change
        # per docs/ML_HARDENING_PHASE6B.md, not an accident).
        assert not np.array_equal(production_out, training_eval_out)

    def test_transform_pipeline_applies_totensor_then_normalize(self):
        """_TRANSFORM must be exactly ToTensor -> Normalize(canonical
        constants), in that order - no additional steps (no extra crop,
        flip, or color jitter at inference time)."""
        steps = list(_TRANSFORM.transforms)
        assert len(steps) == 2
        assert type(steps[0]).__name__ == "ToTensor"
        assert type(steps[1]).__name__ == "Normalize"
        assert list(steps[1].mean) == pytest.approx(_IMAGENET_MEAN)
        assert list(steps[1].std) == pytest.approx(_IMAGENET_STD)

    def test_full_canonical_pipeline_is_deterministic(self):
        """Same input -> byte-identical tensor, every time (no randomness
        anywhere in the production preprocessing path)."""
        img = Image.new("RGB", (500, 333), (77, 88, 99))
        cropped = resize_for_mobilenet(img)
        t1 = _TRANSFORM(cropped)
        t2 = _TRANSFORM(resize_for_mobilenet(img))
        assert torch.equal(t1, t2)


class TestEvaluationScriptUsesRealProductionPath:
    """training/evaluation/evaluate_mobilenet.py (the Phase 6A/6B
    audit-corrected evaluation script) must classify through the real
    models.classifier.SolarFaultClassifier wrapper - never a hand-rolled
    duplicate preprocessing pipeline that could silently drift from
    production and reintroduce exactly the discrepancy this suite exists
    to catch."""

    def test_evaluate_mobilenet_script_imports_the_real_classifier_wrapper(self):
        import training.evaluation.evaluate_mobilenet as module
        source = inspect.getsource(module)
        assert "from models.classifier import SolarFaultClassifier" in source
        # Must not hand-roll its own torchvision transform pipeline.
        assert "transforms.Compose" not in source
        assert "transforms.Resize" not in source
        assert "CenterCrop" not in source

    def test_evaluate_end_to_end_script_imports_the_real_classifier_wrapper(self):
        import training.evaluation.evaluate_end_to_end as module
        source = inspect.getsource(module)
        assert "from models.classifier import SolarFaultClassifier" in source
        assert "transforms.Compose" not in source
