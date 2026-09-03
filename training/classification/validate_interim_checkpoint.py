"""Validate an interim (non-production, class-subset) MobileNet checkpoint against
this project's REAL inference plumbing.

This does NOT wire the checkpoint into the app and does NOT touch
``weights/mobilenet_solar.pth`` or ``configs/settings.yaml``. It exists to answer one
question: does the checkpoint actually work when run through the exact preprocessing,
model-construction, and inference code the production app uses (not a reimplementation
of it that could silently drift)?

It does two things:

1. **Correct-label path**: builds the model exactly as
   :meth:`models.model_manager.ModelManager._load_classifier` does (same
   ``torchvision.models.mobilenet_v2`` construction, same ``classifier[1]`` replacement
   pattern, same ``torch.load(..., weights_only=True)``), preprocesses real held-out test
   images with the actual ``utils.image_utils.resize_for_mobilenet``, and runs them
   through the actual :class:`models.classifier.SolarFaultClassifier` — but with its
   class-label list temporarily overridden to the checkpoint's own (subset) classes
   instead of the six production labels. Reports accuracy on this held-out sample.

2. **Danger demonstration**: runs the exact same checkpoint through
   :class:`SolarFaultClassifier` completely UNMODIFIED (i.e. with the real six
   production labels from ``configs/settings.yaml``) to concretely show why this
   checkpoint must never be swapped in as-is: with only 3 output logits against 6
   labels, ``zip(_LABELS, probs)`` silently truncates and mislabels — a Hotspot
   image would be reported under the label "Bird-Drop" (production label index 2),
   not "Hotspot". This is a demonstration, not a fix; the fix is "never do this."

Usage::

    python training/classification/validate_interim_checkpoint.py \\
        --checkpoint weights/mobilenet_interim_3class.pth \\
        --classes Clean Dusty Hotspot \\
        --test-root D:/solar_ai_training_scratch/prepared/test \\
        --samples-per-class 10
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torchvision import models

# Make the repo root (two levels up from training/classification/) importable so
# `models.*` / `utils.*` resolve to the app's packages, matching scripts/check_runtime_readiness.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import models.classifier as classifier_module
from models.classifier import SolarFaultClassifier
from utils.image_utils import load_pil_image


def _build_model(checkpoint: Path, num_classes: int) -> torch.nn.Module:
    """Mirrors ModelManager._load_classifier's construction exactly."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _sample_test_images(test_root: Path, classes: list[str], n: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for cls in classes:
        files = sorted((test_root / cls).glob("*"))
        chosen = rng.sample(files, min(n, len(files)))
        samples.extend((f, cls) for f in chosen)
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--classes", type=str, nargs="+", required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--samples-per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    model = _build_model(args.checkpoint, len(args.classes))
    samples = _sample_test_images(args.test_root, args.classes, args.samples_per_class, args.seed)
    if not samples:
        raise SystemExit(f"no test images found under {args.test_root} for classes {args.classes}")

    # --- 1. Correct-label path: real preprocessing + real classify(), correct labels ---
    original_labels = classifier_module._LABELS
    classifier_module._LABELS = args.classes  # scoped override, restored below
    try:
        clf = SolarFaultClassifier()
        clf.set_model(model)
        correct = 0
        rows = []
        for path, true_label in samples:
            img = load_pil_image(path)
            result = clf.classify(img)
            ok = result.label == true_label
            correct += int(ok)
            rows.append((path.name, true_label, result.label, result.confidence, ok))
        print(f"=== Correct-label validation (real resize_for_mobilenet + real classify()) ===")
        print(f"classes={args.classes}")
        for name, true_label, pred_label, conf, ok in rows:
            flag = "OK" if ok else "MISMATCH"
            print(f"  [{flag}] {name}: true={true_label} pred={pred_label} conf={conf:.4f}")
        acc = correct / len(samples)
        print(f"accuracy={acc:.4f} ({correct}/{len(samples)})")
    finally:
        classifier_module._LABELS = original_labels

    # --- 2. Danger demonstration: same checkpoint, UNMODIFIED production labels ---
    print()
    print("=== Danger demonstration: same checkpoint through UNMODIFIED production classify() ===")
    print(f"production labels (from configs/settings.yaml) = {classifier_module._LABELS}")
    demo_path, demo_true_label = samples[-1]
    clf2 = SolarFaultClassifier()
    clf2.set_model(model)
    img = load_pil_image(demo_path)
    bad_result = clf2.classify(img)
    print(f"  image: {demo_path.name} (true class: {demo_true_label})")
    print(f"  UNMODIFIED classify() reports label='{bad_result.label}' "
          f"(this is WRONG whenever the true class isn't literally first-N of the "
          f"production order — never inject a subset checkpoint into the real "
          f"classifier without also overriding its label list).")
    print(f"  probabilities dict returned: {bad_result.probabilities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
