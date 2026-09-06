#!/usr/bin/env python3
"""training/detection/build_aerial_eval_sample.py — Deterministically
samples a held-out evaluation subset from the production aerial
dataset's TEST split ONLY (never train or val), for the independent
before/after aerial-regression check in the personal domain-adaptation
experiment. This sample is never used for training or for
checkpoint-selection monitoring - see build_aerial_replay_sample.py for
those (which draw exclusively from the TRAIN split, by design, so the
two scripts can never touch overlapping images)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def build_eval_sample(aerial_test_root: Path, output_root: Path, count: int, seed: int) -> dict:
    images_dir = aerial_test_root / "images"
    labels_dir = aerial_test_root / "labels"
    all_stems = sorted(p.stem for p in images_dir.glob("*.png"))
    if len(all_stems) < count:
        raise ValueError(f"aerial test split only has {len(all_stems)} images, need {count}")

    rng = random.Random(seed)
    shuffled = all_stems[:]
    rng.shuffle(shuffled)
    sample_stems = sorted(shuffled[:count])

    dest_images = output_root / "images"
    dest_labels = output_root / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for stem in sample_stems:
        shutil.copy2(images_dir / f"{stem}.png", dest_images / f"{stem}.png")
        shutil.copy2(labels_dir / f"{stem}.txt", dest_labels / f"{stem}.txt")

    manifest = {
        "source": str(aerial_test_root),
        "seed": seed,
        "count": len(sample_stems),
        "full_test_split_size": len(all_stems),
        "stems": sample_stems,
        "note": "Drawn exclusively from the aerial TEST split - never used in training or checkpoint-selection monitoring.",
    }
    (output_root / "eval_sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aerial-test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    manifest = build_eval_sample(args.aerial_test_root, args.output_root, args.count, args.seed)
    print(f"Eval sample: {manifest['count']} / {manifest['full_test_split_size']} aerial test images")
    print(f"Written under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
