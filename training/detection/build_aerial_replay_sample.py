#!/usr/bin/env python3
"""training/detection/build_aerial_replay_sample.py — Deterministically
samples a small "rehearsal buffer" of aerial images from the existing
production BDAPPV aerial dataset's TRAIN split only (never val or test),
for use as an anti-catastrophic-forgetting replay set when fine-tuning
the production YOLO detector on a new ground-level domain.

This is a personal/experimental training-preparation utility, not a
license/provenance tool - the aerial dataset's own license (CC BY 4.0,
BDAPPV/IGN) and production status are unaffected. It never reads from or
writes to weights/yolo_solar.pt, the aerial val/test splits, or any
production config.

Two disjoint, seeded, non-overlapping samples are drawn from the aerial
train manifest:
  - a "replay" sample: mixed into the domain-adaptation training set
    itself, so the fine-tune keeps seeing real aerial panel examples.
  - a "monitor" sample: mixed into the training-time validation pool
    ONLY (used for checkpoint selection/early stopping during training),
    kept separate from the replay sample so it isn't literally the same
    images the model is trained on in the same run.

Neither sample ever touches the aerial val/test splits - those remain
100% untouched and are used only for the final, independent aerial
regression evaluation after training completes.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def build_replay_sample(
    aerial_train_root: Path,
    output_root: Path,
    replay_count: int,
    monitor_count: int,
    seed: int = 20260906,
) -> dict:
    images_dir = aerial_train_root / "images"
    labels_dir = aerial_train_root / "labels"
    all_stems = sorted(p.stem for p in images_dir.glob("*.png"))
    if len(all_stems) < replay_count + monitor_count:
        raise ValueError(
            f"aerial train split only has {len(all_stems)} images, need "
            f"{replay_count + monitor_count} for replay+monitor"
        )

    rng = random.Random(seed)
    shuffled = all_stems[:]
    rng.shuffle(shuffled)
    replay_stems = sorted(shuffled[:replay_count])
    monitor_stems = sorted(shuffled[replay_count:replay_count + monitor_count])
    assert not (set(replay_stems) & set(monitor_stems)), "replay/monitor overlap - must never happen"

    for subset_name, stems in (("replay", replay_stems), ("monitor", monitor_stems)):
        dest_images = output_root / subset_name / "images"
        dest_labels = output_root / subset_name / "labels"
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.copy2(images_dir / f"{stem}.png", dest_images / f"{stem}.png")
            shutil.copy2(labels_dir / f"{stem}.txt", dest_labels / f"{stem}.txt")

    manifest = {
        "source": str(aerial_train_root),
        "seed": seed,
        "replay_count": len(replay_stems),
        "monitor_count": len(monitor_stems),
        "replay_stems": replay_stems,
        "monitor_stems": monitor_stems,
        "note": (
            "Both samples drawn exclusively from the aerial TRAIN split. "
            "Aerial val/test were never read by this script."
        ),
    }
    (output_root / "replay_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aerial-train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=200)
    parser.add_argument("--monitor-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    manifest = build_replay_sample(
        args.aerial_train_root, args.output_root,
        args.replay_count, args.monitor_count, args.seed,
    )
    print(f"Replay: {manifest['replay_count']} images, Monitor: {manifest['monitor_count']} images")
    print(f"Written under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
