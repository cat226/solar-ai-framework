#!/usr/bin/env python3
"""scripts/build_representative_panel_dataset.py — Phase 14B dataset
assembly CLI.

Scans a directory of raw candidate images (read-only — nothing here ever
modifies, moves, or deletes a source image), builds the source manifest,
groups images into scenes (near-duplicate bursts/re-captures), assigns a
deterministic scene-isolated train/valid/test split, and writes empty YOLO
label-file stubs ready for a human annotator to fill in.

Does NOT annotate anything. Does NOT run the current detector to generate
labels. Does NOT train anything.

Example
-------
    python scripts/build_representative_panel_dataset.py \
        --raw-images-dir "D:\\solar_ai_training_scratch\\Solar Real Images" \
        --capture-source first_party \
        --id-prefix firstparty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv import paths  # noqa: E402
from training.detection.ground_pv.representative_dataset import (  # noqa: E402
    DEFAULT_SPLIT_RATIOS,
    assign_scene_isolated_splits,
    build_source_manifest,
    group_into_scenes,
    save_source_manifest,
    save_split_manifest,
    verify_no_scene_crosses_splits,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-images-dir", type=Path, required=True,
                        help="Directory of source images to register (recursed). Read-only.")
    parser.add_argument("--capture-source", choices=["first_party", "public_supplemental", "unknown"],
                        required=True)
    parser.add_argument("--provenance-status", default="first_party")
    parser.add_argument("--id-prefix", default="img")
    parser.add_argument("--dhash16-merge-hamming", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["valid"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-out", type=Path, default=None,
                        help=f"Default: {paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR}/source_manifest.json")
    parser.add_argument("--split-out", type=Path, default=None,
                        help=f"Default: {paths.REPRESENTATIVE_PANEL_V1_SPLITS_DIR}/split_manifest.csv")
    parser.add_argument("--write-empty-label-stubs", action="store_true",
                        help="Create an empty .txt label file per image (annotation-ready, NOT annotated).")
    parser.add_argument("--labels-out-dir", type=Path, default=None,
                        help=f"Default: {paths.REPRESENTATIVE_PANEL_V1_ANNOTATIONS_DIR}")
    parser.add_argument("--reference-only", action="store_true",
                        help="Register these images (e.g. the six owner photos) as a QUALITATIVE "
                             "REFERENCE SET — every image gets split='reference', never assigned "
                             "into train/valid/test by the proportional splitter (Phase 14B req 23: "
                             "'do not mix them into the training split automatically').")
    args = parser.parse_args(argv)

    if not args.raw_images_dir.is_dir():
        raise SystemExit(f"raw images dir not found: {args.raw_images_dir}")

    ratios = {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio}
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"split ratios must sum to 1.0, got {ratio_sum}")

    print(f"Scanning {args.raw_images_dir} ...")
    entries, corrupt = build_source_manifest(
        args.raw_images_dir, capture_source=args.capture_source,
        provenance_status=args.provenance_status, id_prefix=args.id_prefix,
    )
    if corrupt:
        print(f"WARNING: {len(corrupt)} unreadable file(s) skipped (not silently dropped — see manifest):")
        for c in corrupt:
            print(f"  {c['path']}: {c['error']}")

    print(f"Grouping {len(entries)} image(s) into scenes ...")
    entries, near_pairs = group_into_scenes(entries, dhash16_merge_hamming=args.dhash16_merge_hamming)
    scene_ids = {e.scene_id for e in entries}
    print(f"  {len(scene_ids)} scene(s) from {len(entries)} image(s) "
          f"({len(near_pairs)} near-duplicate pair(s) merged)")

    if args.reference_only:
        print("Reference-only mode: assigning split='reference' to every image "
              "(NOT run through the proportional train/valid/test splitter).")
        image_to_split = {e.image_id: "reference" for e in entries}
        check = {"ok": True, "crossing_scenes": {}}
    else:
        print("Assigning scene-isolated splits ...")
        split_result = assign_scene_isolated_splits(entries, ratios=ratios, seed=args.seed)
        image_to_split = split_result.image_to_split
        check = verify_no_scene_crosses_splits(entries, image_to_split)
        if not check["ok"]:
            raise SystemExit(f"INTERNAL ERROR: split assignment produced a cross-scene leak: {check['crossing_scenes']}")

    manifest_out = args.manifest_out or (paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR / "source_manifest.json")
    split_out = args.split_out or (paths.REPRESENTATIVE_PANEL_V1_SPLITS_DIR / "split_manifest.csv")
    save_source_manifest(entries, manifest_out)
    save_split_manifest(entries, image_to_split, split_out)

    near_dup_out = manifest_out.parent / "near_duplicate_pairs.json"
    near_dup_out.parent.mkdir(parents=True, exist_ok=True)
    near_dup_out.write_text(json.dumps(near_pairs, indent=2), encoding="utf-8")

    if args.write_empty_label_stubs:
        labels_dir = args.labels_out_dir or paths.REPRESENTATIVE_PANEL_V1_ANNOTATIONS_DIR
        labels_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for e in entries:
            stub = labels_dir / f"{Path(e.original_filename).stem}.txt"
            if not stub.exists():
                stub.write_text("", encoding="utf-8")
                written += 1
        print(f"Wrote {written} empty label stub(s) to {labels_dir} (NOT annotated — for a human to fill in)")

    split_counts: dict[str, int] = {}
    for split in image_to_split.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    scenes_per_split = {
        s: len({e.scene_id for e in entries if image_to_split[e.image_id] == s})
        for s in set(image_to_split.values())
    }

    print(f"\nimages: {len(entries)}   scenes: {len(scene_ids)}   corrupt/skipped: {len(corrupt)}")
    print(f"splits (images): {split_counts}")
    print(f"scenes per split: {scenes_per_split}")
    print(f"no scene crosses a split: {check['ok']}")
    print(f"\nmanifest: {manifest_out}")
    print(f"split manifest: {split_out}")
    print(f"near-duplicate pairs: {near_dup_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
