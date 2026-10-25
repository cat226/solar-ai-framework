#!/usr/bin/env python3
"""scripts/import_first_party_panels.py — Phase 14B.1 batch-collection CLI.

Import a batch of freshly-captured (or otherwise first-party) camera images
into the representative panel dataset: preserve originals, compute SHA-256,
reject exact duplicates (against this batch AND every prior batch), extract
metadata, assign deterministic image IDs continuing from the existing
manifest, and merge everything into the source manifest.

Never converts, resizes, or re-encodes a source image on ingestion. Never
overwrites an existing file. Never invents scene grouping it can't prove —
by default every new image starts `scene_status = "needs_review"`; pass
`--scene-id` (whole batch = one scene) or `--scene-map` (per-file CSV) to
record what you, the human importing this batch, actually know. After
import, exact-duplicate / near-duplicate AUTOMATIC grouping still runs on
top (it can only ADD to what you've told it, never override it).

Never annotates anything and never calls the detector — see
scripts/generate_annotation_proposals.py for optional, clearly-labeled
detector-assisted DRAFT proposals (not ground truth).

Examples
--------
    # A batch where every photo is one physical array/session
    python scripts/import_first_party_panels.py \
        --source "D:\\path\\to\\new\\photos" \
        --batch-id batch_001 \
        --scene-id scene_010

    # A batch with mixed scenes, explicitly mapped per file
    python scripts/import_first_party_panels.py \
        --source "D:\\path\\to\\new\\photos" \
        --batch-id batch_002 \
        --scene-map scene_map.csv   # columns: filename,scene_id

    # Scene identity genuinely unknown — leave it for review
    python scripts/import_first_party_panels.py \
        --source "D:\\path\\to\\new\\photos" \
        --batch-id batch_003
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.detection.ground_pv import paths  # noqa: E402
from training.detection.ground_pv.representative_dataset import (  # noqa: E402
    group_into_scenes,
    import_batch_into_manifest,
    load_source_manifest,
    save_source_manifest,
)


def _load_scene_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {row["filename"]: row["scene_id"] for row in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Directory of newly-captured images. Read-only.")
    parser.add_argument("--batch-id", required=True, help="e.g. batch_001 — every image in this run is tagged with it.")
    parser.add_argument("--source-type", choices=["first_party", "public_supplemental", "unknown"],
                        default="first_party")
    parser.add_argument("--provenance-status", default="first_party")
    parser.add_argument("--id-prefix", default="img",
                        help="Must match the prefix already used in the manifest for id numbering to continue "
                             "correctly; a new prefix starts its own independent counter.")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--capture-timestamp", default=None, help="Free-form (e.g. ISO 8601) if known.")
    parser.add_argument("--scene-id", default=None,
                        help="Assign this ONE scene_id to every image in the batch (common case: a whole "
                             "batch is one physical array/session). Mutually exclusive with --scene-map.")
    parser.add_argument("--scene-map", type=Path, default=None,
                        help="CSV with columns filename,scene_id for per-file scene assignment. "
                             "Mutually exclusive with --scene-id.")
    parser.add_argument("--dhash16-merge-hamming", type=int, default=8,
                        help="Threshold for the automatic near-duplicate re-grouping pass run after import.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"Default: {paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR}/source_manifest.json")
    parser.add_argument("--raw-dest-root", type=Path, default=None,
                        help=f"Default: {paths.REPRESENTATIVE_PANEL_V1_RAW_DIR}")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        raise SystemExit(f"source dir not found: {args.source}")
    if args.scene_id and args.scene_map:
        raise SystemExit("pass at most one of --scene-id / --scene-map, not both")

    manifest_path = args.manifest or (paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR / "source_manifest.json")
    raw_dest_root = args.raw_dest_root or paths.REPRESENTATIVE_PANEL_V1_RAW_DIR

    existing_entries = load_source_manifest(manifest_path) if manifest_path.is_file() else []
    print(f"existing manifest: {len(existing_entries)} image(s)" if existing_entries
          else "no existing manifest — starting a new one")

    scene_map = _load_scene_map(args.scene_map) if args.scene_map else None
    if scene_map is None and args.scene_id is None:
        print("NOTE: no --scene-id or --scene-map given — every new image will be "
              "scene_status='needs_review' (Phase 14B.1 req 4: unknown scene grouping is "
              "never pretended to be solved automatically). Automatic near-duplicate "
              "grouping still runs and may resolve some of them.")

    result = import_batch_into_manifest(
        args.source, existing_entries=existing_entries, raw_dest_root=raw_dest_root,
        batch_id=args.batch_id, source_type=args.source_type, provenance_status=args.provenance_status,
        id_prefix=args.id_prefix, camera_id=args.camera_id, capture_timestamp=args.capture_timestamp,
        scene_id_for_batch=args.scene_id, scene_map=scene_map,
    )

    if result.corrupt:
        print(f"\n{len(result.corrupt)} unreadable file(s) skipped (not silently dropped):")
        for c in result.corrupt:
            print(f"  {c['path']}: {c['error']}")
    if result.rejected_duplicates:
        print(f"\n{len(result.rejected_duplicates)} duplicate file(s) rejected (not imported):")
        for d in result.rejected_duplicates:
            print(f"  {d['path']}: {d['reason']}")

    all_entries, near_pairs = group_into_scenes(result.all_entries, dhash16_merge_hamming=args.dhash16_merge_hamming)
    save_source_manifest(all_entries, manifest_path)

    near_dup_out = manifest_path.parent / "near_duplicate_pairs.json"
    near_dup_out.write_text(json.dumps(near_pairs, indent=2), encoding="utf-8")

    needs_review = sum(1 for e in result.imported if e.scene_status == "needs_review")
    print(f"\nimported: {len(result.imported)}   rejected duplicates: {len(result.rejected_duplicates)}   "
          f"corrupt: {len(result.corrupt)}")
    print(f"of the imported images: {needs_review} still scene_status='needs_review' after auto-grouping")
    print(f"manifest now has {len(all_entries)} total image(s) across {len({e.scene_id for e in all_entries})} scene(s)")
    print(f"\nmanifest: {manifest_path}")
    print(f"raw images copied to: {raw_dest_root / args.batch_id}")
    print(f"\nNext: run scripts/audit_representative_panel_dataset.py to see collection progress, "
          "or annotate the new images (one physical panel = one box; see "
          "docs/FIRST_PARTY_COLLECTION_GUIDE.md), then mark annotation_status="
          "'annotated_human_verified' and run sync_panel_count_gt_from_annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
