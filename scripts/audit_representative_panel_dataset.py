#!/usr/bin/env python3
"""scripts/audit_representative_panel_dataset.py — Phase 14B / 14B.1
dataset audit + report CLI.

Loads the source manifest + split manifest built by
scripts/build_representative_panel_dataset.py or
scripts/import_first_party_panels.py, validates any existing annotations
(including a `panel_count_gt` vs actual-verified-box-count disagreement
check), computes panel-count / resolution / duplicate statistics, and
renders TWO reports from the same computed payload:

  - `dataset_report.md` (+ .json) — the full audit narrative.
  - `collection_progress.md` — a shorter, skimmable collection-progress
    view (Phase 14B.1 req 8), for checking status during ongoing
    collection without reading the full report.

Applies the Phase 14B / 14B.1 "Final Decision Rule" exactly: the dataset is
reported ready for Phase 14C ONLY if every hard requirement is met (>=100
VERIFIED independent scenes — not just images registered — at least one
verified 4-panel, one 6-panel, and one 8-panel scene, human-verified
annotations, scene-isolated splits, zero cross-split exact/near
duplicates, a genuine test holdout, difficult-case tag coverage confirmed
by a human reviewer, native aspect ratios, documented provenance) — never
"ready" merely because the code ran without error. Exits non-zero when NOT
ready, mirroring the fail-closed convention of the existing
dataset_audit.py / remediate.py.

Example
-------
    python scripts/audit_representative_panel_dataset.py
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
    audit_panel_count_gt_disagreements,
    build_readiness_checklist,
    compute_duplicate_leakage_report,
    compute_panel_count_stats,
    compute_resolution_and_aspect_stats,
    load_source_manifest,
    validate_representative_annotations,
    verify_no_scene_crosses_splits,
)


def _load_split_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["image_id"]: row["split"] for row in csv.DictReader(fh)}


def _render_dataset_report_md(payload: dict) -> str:
    L: list[str] = []
    L.append("# Representative panel dataset (Phase 14B) — dataset report\n")
    L.append(
        "**Purpose:** assemble a deployment-domain dataset (camera-realistic, "
        "ground-level, individual-panel, native-aspect-ratio) to fix the two "
        "real-world failure modes confirmed in Phase 14A — a missed large "
        "foreground panel, and multiple physical panels represented as one "
        "group box. This report states plainly whether that dataset exists "
        "yet, never merely that the tooling ran.\n"
    )

    c = payload["readiness_checklist"]
    verdict = "READY FOR PHASE 14C" if c["ready_for_phase_14c"] else "NOT READY FOR PHASE 14C"
    L.append(f"## Verdict: **{verdict}**\n")
    if not c["ready_for_phase_14c"]:
        L.append("**Blocking reasons:**\n")
        for r in c["blocking_reasons"]:
            L.append(f"- {r}")
        L.append("")

    L.append("## 1. Source inventory\n")
    inv = payload["inventory"]
    L.append(f"- images registered: **{inv['num_images']}**")
    L.append(f"- scenes: **{inv['num_scenes']}**")
    L.append(f"- verified independent scenes (counts toward Phase 14C readiness): "
             f"**{c['verified_scene_count']}** (spec minimum: 100)")
    L.append(f"- corrupt/unreadable files skipped: {inv['num_corrupt']}")
    L.append(f"- source types: {inv['source_type_counts']}")
    L.append(f"- provenance statuses: {inv['provenance_status_counts']}\n")

    L.append("## 2. Resolution / aspect-ratio distribution\n")
    r = payload["resolution_stats"]
    if r.get("count"):
        L.append(f"- width range: {r['width_min_max']}   height range: {r['height_min_max']}")
        L.append(f"- aspect ratio range: {r['aspect_ratio_min_max']}")
        L.append(f"- orientation counts: {r['orientation_counts']}")
        L.append(f"- unique resolutions: {r['unique_resolutions']}\n")
    else:
        L.append("- no images registered.\n")

    L.append("## 3. Panel-count distribution\n")
    p = payload["panel_count_stats"]
    L.append(f"- images with a VERIFIED panel count (`annotation_status == "
             f"'annotated_human_verified'` only): {p['images_with_known_panel_count']} / {p['images_total']}")
    L.append(f"- images by panel count: {p['images_by_panel_count']}")
    L.append(f"- scenes by panel count: {p['scenes_by_panel_count']}")
    L.append(f"- target counts (4/6/8) covered: {p['target_counts_covered']}")
    L.append(f"- target counts (4/6/8) MISSING: {p['target_counts_missing']}")
    L.append(f"- verified 4-panel scene exists: {c['has_verified_4_panel_scene']}   "
             f"6-panel: {c['has_verified_6_panel_scene']}   8-panel: {c['has_verified_8_panel_scene']}\n")

    disagreements = payload.get("panel_count_gt_disagreements", [])
    if disagreements:
        L.append(f"**WARNING — {len(disagreements)} `panel_count_gt` disagreement(s) found** "
                 "(a stored value doesn't match the actual verified box count — run "
                 "`sync_panel_count_gt_from_annotations` to fix): " + str(disagreements[:10]) + "\n")

    L.append("## 4. Duplicate / near-duplicate analysis\n")
    d = payload["duplicate_report"]
    L.append(f"- exact-duplicate groups: {d['exact_duplicate_groups']} "
             f"({d['exact_duplicate_images']} redundant image(s))")
    L.append(f"- near-duplicate pairs (16x16 dHash): {d['near_duplicate_pairs']}")
    L.append(f"- **cross-split exact-duplicate groups: {d['cross_split_exact_duplicate_groups']}** (must be 0)")
    L.append(f"- **cross-split near-duplicate pairs: {d['cross_split_near_duplicate_pairs']}** (must be 0)\n")

    L.append("## 5. Split methodology\n")
    L.append(
        "Scene-isolated, deterministic (seed=42): images are first grouped into "
        "scenes (connected components over exact SHA-256, 16x16 dHash Hamming "
        "<= 8, OR an explicit human scene assignment from "
        "scripts/import_first_party_panels.py), then WHOLE scenes are greedily "
        "assigned to train/valid/test to approximate 70/20/10 by image count. A "
        "scene never crosses a split — verified explicitly, not merely assumed "
        "from the construction. Splits are NOT created until enough verified "
        "data exists (Phase 14B.1 req 12) — an empty split manifest here is "
        "expected while the dataset is still being collected.\n"
    )
    sc = payload["split_check"]
    L.append(f"- no scene crosses a split: **{sc['ok']}**")
    if sc["crossing_scenes"]:
        L.append(f"- CROSSING SCENES (should be empty): {sc['crossing_scenes']}")
    L.append(f"- split counts (images): {payload['split_counts_images']}")
    L.append(f"- split counts (scenes): {payload['split_counts_scenes']}\n")

    L.append("## 6. Annotation quality\n")
    ar = payload["annotation_report"]
    n_verified = c["num_images_annotated_human_verified"]
    if ar is None:
        L.append("- **No annotation directory found / no label files present — 0 images annotated.**\n")
    else:
        L.append(
            f"- total images with a label FILE present (may be an empty, not-yet-filled-in stub): "
            f"{ar['annotated_images']} / {ar['total_images']}"
        )
        L.append(
            f"- **images with a HUMAN-VERIFIED panel count (`annotation_status == "
            f"'annotated_human_verified'`): {n_verified} / {ar['total_images']}** "
            f"— this is the number that actually counts toward readiness, not the line above; "
            f"an empty stub file existing is not evidence of a verified 0-panel image."
        )
        L.append(f"- missing label files: {ar['missing_label_count']}")
        L.append(f"- images with zero annotations (a label file with no boxes in it — this bucket "
                 f"includes BOTH legitimate verified-empty images and unfilled stubs; see the "
                 f"human-verified count above to tell them apart): {ar['images_with_zero_annotations_count']}")
        L.append(f"- files with mechanical errors: {ar['files_with_errors_count']}")
        L.append(f"- images with overlap/size diagnostics: {ar['images_with_overlap_or_size_diagnostics_count']}")
        L.append(f"- mechanically well-formed (syntax/geometry only — NOT a claim of visual correctness): "
                 f"{ar['ready_for_next_stage']}\n")

    L.append("## 7. Difficult-case coverage\n")
    L.append(
        "- **large foreground panels**: " + str(c["large_foreground_panels_represented"]) +
        " — computed from `difficult_case_tags` on VERIFIED scenes only; a human reviewer "
        "must tag a scene `large_foreground_panel` (scripts/import_first_party_panels.py "
        "or apply_difficult_case_tags) for this to become True. Never guessed from pixels."
    )
    L.append(
        "- **touching / grouped panels**: " + str(c["touching_grouped_panels_represented"]) +
        " — same: real coverage, tagged `touching_panels` or `grouped_panels`, never inferred."
    )
    L.append(f"- 4/6/8-panel scene coverage: {p['target_counts_covered']} present, "
             f"{p['target_counts_missing']} missing\n")

    L.append("## 8. Known limitations\n")
    for reason in c["blocking_reasons"] or ["none — all hard requirements met."]:
        L.append(f"- {reason}")
    L.append("")

    L.append("## 9. Six-image qualitative reference set\n")
    six = payload.get("six_image_reference")
    if six:
        L.append(
            "The original six owner photos "
            "(`D:\\solar_ai_training_scratch\\Solar Real Images`) are included in "
            "this manifest as a QUALITATIVE REFERENCE ONLY (per Phase 14B req 23) — "
            "**not mixed into the training split automatically**, **excluded from the "
            "verified-scene-count / readiness gates** (Phase 14B.1 req 13), and not "
            "treated as sufficient dataset volume on their own."
        )
        L.append(f"- registered as scenes: {six['num_scenes']} from {six['num_images']} image(s)")
        L.append(
            "- known issues (Phase 14A, docs/DYNAMIC_PANEL_DETECTION.md): image 4's "
            "large foreground panel was MISSED entirely; image 6 produced two "
            "overlapping group-like boxes instead of per-panel boxes. Images 1 and "
            "3's earlier `cracked` false positives are irrelevant to panel-only "
            "evaluation (see docs/DYNAMIC_PANEL_DETECTION.md sec.7)."
        )
    else:
        L.append("Not registered in this manifest.")
    L.append("")

    L.append("## 10. Ready for Phase 14C?\n")
    L.append(f"**{verdict}**\n")
    if not c["ready_for_phase_14c"]:
        L.append("Missing before Phase 14C (retraining) can responsibly start:\n")
        for r in c["blocking_reasons"]:
            L.append(f"- {r}")
    L.append("")
    return "\n".join(L) + "\n"


def _render_collection_progress_md(payload: dict) -> str:
    """A short, skimmable status view for tracking ongoing collection —
    same computed payload as dataset_report.md, no duplicate computation."""
    c = payload["readiness_checklist"]
    p = payload["panel_count_stats"]
    r = payload["resolution_stats"]
    inv = payload["inventory"]

    L: list[str] = []
    L.append("# Collection progress\n")
    L.append(f"_Regenerate with `python scripts/audit_representative_panel_dataset.py`._\n")

    L.append("## Counts\n")
    L.append(f"| metric | value |")
    L.append(f"|---|---|")
    L.append(f"| total scenes registered | {inv['num_scenes']} |")
    L.append(f"| total images registered | {inv['num_images']} |")
    L.append(f"| verified images (annotated_human_verified) | {c['num_images_annotated_human_verified']} |")
    n_pending = inv["num_images"] - c["num_images_annotated_human_verified"]
    L.append(f"| pending / not-yet-verified images | {n_pending} |")
    L.append(f"| **verified independent scenes** (counts toward readiness) | **{c['verified_scene_count']}** |")
    L.append(f"| verified panel instances (sum of panel_count_gt) | "
             f"{sum(p['images_by_panel_count'][k] * k for k in p['images_by_panel_count'])} |")
    L.append("")

    L.append("## Panel-count distribution (verified only)\n")
    L.append("| panel count | images |")
    L.append("|---|---|")
    for k in sorted(p["images_by_panel_count"]):
        L.append(f"| {k} | {p['images_by_panel_count'][k]} |")
    if not p["images_by_panel_count"]:
        L.append("| _(none yet)_ | |")
    L.append(f"\nunknown panel counts (not yet annotated/verified): "
             f"{inv['num_images'] - p['images_with_known_panel_count']}\n")

    L.append("## Resolution / aspect ratio\n")
    if r.get("count"):
        L.append(f"- aspect ratio range: {r['aspect_ratio_min_max']}   orientation: {r['orientation_counts']}")
    else:
        L.append("- no images registered.")
    L.append("")

    L.append("## Coverage targets\n")
    L.append(f"| target | met? |")
    L.append(f"|---|---|")
    L.append(f"| >= 100 independent verified scenes ({c['verified_scene_count']}/100) | "
             f"{c['verified_scene_count'] >= 100} |")
    L.append(f"| >= 1 verified 4-panel scene | {c['has_verified_4_panel_scene']} |")
    L.append(f"| >= 1 verified 6-panel scene | {c['has_verified_6_panel_scene']} |")
    L.append(f"| >= 1 verified 8-panel scene | {c['has_verified_8_panel_scene']} |")
    L.append(f"| large-foreground-panel scene tagged | {c['large_foreground_panels_represented']} |")
    L.append(f"| touching/grouped-panel scene tagged | {c['touching_grouped_panels_represented']} |")
    L.append(f"| genuine independent test holdout | {c['genuine_independent_holdout']} |")
    L.append(f"| scene-isolated splits, 0 cross-split duplicates | "
             f"{c['train_valid_test_scene_isolated'] and c['cross_split_exact_duplicates_zero'] and c['cross_split_near_duplicate_families_zero']} |")
    L.append("")

    verdict = "READY FOR PHASE 14C" if c["ready_for_phase_14c"] else "NOT READY FOR PHASE 14C"
    L.append(f"## Verdict: **{verdict}**\n")
    if not c["ready_for_phase_14c"]:
        L.append("Missing:\n")
        for reason in c["blocking_reasons"]:
            L.append(f"- {reason}")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR / "source_manifest.json")
    parser.add_argument("--split-manifest", type=Path,
                        default=paths.REPRESENTATIVE_PANEL_V1_SPLITS_DIR / "split_manifest.csv")
    parser.add_argument("--near-dup-pairs", type=Path,
                        default=paths.REPRESENTATIVE_PANEL_V1_MANIFESTS_DIR / "near_duplicate_pairs.json")
    parser.add_argument("--labels-dir", type=Path, default=paths.REPRESENTATIVE_PANEL_V1_ANNOTATIONS_DIR)
    parser.add_argument("--images-dir", type=Path, default=paths.REPRESENTATIVE_PANEL_V1_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=paths.REPRESENTATIVE_PANEL_V1_REPORTS_DIR)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest} — run build_representative_panel_dataset.py "
                         "or import_first_party_panels.py first")

    entries = load_source_manifest(args.manifest)
    image_to_split = _load_split_manifest(args.split_manifest)
    near_pairs = json.loads(args.near_dup_pairs.read_text(encoding="utf-8")) if args.near_dup_pairs.is_file() else []

    split_check = verify_no_scene_crosses_splits(entries, image_to_split)
    duplicate_report = compute_duplicate_leakage_report(entries, near_pairs, image_to_split)
    panel_count_stats = compute_panel_count_stats(
        entries, labels_dir=args.labels_dir if args.labels_dir.is_dir() else None
    )
    resolution_stats = compute_resolution_and_aspect_stats(entries)

    annotation_report = None
    disagreements: list = []
    if args.images_dir.is_dir() and args.labels_dir.is_dir() and any(args.images_dir.iterdir()):
        annotation_report = validate_representative_annotations(args.images_dir, args.labels_dir)
    if args.labels_dir.is_dir():
        disagreements = audit_panel_count_gt_disagreements(entries, args.labels_dir)

    checklist = build_readiness_checklist(
        entries=entries, image_to_split=image_to_split, split_check=split_check,
        duplicate_report=duplicate_report, panel_count_stats=panel_count_stats,
        annotation_report=annotation_report,
    )

    split_counts_images: dict[str, int] = {}
    for s in image_to_split.values():
        split_counts_images[s] = split_counts_images.get(s, 0) + 1
    scenes_by_split: dict[str, set[str]] = {}
    for e in entries:
        s = image_to_split.get(e.image_id)
        if s is not None:
            scenes_by_split.setdefault(s, set()).add(e.scene_id)
    split_counts_scenes = {s: len(v) for s, v in scenes_by_split.items()}

    source_type_counts: dict[str, int] = {}
    provenance_counts: dict[str, int] = {}
    for e in entries:
        source_type_counts[e.source_type] = source_type_counts.get(e.source_type, 0) + 1
        provenance_counts[e.provenance_status] = provenance_counts.get(e.provenance_status, 0) + 1

    six_image_ref = None
    six_entries = [e for e in entries if "WhatsApp Image" in e.original_filename]
    if six_entries:
        six_image_ref = {
            "num_images": len(six_entries),
            "num_scenes": len({e.scene_id for e in six_entries}),
        }

    payload = {
        "schema_version": "1.0",
        "phase": "14B/14B.1 — representative panel dataset audit",
        "inventory": {
            "num_images": len(entries),
            "num_scenes": len({e.scene_id for e in entries}),
            "num_corrupt": 0,  # corrupt files are never in the manifest to begin with
            "source_type_counts": source_type_counts,
            "provenance_status_counts": provenance_counts,
        },
        "resolution_stats": resolution_stats,
        "panel_count_stats": panel_count_stats,
        "panel_count_gt_disagreements": disagreements,
        "duplicate_report": duplicate_report,
        "split_check": split_check,
        "split_counts_images": split_counts_images,
        "split_counts_scenes": split_counts_scenes,
        "annotation_report": annotation_report,
        "readiness_checklist": checklist,
        "six_image_reference": six_image_ref,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "dataset_report.md").write_text(_render_dataset_report_md(payload), encoding="utf-8")
    (args.out_dir / "collection_progress.md").write_text(_render_collection_progress_md(payload), encoding="utf-8")

    print(f"images: {len(entries)}   scenes: {len({e.scene_id for e in entries})}   "
          f"verified scenes: {checklist['verified_scene_count']}")
    print(f"split counts (images): {split_counts_images}")
    print(f"no scene crosses a split: {split_check['ok']}")
    print(f"cross-split exact duplicates: {duplicate_report['cross_split_exact_duplicate_groups']}")
    print(f"cross-split near duplicates: {duplicate_report['cross_split_near_duplicate_pairs']}")
    if disagreements:
        print(f"WARNING: {len(disagreements)} panel_count_gt disagreement(s) — see dataset_report.md")
    verdict = "READY FOR PHASE 14C" if checklist["ready_for_phase_14c"] else "NOT READY FOR PHASE 14C"
    print(f"\nVERDICT: {verdict}")
    if not checklist["ready_for_phase_14c"]:
        print("Blocking reasons:")
        for r in checklist["blocking_reasons"]:
            print(f"  - {r}")
    print(f"\ndataset report: {args.out_dir / 'dataset_report.md'}")
    print(f"collection progress: {args.out_dir / 'collection_progress.md'}")
    return 0 if checklist["ready_for_phase_14c"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
