"""training/detection/ground_pv/representative_dataset.py — tooling for the
Phase 14B representative deployment-domain panel dataset.

Phase 14A ([[phase14a_dynamic_panel_detection]] / docs/DYNAMIC_PANEL_DETECTION.md)
built the software architecture for detecting an arbitrary number of panels
per capture. It also confirmed the underlying detector
(`weights/yolo_solar_ground_research_v1.pt`) is NOT production-ready: it can
miss a large foreground panel entirely, and it can represent several
physical panels as one "group" box. Phase 14B is DATA-FIRST: this module
builds the reusable tooling to assemble, validate, and split a
deployment-domain dataset that could fix those two failure modes — it does
NOT train anything, and it does NOT invent data. See
docs/DYNAMIC_PANEL_DETECTION.md and this module's own dataset report
(`datasets/ground_pv_research/representative_panel_v1/reports/dataset_report.md`)
for whether enough real data actually exists yet.

Phase 14B.1 extends the same manifest/grouping/split machinery with a real
batch-COLLECTION workflow (`scripts/import_first_party_panels.py`):
`capture_batch_id`/`camera_id`/`capture_timestamp` provenance, EXPLICIT
scene assignment (a `scene_status` of `"needs_review"` when scene identity
truly isn't known — auto-grouping only proves two images are
near-duplicate FRAMES of the same moment, it can never prove two
differently-framed photos are "the same physical array session"; only a
human importing a batch knows that), human-review difficult-case tags
(`DIFFICULT_CASE_TAGS`) so large-foreground/touching-panel coverage
becomes a real computed check instead of a permanent "unknown", and a
`panel_count_gt` that is DERIVED from verified annotation boxes (never
typed in as a separate, potentially-wrong number) with a disagreement
audit.

Scope of this module (pure orchestration/validation, reuses existing
project utilities rather than duplicating them):

  1. Source manifest — one record per candidate image (schema matches the
     Phase 14B spec exactly): image_id, source_family_id, filenames/paths,
     dimensions, aspect ratio, sha256, capture_source, capture_date,
     panel_count_if_known, scene_id, notes, provenance_status,
     annotation_status.
  2. YOLO annotation validation — REUSES
     `training.detection.validate_yolo_annotations` for the mechanical
     checks it already does well (malformed lines, out-of-range/negative
     coordinates, zero-area boxes, box edges outside [0,1], exact-duplicate
     boxes, missing/orphaned label files) and ADDS the checks that module
     doesn't cover: excessive pairwise overlap between two DIFFERENT boxes
     in the same image, and suspiciously tiny/huge boxes relative to the
     image.
  3. Scene / source-family grouping — REUSES the union-find + 16x16-dHash
     near-duplicate pattern from `training.detection.ground_pv.remediate`
     (generalized here: no Roboflow-specific basename stripping, since
     these are first-party captures, not a re-exported public dataset).
  4. Scene-isolated, deterministic split assignment — REUSES
     `remediate.assign_leakage_clean_splits` directly (it is already
     generic: family sizes + ratios + seed, no Roboflow-specific logic).
  5. Panel-count / resolution / aspect-ratio statistics.
  6. Count-error evaluation-protocol helper (Phase 14B defines HOW a future
     detector will be evaluated; it does not run one).

ANNOTATION POLICY (documented per Phase 14B req 8-9; enforced by the
validator above, not re-derived per file):

  - ONE PHYSICAL PANEL = ONE BOUNDING BOX. A box drawn around a group of
    several physical panels is a mis-annotation, not a valid label — this
    module cannot detect that automatically (it is a semantic judgment,
    not a geometry check), so it is a human-review requirement, documented
    here and in the dataset report.
  - A PARTIALLY VISIBLE panel IS annotated when a human annotator can
    confidently identify it as a panel (i.e. enough of its frame/cells/
    edge is visible to be unambiguous), even if cropped by the image edge
    or another object.
  - An AMBIGUOUS FRAGMENT (a human annotator cannot confidently tell it is
    a panel from what's visible) is NOT annotated, and the reason should be
    recorded in that image's manifest `notes` field where practical. This
    prevents the model from learning arbitrary image fragments as panels.
  - Ground truth is HUMAN-VERIFIED ONLY. Model predictions (from this or
    any other checkpoint) may be used as annotation assistance/drafts, but
    are NEVER written into a label file as ground truth without a human
    confirming each box. This module never calls a detector to generate or
    modify a label file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from training.detection.ground_pv.remediate import assign_leakage_clean_splits
from training.detection.validate_yolo_annotations import validate_directory, validate_label_file

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_DHASH16_MERGE_HAMMING = 8  # same value justified/validated in remediate.py
DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "valid": 0.20, "test": 0.10}
DEFAULT_SEED = 42

DEFAULT_OVERLAP_IOU_WARNING = 0.85  # near-duplicate boxes on the SAME panel, not adjacent panels
DEFAULT_MIN_RELATIVE_AREA = 0.0005
DEFAULT_MAX_RELATIVE_AREA = 0.90

VALID_SOURCE_TYPES = {"first_party", "public_supplemental", "unknown"}
VALID_PROVENANCE_STATUSES = {"first_party", "public_supplemental", "unknown"}
# The Phase 14B.1 spec's required minimum vocabulary is exactly
# {pending, annotated_human_verified, rejected, needs_review}; in_progress
# and excluded_ambiguous are additional, compatible refinements kept from
# Phase 14B (a finer-grained "still being worked on" / "deliberately
# skipped as ambiguous" state, neither of which contradicts the spec).
#
# annotated_ai_assisted_unverified: boxes proposed by a segmentation model and
#   accepted/corrected by an AI reviewer looking at each overlay (Phase 19). It is
#   deliberately NOT annotated_human_verified: it never counts toward the
#   human-verified readiness gates and needs a person's spot-check before it
#   can be promoted.
VALID_ANNOTATION_STATUSES = {
    "pending", "in_progress", "annotated_human_verified", "excluded_ambiguous",
    "rejected", "needs_review", "annotated_ai_assisted_unverified",
}
# needs_review: scene identity genuinely unknown (the honest default — see
#   group_into_scenes()'s docstring on why auto-grouping cannot prove this).
# auto_grouped: >=2 images merged by exact-SHA or dHash near-duplicate
#   detection — a real signal, not a guess.
# manually_assigned: a human importer explicitly assigned this scene_id
#   (single-image or whole-batch) via scripts/import_first_party_panels.py.
VALID_SCENE_STATUSES = {"needs_review", "auto_grouped", "manually_assigned"}

# Human-review tags a reviewer can attach to a scene to record which known
# difficult failure modes ([[phase14a_dynamic_panel_detection]]) it covers.
# Metadata only — never a detector class, never inferred automatically.
DIFFICULT_CASE_TAGS = {
    "large_foreground_panel", "touching_panels", "grouped_panels",
    "strong_perspective", "partial_panel", "occlusion", "low_light",
    "backlighting", "shadow", "distant_array",
}

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


# ---------------------------------------------------------------------------
# 1. Image metadata + source manifest
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash16_bytes(image: Image.Image) -> bytes:
    g = image.convert("L").resize((17, 16), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten().astype(np.uint8)
    return np.packbits(bits).tobytes()


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


@dataclass
class ImageMetadata:
    """Raw, objective facts about one image file — no judgment calls."""

    path: Path
    width: int
    height: int
    sha256: str
    dhash16: bytes

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 4) if self.height > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "sha256": self.sha256,
            "dhash16": self.dhash16.hex(),
        }


def extract_image_metadata(path: Path) -> ImageMetadata:
    """Read one image's objective metadata. Raises on a corrupt/unreadable
    file rather than silently skipping it — the caller decides how to
    report that."""
    path = Path(path)
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        w, h = im.size
        dh = _dhash16_bytes(im.convert("RGB"))
    return ImageMetadata(path=path, width=w, height=h, sha256=_sha256_file(path), dhash16=dh)


@dataclass
class SourceManifestEntry:
    """One record per candidate image. Field set matches the Phase 14B /
    14B.1 spec exactly. `scene_id` / `source_family_id` start out equal to
    each other (each image is its own singleton scene, `scene_status ==
    "needs_review"`) until `group_into_scenes()` and/or an explicit human
    scene assignment resolves it."""

    image_id: str
    source_family_id: str
    original_filename: str
    source_path: str
    width: int
    height: int
    aspect_ratio: float
    sha256: str
    source_type: str  # "first_party" | "public_supplemental" | "unknown"
    scene_id: str
    provenance_status: str = "unknown"
    annotation_status: str = "pending"
    scene_status: str = "needs_review"
    capture_batch_id: str | None = None
    capture_timestamp: str | None = None
    camera_id: str | None = None
    relative_path: str = ""
    panel_count_gt: int | None = None
    difficult_case_tags: list[str] = field(default_factory=list)
    notes: str = ""
    dhash16: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_family_id": self.source_family_id,
            "original_filename": self.original_filename,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "source_type": self.source_type,
            "capture_batch_id": self.capture_batch_id,
            "capture_timestamp": self.capture_timestamp,
            "camera_id": self.camera_id,
            "panel_count_gt": self.panel_count_gt,
            "scene_id": self.scene_id,
            "scene_status": self.scene_status,
            "difficult_case_tags": list(self.difficult_case_tags),
            "notes": self.notes,
            "sha256": self.sha256,
            "dhash16": self.dhash16,
            "provenance_status": self.provenance_status,
            "annotation_status": self.annotation_status,
        }


def build_source_manifest(
    raw_dir: Path,
    *,
    source_type: str,
    provenance_status: str = "first_party",
    id_prefix: str = "img",
) -> tuple[list[SourceManifestEntry], list[dict[str, str]]]:
    """Scan `raw_dir` for images and build one manifest entry per image (a
    one-shot fresh scan — see `import_batch_into_manifest` for the
    incremental, merge-with-existing-manifest batch-collection workflow).

    Every image starts as its own singleton scene (`scene_id == image_id`,
    `scene_status == "needs_review"`) — call `group_into_scenes()`
    afterward to merge genuine near-duplicate bursts/re-captures, or pass
    an explicit scene assignment via `import_batch_into_manifest`. Returns
    (entries, corrupt_files) — a corrupt/unreadable image is reported,
    never silently dropped or fabricated.
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {VALID_SOURCE_TYPES}, got {source_type!r}")
    if provenance_status not in VALID_PROVENANCE_STATUSES:
        raise ValueError(f"provenance_status must be one of {VALID_PROVENANCE_STATUSES}, got {provenance_status!r}")

    raw_dir = Path(raw_dir)
    entries: list[SourceManifestEntry] = []
    corrupt: list[dict[str, str]] = []
    for idx, path in enumerate(iter_images(raw_dir), start=1):
        try:
            meta = extract_image_metadata(path)
        except Exception as exc:  # noqa: BLE001
            corrupt.append({"path": str(path), "error": str(exc)})
            continue
        image_id = f"{id_prefix}_{idx:04d}"
        entries.append(SourceManifestEntry(
            image_id=image_id,
            source_family_id=image_id,  # singleton until grouped
            original_filename=path.name,
            source_path=str(path),
            relative_path=path.name,
            width=meta.width,
            height=meta.height,
            aspect_ratio=meta.aspect_ratio,
            sha256=meta.sha256,
            source_type=source_type,
            scene_id=image_id,  # singleton until grouped
            scene_status="needs_review",
            provenance_status=provenance_status,
            annotation_status="pending",
            dhash16=meta.dhash16.hex(),
        ))
    return entries, corrupt


def save_source_manifest(entries: Iterable[SourceManifestEntry], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.as_dict() for e in entries]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_source_manifest(path: Path) -> list[SourceManifestEntry]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SourceManifestEntry(
            image_id=r["image_id"], source_family_id=r["source_family_id"],
            original_filename=r["original_filename"], source_path=r["source_path"],
            relative_path=r.get("relative_path", ""),
            width=r["width"], height=r["height"], aspect_ratio=r["aspect_ratio"],
            sha256=r["sha256"], source_type=r.get("source_type", r.get("capture_source", "unknown")),
            scene_id=r["scene_id"],
            provenance_status=r.get("provenance_status", "unknown"),
            annotation_status=r.get("annotation_status", "pending"),
            scene_status=r.get("scene_status", "needs_review"),
            capture_batch_id=r.get("capture_batch_id"),
            capture_timestamp=r.get("capture_timestamp", r.get("capture_date")),
            camera_id=r.get("camera_id"),
            panel_count_gt=r.get("panel_count_gt", r.get("panel_count_if_known")),
            difficult_case_tags=list(r.get("difficult_case_tags", [])),
            notes=r.get("notes", ""),
            dhash16=r.get("dhash16", ""),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# 1b. Batch import — incremental, merges into an EXISTING manifest rather
#     than a one-shot fresh scan (see scripts/import_first_party_panels.py)
# ---------------------------------------------------------------------------

_ID_INDEX_RE = re.compile(r"_(\d+)$")


@dataclass
class BatchImportResult:
    imported: list[SourceManifestEntry] = field(default_factory=list)
    rejected_duplicates: list[dict[str, Any]] = field(default_factory=list)
    corrupt: list[dict[str, str]] = field(default_factory=list)
    all_entries: list[SourceManifestEntry] = field(default_factory=list)


def _next_id_index(existing: list[SourceManifestEntry], id_prefix: str) -> int:
    """Continue numbering from the existing manifest's highest index for
    this prefix — deterministic across repeated imports, never restarts at
    1 and silently collides with an earlier batch's ids."""
    best = 0
    for e in existing:
        if not e.image_id.startswith(f"{id_prefix}_"):
            continue
        m = _ID_INDEX_RE.search(e.image_id)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def import_batch_into_manifest(
    source_dir: Path,
    *,
    existing_entries: list[SourceManifestEntry],
    raw_dest_root: Path,
    batch_id: str,
    source_type: str = "first_party",
    provenance_status: str = "first_party",
    id_prefix: str = "img",
    camera_id: str | None = None,
    capture_timestamp: str | None = None,
    scene_id_for_batch: str | None = None,
    scene_map: dict[str, str] | None = None,
) -> BatchImportResult:
    """Import one collection batch: copy originals (byte-for-byte, never
    converted/resized) into `raw_dest_root/<batch_id>/`, compute metadata,
    reject any image whose SHA-256 already exists ANYWHERE in
    `existing_entries` (a prior batch) or elsewhere in THIS batch (never
    silently re-imports or silently overwrites), and merge the newly
    accepted entries into the returned combined manifest.

    Scene assignment (Phase 14B.1 req 4 — "do not pretend unknown scene
    grouping is solved automatically"):
      - `scene_id_for_batch`: every image in this batch gets this one
        scene_id, `scene_status = "manually_assigned"` (the common case: a
        whole batch is one physical array/session).
      - `scene_map`: {original_filename: scene_id} for finer per-file
        control, same status.
      - neither given: every new image starts as its own singleton scene
        with `scene_status = "needs_review"` — the honest default. Run
        `group_into_scenes()` on the returned `all_entries` afterward to
        pick up any automatic near-duplicate merges on top of this.

    Never modifies or moves the ORIGINAL source files in `source_dir`.
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {VALID_SOURCE_TYPES}, got {source_type!r}")
    if scene_id_for_batch and scene_map:
        raise ValueError("pass at most one of scene_id_for_batch / scene_map, not both")

    source_dir = Path(source_dir)
    raw_dest_root = Path(raw_dest_root)
    batch_dest = raw_dest_root / batch_id
    batch_dest.mkdir(parents=True, exist_ok=True)

    # Frozen snapshot of shas that existed BEFORE this batch — kept separate
    # from `seen_in_this_batch` so a within-batch duplicate always gets the
    # more specific "same batch" reason, never misreported as "already
    # imported" just because an earlier file in THIS batch happened to be
    # accepted first.
    known_sha256: set[str] = {e.sha256 for e in existing_entries}
    result = BatchImportResult(all_entries=list(existing_entries))
    next_idx = _next_id_index(existing_entries, id_prefix)

    seen_in_this_batch: set[str] = set()
    for path in iter_images(source_dir):
        try:
            meta = extract_image_metadata(path)
        except Exception as exc:  # noqa: BLE001
            result.corrupt.append({"path": str(path), "error": str(exc)})
            continue

        if meta.sha256 in seen_in_this_batch:
            result.rejected_duplicates.append({
                "path": str(path), "sha256": meta.sha256,
                "reason": "exact duplicate of another file in this same import batch",
            })
            continue
        if meta.sha256 in known_sha256:
            result.rejected_duplicates.append({
                "path": str(path), "sha256": meta.sha256,
                "reason": "exact duplicate of an already-imported image",
            })
            continue
        seen_in_this_batch.add(meta.sha256)

        dest_name = path.name
        dest_path = batch_dest / dest_name
        if dest_path.exists():
            # Never silently overwrite. Same content already there (a
            # re-run of the same import) is a safe no-op; different
            # content under the same filename gets a disambiguated name.
            if _sha256_file(dest_path) == meta.sha256:
                pass
            else:
                stem, suffix = Path(dest_name).stem, Path(dest_name).suffix
                n = 2
                while (batch_dest / f"{stem}__dup{n}{suffix}").exists():
                    n += 1
                dest_path = batch_dest / f"{stem}__dup{n}{suffix}"
                shutil.copy2(path, dest_path)
        else:
            shutil.copy2(path, dest_path)

        image_id = f"{id_prefix}_{next_idx:04d}"
        next_idx += 1

        if scene_id_for_batch:
            scene_id, scene_status = scene_id_for_batch, "manually_assigned"
        elif scene_map and path.name in scene_map:
            scene_id, scene_status = scene_map[path.name], "manually_assigned"
        else:
            scene_id, scene_status = image_id, "needs_review"

        entry = SourceManifestEntry(
            image_id=image_id,
            source_family_id=scene_id,
            original_filename=path.name,
            source_path=str(path),
            relative_path=f"{batch_id}/{dest_path.name}",
            width=meta.width, height=meta.height, aspect_ratio=meta.aspect_ratio,
            sha256=meta.sha256, dhash16=meta.dhash16.hex(),
            source_type=source_type, provenance_status=provenance_status,
            scene_id=scene_id, scene_status=scene_status,
            capture_batch_id=batch_id, capture_timestamp=capture_timestamp, camera_id=camera_id,
            annotation_status="pending",
        )
        result.imported.append(entry)
        result.all_entries.append(entry)

    return result


# ---------------------------------------------------------------------------
# 2. Scene / source-family grouping (reuses remediate.py's union-find +
#    16x16 dHash pattern; no Roboflow-specific basename logic — first-party
#    captures don't have re-export suffixes to strip)
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_into_scenes(
    entries: list[SourceManifestEntry],
    *,
    dhash16_merge_hamming: int = DEFAULT_DHASH16_MERGE_HAMMING,
) -> tuple[list[SourceManifestEntry], list[dict[str, Any]]]:
    """Group images into scenes: a connected component over (identical
    SHA-256) UNION (16x16 dHash Hamming <= threshold) UNION (already sharing
    a human-assigned `scene_id` with `scene_status == "manually_assigned"`).
    Mutates and returns `entries` with `scene_id`/`source_family_id` set to
    a stable "sceneNNNN" id per group, and `scene_status` resolved per the
    rules below; also returns the list of near-duplicate pairs found (for
    the duplicate-analysis section of the dataset report).

    A scene groups images that are EITHER the same physical capture moment/
    burst (near-identical framing, detected automatically) OR that a human
    importer explicitly said belong together (e.g. multiple angles of one
    physical array/session) — auto-grouping alone can NEVER prove the
    latter, so a human-assigned group is honored and preserved (never
    silently overridden), and additional near-duplicate matches on top of
    it only ever ADD members, never remove the human's assignment.

    Resulting `scene_status` per merged group:
      - "manually_assigned" if ANY member entering the merge already had
        scene_status == "manually_assigned" (a human confirmed at least
        part of this group; propagates to the whole merged group).
      - "auto_grouped" if the group has >1 member purely from automated
        exact-SHA / dHash matching, with no human input at all.
      - "needs_review" for an untouched singleton — the honest default
        when scene identity is genuinely unknown (Phase 14B.1 req 4: "do
        not pretend unknown scene grouping is solved automatically").
    """
    if not entries:
        return entries, []

    uf = _UnionFind()
    ids = [e.image_id for e in entries]
    for i in ids:
        uf.find(i)

    # Seed unions from existing human-confirmed scene assignments FIRST, so
    # they anchor the group before any automated signal is layered on top.
    manually_assigned_scene_of: dict[str, str] = {
        e.image_id: e.scene_id for e in entries if e.scene_status == "manually_assigned"
    }
    by_manual_scene: dict[str, list[str]] = defaultdict(list)
    for image_id, scene_id in manually_assigned_scene_of.items():
        by_manual_scene[scene_id].append(image_id)
    for members in by_manual_scene.values():
        for other in members[1:]:
            uf.union(members[0], other)

    by_sha: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_sha[e.sha256].append(e.image_id)
    exact_groups = [sorted(v) for v in by_sha.values() if len(v) > 1]
    for grp in exact_groups:
        for other in grp[1:]:
            uf.union(grp[0], other)

    hashes = np.frombuffer(
        b"".join(bytes.fromhex(e.dhash16) for e in entries), dtype=np.uint8
    ).reshape(len(entries), 32)
    near_pairs: list[dict[str, Any]] = []
    for i in range(len(entries)):
        if i + 1 >= len(entries):
            continue
        xor = np.bitwise_xor(hashes[i + 1:], hashes[i])
        dist = _POPCOUNT[xor].sum(axis=1)
        for off in np.where(dist <= dhash16_merge_hamming)[0]:
            j = i + 1 + int(off)
            h = int(dist[off])
            near_pairs.append({"a": ids[i], "b": ids[j], "dhash16_hamming": h})
            uf.union(ids[i], ids[j])

    fam_of = {i: uf.find(i) for i in ids}
    sizes = Counter(fam_of.values())
    ordered = sorted(sizes, key=lambda r: (-sizes[r], r))
    rename = {root: f"scene{idx:04d}" for idx, root in enumerate(ordered)}

    had_manual_seed: dict[str, bool] = defaultdict(bool)
    for e in entries:
        if e.scene_status == "manually_assigned":
            had_manual_seed[fam_of[e.image_id]] = True

    for e in entries:
        root = fam_of[e.image_id]
        new_id = rename[root]
        e.scene_id = new_id
        e.source_family_id = new_id
        if had_manual_seed[root]:
            e.scene_status = "manually_assigned"
        elif sizes[root] > 1:
            e.scene_status = "auto_grouped"
        else:
            e.scene_status = "needs_review"

    return entries, near_pairs


# ---------------------------------------------------------------------------
# 3. Scene-isolated, deterministic split assignment
# ---------------------------------------------------------------------------


@dataclass
class SplitAssignmentResult:
    image_to_split: dict[str, str]
    scene_to_split: dict[str, str]
    scene_sizes: dict[str, int]
    ratios: dict[str, float]
    seed: int


def assign_scene_isolated_splits(
    entries: list[SourceManifestEntry],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> SplitAssignmentResult:
    """Assign every image a split such that all images sharing a `scene_id`
    get the SAME split — reuses `remediate.assign_leakage_clean_splits`
    directly (already generic: scene sizes + ratios + seed in, split
    assignment out; no dataset-specific logic in that function)."""
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    scene_sizes: dict[str, int] = Counter(e.scene_id for e in entries)
    scene_to_split = assign_leakage_clean_splits(dict(scene_sizes), ratios=ratios, seed=seed)
    image_to_split = {e.image_id: scene_to_split[e.scene_id] for e in entries}
    return SplitAssignmentResult(
        image_to_split=image_to_split, scene_to_split=scene_to_split,
        scene_sizes=dict(scene_sizes), ratios=ratios, seed=seed,
    )


def verify_no_scene_crosses_splits(
    entries: list[SourceManifestEntry], image_to_split: dict[str, str]
) -> dict[str, Any]:
    """Hard check: no `scene_id` may have images assigned to more than one
    split. Returns a report dict; `ok` is False if ANY scene crosses."""
    scene_to_splits: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        split = image_to_split.get(e.image_id)
        if split is not None:
            scene_to_splits[e.scene_id].add(split)
    crossing = {s: sorted(v) for s, v in scene_to_splits.items() if len(v) > 1}
    return {"ok": not crossing, "crossing_scenes": crossing}


def save_split_manifest(
    entries: list[SourceManifestEntry], image_to_split: dict[str, str], out_path: Path
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "scene_id", "source_family_id", "split"])
        for e in sorted(entries, key=lambda e: e.image_id):
            w.writerow([e.image_id, e.scene_id, e.source_family_id, image_to_split.get(e.image_id, "")])
    return out_path


# ---------------------------------------------------------------------------
# 4. YOLO annotation validation — reuses validate_yolo_annotations.py for
#    the mechanical per-line checks, adds overlap / size diagnostics on top.
# ---------------------------------------------------------------------------


def _box_xyxy(b: dict[str, float]) -> tuple[float, float, float, float]:
    return (b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_overlap_and_size(
    boxes: list[dict[str, float]],
    *,
    overlap_iou_warning: float = DEFAULT_OVERLAP_IOU_WARNING,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
    max_relative_area: float = DEFAULT_MAX_RELATIVE_AREA,
) -> list[str]:
    """Extra diagnostics beyond validate_yolo_annotations.py's mechanical
    checks: excessive overlap between two DIFFERENT boxes (near-duplicate
    box on the same physical panel — a real annotation mistake, distinct
    from two adjacent panels which legitimately share an edge but should
    have LOW IoU), and suspiciously tiny/huge boxes. Diagnostics only —
    never mutates or drops a box; the caller/human decides."""
    issues: list[str] = []
    for i, b in enumerate(boxes):
        area = b["w"] * b["h"]
        if area < min_relative_area:
            issues.append(f"box {i} (line {b.get('line', '?')}): suspiciously tiny (area={area:.6f})")
        if area > max_relative_area:
            issues.append(f"box {i} (line {b.get('line', '?')}): suspiciously huge (area={area:.6f})")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = _iou(_box_xyxy(boxes[i]), _box_xyxy(boxes[j]))
            if iou >= overlap_iou_warning:
                issues.append(
                    f"boxes {i} & {j} (lines {boxes[i].get('line','?')}, {boxes[j].get('line','?')}): "
                    f"excessive overlap (IoU={iou:.3f}) — likely a duplicate annotation of the same panel"
                )
    return issues


def validate_representative_annotations(
    images_dir: Path,
    labels_dir: Path,
    *,
    allowed_class_id: int = 0,
    overlap_iou_warning: float = DEFAULT_OVERLAP_IOU_WARNING,
    min_relative_area: float = DEFAULT_MIN_RELATIVE_AREA,
    max_relative_area: float = DEFAULT_MAX_RELATIVE_AREA,
) -> dict[str, Any]:
    """Full annotation QC for the representative dataset: the mechanical
    checks from validate_yolo_annotations.py, PLUS per-image overlap/size
    diagnostics on top. Never treats the current detector's predictions as
    ground truth — this only validates whatever is already saved in
    `labels_dir` (human-authored label files)."""
    base = validate_directory(images_dir, labels_dir, allowed_class_id)

    extra_issues: dict[str, list[str]] = {}
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    for img_path in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            continue
        result = validate_label_file(label_path, allowed_class_id)
        issues = check_overlap_and_size(
            result["boxes"], overlap_iou_warning=overlap_iou_warning,
            min_relative_area=min_relative_area, max_relative_area=max_relative_area,
        )
        if issues:
            extra_issues[img_path.name] = issues

    base["overlap_and_size_diagnostics"] = extra_issues
    base["images_with_overlap_or_size_diagnostics_count"] = len(extra_issues)
    return base


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------


def compute_panel_count_stats(
    entries: list[SourceManifestEntry],
    labels_dir: Path | None = None,
) -> dict[str, Any]:
    """Panel-count distribution by image and by scene.

    A count is trusted ONLY when that image's `annotation_status ==
    "annotated_human_verified"` — an empty (or any other) label file that
    merely EXISTS is not evidence of a verified panel count, and neither is
    a `panel_count_gt` value on an unverified entry. This matters
    concretely: `--write-empty-label-stubs` creates a 0-byte placeholder
    `.txt` per image so a human has somewhere to draw boxes; without this
    gate, every un-annotated image with a stub would silently read back as
    "0 panels", which is not true — it is "not yet annotated". Prefers the
    manifest's `panel_count_gt` (kept in sync with the actual verified box
    count by `sync_panel_count_gt_from_annotations`); falls back to
    counting the label file live when `labels_dir` is given and
    `panel_count_gt` hasn't been synced yet. An image with neither is
    excluded from the distribution (never assumed to be 0, 4, or anything
    else)."""
    labels_dir = Path(labels_dir) if labels_dir else None
    counts_by_image: dict[str, int] = {}
    for e in entries:
        if e.annotation_status != "annotated_human_verified":
            continue
        n: int | None = e.panel_count_gt
        if n is None and labels_dir is not None:
            label_path = labels_dir / f"{Path(e.original_filename).stem}.txt"
            if label_path.is_file():
                n = len(validate_label_file(label_path)["boxes"])
        if n is not None:
            counts_by_image[e.image_id] = n

    dist_images = Counter(counts_by_image.values())
    scene_of = {e.image_id: e.scene_id for e in entries}
    counts_by_scene: dict[str, set[int]] = defaultdict(set)
    for image_id, n in counts_by_image.items():
        counts_by_scene[scene_of[image_id]].add(n)
    dist_scenes = Counter(next(iter(v)) for v in counts_by_scene.values() if len(v) == 1)

    return {
        "images_with_known_panel_count": len(counts_by_image),
        "images_total": len(entries),
        "images_by_panel_count": dict(sorted(dist_images.items())),
        "scenes_by_panel_count": dict(sorted(dist_scenes.items())),
        "target_counts_covered": sorted(set(dist_images) & {4, 6, 8}),
        "target_counts_missing": sorted({4, 6, 8} - set(dist_images)),
    }


def sync_panel_count_gt_from_annotations(
    entries: list[SourceManifestEntry], labels_dir: Path
) -> list[dict[str, Any]]:
    """For every `annotated_human_verified` entry, set `panel_count_gt =
    len(verified boxes)` from its label file — the ONLY sanctioned way
    `panel_count_gt` gets populated (Phase 14B.1 req 7: never type it in
    separately). Mutates `entries` in place. Returns a list of
    disagreement records for any entry whose PRE-EXISTING `panel_count_gt`
    did not match the freshly-computed box count (so a stale/incorrectly
    hand-typed value is surfaced, never silently overwritten without a
    trace)."""
    labels_dir = Path(labels_dir)
    disagreements: list[dict[str, Any]] = []
    for e in entries:
        if e.annotation_status != "annotated_human_verified":
            continue
        label_path = labels_dir / f"{Path(e.original_filename).stem}.txt"
        if not label_path.is_file():
            continue
        actual = len(validate_label_file(label_path)["boxes"])
        if e.panel_count_gt is not None and e.panel_count_gt != actual:
            disagreements.append({
                "image_id": e.image_id, "stored_panel_count_gt": e.panel_count_gt,
                "actual_verified_box_count": actual,
            })
        e.panel_count_gt = actual
    return disagreements


def audit_panel_count_gt_disagreements(
    entries: list[SourceManifestEntry], labels_dir: Path
) -> list[dict[str, Any]]:
    """Read-only version of the disagreement check in
    `sync_panel_count_gt_from_annotations` — reports mismatches between the
    STORED `panel_count_gt` and the actual verified box count WITHOUT
    mutating `entries` (for the audit CLI, which should never silently fix
    data it's only supposed to report on)."""
    labels_dir = Path(labels_dir)
    disagreements: list[dict[str, Any]] = []
    for e in entries:
        if e.annotation_status != "annotated_human_verified":
            continue
        label_path = labels_dir / f"{Path(e.original_filename).stem}.txt"
        if not label_path.is_file():
            continue
        actual = len(validate_label_file(label_path)["boxes"])
        if e.panel_count_gt != actual:
            disagreements.append({
                "image_id": e.image_id, "stored_panel_count_gt": e.panel_count_gt,
                "actual_verified_box_count": actual,
            })
    return disagreements


# ---------------------------------------------------------------------------
# 5b. Difficult-case review tags
# ---------------------------------------------------------------------------


def apply_difficult_case_tags(
    entries: list[SourceManifestEntry], scene_tags: dict[str, list[str]]
) -> None:
    """Attach human-review difficult-case tags to every entry sharing a
    tagged `scene_id`. `scene_tags` is scene_id -> list of tags (each must
    be in `DIFFICULT_CASE_TAGS`). Mutates `entries` in place; raises on an
    unrecognized tag rather than silently accepting a typo."""
    for scene_id, tags in scene_tags.items():
        unknown = set(tags) - DIFFICULT_CASE_TAGS
        if unknown:
            raise ValueError(f"unrecognized difficult-case tag(s) {sorted(unknown)} for scene {scene_id!r} — "
                             f"must be one of {sorted(DIFFICULT_CASE_TAGS)}")
    for e in entries:
        new_tags = scene_tags.get(e.scene_id)
        if new_tags:
            merged = sorted(set(e.difficult_case_tags) | set(new_tags))
            e.difficult_case_tags = merged


def compute_resolution_and_aspect_stats(entries: list[SourceManifestEntry]) -> dict[str, Any]:
    if not entries:
        return {"count": 0}
    widths = [e.width for e in entries]
    heights = [e.height for e in entries]
    ratios = [e.aspect_ratio for e in entries]
    orientation = Counter("portrait" if r < 0.95 else ("landscape" if r > 1.05 else "square") for r in ratios)
    return {
        "count": len(entries),
        "width_min_max": [min(widths), max(widths)],
        "height_min_max": [min(heights), max(heights)],
        "aspect_ratio_min_max": [min(ratios), max(ratios)],
        "orientation_counts": dict(orientation),
        "unique_resolutions": len({(e.width, e.height) for e in entries}),
    }


# ---------------------------------------------------------------------------
# 6. Evaluation protocol (Phase 14B defines this; nothing here runs a model)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Duplicate/leakage summary + readiness checklist (Final Decision Rule)
# ---------------------------------------------------------------------------


def compute_duplicate_leakage_report(
    entries: list[SourceManifestEntry],
    near_pairs: list[dict[str, Any]],
    image_to_split: dict[str, str],
) -> dict[str, Any]:
    """Exact + near-duplicate counts, and — critically — whether any of
    them cross a split boundary. `cross_split_*` must be 0 for the dataset
    to be leakage-safe; this is a defensive re-check independent of the
    scene-grouping-before-splitting construction (belt and suspenders, like
    remediate.py's `residual_cross_split_near_dup_pairs`)."""
    by_sha: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_sha[e.sha256].append(e.image_id)
    exact_dup_groups = [v for v in by_sha.values() if len(v) > 1]
    cross_split_exact = sum(
        1 for grp in exact_dup_groups
        if len({image_to_split.get(i) for i in grp}) > 1
    )

    cross_split_near = sum(
        1 for p in near_pairs
        if image_to_split.get(p["a"]) is not None
        and image_to_split.get(p["b"]) is not None
        and image_to_split.get(p["a"]) != image_to_split.get(p["b"])
    )

    return {
        "exact_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_images": sum(len(g) - 1 for g in exact_dup_groups),
        "near_duplicate_pairs": len(near_pairs),
        "cross_split_exact_duplicate_groups": cross_split_exact,
        "cross_split_near_duplicate_pairs": cross_split_near,
    }


def build_readiness_checklist(
    *,
    entries: list[SourceManifestEntry],
    image_to_split: dict[str, str],
    split_check: dict[str, Any],
    duplicate_report: dict[str, Any],
    panel_count_stats: dict[str, Any],
    annotation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """The Phase 14B / 14B.1 "Final Decision Rule" as a computed checklist
    — every item is a real, independently-verifiable boolean; nothing is
    assumed ready just because the code runs cleanly, and the two
    difficult-case coverage checks are computed from actual human-applied
    `difficult_case_tags`, never guessed.

    "Verified scene" (the unit the >=100 / >=1-per-4-6-8 gates count in)
    means: at least one image in that scene has `annotation_status ==
    "annotated_human_verified"`, the scene's `scene_status !=
    "needs_review"` (an unresolved scene identity doesn't count as a
    dataset-ready scene), AND the scene is not in the `reference` split
    (the six-owner-photo qualitative reference set is explicitly excluded
    from counting toward training-readiness, Phase 14B.1 req 13)."""
    n = len(entries)
    n_annotated = sum(1 for e in entries if e.annotation_status == "annotated_human_verified")
    individual_panel_annotations_available = n_annotated > 0

    verified_scene_ids = {
        e.scene_id for e in entries
        if e.annotation_status == "annotated_human_verified"
        and e.scene_status != "needs_review"
        and image_to_split.get(e.image_id) != "reference"
    }
    verified_scene_count = len(verified_scene_ids)

    verified_count_by_scene: dict[str, int] = {}
    for e in entries:
        if e.scene_id in verified_scene_ids and e.panel_count_gt is not None:
            verified_count_by_scene[e.scene_id] = e.panel_count_gt
    has_verified_4_panel_scene = any(c == 4 for c in verified_count_by_scene.values())
    has_verified_6_panel_scene = any(c == 6 for c in verified_count_by_scene.values())
    has_verified_8_panel_scene = any(c == 8 for c in verified_count_by_scene.values())

    source_families_tracked = all(e.source_family_id for e in entries) if entries else False
    scene_isolated = bool(split_check.get("ok"))
    cross_split_exact_zero = duplicate_report.get("cross_split_exact_duplicate_groups", 1) == 0
    cross_split_near_zero = duplicate_report.get("cross_split_near_duplicate_pairs", 1) == 0

    annotation_audit_passes: bool | str
    if annotation_report is None:
        annotation_audit_passes = "no_annotations_to_audit"
    else:
        annotation_audit_passes = annotation_report.get("ready_for_next_stage", False)

    scenes_by_split: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        split = image_to_split.get(e.image_id)
        if split is not None:
            scenes_by_split[split].add(e.scene_id)
    genuine_independent_holdout = len(scenes_by_split.get("test", set())) > 0

    counts_covered = set(panel_count_stats.get("target_counts_covered", []))
    target_counts_represented = counts_covered == {4, 6, 8}

    provenance_documented = all(
        e.provenance_status in VALID_PROVENANCE_STATUSES and e.provenance_status != "unknown"
        for e in entries
    ) if entries else False

    native_aspect_ratios_preserved = True  # by construction: this tooling never resizes/pads source images

    # Difficult-case coverage: a REAL computed check now that tags exist —
    # true only if at least one VERIFIED, non-reference scene carries the
    # tag. Never guessed; an untagged dataset correctly reports False, not
    # "unknown".
    tagged_scenes: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        if e.scene_id in verified_scene_ids:
            tagged_scenes[e.scene_id] |= set(e.difficult_case_tags)
    large_foreground_panels_represented = any(
        "large_foreground_panel" in tags for tags in tagged_scenes.values()
    )
    touching_grouped_panels_represented = any(
        ("touching_panels" in tags or "grouped_panels" in tags) for tags in tagged_scenes.values()
    )

    checklist = {
        "individual_panel_annotations_available": individual_panel_annotations_available,
        "verified_scene_count": verified_scene_count,
        "has_verified_4_panel_scene": has_verified_4_panel_scene,
        "has_verified_6_panel_scene": has_verified_6_panel_scene,
        "has_verified_8_panel_scene": has_verified_8_panel_scene,
        "source_families_tracked": source_families_tracked,
        "train_valid_test_scene_isolated": scene_isolated,
        "cross_split_exact_duplicates_zero": cross_split_exact_zero,
        "cross_split_near_duplicate_families_zero": cross_split_near_zero,
        "annotation_audit_passes": annotation_audit_passes,
        "genuine_independent_holdout": genuine_independent_holdout,
        "target_4_6_8_panel_scenes_represented": target_counts_represented,
        "large_foreground_panels_represented": large_foreground_panels_represented,
        "touching_grouped_panels_represented": touching_grouped_panels_represented,
        "native_aspect_ratios_preserved": native_aspect_ratios_preserved,
        "provenance_documented": provenance_documented,
        "num_images": n,
        "num_images_annotated_human_verified": n_annotated,
    }

    hard_requirements = [
        individual_panel_annotations_available,
        verified_scene_count >= 100,
        has_verified_4_panel_scene,
        has_verified_6_panel_scene,
        has_verified_8_panel_scene,
        source_families_tracked,
        scene_isolated,
        cross_split_exact_zero,
        cross_split_near_zero,
        annotation_audit_passes is True,
        genuine_independent_holdout,
        target_counts_represented,
        large_foreground_panels_represented,
        touching_grouped_panels_represented,
        native_aspect_ratios_preserved,
        provenance_documented,
    ]
    checklist["ready_for_phase_14c"] = all(hard_requirements)
    checklist["blocking_reasons"] = _explain_blockers(checklist)
    return checklist


def _explain_blockers(checklist: dict[str, Any]) -> list[str]:
    reasons = []
    if checklist["verified_scene_count"] < 100:
        reasons.append(
            f"only {checklist['verified_scene_count']} verified independent scene(s) "
            f"(annotated_human_verified, scene_status != needs_review, not in the reference "
            f"split) — spec minimum is 100"
        )
    if not checklist["individual_panel_annotations_available"]:
        reasons.append("no human-verified individual-panel annotations exist yet")
    if not checklist["has_verified_4_panel_scene"]:
        reasons.append("no verified 4-panel scene yet")
    if not checklist["has_verified_6_panel_scene"]:
        reasons.append("no verified 6-panel scene yet")
    if not checklist["has_verified_8_panel_scene"]:
        reasons.append("no verified 8-panel scene yet")
    if not checklist["train_valid_test_scene_isolated"]:
        reasons.append("a scene crosses a split boundary")
    if not checklist["cross_split_exact_duplicates_zero"]:
        reasons.append("exact-duplicate images cross a split boundary")
    if not checklist["cross_split_near_duplicate_families_zero"]:
        reasons.append("near-duplicate images cross a split boundary")
    if checklist["annotation_audit_passes"] not in (True,):
        reasons.append(f"annotation audit status: {checklist['annotation_audit_passes']}")
    if not checklist["genuine_independent_holdout"]:
        reasons.append("no test-split scenes exist yet — no genuine independent holdout")
    if not checklist["target_4_6_8_panel_scenes_represented"]:
        reasons.append("4/6/8-panel scenes are not all represented (with known panel counts)")
    if not checklist["large_foreground_panels_represented"]:
        reasons.append("large-foreground-panel coverage not yet confirmed by a human reviewer's tags")
    if not checklist["touching_grouped_panels_represented"]:
        reasons.append("touching/grouped-panel coverage not yet confirmed by a human reviewer's tags")
    if not checklist["provenance_documented"]:
        reasons.append("one or more images have undocumented provenance_status")
    return reasons


def compute_count_error_stats(
    predictions: dict[str, int], ground_truth: dict[str, int]
) -> dict[str, Any]:
    """Per-image panel-COUNT evaluation (the actual product requirement is
    dynamic counting, not just aggregate mAP — see docs). `predictions` and
    `ground_truth` are image_id -> panel_count; only image_ids present in
    BOTH are scored (never invents a ground-truth count)."""
    common = sorted(set(predictions) & set(ground_truth))
    per_image = []
    errors = []
    exact = 0
    over = 0
    under = 0
    for image_id in common:
        gt, pred = ground_truth[image_id], predictions[image_id]
        err = pred - gt
        errors.append(err)
        if err == 0:
            exact += 1
        elif err > 0:
            over += 1
        else:
            under += 1
        per_image.append({"image_id": image_id, "ground_truth_panel_count": gt,
                           "predicted_panel_count": pred, "count_error": err})
    n = len(common)
    return {
        "num_images_scored": n,
        "num_predictions_without_ground_truth": len(set(predictions) - set(ground_truth)),
        "num_ground_truth_without_prediction": len(set(ground_truth) - set(predictions)),
        "exact_count_accuracy": round(exact / n, 4) if n else None,
        "overcount_rate": round(over / n, 4) if n else None,
        "undercount_rate": round(under / n, 4) if n else None,
        "mean_absolute_count_error": round(sum(abs(e) for e in errors) / n, 4) if n else None,
        "mean_count_error": round(sum(errors) / n, 4) if n else None,  # signed — bias direction
        "per_image": per_image,
    }
