"""training/evaluation/dataset_candidate_gate.py — Reusable, deterministic
tooling for vetting a candidate external dataset before it is ever used for
training, per the licensing/provenance discipline established in
docs/ML_HARDENING_PHASE6C.md (and, before it, training/classification/
DATASET_SOURCES.md and training/detection/PROVENANCE_VERIFICATION.md).

No candidate dataset passed this project's gate in Phase 6C (see
docs/ML_HARDENING_PHASE6C.md Task 4/9) - this module exists so the *rules*
that decision was made under are encoded as real, testable logic, ready to
be reused the moment a genuine candidate is found in a future phase,
rather than re-litigated informally each time.

Never downloads, creates, or fabricates dataset content. Pure schema/logic
only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Licenses this project accepts as "explicitly permits ML training /
# derivative model use" (docs/ML_HARDENING_PHASE6C.md Task 2, criterion 4).
# "Unknown", "All rights reserved", a bare "free to download" claim, or an
# unstated license are all deliberately absent - see non-negotiable rule 8.
PERMISSIVE_LICENSES = frozenset({
    "CC0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0",
    "MIT", "APACHE-2.0", "PUBLIC-DOMAIN",
})

VERDICTS = frozenset({"ACCEPT", "CONDITIONAL", "REJECT"})


@dataclass
class DatasetCandidate:
    """One row of a Task-4-style candidate gate table. Every field mirrors
    a column this project has required for every accepted dataset so far
    (BDAPPV, SolNET, PVMD) - see docs/ML_HARDENING_PHASE6C.md Task 3/4."""

    name: str
    source_url: str
    license_id: str  # one of PERMISSIVE_LICENSES, or "UNKNOWN"/free text if not
    acquisition_date: str  # ISO date the candidate was reviewed, not downloaded
    image_type: str  # e.g. "aerial", "ground-level close-up", "thermal", "EL cell-level"
    has_bounding_boxes: bool
    target_domain_relevant: bool
    verdict: str
    verdict_reason: str
    provenance_notes: str = ""
    doi_or_repository: Optional[str] = None


def validate_candidate(candidate: DatasetCandidate) -> list[str]:
    """Return a list of problems with this candidate record (empty = valid
    record - NOT the same as "safe to use"; see gate_verdict_is_defensible
    for the actual accept/reject logic). Every required field must be
    genuinely filled in, matching this project's own established column
    set - a candidate with any blank required field cannot be recorded as
    reviewed."""
    problems: list[str] = []
    if not candidate.name.strip():
        problems.append("name is empty")
    if not candidate.source_url.strip():
        problems.append("source_url is empty")
    if not candidate.acquisition_date.strip():
        problems.append("acquisition_date is empty")
    if not candidate.image_type.strip():
        problems.append("image_type is empty")
    if candidate.verdict not in VERDICTS:
        problems.append(f"verdict must be one of {sorted(VERDICTS)}, got {candidate.verdict!r}")
    if not candidate.verdict_reason.strip():
        problems.append("verdict_reason is empty - every verdict must be justified")
    return problems


def gate_verdict_is_defensible(candidate: DatasetCandidate) -> tuple[bool, str]:
    """The actual ACCEPT/CONDITIONAL/REJECT logic, applied mechanically so
    a verdict can never silently drift from the stated policy (non-
    negotiable rules 7, 8, 10; docs/ML_HARDENING_PHASE6C.md Task 2/4).

    Returns (is_defensible, reason). An ACCEPT verdict is defensible only
    when the license is on the permissive allowlist AND the dataset has
    real bounding boxes AND it is marked target-domain relevant. Anything
    with an unknown/non-permissive license can be at most CONDITIONAL,
    never ACCEPT - this is the concrete enforcement of "do not use a
    dataset merely because a page claims it's free" and "reject datasets
    whose licensing cannot be established"."""
    license_ok = candidate.license_id.upper() in PERMISSIVE_LICENSES

    if candidate.verdict == "ACCEPT":
        if not license_ok:
            return False, "ACCEPT requires a license on the permissive allowlist"
        if not candidate.has_bounding_boxes:
            return False, "ACCEPT requires real bounding-box annotations"
        if not candidate.target_domain_relevant:
            return False, "ACCEPT requires target-domain relevance"
        return True, "license, annotations, and domain relevance all satisfied"

    if candidate.verdict == "REJECT":
        return True, "REJECT is always defensible as a conservative default"

    # CONDITIONAL
    return True, "CONDITIONAL is defensible pending further verification"


def promote_conditional_to_accept(
    candidate: DatasetCandidate, new_evidence: str, new_license_id: Optional[str] = None
) -> DatasetCandidate:
    """The ONLY sanctioned way to move a CONDITIONAL candidate to ACCEPT -
    requires fresh, non-empty evidence to be recorded (never a silent
    verdict flip - non-negotiable rule: "Do not silently convert
    CONDITIONAL into ACCEPT")."""
    if candidate.verdict != "CONDITIONAL":
        raise ValueError(f"can only promote a CONDITIONAL candidate, got verdict={candidate.verdict!r}")
    if not new_evidence.strip():
        raise ValueError("promoting a CONDITIONAL candidate requires non-empty new_evidence")
    license_id = new_license_id if new_license_id is not None else candidate.license_id
    promoted = DatasetCandidate(
        name=candidate.name, source_url=candidate.source_url, license_id=license_id,
        acquisition_date=candidate.acquisition_date, image_type=candidate.image_type,
        has_bounding_boxes=candidate.has_bounding_boxes,
        target_domain_relevant=candidate.target_domain_relevant,
        verdict="ACCEPT", verdict_reason=new_evidence,
        provenance_notes=candidate.provenance_notes, doi_or_repository=candidate.doi_or_repository,
    )
    ok, reason = gate_verdict_is_defensible(promoted)
    if not ok:
        raise ValueError(f"new evidence does not make ACCEPT defensible: {reason}")
    return promoted


# ---------------------------------------------------------------------------
# YOLO annotation validation
# ---------------------------------------------------------------------------

def validate_yolo_annotation_line(line: str, expected_class_id: int = 0) -> Optional[str]:
    """Validate one line of a YOLO-format label file. Returns None if
    valid, or a description of the problem. Enforces this project's
    single-class ("solar panel", id 0) taxonomy - a label file using any
    other class id is flagged, not silently accepted (guards against
    accidentally mixing an incompatible taxonomy in - non-negotiable rule
    11)."""
    parts = line.split()
    if len(parts) != 5:
        return f"expected 5 fields (class cx cy w h), got {len(parts)}: {line!r}"
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError as exc:
        return f"non-numeric field: {exc}"
    if class_id != expected_class_id:
        return f"unexpected class id {class_id}, expected {expected_class_id} (single-class 'solar panel' taxonomy)"
    for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not (0.0 <= value <= 1.0):
            return f"{name}={value} is outside the normalized [0,1] range"
    if w <= 0.0 or h <= 0.0:
        return f"degenerate box: w={w}, h={h}"
    return None


def validate_yolo_label_file(path: Path, expected_class_id: int = 0) -> list[str]:
    """Validate every line of a YOLO-format label file. An empty/absent
    file is valid (a real negative/no-object image) - not an error."""
    if not path.is_file():
        return []
    problems = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        problem = validate_yolo_annotation_line(line, expected_class_id)
        if problem:
            problems.append(f"line {i + 1}: {problem}")
    return problems


# ---------------------------------------------------------------------------
# Protected-artifact guard
# ---------------------------------------------------------------------------

# The real, released v1.0.0 production artifacts - never to be overwritten
# by any candidate training output (non-negotiable rules 4, 5).
PROTECTED_ARTIFACT_NAMES = frozenset({"yolo_solar.pt", "mobilenet_solar_v1.pth"})


def assert_candidate_path_is_safe(candidate_output_path: Path) -> None:
    """Raise ValueError if a proposed candidate-artifact output path would
    collide with (or be indistinguishable from) a protected v1 production
    artifact name. Call this before ever writing a training output path."""
    name = candidate_output_path.name
    if name in PROTECTED_ARTIFACT_NAMES:
        raise ValueError(
            f"refusing to write candidate output to {candidate_output_path} - "
            f"{name!r} is a protected v1 production artifact name and must never be overwritten"
        )
    if "candidate" not in name.lower() and "domain" not in name.lower():
        raise ValueError(
            f"candidate artifact path {candidate_output_path} does not clearly identify "
            "itself as a non-v1 candidate (expected a name containing 'candidate' or 'domain', "
            "e.g. weights/yolo_solar_domain_candidate.pt) - naming must make v1 vs. candidate "
            "unambiguous at a glance"
        )
