"""training/detection/ground_pv/panel_identity.py — persistent panel
identity across Ground Inspection runs (Phase 17).

Problem this solves: `dynamic_panel_detection.py`'s P1..PN ids are ONLY
spatial identity within a single capture — nothing ties "P2 in today's
photo" to "P2 in last week's photo" of the same array. Without that link,
there is no way to say "Panel P2: CLEAN -> DUSTY since Tuesday", which is
the actual product vision (see the project context doc's Definition of
Success).

Approach — geometry matching under a fixed-camera assumption (the
documented, deliberately un-ambitious choice from
docs/PHASE_16_ARCHITECTURE_REVIEW.md §12, not appearance/feature matching
or tracking): if the camera is genuinely fixed between captures, a panel
photographed today should be at roughly the same NORMALIZED position in
frame as it was last time. This module matches panels between one run's
captures and the most recent PREVIOUS run's captures, per capture slot
(the Nth photo this run is compared against the Nth photo last run — see
`match_run_to_previous`'s docstring for exactly what that assumes and how
it degrades when the assumption doesn't hold).

This is a BEST-EFFORT ESTIMATED match, never a verified one. It is never
presented as certain identity — every match carries the geometric distance
it was matched at, and callers must surface that a persistent id is an
estimate (see docs/PANEL_IDENTITY_TRACKING.md).

No third-party dependency (pure python), matching this codebase's
convention for matching/geometry modules (dynamic_panel_detection.py,
panel_condition.py) — directly unit-testable without a GPU or model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"

# Normalized (0-1, relative to each photo's own width/height) Euclidean
# distance between two panel centers, beyond which they are considered
# DIFFERENT panels rather than the same panel photographed again. Panels in
# these captures typically span a large fraction of frame width side by
# side (see the six real reference photos: 1-2 panels filling most of the
# frame), so a generous-but-bounded cutoff is used — tunable, not a magic
# number buried in logic.
_DEFAULT_MAX_MATCH_DISTANCE = 0.25


@dataclass
class PanelRef:
    """One panel detection, reduced to exactly what identity-matching
    needs: its normalized position. `source_id` is the caller's own
    reference (e.g. a within-run "C1-P2" display id, or a persistent id
    string) — this module never invents or parses ids, only matches
    positions and hands back the caller's own references paired up."""

    source_id: str
    center_x: float  # normalized 0-1, relative to that panel's own photo width
    center_y: float  # normalized 0-1, relative to that panel's own photo height

    @property
    def as_tuple(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)


@dataclass
class MatchResult:
    """One current panel's match outcome against the previous run's known
    panels in the same capture slot."""

    current_source_id: str
    matched_previous_source_id: str | None
    distance: float | None  # None when unmatched (no previous panel within the cutoff)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_source_id": self.current_source_id,
            "matched_previous_source_id": self.matched_previous_source_id,
            "distance": round(self.distance, 4) if self.distance is not None else None,
        }


def normalized_center(bbox: Sequence[float], image_width: float, image_height: float) -> tuple[float, float]:
    """Panel bbox [x1,y1,x2,y2] in pixel coords -> normalized (cx, cy) in
    [0,1] relative to its own photo's dimensions. Normalizing per-photo
    (not to some shared canvas) is what makes this tolerant of the photo
    resolution differing slightly between two captures of a genuinely fixed
    camera (auto-exposure/auto-focus can change encoded resolution even
    when framing does not)."""
    x1, y1, x2, y2 = bbox
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image_width/image_height must be positive, got {image_width}x{image_height}")
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    return cx, cy


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def match_panels(
    current: Sequence[PanelRef],
    previous: Sequence[PanelRef],
    *, max_distance: float = _DEFAULT_MAX_MATCH_DISTANCE,
) -> list[MatchResult]:
    """Greedy nearest-neighbor bipartite matching between two panel sets
    from the SAME capture slot at two different points in time.

    Deterministic: candidate pairs are considered in order of (distance
    ascending, current_source_id, previous_source_id) so ties never depend
    on input order or dict/set iteration order. Each previous panel can be
    claimed by at most one current panel (closest wins); any current panel
    left over after all previous panels are claimed, or whose closest
    remaining previous panel is farther than `max_distance`, is reported
    unmatched (a genuinely new panel, or the camera/framing changed enough
    that this module is honest it can't tell).

    This does NOT attempt one-to-many or many-to-one matches, and does NOT
    use appearance/pixel content — position only. See the module docstring
    for why that's the deliberate scope for this phase.
    """
    if not current:
        return []
    if not previous:
        return [MatchResult(c.source_id, None, None) for c in current]

    candidates: list[tuple[float, str, str]] = []
    for c in current:
        for p in previous:
            d = _distance(c.as_tuple, p.as_tuple)
            if d <= max_distance:
                candidates.append((d, c.source_id, p.source_id))
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    claimed_previous: set[str] = set()
    matched_current: dict[str, tuple[str, float]] = {}
    for d, cur_id, prev_id in candidates:
        if cur_id in matched_current or prev_id in claimed_previous:
            continue
        matched_current[cur_id] = (prev_id, d)
        claimed_previous.add(prev_id)

    return [
        MatchResult(c.source_id, *matched_current[c.source_id]) if c.source_id in matched_current
        else MatchResult(c.source_id, None, None)
        for c in current
    ]


@dataclass
class CaptureSlotPanels:
    """One capture's panels, reduced for matching: which capture slot
    (0-based index within its run) they belong to, plus each panel's
    normalized position and the caller's own reference id for it."""

    capture_slot: int
    panels: list[PanelRef] = field(default_factory=list)


def match_run_to_previous(
    current_slots: Sequence[CaptureSlotPanels],
    previous_slots: Sequence[CaptureSlotPanels],
    *, max_distance: float = _DEFAULT_MAX_MATCH_DISTANCE,
) -> dict[int, list[MatchResult]]:
    """Match every capture slot present in BOTH the current and previous
    run, independently, by calling `match_panels` per slot.

    What this assumes, and how it degrades:
      - It assumes captures are taken in the SAME order/position each run
        (e.g. "capture 1" is always the left half of the array) — a
        genuinely fixed camera setup, matching the project's target
        deployment (a pole-mounted camera capturing on a schedule).
      - If the current run has MORE capture slots than the previous run
        (e.g. array coverage was extended), the extra slots have no
        previous panels to match against — every panel in them comes back
        unmatched (new), which is correct, not an error.
      - If the current run has FEWER capture slots than the previous run
        (e.g. one photo was skipped this time), those missing slots are
        simply not present in the returned dict — their previously-known
        panels are not "lost", just not observed this run.
      - It does NOT attempt to match a panel from slot 2 against a panel
        that was previously seen in slot 1 (no cross-slot matching) — if
        capture order genuinely varies between runs, this will report
        spurious "new" panels rather than silently guessing. That's a
        known, documented limitation, not a bug: this module has no way to
        tell that apart from a real camera move without appearance
        matching, which is explicitly out of scope for this phase (see the
        module docstring).

    Returns a dict of capture_slot -> list[MatchResult], only for slots
    present in `current_slots`.
    """
    previous_by_slot = {s.capture_slot: s.panels for s in previous_slots}
    results: dict[int, list[MatchResult]] = {}
    for slot in current_slots:
        prev_panels = previous_by_slot.get(slot.capture_slot, [])
        results[slot.capture_slot] = match_panels(slot.panels, prev_panels, max_distance=max_distance)
    return results
