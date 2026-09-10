"""training/detection/ground_pv/condition_taxonomy.py — canonical ACTIVE
4-class RGB panel-condition taxonomy (Phase 15A).

    0 = CLEAN
    1 = DUSTY
    2 = BIRD_DROP
    3 = PHYSICAL_DAMAGE

This is the ONLY active taxonomy for panel-level RGB condition analysis.
Nothing downstream of this module should invent or accept a different
class set.

Explicitly and permanently EXCLUDED from the active taxonomy:

  - ELECTRICAL_DAMAGE — present in the existing 5-class research dataset
    (`datasets/ground_pv_research/condition/`, taxonomy defined in
    `training/detection/ground_pv/build_condition_dataset.py`) and the
    existing 5-class checkpoint (`weights/yolo_solar_condition_research_v1.pt`,
    kept frozen/untouched — see [[ground_pv_condition_model_baseline]]).
    Removed because the source audit found those annotations visually
    correspond to CELL CRACKS, not confirmed electrical failure — a
    semantic-risk finding recorded on every such instance
    (`electrical_damage_semantic_risk = HIGH`). This phase does NOT
    silently relabel those instances as PHYSICAL_DAMAGE (that would be
    fabricating a diagnosis the source data never supported) — see
    `build_condition_4class_dataset.py` for the exact removal policy.
  - SNOW — out of scope; was already excluded from the 5-class dataset.
  - Generic "DEFECTIVE" — ambiguous; was already excluded from the
    5-class dataset (and never present in either source anyway).
  - PANEL — panel localization is a SEPARATE stage
    ([[phase14a_dynamic_panel_detection]], `dynamic_panel_detection.py`).
    The condition taxonomy only describes what is happening on an
    ALREADY-LOCATED panel crop; it never re-answers "is this a panel".
  - HOTSPOT — requires thermal/IR imagery. RGB photographs cannot validate
    thermal-hotspot semantics (same reasoning as the ground_pv research
    detector's existing panel/defect taxonomy — see
    training/detection/ground_pv/__init__.py). A future separate thermal/
    IR modality, not this one.

No third-party dependency — importable without pulling in torch/ultralytics,
matching the project's convention for taxonomy/class-mapping modules (see
class_map.py).
"""
from __future__ import annotations

# Canonical active taxonomy — fixed order, contiguous class ids. Do not reorder.
CONDITION_4CLASS_ORDER: tuple[str, ...] = ("CLEAN", "DUSTY", "BIRD_DROP", "PHYSICAL_DAMAGE")
CONDITION_4CLASS_NAMES: dict[int, str] = {i: name for i, name in enumerate(CONDITION_4CLASS_ORDER)}
CONDITION_4CLASS_IDS: dict[str, int] = {name: i for i, name in CONDITION_4CLASS_NAMES.items()}

# Explicitly rejected — never valid as an active condition class. Checked by
# `assert_active_taxonomy` / `is_active_class` so a stray class from an older
# dataset or model can never silently pass through as if it were one of the
# four active classes.
REJECTED_CLASSES: frozenset[str] = frozenset({
    "ELECTRICAL_DAMAGE", "SNOW", "SNOW-COVERED", "DEFECTIVE", "PANEL", "HOTSPOT",
})

SCHEMA_VERSION = "1.0"


def is_active_class(name: str) -> bool:
    """True iff `name` (case-sensitive, canonical form) is one of the four
    active condition classes."""
    return name in CONDITION_4CLASS_IDS


def assert_active_taxonomy(names: dict[int, str] | list[str]) -> None:
    """Fail closed if `names` (a model's or dataset's class-id -> name
    mapping) is anything other than EXACTLY the four active classes in the
    canonical order. Raises ValueError naming exactly what's wrong — never
    silently accepts a 5-class (or otherwise mismatched) taxonomy as if it
    were the active 4-class one.
    """
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names)]
    else:
        ordered = list(names)

    if tuple(ordered) != CONDITION_4CLASS_ORDER:
        rejected_present = [n for n in ordered if n in REJECTED_CLASSES]
        detail = (
            f" — contains explicitly-rejected class(es): {rejected_present}"
            if rejected_present else ""
        )
        raise ValueError(
            f"not the active 4-class condition taxonomy: got {ordered}, "
            f"expected {list(CONDITION_4CLASS_ORDER)}{detail}. "
            "A 5-class (or other) checkpoint/dataset must never be silently "
            "treated as if it were the active 4-class taxonomy."
        )
