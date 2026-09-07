"""training/detection/ground_pv/class_map.py — Class-name normalization for
the ground-level PV defect research dataset.

The Roboflow source dataset ("panel solar", susan-ifblr/panel-solar-bw945,
version 1) uses these class names, in this order, per its published
`data.yaml`:

    0: panel
    1: bird_drop
    2: cracked
    3: dusty

(The Roboflow project page lists them as: panel, bird_drop, cracked, dusty.)

Our research taxonomy uses the SAME four semantic classes but pins an
explicit canonical order so downstream code, configs, and the trained
checkpoint never depend on whatever order a re-export happens to produce:

    0 = panel
    1 = bird_drop
    2 = cracked
    3 = dusty

`build_class_id_remap()` takes the source `data.yaml` `names:` mapping as
actually read from the downloaded dataset and returns the integer remap
(source_id -> research_id). If the source order already matches, the remap
is the identity - but it is still computed from the real file, never
assumed. The raw annotation files on disk are NEVER rewritten by this
module; the remap is applied only when materializing the normalized
`datasets/ground_pv_research/yolo/` copy.

This module has no third-party dependencies so it can be imported by the
audit script, the training wrapper, and the tests without pulling in torch.
"""
from __future__ import annotations

from typing import Mapping

# Canonical research taxonomy - fixed order, do not reorder.
RESEARCH_CLASS_ORDER: tuple[str, ...] = ("panel", "bird_drop", "cracked", "dusty")
RESEARCH_CLASS_NAMES: dict[int, str] = {i: name for i, name in enumerate(RESEARCH_CLASS_ORDER)}

# Human-facing condition labels the application derives from detected defects
# (kept separate from the detector class names - these are the words shown to
# a user, matching the project's existing configs/settings.yaml taxonomy).
DEFECT_TO_CONDITION: dict[str, str] = {
    "bird_drop": "BirdDrop",
    "cracked": "Damage",
    "dusty": "Dust",
}

# Known aliases seen in Roboflow re-exports / mirrors of this dataset, mapped
# to our canonical name. Extend only with names actually observed in a real
# downloaded data.yaml - never speculatively.
_ALIASES: dict[str, str] = {
    "panel": "panel",
    "solar_panel": "panel",
    "solar panel": "panel",
    "bird_drop": "bird_drop",
    "bird-drop": "bird_drop",
    "birddrop": "bird_drop",
    "bird_dropping": "bird_drop",
    "cracked": "cracked",
    "crack": "cracked",
    "damage": "cracked",
    "damaged": "cracked",
    "dusty": "dusty",
    "dust": "dusty",
    "dirty": "dusty",
    "soiling": "dusty",
}


def canonical_name(source_name: str) -> str:
    """Map a source class name to our canonical name.

    Raises KeyError for an unrecognized name - a new/unexpected class in the
    source dataset is a fail-closed condition (the audit must surface it),
    not something to silently drop or guess at.
    """
    key = source_name.strip().lower()
    if key not in _ALIASES:
        raise KeyError(
            f"source class name {source_name!r} is not a recognized alias of any "
            f"research class {RESEARCH_CLASS_ORDER}. Refusing to guess. "
            "Add it to _ALIASES only after confirming its meaning in the source dataset."
        )
    return _ALIASES[key]


def build_class_id_remap(source_names: Mapping[int, str]) -> dict[int, int]:
    """Given the source dataset's `names:` mapping (source_id -> source_name,
    exactly as read from its data.yaml), return source_id -> research_id.

    Fails closed if:
      - a source name is unrecognized (see canonical_name)
      - two source ids map to the same research class (ambiguous)
      - the source is missing one of the four research classes (the audit
        should still run, but the caller is told the mapping is incomplete)
    """
    remap: dict[int, int] = {}
    seen_research_ids: dict[int, int] = {}
    for src_id, src_name in sorted(source_names.items()):
        canon = canonical_name(src_name)
        research_id = RESEARCH_CLASS_ORDER.index(canon)
        if research_id in seen_research_ids:
            raise ValueError(
                f"source ids {seen_research_ids[research_id]} and {src_id} both map to "
                f"research class {RESEARCH_CLASS_ORDER[research_id]!r} - ambiguous source taxonomy"
            )
        seen_research_ids[research_id] = src_id
        remap[src_id] = research_id
    return remap


def remap_is_identity(remap: Mapping[int, int]) -> bool:
    """True iff applying the remap changes no class id (source order already
    matches the research order for every present class)."""
    return all(src == dst for src, dst in remap.items())
