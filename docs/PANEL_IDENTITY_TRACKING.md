# Phase 17 — Persistent panel identity across runs

Research feature, not production-validated. Answers the question the
project vision names as the actual point of Solar AI (context doc §41-42,
§66): not "what does this panel look like right now," but "Panel P2:
CLEAN → DUSTY, first observed capture 3."

## Why this didn't already exist

`training/detection/ground_pv/dynamic_panel_detection.py`'s `P1..PN` ids
are explicitly documented as spatial identity **within one capture only**
— nothing tied "P2 in today's photo" to "P2 in last week's photo." Building
that link was always future work (`docs/PHASE_16_ARCHITECTURE_REVIEW.md`
§12), deliberately deferred until the rest of the pipeline (panel
detection, condition classification, history persistence) was real and
working.

## Method: geometry matching, not appearance matching

Per the architecture review's own recommendation: given the target
deployment is a **fixed, pole-mounted camera**, the pragmatic first
approach is comparing each panel's **normalized position in frame** to the
previous run — not pixel/feature/appearance matching, and not a tracking
algorithm. This is a deliberately narrow, honest scope, not a shortcut
disguised as the full solution.

`training/detection/ground_pv/panel_identity.py` (pure Python, no model,
no GPU — directly unit-tested, 17 tests):

1. `normalized_center(bbox, image_width, image_height)` — a panel's pixel
   bbox becomes a `(cx, cy)` pair in `[0,1]`, relative to **that photo's
   own dimensions** (tolerant of the encoded resolution changing slightly
   between captures of a genuinely fixed camera — auto-exposure/auto-focus
   can do that even when framing doesn't change).
2. `match_panels(current, previous, max_distance=0.25)` — greedy
   nearest-neighbor bipartite matching between two panel sets from the
   SAME capture slot at two points in time. Deterministic tie-breaking
   (distance, then source id) — never depends on input order. A previous
   panel can be claimed by at most one current panel; anything beyond
   `max_distance` is reported unmatched, never forced.
3. `match_run_to_previous(current_slots, previous_slots)` — applies that
   per **capture slot** (the Nth photo this run vs. the Nth photo last
   run). No cross-slot matching: a panel in "capture 2" is never compared
   against a panel previously seen in "capture 1," even at an identical
   position — see the "what this assumes" list below.

## What this assumes, stated explicitly

- **The camera is genuinely fixed** between runs (same position, same
  framing). This is the project's own target deployment, not an
  assumption invented for convenience.
- **Captures are taken in the same order each run** — "photo 1" is always
  the same physical section of the array. If you photograph the array in
  a different order next time, this module has no way to tell that apart
  from the panels actually having moved, and will report spurious "new"
  panels rather than silently guessing which old panel each one really is.
- **No appearance/content matching.** Two visually identical panels at
  different positions are never confused with each other by content, but
  two DIFFERENT panels that happen to land at nearly the same normalized
  position between runs (e.g. the array was slightly reconfigured) could
  be matched incorrectly — the geometry has no way to know that. This is
  the honest cost of staying in scope for this phase; see "Future work"
  below for what would close this gap.

## What "new" vs. "matched" actually means

- **Matched**: the closest previous panel in the same capture slot was
  within `max_distance` (0.25 normalized units, a generous-but-bounded
  default — panels in this project's real reference photos typically
  span a large fraction of frame width). The persistent id is reused.
- **New**: either no previous panel existed in that slot yet, or nothing
  was close enough. A **new persistent id** is assigned
  (`PANEL-001`, `PANEL-002`, ... — a simple incrementing counter, reserved
  atomically within the same database transaction so two new panels in
  one run can never collide).

Neither outcome is ever presented as more certain than it is — every
match carries its geometric distance, and the UI (`app.py`,
`pages/11_🕰️_Panel_History.py`) labels this whole feature "estimated,"
not verified.

## Condition changes — the actual payoff

A condition change (`services/storage.py::record_panel_identities`) is
only ever recorded when **both** the previous and the new read are a
CONFIDENT (`"predicted"`) read, using the CLASSIFIER's condition (the
primary signal — see `docs/CONDITION_MODEL_4CLASS_CLASSIFIER_V1.md`), and
the label differs. An `"uncertain"` read is never compared as if it were
a confirmed condition — the same no-fabrication discipline already
applied everywhere else in this page's attention summary.

## Storage

Three new SQLite tables (`services/storage.py`), separate from
`ground_inspections` (which stores one row per *run*; these store
per-*panel* state and history):

- `panel_identities` — current known state of every tracked panel (one
  row per persistent id, always the MOST RECENT observation).
- `panel_observations` — full history, one row per (panel, run) it was
  actually seen in.
- `panel_condition_changes` — detected transitions, computed once at
  write time (same pattern as `ground_inspections`' precomputed counts)
  rather than re-derived from `panel_observations` on every page load.

## Testing

- `tests/test_ground_pv_panel_identity.py` (17 tests) — the pure matching
  algorithm: exact-position matching, closest-pair-wins (not
  first-in-list), one previous panel can't be double-claimed, deterministic
  tie-breaking, cutoff behavior, no cross-slot matching, missing/extra
  capture slots.
- `tests/test_storage_panel_identity.py` (10 tests) — the full
  match-and-persist pipeline against a real (temp) SQLite database: first
  run assigns new ids, a second run at the same position reuses them,
  condition changes are detected only between confident reads, a
  far-away panel is correctly treated as new rather than force-matched,
  and cross-slot isolation holds end to end.

**Not yet tested against real multi-day captures** — no such data exists
yet (see the project context doc §38, §58). Every test above uses
synthetic or same-session data. The honest status is: the matching
algorithm and persistence pipeline are verified correct against their own
stated logic; whether that logic holds up against a real fixed camera
photographing a real array over real weeks is unknown until that data
exists.

## Future work (explicitly out of scope for this phase)

- **Appearance/feature matching** as a second signal when geometry is
  ambiguous (e.g. two candidate matches within `max_distance` of each
  other) — would need real multi-day data to validate against, not
  buildable honestly right now.
- **Cross-slot matching** for when capture order genuinely varies.
- **A confidence-decay or "not seen recently" status** for a persistent
  panel that stops appearing in matches (currently it just silently stays
  at its last known state forever — not wrong, but not flagged either).
