# Collection progress

_Regenerate with `python scripts/audit_representative_panel_dataset.py`._

## Counts

| metric | value |
|---|---|
| total scenes registered | 6 |
| total images registered | 6 |
| verified images (annotated_human_verified) | 0 |
| pending / not-yet-verified images | 6 |
| **verified independent scenes** (counts toward readiness) | **0** |
| verified panel instances (sum of panel_count_gt) | 0 |

## Panel-count distribution (verified only)

| panel count | images |
|---|---|
| _(none yet)_ | |

unknown panel counts (not yet annotated/verified): 6

## Resolution / aspect ratio

- aspect ratio range: [0.4477, 2.2339]   orientation: {'portrait': 5, 'landscape': 1}

## Coverage targets

| target | met? |
|---|---|
| >= 100 independent verified scenes (0/100) | False |
| >= 1 verified 4-panel scene | False |
| >= 1 verified 6-panel scene | False |
| >= 1 verified 8-panel scene | False |
| large-foreground-panel scene tagged | False |
| touching/grouped-panel scene tagged | False |
| genuine independent test holdout | False |
| scene-isolated splits, 0 cross-split duplicates | True |

## Verdict: **NOT READY FOR PHASE 14C**

Missing:

- only 0 verified independent scene(s) (annotated_human_verified, scene_status != needs_review, not in the reference split) — spec minimum is 100
- no human-verified individual-panel annotations exist yet
- no verified 4-panel scene yet
- no verified 6-panel scene yet
- no verified 8-panel scene yet
- no test-split scenes exist yet — no genuine independent holdout
- 4/6/8-panel scenes are not all represented (with known panel counts)
- large-foreground-panel coverage not yet confirmed by a human reviewer's tags
- touching/grouped-panel coverage not yet confirmed by a human reviewer's tags

