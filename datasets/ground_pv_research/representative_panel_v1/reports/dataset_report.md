# Representative panel dataset (Phase 14B) — dataset report

**Purpose:** assemble a deployment-domain dataset (camera-realistic, ground-level, individual-panel, native-aspect-ratio) to fix the two real-world failure modes confirmed in Phase 14A — a missed large foreground panel, and multiple physical panels represented as one group box. This report states plainly whether that dataset exists yet, never merely that the tooling ran.

## Verdict: **NOT READY FOR PHASE 14C**

**Blocking reasons:**

- only 6 image(s) registered — spec minimum is 100 independent scenes
- no human-verified individual-panel annotations exist yet
- no test-split scenes exist yet — no genuine independent holdout
- 4/6/8-panel scenes are not all represented (with known panel counts)
- large-foreground-panel coverage not yet confirmed by a human reviewer
- touching/grouped-panel coverage not yet confirmed by a human reviewer

## 1. Source inventory

- images registered: **6**
- scenes: **6**
- corrupt/unreadable files skipped: 0
- capture sources: {'first_party': 6}
- provenance statuses: {'first_party': 6}

## 2. Resolution / aspect-ratio distribution

- width range: [573, 1280]   height range: [573, 1280]
- aspect ratio range: [0.4477, 2.2339]
- orientation counts: {'portrait': 5, 'landscape': 1}
- unique resolutions: 2

## 3. Panel-count distribution

- images with a KNOWN panel count (human-annotated or manually recorded): 0 / 6
- images by panel count: {}
- scenes by panel count: {}
- target counts (4/6/8) covered: []
- target counts (4/6/8) MISSING: [4, 6, 8]

## 4. Duplicate / near-duplicate analysis

- exact-duplicate groups: 0 (0 redundant image(s))
- near-duplicate pairs (16x16 dHash): 0
- **cross-split exact-duplicate groups: 0** (must be 0)
- **cross-split near-duplicate pairs: 0** (must be 0)

## 5. Split methodology

Scene-isolated, deterministic (seed=42): images are first grouped into scenes (connected components over exact SHA-256 OR 16x16 dHash Hamming <= 8), then WHOLE scenes are greedily assigned to train/valid/test to approximate 70/20/10 by image count. A scene never crosses a split — verified explicitly, not merely assumed from the construction.

- no scene crosses a split: **True**
- split counts (images): {'reference': 6}
- split counts (scenes): {'reference': 6}

## 6. Annotation quality

- total images with a label FILE present (may be an empty, not-yet-filled-in stub): 6 / 6
- **images with a HUMAN-VERIFIED panel count (`annotation_status == 'annotated_human_verified'`): 0 / 6** — this is the number that actually counts toward readiness, not the line above; an empty stub file existing is not evidence of a verified 0-panel image.
- missing label files: 0
- images with zero annotations (a label file with no boxes in it — this bucket includes BOTH legitimate verified-empty images and unfilled stubs; see the human-verified count above to tell them apart): 6
- files with mechanical errors: 0
- images with overlap/size diagnostics: 0
- mechanically well-formed (syntax/geometry only — NOT a claim of visual correctness): True

## 7. Difficult-case coverage

- **large foreground panels**: unknown_needs_manual_tagging — requires a human to tag qualifying scenes; this tooling cannot infer "is this a large, clearly-bordered foreground panel" from pixels alone.
- **touching / grouped panels**: unknown_needs_manual_tagging — same: requires human tagging, never inferred.
- 4/6/8-panel scene coverage: [] present, [4, 6, 8] missing

## 8. Known limitations

- only 6 image(s) registered — spec minimum is 100 independent scenes
- no human-verified individual-panel annotations exist yet
- no test-split scenes exist yet — no genuine independent holdout
- 4/6/8-panel scenes are not all represented (with known panel counts)
- large-foreground-panel coverage not yet confirmed by a human reviewer
- touching/grouped-panel coverage not yet confirmed by a human reviewer

## 9. Six-image qualitative reference set

The original six owner photos (`D:\solar_ai_training_scratch\Solar Real Images`) are included in this manifest as a QUALITATIVE REFERENCE ONLY (per Phase 14B req 23) — **not mixed into the training split automatically**, and not treated as sufficient dataset volume on their own.
- registered as scenes: 6 from 6 image(s)
- known issues (Phase 14A, docs/DYNAMIC_PANEL_DETECTION.md): image 4's large foreground panel was MISSED entirely; image 6 produced two overlapping group-like boxes instead of per-panel boxes. Images 1 and 3's earlier `cracked` false positives are irrelevant to panel-only evaluation (see docs/DYNAMIC_PANEL_DETECTION.md sec.7).

## 10. Ready for Phase 14C?

**NOT READY FOR PHASE 14C**

Missing before Phase 14C (retraining) can responsibly start:

- only 6 image(s) registered — spec minimum is 100 independent scenes
- no human-verified individual-panel annotations exist yet
- no test-split scenes exist yet — no genuine independent holdout
- 4/6/8-panel scenes are not all represented (with known panel counts)
- large-foreground-panel coverage not yet confirmed by a human reviewer
- touching/grouped-panel coverage not yet confirmed by a human reviewer

