# Phase 20: dataset search (Oct 2026) and owner-marks label proposal

Date: 2026-10-21. **Analysis only: no training, no label change, no weight change, nothing downloaded.**
Verified = checked on the source page or file listing this session. Inferred / not verified is stated per row.

## 1. New-dataset search: result is NO suitable public dataset (agrees with Phase 12)

Target: ground-level / close-up RGB photos of rooftop PV with panel boxes and
CLEAN / DUSTY / BIRD_DROP / PHYSICAL_DAMAGE-type labels, with a usable licence. Aerial/UAV data excluded by the owner.

| Candidate | What it is (verified unless noted) | Verdict |
|---|---|---|
| Kaggle `ucimachinelearning/solar-panel-fault-dataset` (new, 2026-10-13) | File listing: `Bird-Drop/bd_NNN.JPG`, 107 MB, no licence/description in metadata | **Reject**: renumbered re-pack of the Afroz scrape (inferred from taxonomy + file layout); provenance unknown |
| Kaggle `alicjalena/pv-panel-defect-dataset`, `mohammadhossini/dataset-solar` | Same 6 folder names and `Bird (N).jpg` naming as `pythonafroz/solar-panel-images`; licence fields empty | **Reject**: Afroz lineage, licence "Unknown" upstream; already overlap-audited as existing/duplicate pool |
| Kaggle `salonipandagale/...-dl-project`, `pythonafroz/*` | Sizes 319-329 MB = Afroz re-packs | **Reject** (same lineage) |
| SolarFCD (arXiv 2604.23662) | Compiles 3 sources: Garladinne dust (CC0 per Kaggle), **Afroz (licence Unknown, 873 imgs)**, PVMD thermal; image-level labels only; paper itself says annotations don't support detection | **Reject** for boxes; re-ships the Afroz images |
| IEEE DataPort "Solar Panel Anomaly Detection... Insecticidal Lamp IoT" (Yang, Fang; 2025-12-03) | cracks / bird droppings / dust, 1.11 GB; **login required, no licence stated, counts/format not stated** | **Blocked**: cannot inspect without bypassing the access gate; same first author as the Mendeley set Phase 12 found ~90% scraped stock |
| Zenodo 15525194 "Solar panel defects" (CC BY 4.0) | 4,447 imgs; classes are *fingerprints, black core, thick line, star crack, short circuit* | **Reject**: these are electroluminescence cell-defect terms, wrong modality (capture method not stated on page - inferred) |
| Zenodo 21447895 | A journal PDF, not a dataset | n/a |
| Garladinne dust (CC0 on Kaggle), DeepSolarEye (IEEE DataPort) | Web-scraped (per SolarFCD) / fixed camera, no boxes | **Reject** (CC0 tag is a platform claim, not image-level rights; no boxes) |
| DuraMAT hail / storm / EL sets | EL or satellite imagery; hail photo archive page unreachable (404) | **Reject** / unverified |
| Roboflow "solar-panels-with-bird-droppings" (AIOT Team 1), MDPI 17/20/5238 Figshare, PMC11821171 | Pages returned HTTP 403 to my fetch; UAV for PMC11821171 | **Not verified**. Roboflow CC BY is a re-uploader claim (same rule as Phases 7/12) |

Not searchable by me: Roboflow Universe full-text search (no public API). Everything found is either
the known unlicensed Afroz lineage, wrong modality, aerial, or gated.

**Conclusion: the only legitimate, in-domain data is the owner's own photos.**

## 2. Working with the owner's marks (6 photos, 7 boxes)

Files: `rooftop_condition_v1/reports/owner_marks_audit.md`, `owner_marks_extraction.json`, `owner_marks_label_proposal.csv`.
Pristine mark-region crops (7, owner-private) are outside the repo: `F:\Solar AI Derived\owner_marks_v1\crops\`.

Proposal (not applied): 5 panel relabels where a mark is >=90% inside exactly one panel box:

| Image | Scene | Split | Current -> Proposed |
|---|---|---|---|
| IMG_5907 p1 | 0006 | valid | CLEAN -> BIRD_DROP |
| WhatsApp 01.03.35 p6 | 0003 | train | CLEAN -> BIRD_DROP |
| WhatsApp 01.03.37 p1 | 0004 | train | CLEAN -> BIRD_DROP |
| WhatsApp 01.03.37 (1) p2 | 0004 | train | BIRD_DROP -> PHYSICAL_DAMAGE |
| WhatsApp 01.03.46 p3 | 0004 | train | BIRD_DROP -> PHYSICAL_DAMAGE |

IMG_5888 (scene 0000, two BIRD DROP boxes) is **ambiguous** (each box lies inside or straddles two
overlapping panel boxes) and is marked NEEDS_OWNER_DECISION.

### The honest sample size is smaller than the row count
- All 7 boxes fall in only **4 scenes** (0000, 0003, 0004, 0006). Scene 0004 holds 4 of them.
- The two PHYSICAL_DAMAGE marks are in the same scene (0004) and, from the two images I viewed,
  look like the **same physical panel in two video frames** (inferred, not confirmed). That is one
  defect, not two.
- Marked scenes also contain unmarked sibling frames (listed per row in the CSV). Whether the panels
  in those frames carry the same defect is unknown; do not propagate labels to them automatically.
- Only scene 0006 (one mark) is in the valid split; the test split has none. **No held-out claim about
  defect detection can be made from this**, and retraining on ~4 independent scenes would repeat the
  Phase 19-B finding (more retraining on a handful of non-CLEAN crops will not fix it).

## 3. What would actually move the project forward (needs the owner)
1. Confirm the marks are the owner's and the proposed panel mapping (esp. small marks covering 9-36% of a panel,
   and IMG_5888).
2. Review the remaining 42 photos and mark them too, **including an explicit "no defect" tag** so absence of
   a mark is evidence rather than silence.
3. New photos targeted at defects: different panels, days and angles, with each distinct defect photographed
   from several distances. Aim for defect examples in **different scenes** (scene count, not image count, is the
   limiting number), and keep whole scenes out of train when held out.
4. Only after 1-3: rebuild the labeling sheet from owner labels, re-split by scene, and evaluate the active
   classifier on the owner-labeled held-out scenes **before** any retraining (lesson 2.3 of the handoff).

## 4. Diagnostic: what the classifiers say about the marked panels (read-only, 2026-10-21)

Pristine panel crops (built before the marks existed) and pristine mark-region crops, 224px, argmax.
Active = rooftop MIX x1 (`mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth`); previous = Phase 16-B.

| Panel (split) | Owner mark | Active | Previous |
|---|---|---|---|
| IMG_5888 x3 candidate panels (train) | BIRD_DROP (ambiguous) | CLEAN .99 / .99 / .97 | CLEAN |
| **IMG_5907 p1 (valid)** | BIRD_DROP | **CLEAN 1.00** (mark region: CLEAN .91) | CLEAN .98 |
| 01.03.35 p6 (train) | BIRD_DROP | CLEAN .95 (mark region: CLEAN .97) | DUSTY .79 |
| 01.03.37 p1 (train) | BIRD_DROP | CLEAN .99 (mark region: **BIRD_DROP .68**) | CLEAN .68 |
| 01.03.37 (1) p2 (train) | PHYSICAL_DAMAGE | BIRD_DROP .98 | PHYSICAL_DAMAGE .65 |
| 01.03.46 p3 (train) | PHYSICAL_DAMAGE | BIRD_DROP .94 | DUSTY .53 |

Reading it honestly:
- The active model agrees with the owner mark on 0 of 8 whole-panel crops and on 1 of 7 mark-region crops.
- **Mostly not independent evidence**: 7 of 8 panels are in the TRAIN split with the AI labels (CLEAN / BIRD_DROP) that the
  model was fitted to, so reproducing those labels at 0.94-0.99 is training fit, not a test. Only **IMG_5907 (valid) is a true
  held-out check, n=1**: the active model calls a panel the owner marked BIRD_DROP "CLEAN" at 1.00.
- Consistent with the Phase 19-B caveat that defect detection on this rooftop is unmeasured; it does not by itself show the
  classifier is bad (the mapping of marks to labels is unconfirmed, and a small mark inside a whole-panel crop is hard at 224px).
- The previous Phase 16-B model gets PHYSICAL_DAMAGE on 1 of the 2 damage panels, but it also scored 27% on the owner's
  held-out rooftop crops, so this does not argue for reverting. No decision on model changes is made here.
