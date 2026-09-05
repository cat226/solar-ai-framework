# YOLO Detection Dataset — Audit Report

Generated: 2026-09-04
Source: BDAPPV IGN subset, converted via `training/detection/prepare_dataset.py`
Manifest: `E:/Solar AI Training Images/yolo_prepared/manifest.json` (not committed — local data artifact)

## Summary

| Check | Result |
|---|---|
| Decode failures (images) | 0 |
| Malformed/non-binary masks | 0 |
| Out-of-bounds coordinates | 0 (bboxes are derived from pixel positions within the 400×400 mask by construction) |
| Zero-area boxes | 2 rows discarded (has_mask=True but the labeled component had zero extractable area after connected-component analysis) — 0.01% of the dataset, not investigated further |
| Duplicate images (global, source-level) | 218 found and excluded (kept first-seen only) |
| Cross-split leakage in **written output** | **0** — verified twice: (1) source-level SHA-256 analysis found 8 raw duplicate groups spanning splits in the *source* parquet data, one of which was a single blank white "no imagery available" placeholder tile duplicated across 92 different installation records (has_mask=False, 1 unique RGB color, confirmed by direct decode); (2) direct SHA-256 hashing of the actual **written** image files in `train/`, `val/`, `test/` confirmed zero hashes appear in more than one split — the pipeline's global (not per-shard, not per-split) duplicate detection during conversion eliminates this before any file is written. |
| Unexpected image dimensions | 0 (all 400×400, matching the dataset's documented fixed size) |

## Final dataset composition

| Split | Images | Positive (≥1 panel) | Negative (validated, no panel) | Total instances |
|---|---|---|---|---|
| train | 11,347 | 5,077 | 6,270 | 6,926 |
| val | 3,179 | 1,351 | 1,828 | 1,745 |
| test | 2,581 | 1,238 | 1,343 | 1,640 |
| **Total** | **17,107** | **7,666** | **9,441** | **10,311** |

17,107 records + 220 excluded anomalies ≈ 17,327, consistent with the ~17,325 IGN total reported by both the Zenodo record and the Hugging Face dataset card (small variance from the 218 duplicates found).

Negative (no-panel) images outnumber positive images in every split (55–58% negative) — expected and useful for a detector (reduces false-positive rate), not a class-imbalance problem requiring correction, since "negative" here means zero instances rather than an under-represented positive class.

## A real bug caught and fixed during this audit

The first conversion run wrote 0 validation images — `_SPLIT_MAP` assumed the parquet `split` column's value matched the shard filename pattern (`"validation"`) without verifying it against the actual downloaded data. The real value is `"val"`. Fixed in commit `f2813fc`, re-run confirmed correct. Documented here per this project's policy of disclosing real issues found during work, not just final clean results.

## Conclusion

**Audit PASSED.** No provenance, schema, or data-integrity problem serious enough to block proceeding. The one dramatic-looking finding (92-way duplicate) resolved to a confirmed-benign blank-placeholder artifact with zero information content, correctly collapsed to a single copy by the existing duplicate-detection logic — not a defect requiring further action. Proceeding to training.
