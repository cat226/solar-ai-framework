# ground_condition_4class_classifier_dataset/ — staging for a PRIVATE Kaggle Dataset

Kaggle id: `edithstark/solar-ai-ground-condition-4class-classifier-v1` (PRIVATE).

Built by `scripts/build_condition_4class_classifier_dataset.py` from `datasets/ground_pv_research/condition_4class_v1/` (Phase 16-B). The materialised image tree (train/valid/test/<CLASS>/*.jpg) is gitignored; `dataset-metadata.json`, this README, and `provenance/` are committed.

PERSONAL NON-COMMERCIAL RESEARCH ONLY — provenance grade C, unverified (inherited unchanged from the source detection dataset).

- images: 2411 (train 1686 / valid 483 / test 242) — every image from the source dataset kept, none dropped
- label derivation: dominant class by box-instance count, ties by total box area, then alphabetical (`training/detection/ground_pv/condition_taxonomy.py::dominant_class`)
- images with >1 distinct class in their original annotations (dominant-only label kept): 910 / 2411 — see `provenance/condition_4class_classifier_v1_report.json`
- split assignment: verbatim from the source dataset's leakage-clean split — never re-computed
