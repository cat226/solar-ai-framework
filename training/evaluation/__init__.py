"""training/evaluation — dataset/leakage evaluation tooling.

The scripts that evaluated the legacy aerial pipeline's production wrappers
(models.detector, models.classifier, models.model_manager) were removed
alongside that pipeline — see docs/PHASE_16_ARCHITECTURE_REVIEW.md. What
remains here (common.py, dataset_candidate_gate.py, leakage_audit.py) is
generic dataset-quality tooling with no dependency on either pipeline.
Nothing here changes model weights, training data, or any live pipeline.
"""
