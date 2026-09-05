"""training/evaluation — independent evaluation tooling for released Solar AI
model artifacts.

Deliberately separate from production inference code (models/, services/):
these scripts *consume* the real production wrappers (models.detector,
models.classifier, models.model_manager) to measure real performance against
held-out data, but must never be imported by the application itself. Nothing
here changes model weights, training data, or the production pipeline.
"""
