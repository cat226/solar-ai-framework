"""training/cloud — Provider-independent cloud training orchestration.

Keeps the existing, working training scripts (training/classification/,
training/detection/) untouched. This package only adds a layer around them
for running the same jobs on remote GPU workers: a serializable job
specification, an experiment registry, artifact validation helpers, and
provider-specific adapters (currently: Kaggle).

See training/cloud/README.md for what is actually verified to work versus
what is designed but untested.
"""
