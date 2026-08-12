# Dependency Reproducibility

## Supported environment

The supported CI and development interpreter is **Python 3.12**. The application stack includes PyTorch and Ultralytics, so the supported interpreter must be one for which the complete dependency set is available.

## Dependency policy

`requirements.txt` intentionally expresses minimum supported versions rather than pretending that those lower bounds constitute a fully locked environment. CI installs the current compatible releases from those requirements and then validates the resulting environment with `python -m pip check`.

This gives us two separate guarantees:

1. **Compatibility:** the repository declares minimum versions and CI verifies that the current dependency resolver can produce a coherent environment.
2. **Reproducibility:** deployments that require byte-for-byte dependency reproduction should consume a separately generated, reviewed lock/constraints file from a known-good Python 3.12 environment. Such a file must be generated from actual verified installations; versions must not be guessed or fabricated in source control.

## CI contract

CI performs, in order:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip check
python verify_imports.py
python -m pytest -q
```

`pip check` is deliberately a separate gate. Import tests can pass while package metadata still contains an incompatible dependency relationship.

## Future lockfile work

Before creating a committed lock/constraints file, capture a successful Python 3.12 installation from CI or a controlled deployment environment, review the complete resolved dependency graph, and verify the lock on the supported Windows CI runner. PyTorch/torchvision compatibility should be validated as a pair rather than pinned independently.
