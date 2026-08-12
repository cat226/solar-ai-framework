# Dependency reproducibility

## Supported environment

The supported interpreter is **Python 3.12**. The application stack includes PyTorch and Ultralytics, so the supported interpreter must be one for which the complete dependency set is available.

## Dependency policy

`requirements.txt` intentionally expresses minimum supported versions rather than pretending those lower bounds are a fully locked environment. CI resolves those requirements on every run and validates the resulting environment with `python -m pip check`, import checks, and the full test suite.

A reproducible deployment should consume a separately generated, reviewed constraints/lock set from a known-good Python 3.12 environment. Versions must never be guessed or fabricated in source control.

## Resolution procedure

From a clean Python 3.12 environment:

```text
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip freeze --all > requirements-constraints.txt
python -m pip check
python -m pytest -q --cov=. --cov-report=term-missing
```

Then run the repository Docker build and smoke test. Review the generated set for Python 3.12 compatibility, platform-specific packages, duplicate/conflicting requirements, unexpected direct dependencies, and PyTorch/torchvision compatibility.

Only a resolution that passes those checks should replace the placeholder constraints file.

## CI contract

CI continues to use the canonical lower-bound requirements until a validated resolved environment is captured. This preserves a meaningful compatibility test without creating a false claim of reproducibility.

## Updating the constraints set

When direct dependencies change, regenerate from a clean Python 3.12 environment and repeat `pip check`, the complete test suite, and the Docker smoke test. Record the resolution date and source environment in the pull request description. Validate the resolved set on the supported CI runner before merging.
