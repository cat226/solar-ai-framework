# Repository Audit — 2026-08-12

## Scope

Audit of `main`, all feature branches, pull requests, GitHub Actions, CI configuration, model artifacts, application startup, tests, and repository documentation.

## Baseline

`main` is `870bed041ba4fa2e186b17656881014813777aca`.

The latest merged testing commit reports 676 tests and approximately 94% statement coverage. The verified GitHub Actions run for PR #3 also completed `verify_imports.py` 14/14 and `pytest -q` with 676 passed and 94% total coverage.

## Branches

Branches present at audit time:

- `main`
- `feature/coverage-gate`
- `feature/input-validation-hardening`
- `feature/task-002-physics`
- `feature/testing-infrastructure`
- `feature/verona`
- `audit/ci-and-artifact-hardening` (audit branch)

PR #3 is closed and merged. PR #2 and PR #1 are also historical completed work; there are no unmerged changes from those PRs requiring recovery.

## Blockers found

### 1. CI did not run on `main` pushes

The workflow only triggered push validation for `feature/testing-infrastructure`. Pull requests into `main` were covered, but direct changes landing on `main` were not automatically revalidated.

**Fix:** CI now runs on `main`, `feature/**`, and `audit/**` pushes, plus pull requests to `main` and manual dispatch.

### 2. Import verification could report failures while exiting successfully

`verify_imports.py` collected errors but did not terminate with a non-zero status. That made the CI gate weaker than its output suggested.

**Fix:** verification now exits with status 1 whenever any check fails, and missing dependencies are failures rather than skips.

### 3. Model weights are intentionally absent

The repository does not contain trained YOLO, MobileNet, or XGBoost artifacts. This is a real deployment prerequisite, not a test failure. No synthetic or fabricated weights were added.

The model manager already raises typed `ModelLoadError` exceptions for missing artifacts. The audit branch additionally exposes `ModelManager.artifact_status` so callers can inspect configured paths and readiness without attempting to create or substitute artifacts.

### 4. Live end-to-end inference remains blocked by external artifacts/credentials

Unit and integration-style tests deliberately mock model weights and network services. A genuine inference run requires the trained model files and an OpenWeatherMap API key supplied outside the repository.

## Application startup

The Streamlit entry point is parseable and imports successfully under the existing CI environment. Model loading remains lazy, so absent weights do not make the application source itself fail at startup. The first real inference attempt correctly surfaces the missing-artifact error.

## Test policy

Tests must remain deterministic and network-free. Model artifacts must be supplied externally; tests must never generate fake production weights to make an inference path appear healthy.

## Verification

The audit branch uses the same Python 3.12 dependency installation and test commands as the existing CI workflow:

```text
python verify_imports.py
python -m pytest -q
```

The branch-specific CI run is the authoritative verification for the changes made here.
