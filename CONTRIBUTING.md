# Contributing to Solar AI Framework

## Development Environment

- Python 3.12 (canonical test environment)
- Windows, Linux, or macOS for local development
- Virtual environment recommended

Setup:

```bash
git clone <repository-url>
cd solar-ai-framework
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Branching and Commits

- Work only on the active feature branch.
- Commit only the current task.
- Keep commit messages task-based.
- Do not force-push, rebase, or amend without explicit instruction.
- Do not merge branches.
- Do not resolve merge conflicts automatically.

## Test-Before-Commit

Run the full suite before committing:

```bash
py -3.12 -m pytest -q
```

Run import verification:

```bash
py -3.12 verify_imports.py
```

The suite must pass locally before pushing. CI will re-run the same checks.

## Test Isolation Rules

The test suite must remain order-independent. Contributors must not introduce global import-state pollution.

Specifically:

- Do not install fake modules into `sys.modules` at module import time.
- Do not leave `sys.modules` mutations in place after a test completes.
- Use pytest `monkeypatch` for per-test mocking; it is automatically restored.
- If a test needs to simulate a missing dependency, intercept the import boundary inside the test or fixture, not at module scope.

These rules were established in Sprint 3.3.26 and enforced in Sprint 3.3.30.

## CI Expectations

- CI runs on every push to `feature/testing-infrastructure` and on pull requests to `main`.
- CI installs the full dependency set from `requirements.txt` and `requirements-dev.txt`.
- A passing local run does not guarantee CI passes if test isolation is broken.
- Do not modify `.github/workflows/ci.yml` to hide test failures.

## Pull Requests

- Ensure the working tree is clean.
- Push only after local validation passes.
- Wait for architecture approval before merging to main.

## Configuration

- All tunable values live in `configs/settings.yaml`.
- API keys live in `.env` or `.streamlit/secrets.toml` (never commit secrets).
- Do not hardcode constants in business logic modules.

## Architecture Rules

- `app.py` must only contain UI logic.
- `services/pipeline.py` is the only file `app.py` imports from the framework.
- `models/` modules must not import from `services/`.
- `utils/` modules must not import from `models/` or `services/`.
- No new top-level directories may be created without architecture review.
