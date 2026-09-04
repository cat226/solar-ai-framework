# Runtime Security and Observability

## Secrets

- Keep API keys and credentials out of `configs/settings.yaml` and source control.
- `OPENWEATHER_API_KEY` is resolved at call time through `utils.config.get_secret()`.
- Streamlit secrets are preferred, followed by environment variables.
- `.streamlit/secrets.toml` and `.env` are excluded from Docker build context.

## User input

- Streamlit limits image uploads to 10 MB and validates image bytes before decoding.
- City input is capped at 100 characters and control characters are removed before API calls and logging.
- Pipeline/model failures are returned through typed results rather than exposing uncontrolled tracebacks in the UI.

## Model artifact deserialization

- YOLO (`ultralytics.YOLO(...)`) and MobileNet (`torch.load(..., weights_only=True)`)
  loading is hardened against arbitrary-code-execution-on-load; `weights_only=True`
  specifically restricts the MobileNet checkpoint to tensor data only.
- The XGBoost predictor loads via `joblib.load(...)`, which is pickle-based and
  therefore capable of executing arbitrary code if given an untrusted file. No
  XGBoost artifact exists for the v1 release (this load path is never reached in
  a v1 deployment), and the project's supply-chain policy already requires every
  artifact to come from a controlled, reviewed process before deployment (see
  `docs/ARTIFACT_SUPPLY_CHAIN.md`) — never a substitute for pickle safety, but the
  existing mitigation for this specific gap. Sandboxing/format hardening this load
  path is a real future improvement, out of scope for this release since XGBoost
  itself is not shipped in v1.

## Container security

- The production container uses Python 3.12 slim Bookworm.
- The application runs as an unprivileged `appuser`.
- A container healthcheck targets Streamlit's health endpoint.
- No trained model artifacts are embedded in the image; trusted artifacts must be supplied separately.

## Observability

Application modules use the centralized logger from `utils.logger`. Operational events include weather fallback, invalid uploads, pipeline status, and model-loading failures. Logs must never contain API keys or other credentials.

## Validation

CI runs the normal test suite and a Docker build/startup smoke test. The smoke test verifies that Streamlit starts and its health endpoint becomes available without requiring model artifacts.
