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

## Container security

- The production container uses Python 3.12 slim Bookworm.
- The application runs as an unprivileged `appuser`.
- A container healthcheck targets Streamlit's health endpoint.
- No trained model artifacts are embedded in the image; trusted artifacts must be supplied separately.

## Observability

Application modules use the centralized logger from `utils.logger`. Operational events include weather fallback, invalid uploads, pipeline status, and model-loading failures. Logs must never contain API keys or other credentials.

## Validation

CI runs the normal test suite and a Docker build/startup smoke test. The smoke test verifies that Streamlit starts and its health endpoint becomes available without requiring model artifacts.
