# Deployment

The reference production runtime is a Dockerized Streamlit application.

## Build

```bash
docker build -t solar-ai-framework:latest .
```

## Run

The application listens on port `8501`.

Model artifacts are intentionally excluded from Git and the image. Mount a
trusted directory containing the trained artifacts at `/app/weights`:

```bash
docker run --rm -p 8501:8501 \
  -e OPENWEATHER_API_KEY="$OPENWEATHER_API_KEY" \
  -v "$(pwd)/weights:/app/weights:ro" \
  solar-ai-framework:latest
```

Required files for real inference:

- `yolo_solar.pt`
- `mobilenet_solar.pth`
- `xgboost_solar.joblib`

Do not substitute generated, placeholder, or unverified model files.

## Model artifact verification

Trusted deployment artifacts should be accompanied by a reviewed JSON
manifest containing the SHA-256 digest of every artifact. Verify the mounted
artifacts before starting an inference-capable deployment:

```bash
python scripts/verify_model_artifacts.py --manifest /app/weights/manifest.json
```

The verifier reads local files only. It never downloads or creates model
artifacts and fails on missing files or digest mismatches.

## Health and readiness

The container exposes Streamlit's process health endpoint through Docker's
`HEALTHCHECK`. A healthy container means the web process is accepting health
requests; it does **not** mean trained model artifacts are present.

For inference readiness, run:

```bash
python scripts/check_runtime_readiness.py
```

This reports liveness and inference readiness separately and exits non-zero
when required model artifacts are missing. This distinction allows a service
orchestrator to keep process health separate from model availability.

## Secrets

Provide `OPENWEATHER_API_KEY` through the deployment platform's secret store
or environment configuration. Never bake secrets into the image or commit
`.streamlit/secrets.toml`.

## Production notes

- Python 3.12 is the supported runtime.
- The image runs as an unprivileged `appuser`.
- Uploaded files are capped at 10 MB by Streamlit configuration and are
  validated before decoding.
- XSRF protection is enabled and CORS is disabled.
- Model loading remains lazy so the application can start without model
  artifacts; an inference request fails with a typed `ModelLoadError` until
  the required artifacts are supplied.
- Dependency requirements remain minimum-version declarations. A reviewed
  lock/constraints strategy is still a separate reproducibility task.
