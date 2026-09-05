FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Runtime libraries required by OpenCV / image processing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Model artifacts are intentionally not bundled. Provide the trained files
# under /app/weights at deployment time when inference is required.
#
# Only the one directory the app actually writes to at runtime
# (services/storage.py's SQLite history file, data/inspections.db) is
# owned by the runtime user. Application source stays root-owned and
# read-only even to appuser - if the running app were ever compromised via
# a code-execution bug, the attacker still could not rewrite the app's own
# source files to persist a backdoor, only write within data/.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown appuser:appuser /app/data
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"

ENTRYPOINT ["streamlit", "run", "app.py"]
