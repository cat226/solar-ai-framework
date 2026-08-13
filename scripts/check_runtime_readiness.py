#!/usr/bin/env python3
"""Report inference readiness without loading or fabricating model artifacts.

Exit codes:
  0 - all expected artifacts are present
  2 - one or more expected artifacts are missing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is importable when this script is executed directly
# (e.g. ``python scripts/check_runtime_readiness.py`` from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_manager import model_manager


def main() -> int:
    status = model_manager.artifact_status
    missing = [name for name, entry in status.items() if not entry["exists"]]

    payload = {
        "liveness": "ok",
        "inference_readiness": "ready" if not missing else "not_ready",
        "missing_artifacts": missing,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
