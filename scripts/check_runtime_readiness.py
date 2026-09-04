#!/usr/bin/env python3
"""Report inference readiness without loading or fabricating model artifacts.

Exit codes:
  0 - all expected artifacts are present
  2 - one or more expected artifacts are missing

"MobileNet" is ready whenever either the v1 release artifact or the future
six-class artifact is present - see ModelManager.artifact_status. "XGBoost"
is checked against its one real path and, for the current Solar AI v1
release, is *expected* to be reported missing: no legitimate training
dataset exists yet (see training/prediction/DATASET_SOURCES.md), so a v1
deployment with real YOLO + MobileNet artifacts correctly reports
not_ready/exit 2 with only ["XGBoost"] missing. That is honest, accurate
reporting of a real v1 capability boundary, not a defect - this script
never treats a permanently-scoped-out artifact as "ready" just because a
release doesn't need it.
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
