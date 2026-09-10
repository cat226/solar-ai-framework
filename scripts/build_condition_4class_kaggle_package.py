"""scripts/build_condition_4class_kaggle_package.py — CLI wrapper around
training.detection.ground_pv.build_condition_4class_kaggle_package.

Stages datasets/ground_pv_research/condition_4class_v1/ into
kaggle/ground_condition_4class_dataset/ for upload as a PRIVATE Kaggle
Dataset. Local only — does not call the Kaggle API.

    python scripts/build_condition_4class_kaggle_package.py --overwrite
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.detection.ground_pv.build_condition_4class_kaggle_package import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
