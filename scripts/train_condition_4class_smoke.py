#!/usr/bin/env python3
"""scripts/train_condition_4class_smoke.py — Phase 15A LOCAL SMOKE training
for the active 4-class condition model.

This is explicitly NOT a production training run. Its only purpose is to
verify the architecture, the 4-class label mapping, and end-to-end
inference plumbing with a REAL checkpoint — not to claim production
accuracy (the task's own Final Decision Rule: "A trained model is NOT
required to declare the software phase successful"). Local CPU training is
only practical at this tiny scale; the equivalent full-scale experiment
for the 5-class research model ran on Kaggle GPU (~1h36m — see
[[ground_pv_condition_model_baseline]]).

Reuses the SAME safety pattern as scripts/train_ground_pv.py:
  - never initializes from weights/yolo_solar.pt (or anything with
    "yolo_solar" in the name that isn't explicitly this experiment's own
    output path),
  - a pretrained COCO base (yolo11n.pt), never the aerial production model,
  - validates the data.yaml's taxonomy is EXACTLY the active 4-class set
    before training even starts (assert_active_taxonomy) - refuses to
    silently train against a mismatched/5-class dataset.
  - writes to a NEW checkpoint filename
    (weights/yolo_solar_condition_4class_research_v1.pt), never overwrites
    the existing 5-class weights/yolo_solar_condition_research_v1.pt.

Example
-------
    python scripts/train_condition_4class_smoke.py \
        --data-yaml datasets/ground_pv_research/condition_4class_v1/data.yaml \
        --epochs 2 --fraction 0.05 --imgsz 320 --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from training.detection.ground_pv.condition_taxonomy import assert_active_taxonomy  # noqa: E402

_PRODUCTION_WEIGHTS_NAMES = {"yolo_solar.pt", "yolo_solar_condition_research_v1.pt"}


def _resolve_class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=_REPO_ROOT / "weights" / "yolo_solar_condition_4class_research_v1.pt")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="Train on this random FRACTION of the dataset each epoch (Ultralytics native "
                             "support) — keeps a CPU smoke run bounded without physically subsetting files.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=Path, default=_REPO_ROOT / "research_runs" / "ground_pv" / "condition_4class_smoke")
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {args.data_yaml}. Run build_condition_4class_dataset.py first.")

    if args.output.name in _PRODUCTION_WEIGHTS_NAMES or "yolo_solar.pt" in str(args.output):
        raise SystemExit(f"refusing to write to {args.output} — would overwrite a protected checkpoint.")
    if Path(args.base_model).name in _PRODUCTION_WEIGHTS_NAMES:
        raise SystemExit(f"refusing to initialize training from {args.base_model} — must be a pretrained COCO checkpoint.")

    class_names = _resolve_class_names(args.data_yaml)
    try:
        assert_active_taxonomy(class_names)
    except ValueError as exc:
        raise SystemExit(f"REFUSING to train: {exc}") from exc

    from ultralytics import YOLO
    import ultralytics

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        torch = None
        cuda = False

    train_kwargs = dict(
        data=str(args.data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        seed=42,
        project=str(args.project),
        name="smoke_v1",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        patience=0,  # smoke run — no early-stopping patience needed for 1-2 epochs
        # conservative augmentation (mirrors the project's existing ground-PV convention)
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.3,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0,
    )

    print(json.dumps({
        "smoke_run": True, "not_production": True, "base_model": args.base_model,
        "resolved_train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
        "ultralytics": ultralytics.__version__, "cuda_available": cuda,
    }, indent=2))

    start = time.time()
    model = YOLO(args.base_model)
    results = model.train(**train_kwargs)
    elapsed = time.time() - start

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise SystemExit(f"training produced no checkpoint at {best_pt}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(best_pt.read_bytes())
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    manifest = {
        "smoke_run": True,
        "purpose": "verify architecture / 4-class label mapping / end-to-end inference plumbing — "
                  "NOT a claim of production accuracy",
        "base_model": args.base_model,
        "data_yaml": str(args.data_yaml),
        "class_names": class_names,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "fraction": args.fraction,
        "device": args.device, "cuda_available": cuda,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "ultralytics_version": ultralytics.__version__,
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "checkpoint_saved_to": str(args.output),
        "checkpoint_sha256": sha,
    }
    manifest_out = args.manifest_out or (run_dir / "smoke_training_manifest.json")
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\ncheckpoint -> {args.output}\nsha256 {sha}\nmanifest -> {manifest_out}\nelapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
