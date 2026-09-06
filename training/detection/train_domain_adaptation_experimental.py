#!/usr/bin/env python3
"""training/detection/train_domain_adaptation_experimental.py

PERSONAL / NON-COMMERCIAL / EXPERIMENTAL training run only.

Fine-tunes the existing production aerial YOLO checkpoint
(weights/yolo_solar.pt) on a mixed aerial+ground dataset to produce an
EXPERIMENTAL domain-adapted candidate. This script:

  - NEVER writes to weights/yolo_solar.pt or any path under weights/
    that the production runtime reads by default - it only ever writes
    to an explicit --output path the caller chooses.
  - NEVER trains from scratch - always starts from --base-model, which
    defaults to the production checkpoint.
  - Uses conservative settings by default (low learning rate, frozen
    early backbone layers, early stopping via --patience) specifically
    to reduce the risk of catastrophically forgetting the aerial domain
    while adapting to ground-level imagery - this is a fine-tune, not a
    full retrain.
  - Does not evaluate anything itself - see
    training/detection/evaluate_domain_adaptation_experimental.py for
    the separate, independent before/after metrics computation against
    untouched held-out data.

This is not a general-purpose training entrypoint (unlike train_yolo.py,
which trains the production aerial-only model from a single data root).
It exists specifically for the mixed-domain-list data.yaml format this
personal experiment needs and is not wired into any CI/production path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("weights/yolo_solar.pt"))
    parser.add_argument("--output", type=Path, required=True, help="Where to copy the resulting best.pt (must NOT be under weights/ using the production filename)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=10, help="Freeze this many leading layers (backbone) to reduce catastrophic forgetting + compute cost")
    parser.add_argument("--lr0", type=float, default=0.0005, help="Conservative initial learning rate (production default fine-tune uses 0.01)")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", type=str, default="domain_adapt_experimental")
    args = parser.parse_args()

    if args.output.resolve() == Path("weights/yolo_solar.pt").resolve():
        raise SystemExit("Refusing to write to weights/yolo_solar.pt - use a distinct --output path.")

    base_model_sha256 = _sha256_file(args.base_model)

    model = YOLO(str(args.base_model))
    results = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        lr0=args.lr0,
        optimizer=args.optimizer,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"training did not produce a checkpoint at {best_pt}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(best_pt.read_bytes())
    output_sha256 = _sha256_file(args.output)

    summary = {
        "PERSONAL_NON_COMMERCIAL_EXPERIMENTAL": True,
        "base_model": str(args.base_model),
        "base_model_sha256": base_model_sha256,
        "data_yaml": str(args.data_yaml),
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "checkpoint_saved_to": str(args.output),
        "output_sha256": output_sha256,
    }
    summary_path = args.output.with_suffix(".train_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
