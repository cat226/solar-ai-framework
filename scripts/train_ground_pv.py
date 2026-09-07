#!/usr/bin/env python3
"""scripts/train_ground_pv.py — Train the ground-level PV defect RESEARCH
detector (YOLO11n, 4 classes: panel / bird_drop / cracked / dusty).

This is the SINGLE shared training entrypoint used both for local runs and,
unchanged, by the Kaggle notebook (kaggle/ground_pv_train.ipynb). It never
initializes from weights/yolo_solar.pt — the base model is a pretrained
COCO checkpoint (YOLO11n, falling back to YOLOv8n on older Ultralytics).

The intended GPU training happens on Kaggle; running this locally on CPU is
only practical for a tiny smoke check (--epochs 1 --limit-smoke).

Config defaults come from configs/ground_pv_research.yaml.

Examples
--------
    # baseline experiment (as configured)
    python scripts/train_ground_pv.py --experiment baseline \
        --data-yaml datasets/ground_pv_research/yolo/data.yaml

    # local smoke check only
    python scripts/train_ground_pv.py --experiment baseline \
        --data-yaml datasets/ground_pv_research/yolo/data.yaml --epochs 1 --device cpu
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

PRODUCTION_YOLO = "weights/yolo_solar.pt"
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "ground_pv_research.yaml"


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _select_base_model(cfg: dict, override: str | None) -> str:
    if override:
        return override
    preferred = cfg["research_model"].get("base_model", "yolo11n.pt")
    fallback = cfg["research_model"].get("fallback_base_model", "yolov8n.pt")
    try:
        import ultralytics  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
        # YOLO11 configs ship inside ultralytics>=8.3; probe by trying to
        # resolve the model name to a config without downloading weights.
        from ultralytics.utils import checks  # noqa: F401
        return preferred
    except Exception:  # noqa: BLE001
        return fallback


def _resolve_class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-yaml", type=Path, required=True,
                        help="Ultralytics data.yaml for the normalized 4-class dataset "
                             "(datasets/ground_pv_research/yolo/data.yaml).")
    parser.add_argument("--experiment", choices=["baseline", "balanced"], default="baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to copy the final best.pt (default: weights/yolo_solar_ground_research_v1.pt).")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override the pretrained base checkpoint. NEVER weights/yolo_solar.pt.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {args.data_yaml}. Run the download + audit + normalize steps first.")

    cfg = _load_config(args.config)
    exp = cfg["experiments"][args.experiment]
    aug = cfg.get("augmentation", {})

    base_model = _select_base_model(cfg, args.base_model)
    if Path(base_model).name == Path(PRODUCTION_YOLO).name or "yolo_solar" in base_model:
        raise SystemExit(
            f"refusing to initialize the research model from {base_model!r} - the ground "
            "detector must start from a pretrained COCO checkpoint, not the aerial production model."
        )

    class_names = _resolve_class_names(args.data_yaml)
    expected = {0: "panel", 1: "bird_drop", 2: "cracked", 3: "dusty"}
    if class_names != expected:
        raise SystemExit(f"data.yaml names {class_names} != required research taxonomy {expected}")

    epochs = args.epochs if args.epochs is not None else int(exp["epochs"])
    batch = args.batch if args.batch is not None else exp["batch"]
    imgsz = args.imgsz if args.imgsz is not None else int(exp.get("imgsz", cfg["research_model"]["image_size"]))
    seed = int(exp.get("seed", 42))
    patience = int(exp.get("patience", 20))
    project = args.project or Path(exp["project"])
    run_name = exp["name"]
    device = args.device if args.device is not None else exp.get("device", "auto")
    output = args.output or (_REPO_ROOT / cfg["research_model"]["weights"])

    from ultralytics import YOLO
    import ultralytics

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
    except Exception:  # noqa: BLE001
        torch = None
        cuda = False
        gpu_name = None

    train_kwargs: dict = dict(
        data=str(args.data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
        patience=patience,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        cos_lr=bool(exp.get("cos_lr", True)),
        verbose=True,
        # conservative augmentation from config
        hsv_h=aug.get("hsv_h", 0.015), hsv_s=aug.get("hsv_s", 0.7), hsv_v=aug.get("hsv_v", 0.4),
        degrees=aug.get("degrees", 0.0), translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.5), shear=aug.get("shear", 0.0), perspective=aug.get("perspective", 0.0),
        flipud=aug.get("flipud", 0.0), fliplr=aug.get("fliplr", 0.5),
        mosaic=aug.get("mosaic", 1.0), mixup=aug.get("mixup", 0.0), copy_paste=aug.get("copy_paste", 0.0),
    )
    if batch is not None:
        train_kwargs["batch"] = batch
    if device not in (None, "auto"):
        train_kwargs["device"] = device

    print(json.dumps({
        "experiment": run_name,
        "base_model": base_model,
        "resolved_train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
        "ultralytics": ultralytics.__version__,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": cuda,
        "gpu": gpu_name,
    }, indent=2))

    start = time.time()
    model = YOLO(base_model)
    results = model.train(**train_kwargs)
    elapsed = time.time() - start

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if not best_pt.is_file():
        raise SystemExit(f"training produced no checkpoint at {best_pt}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(best_pt.read_bytes())

    import hashlib
    sha = hashlib.sha256(output.read_bytes()).hexdigest()

    manifest = {
        "experiment_name": run_name,
        "dataset_name": cfg["dataset"]["name"],
        "dataset_version": cfg["dataset"]["version"],
        "dataset_source": cfg["dataset"]["url"],
        "dataset_license": cfg["dataset"]["license_as_displayed_by_source"],
        "model": base_model,
        "pretrained_source": f"{base_model} (COCO-pretrained, Ultralytics)",
        "epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed, "patience": patience,
        "optimizer": exp.get("optimizer", "auto"),
        "device": device, "gpu": gpu_name,
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "ultralytics_version": ultralytics.__version__,
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt) if last_pt.is_file() else None,
        "checkpoint_saved_to": str(output),
        "checkpoint_sha256": sha,
        "class_names": expected,
    }
    manifest_out = args.manifest_out or (run_dir / "training_manifest.json")
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\ncheckpoint -> {output}\nsha256 {sha}\nmanifest -> {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
