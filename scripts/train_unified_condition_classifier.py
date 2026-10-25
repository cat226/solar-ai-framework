#!/usr/bin/env python3
"""scripts/train_unified_condition_classifier.py — train an IMAGE-LEVEL
4-class condition classifier (MobileNetV2, ImageNet-pretrained) on the family-
split dataset built by scripts/build_unified_condition_dataset.py.

Differences from scripts/train_condition_4class_classifier.py (kept for
reproducibility of Phase 16-B, untouched):
  * checkpoint selection + early stopping on validation MACRO-F1, not accuracy
    (the pool is ~85x imbalanced CLEAN : PHYSICAL_DAMAGE; accuracy would reward
    predicting the majority class);
  * whole-frame input: Resize((224, 224)) with NO centre crop (photos are
    landscape; a centre crop would cut off the sides) - recorded in the
    checkpoint metadata so inference can reproduce it;
  * stronger scale/colour augmentation (web photos -> a different camera);
  * report: per-class P/R/F1 with Wilson 95% intervals, balanced accuracy,
    confusion matrix, per-lineage breakdown, and saved test predictions.

Writes NEW artifacts only; no production weight is touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from training.detection.ground_pv.condition_taxonomy import CONDITION_4CLASS_ORDER  # noqa: E402

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_SIZE = 224


def _tf(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop((_SIZE, _SIZE), scale=(0.6, 1.0), ratio=(0.9, 1.6)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    return transforms.Compose([transforms.Resize((_SIZE, _SIZE)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])


class Pool(Dataset):
    """Reads the split from the manifest so each sample keeps its lineage."""

    def __init__(self, root: Path, manifest: list[dict], split: str, classes: list[str], train: bool):
        self.items = [(root / r["out_rel"], classes.index(r["label"]), r["lineage"], r["image_id"])
                      for r in manifest if r["split"] == split]
        self.tf = _tf(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, _, _ = self.items[i]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def metrics_from_cm(cm: list[list[int]], classes: list[str]) -> dict:
    k = len(classes)
    per, recalls, f1s = {}, [], []
    for i, c in enumerate(classes):
        tp = cm[i][i]
        n_true = sum(cm[i])
        n_pred = sum(cm[r][i] for r in range(k))
        prec = tp / n_pred if n_pred else None
        rec = tp / n_true if n_true else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if n_true else None)
        per[c] = {"n": n_true, "tp": tp, "precision": prec, "recall": rec, "recall_wilson95": wilson(tp, n_true), "f1": f1}
        if rec is not None:
            recalls.append(rec)
        if f1 is not None:
            f1s.append(f1)
    n = sum(map(sum, cm))
    return {"accuracy": sum(cm[i][i] for i in range(k)) / n if n else None,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None, "per_class": per}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    return torch.cat(probs), torch.cat(ys)


def confusion(probs, ys, k):
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(ys.tolist(), probs.argmax(1).tolist()):
        cm[t][p] += 1
    return cm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("F:/Solar AI Datasets/condition_unified_v1"))
    ap.add_argument("--manifest", type=Path, default=_REPO / "datasets/ground_pv_research/unified_condition_v1/manifest_representatives.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="unified_v1")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--report-dir", type=Path, default=_REPO / "datasets/ground_pv_research/unified_condition_v1")
    args = ap.parse_args(argv)
    ckpt_path = args.checkpoint_out or (_REPO / f"weights/mobilenet_condition_{args.tag}.pth")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    classes = list(CONDITION_4CLASS_ORDER)  # canonical order (NOT alphabetical) - index == class id everywhere
    sets = {s: Pool(args.dataset, manifest, s, classes, train=(s == "train")) for s in ("train", "valid", "test")}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"), num_workers=args.workers,
                             persistent_workers=args.workers > 0) for s, d in sets.items()}
    print(f"device={device} classes={classes} " + " ".join(f"{s}={len(d)}" for s, d in sets.items()), flush=True)

    counts = Counter(y for _, y, _, _ in sets["train"].items)
    n_tr = len(sets["train"])
    weights = torch.tensor([n_tr / (len(classes) * max(1, counts[i])) for i in range(len(classes))], dtype=torch.float32)
    print("train counts:", {classes[i]: counts[i] for i in range(len(classes))}, "loss weights:", [round(w, 2) for w in weights.tolist()], flush=True)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_f1, best_ep, stale = [], -1.0, -1, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        e0 = time.time()
        model.train()
        tl, tn = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += float(loss) * len(y)
            tn += len(y)
        sched.step()
        vp, vy = predict(model, loaders["valid"], device)
        vm = metrics_from_cm(confusion(vp, vy, len(classes)), classes)
        history.append({"epoch": ep, "train_loss": tl / tn, "val_acc": vm["accuracy"], "val_balanced_acc": vm["balanced_accuracy"],
                        "val_macro_f1": vm["macro_f1"], "seconds": time.time() - e0})
        print(f"epoch {ep:2d}/{args.epochs} loss={tl / tn:.4f} val_acc={vm['accuracy']:.3f} val_balacc={vm['balanced_accuracy']:.3f} "
              f"val_macroF1={vm['macro_f1']:.3f} ({time.time() - e0:.0f}s)", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, stale = vm["macro_f1"], ep, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop (best epoch {best_ep}, val macro-F1 {best_f1:.3f})", flush=True)
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    out = {}
    for split in ("valid", "test"):
        p, y = predict(model, loaders[split], device)
        cm = confusion(p, y, len(classes))
        out[split] = {"confusion_matrix": cm, **metrics_from_cm(cm, classes)}
        if split == "test":
            test_probs, test_y = p, y
    items = sets["test"].items
    by_lin = {}
    for lin in sorted({it[2] for it in items}):
        idx = [i for i, it in enumerate(items) if it[2] == lin]
        k = sum(int(test_probs[i].argmax() == test_y[i]) for i in idx)
        by_lin[lin] = {"n": len(idx), "accuracy": k / len(idx), "accuracy_wilson95": wilson(k, len(idx)),
                       "class_counts": dict(Counter(classes[int(test_y[i])] for i in idx))}
    out["test_by_lineage"] = by_lin

    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / f"test_predictions_{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "lineage", "true", "pred", *[f"p_{c}" for c in classes]])
        for i, (_, y, lin, iid) in enumerate(items):
            w.writerow([iid, lin, classes[y], classes[int(test_probs[i].argmax())], *[f"{v:.4f}" for v in test_probs[i].tolist()]])

    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    meta = {"classes_index_order": classes, "architecture": "torchvision mobilenet_v2, classifier[1]=Linear(1280,4)",
            "preprocessing": f"Resize(({_SIZE},{_SIZE})) whole frame, NO centre crop; ImageNet mean/std",
            "checkpoint_sha256": sha}
    ckpt_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = {"tag": args.tag, "dataset": str(args.dataset), "device": str(device), "seed": args.seed,
              "hyperparameters": {"epochs_requested": args.epochs, "epochs_run": len(history), "batch_size": args.batch_size,
                                  "lr": args.lr, "optimizer": "AdamW wd=1e-4 + cosine", "patience": args.patience,
                                  "selection_metric": "val macro-F1"},
              "train_class_counts": {classes[i]: counts[i] for i in range(len(classes))},
              "class_loss_weights": dict(zip(classes, weights.tolist())),
              "best_epoch": best_ep, "best_val_macro_f1": best_f1, "training_seconds": time.time() - t0,
              "history": history, "results": out, "checkpoint": str(ckpt_path), "checkpoint_sha256": sha,
              "caveats": ["test PHYSICAL_DAMAGE / BIRD_DROP n is tiny: read the Wilson intervals, not the point estimates",
                          "all sources are web-scraped / licence-unresolved: research only",
                          "test is family-disjoint but drawn from the SAME web distribution - it is NOT a rooftop-camera benchmark"]}
    (args.report_dir / f"training_report_{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = out["test"]
    print(f"\nTEST acc={t['accuracy']:.3f} balanced_acc={t['balanced_accuracy']:.3f} macroF1={t['macro_f1']:.3f}")
    for r in t["confusion_matrix"]:
        print("  ", r)
    for c, m in t["per_class"].items():
        print(f"  {c:16s} n={m['n']:4d} recall={m['recall']} wilson95={m['recall_wilson95']} precision={m['precision']}")
    print("by lineage:", json.dumps(by_lin))
    print("checkpoint:", ckpt_path, sha[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
