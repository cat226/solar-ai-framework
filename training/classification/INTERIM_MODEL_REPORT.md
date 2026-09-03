# Interim 3-Class MobileNet Model Report

Generated: 2026-09-03
Repository: https://github.com/cat226/solar-ai-framework
Branch: feat/mobilenet-training-pipeline

## Status: NOT the production model

This report documents an **interim, non-production** MobileNetV2 checkpoint trained on
three of the six required classes (`Clean`, `Dusty`, `Hotspot`). It exists to validate the
training pipeline end-to-end on genuine data while `Bird-Drop`, `Electrical-Damage`, and
`Physical-Damage` remain blocked on data sourcing/access (see `DATASET_SOURCES.md`).

**This checkpoint is not compatible with the production `ModelManager`** (which requires
all six classes in the fixed production order) and must never be saved to or loaded from
`weights/mobilenet_solar.pth`. It is saved to `weights/mobilenet_interim_3class.pth`
(gitignored, not committed) purely as a local validation artifact.

## Critical bug found and fixed (2026-09-03, pre-merge review)

A `/code-review` pass on this PR, independently verified by direct execution (not just
trusted), found that **the class-order remapping this entire PR exists to add did not
actually work**. `train_mobilenet.py`'s `_dataset()` and `evaluate_mobilenet.py`'s
`_map_dataset_to_production()` both reassigned `ds.targets` to the remapped
production-order indices, but `torchvision.datasets.ImageFolder.__getitem__` reads labels
from `self.samples`, not `self.targets` — confirmed directly against torchvision's source
and with a live repro (`ds.targets` said `[1, 0, 2]` after remapping; `ds[i]` still
returned the original alphabetical targets `[0, 1, 2]`). Reassigning `.targets` alone is
dead code as far as any `DataLoader` is concerned.

**Impact**: any training or evaluation run where alphabetical class order differs from the
requested order — which includes every real six-class production run (alphabetical:
Bird-Drop, Clean, Dusty, Electrical-Damage, Hotspot, Physical-Damage vs. production:
Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Hotspot) — would have silently
trained/evaluated against scrambled labels while every fail-closed check still passed.

**Why this checkpoint's own results are unaffected**: `Clean, Dusty, Hotspot` happens to
already be alphabetically sorted, so the (broken) identity-adjacent remapping coincidentally
matched the correct one for this specific 3-class combination. The 100% test accuracy and
the app-integration validation reported elsewhere in this file are genuine and unaffected —
verified by re-running both after the fix and confirming byte-identical results. But the bug
would have silently corrupted the very training run this PR was created to make correct
(the full six-class production order), and would corrupt any future non-alphabetical
`--classes` subset too.

**Fix**: both `_dataset()`/`_map_dataset_to_production()` now return a proper dataset
wrapper whose `__getitem__` remaps the target at access time (delegating to the wrapped
`ImageFolder.__getitem__` for the sample, then mapping the *original* returned target
through the production-order mapping), instead of mutating an attribute the loader never
reads. Added regression tests that call `ds[idx]` directly with a deliberately
non-alphabetical requested order and assert it matches both `.targets` and the true source
folder — the previous test suite only ever exercised alphabetically-ordered class sets and
so never caught this. A third, related bug was found and fixed in the same pass:
`prepare_dataset.py`'s SHA-256 duplicate detection used a fresh dict per `--source` root,
so a duplicate image spanning two different source roots (not exercised by this project's
actual single-root runs, but supported by the CLI) would not have been caught.

## Dataset provenance

| Class | Source | License | Raw downloaded | After dedup |
|---|---|---|---|---|
| Clean | SolNET (MDPI Energies 2023), Google Drive | CC BY 4.0 | 722 | 713 |
| Dusty | SolNET (MDPI Energies 2023), Google Drive | CC BY 4.0 | 718 | 635 |
| Hotspot | PVMD (Mendeley, DOI 10.17632/5ssmfpgrpc.1) | CC BY 4.0 | 350 | 187 |

Hotspot's PVMD archive was downloaded directly from Mendeley's public API and SHA-256
verified against the API's own manifest (`484ade84fa513012c67de58dfc6d372ebd333604ccf940b0aab8bcfca684ae69`)
before extraction — the archive itself is untampered. The drop from 350 to 187 unique
images happened during deduplication (see below), not during download.

## Deduplication finding

`prepare_dataset.py`'s SHA-256 duplicate check (by design, fail-closed on any duplicate
anywhere in the source) caught real exact-duplicate files within the downloaded data:
**190 duplicate groups, 255 redundant files removed in total**, across all three classes.
All duplicates were confirmed **intra-class only** (zero cross-class conflicts) — i.e.
redundant copies of the same image under different filenames (e.g.
`20210917_151334.jpg` / `20210917_151334(1).jpg`), not a labeling problem. This is most
likely an artifact of how the original SolNET/PVMD datasets were assembled (duplicate
uploads), not something introduced by this download.

Hotspot in particular lost 163/350 images (46.6%) to deduplication — a notably higher
rate than Clean/Dusty, plausibly because the PVMD thermal captures include
near-consecutive drone frames that ended up byte-identical. This is disclosed here
rather than smoothed over; the true, deduplicated Hotspot count for this run is 187,
not the ~350 figure quoted in the source paper/DATASET_SOURCES.md.

## Prepared dataset

Manifest: `E:\Solar AI Training Images\prepared\manifest.json` (local only, not committed —
see `.gitignore` `training/data/` and note this run used a local `E:\` path outside the
repo, not `training/data/`, for the same reason: real image data must never enter git).

| Split | Clean | Dusty | Hotspot |
|---|---|---|---|
| train | 570 | 508 | 149 |
| val | 71 | 63 | 18 |
| test | 72 | 64 | 20 |

Split: deterministic, stratified per class (no real grouping metadata available), seed 42,
80/10/10 target proportions, per `prepare_dataset.py`.

## Training configuration

- Script: `training/classification/train_mobilenet.py --classes Clean Dusty Hotspot`
- Architecture: MobileNetV2, ImageNet-pretrained backbone (`MobileNet_V2_Weights.DEFAULT`),
  final classifier layer replaced with a 3-way linear head
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Loss: CrossEntropyLoss
- Sampling: WeightedRandomSampler (inverse class frequency) to counter class imbalance
- Augmentation (train only): RandomResizedCrop(224, scale 0.75-1.0), RandomHorizontalFlip,
  ColorJitter(brightness/contrast/saturation=0.15), ImageNet normalization
- Epochs: 10, batch size: 32, seed: 42
- Device: CPU (no CUDA GPU available on this machine)
- Checkpoint selection: best validation accuracy across epochs

## Results

Trained successfully 2026-09-03 (an earlier attempt segfaulted with no Python-level traceback
before any epoch completed; corrupt-image and pretrained-cache checks both came back clean,
and a retry with `PYTHONFAULTHANDLER=1` completed normally — treated as transient, not
reproduced on retry).

Validation accuracy by epoch (best: epoch 2/4/6/7/8, 0.993421, checkpoint selection keeps the
first best):

```
epoch=1  val_accuracy=0.973684
epoch=2  val_accuracy=0.993421
epoch=3  val_accuracy=0.986842
epoch=4  val_accuracy=0.993421
epoch=5  val_accuracy=0.986842
epoch=6  val_accuracy=0.993421
epoch=7  val_accuracy=0.993421
epoch=8  val_accuracy=0.993421
epoch=9  val_accuracy=0.993421
epoch=10 val_accuracy=0.986842
```

Untouched test-set evaluation (`evaluate_mobilenet.py`, 156 test images):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Clean | 1.0 | 1.0 | 1.0 | 72 |
| Dusty | 1.0 | 1.0 | 1.0 | 64 |
| Hotspot | 1.0 | 1.0 | 1.0 | 20 |
| **Accuracy** | | | **1.0** | 156 |
| Macro F1 | | | 1.0 | |
| Weighted F1 | | | 1.0 | |

Confusion matrix (rows/cols = Clean, Dusty, Hotspot): `[[72,0,0],[0,64,0],[0,0,20]]` — zero
misclassifications.

**Caveat on the 100% figure — read before treating this as a generalization result.** This is
a real, unmodified `evaluate_mobilenet.py` run against files that never appear in `train/`
(exact-duplicate leakage is structurally ruled out — `prepare_dataset.py` fails closed on any
duplicate SHA-256 anywhere in the source, and this run completed without that error). It is
**not** independent confirmation the model generalizes to new solar sites:

- Hotspot is thermal imagery, trivially separable from RGB Clean/Dusty on color/texture alone
  — a large part of this task is easier than the real six-class problem will be.
- Per `README.md`'s documented split policy, "class labels are NOT grouping identifiers" and
  per-class splitting is used whenever "no real panel/module/source identity metadata exists"
  — which is the case for SolNET and PVMD here. Multiple images from the same physical
  capture session/panel may exist in this data (SolNET's field collection, PVMD's drone
  frames), and without grouping metadata such near-duplicates (visually similar but not
  byte-identical) could land in both train and test, inflating this score. This is a known,
  disclosed limitation of training on data without real provenance-level grouping, not a bug.

Treat 100% as "the pipeline and this specific held-out split show no errors," not as a
validated real-world accuracy claim. A true generalization estimate would need either
grouping metadata from the original sources or evaluation on a genuinely independent
external dataset.

## App-integration validation (2026-09-03)

Run via `training/classification/validate_interim_checkpoint.py`, which does **not** wire
the checkpoint into the app — it validates the checkpoint against the app's *actual*
inference code, not a reimplementation of it:

1. **Correct-label path**: built the model with the exact same construction pattern as
   `ModelManager._load_classifier` (`torchvision.models.mobilenet_v2` +
   `classifier[1]` replaced with `nn.Linear`), loaded via the same
   `torch.load(..., weights_only=True)` call, preprocessed 45 real held-out test images
   (15 per class) with the real `utils.image_utils.resize_for_mobilenet`, and ran them
   through the real `models.classifier.SolarFaultClassifier.classify()` — with its label
   list temporarily overridden to `['Clean', 'Dusty', 'Hotspot']` instead of the six
   production labels. Result: **45/45 correct (100%)**, confidences 0.80–1.00.
2. **Danger demonstration — why this checkpoint can never be swapped in as-is**: ran the
   identical checkpoint through `SolarFaultClassifier.classify()` completely
   *unmodified*, i.e. with the real six production labels from `configs/settings.yaml`.
   A real Hotspot test image (`H122.jpeg`) was reported as **`Bird-Drop`, confidence
   0.972** — because the checkpoint only emits 3 logits and `zip(_LABELS, probs)`
   silently pairs them with the first 3 production labels (Clean, Dusty, Bird-Drop)
   instead of this checkpoint's actual classes (Clean, Dusty, Hotspot). This is why the
   interim checkpoint is saved outside `weights/mobilenet_solar.pth`, is gitignored, and
   must never be treated as a drop-in production artifact — confirmed concretely, not just
   asserted by convention.

**Conclusion:** the checkpoint is fully compatible with the app's real model-construction
and preprocessing code when used with its own (correct) class list, and is conclusively
proven incompatible with the app's default six-label configuration. It remains a
pipeline-validation artifact, not an integration target.

## Absolute rule compliance

- No fabricated data, metrics, or provenance
- No relabeling; class definitions match the production contract exactly for the three
  classes included
- Real duplicate/quality issues found are disclosed, not hidden
- Checkpoint is clearly marked non-production and stored outside the production weights path
- Electrical-Damage, Physical-Damage, and Bird-Drop remain genuinely blocked — this report
  does not claim otherwise
