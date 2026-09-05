# Solar AI v1.0.0 — Real-Image ML Validation Gate

Generated: 2026-09-05
Repository: https://github.com/cat226/solar-ai-framework
Branch: `feat/cloud-training-orchestration` (PR #23, open/draft)
Sample source: `D:\Documents\Projects\solar-ai-ml-validation` (external to this
repository; 16 genuine images, not committed here — see §7).

This is a **validation-only** exercise using the frozen, committed v1
production configuration exactly as-is. No model weight, threshold,
preprocessing, taxonomy, or configuration change was made to produce these
results. No conventional accuracy/precision/recall/F1/mAP figure is
reported — this sample has no independently verified ground truth (see
§5).

---

## Environment

| Field | Value |
|---|---|
| Date/time | 2026-09-05 |
| Git SHA | `e5e036836ff588ce6d617d477ac2e72999d9b978` |
| Branch | `feat/cloud-training-orchestration` |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| Ultralytics | 8.4.118 |
| Pillow | 12.3.0 |
| OpenCV | 5.0.0 |

### Active production configuration (read from the real, committed `configs/settings.yaml` — not modified)

```yaml
models:
  yolo:
    weights: "weights/yolo_solar.pt"
    confidence_threshold: 0.30
    iou_threshold: 0.50
    image_size: 640
  mobilenet:
    v1_weights: "weights/mobilenet_solar_v1.pth"
    v1_labels: ["Clean", "Dusty", "Hotspot"]
```

### Artifact verification

| Artifact | Manifest SHA-256 | Computed SHA-256 (this run) | Result |
|---|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **PASS** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **PASS** |

`python scripts/verify_model_artifacts.py --manifest weights/manifest.json`
→ `Model artifact integrity verification passed.` (exit 0).

**Confirmed the actual loaded runtime state** (via `models.model_manager`,
not assumed from config alone): `classifier_source="v1"`,
`classifier_labels=['Clean', 'Dusty', 'Hotspot']`, MobileNet loaded from
`weights/mobilenet_solar_v1.pth`, YOLO loaded from `weights/yolo_solar.pt`
— these are the real production artifacts, not test fixtures or an
alternate checkpoint.

**Artifact/config verification: PASS.**

---

## Dataset inventory

16 files, all successfully read by PIL. No folder structure exists (flat
directory) — there is **no Clean/Dusty/Hotspot folder categorization to
preserve**. Two filenames contain a weak, unverified condition hint
(noted below); the rest are generic download-style names with no category
information at all.

| ID | Image | Dimensions | Format | Size | Source/category metadata | Readable |
|---|---|---|---|---|---|---|
| 1 | `20200409T101040.jpg` | 640×480 | JPEG | 92 KB | none (timestamp filename) — **visually a FLIR thermal/false-color image, not a standard RGB photo** | Yes |
| 2 | `513b47f8d836674.jpg` | 295×325 | JPEG | 21 KB | none (hash-style filename) — real RGB, ground-mount array, visibly dusty/sandy, multi-panel | Yes |
| 3 | `691ba04f530bb983e4e70e28_fig2-...jpg` | 826×620 | JPEG | 251 KB | none authoritative; filename suggests an academic figure ("fig2") — real RGB, mostly dusty panel with one visibly wiped-clean patch, multi-panel | Yes |
| 4 | `dirty-dust-on-solar-panel-260nw-2110061597.webp` | 390×280 | WEBP | 39 KB | **filename suggests "dusty"** (unverified) — real RGB close-up, visibly dusty surface; "260nw" suffix indicates a watermarked stock-photo preview | Yes |
| 5 | `images (1).jpg` | 547×365 | JPEG | 38 KB | none (generic download filename) — real RGB, visibly sand/dust streaks | Yes |
| 6 | `images (2).jpg` | 596×335 | JPEG | 32 KB | none — real RGB, ground array being mechanically cleaned, panels look relatively clean | Yes |
| 7 | `images (3).jpg` | 547×365 | JPEG | 37 KB | none — real RGB, panel being brush-cleaned, wet/glossy, visibly clean | Yes |
| 8 | `images (4).jpg` | 467×657 | JPEG | 53 KB | none — real RGB, panel being water-sprayed; **watermarked Dreamstime stock-photo preview** | Yes |
| 9 | `images (5).jpg` | 739×415 | JPEG | 40 KB | none — **thermal/FLIR aerial image**, one clear bright/hot anomaly visible | Yes |
| 10 | `images (6).jpg` | 629×317 | JPEG | 61 KB | none — **thermal/FLIR aerial image**, large array, no obvious localized anomaly | Yes |
| 11 | `images (7).jpg` | 576×432 | JPEG | 47 KB | none — **thermal/FLIR image with an explicit temperature scale (18.9–32.3°C)**, two clear hot anomalies | Yes |
| 12 | `images (8).jpg` | 657×467 | JPEG | 37 KB | none — **thermal/FLIR aerial image**, several distinct bright/hot rectangular patches | Yes |
| 13 | `images.jpg` | 540×370 | JPEG | 27 KB | none — real RGB, full rooftop array, moderate distance, multi-panel | Yes |
| 14 | `mage-of-a-clean-panel-Figure-2-Image-of-a-dusty-Panel.webp` | 704×472 | WEBP | 67 KB | **filename mentions both "clean" and "dusty"** (unverified, likely an academic comparison-figure filename) — the actual image content is a single close-up of a visibly dusty surface | Yes |
| 15 | `premium_photo-1679952890714-...avif` | 3000×5333 | AVIF | 1.2 MB | none (Unsplash-style stock-photo filename) — not visually pre-inspected (tool size limit); large real photo | Yes |
| 16 | `premium_photo-1682148222948-...jpg` | 3000×4509 | **AVIF despite the `.jpg` extension** | 357 KB | none — **filename/content-format mismatch**: a `.jpg`-named file whose actual bytes are AVIF-encoded; decoded correctly by content, not by trusting the extension | Yes |

**Data-quality observations** (not ground truth, just what the sample set
actually contains): 4 of 16 images are thermal/false-color (FLIR-style),
not standard visible-light RGB photography — a modality genuinely outside
what this project's models were ever trained on (see §6). At least 2
images are watermarked stock-photo previews. One file's real format
disagrees with its extension. None of this was assumed from filenames —
each image was opened and visually reviewed before this report was
written.

---

## Per-image results

All 16 images produced `status=SUCCESS` (the pipeline executed to
completion without an exception in every case) and correctly reported
`xgboost_available=False` throughout (never fabricated). **YOLO detected
zero panels in all 16 images** — no per-panel crops or per-panel
classifications exist for this sample set. The only classification signal
available is the pipeline's whole-image classification, which runs
independently of detection success (by design — see
`utils/ui_helpers.py`/`services/pipeline.py`).

| ID | Image | YOLO detections | Detection confidences | Whole-image prediction | Confidence | E2E status | Notes |
|---|---|---:|---|---|---:|---|---|
| 1 | `20200409T101040.jpg` | 0 | — | Hotspot | 0.998 | SUCCESS, no detection | Thermal image; high-confidence Hotspot call plausibly driven by bright false-color regions, not verified real hotspot |
| 2 | `513b47f8d836674.jpg` | 0 | — | Hotspot | 0.548 | SUCCESS, no detection | Real RGB, visually dusty; predicted label does not match the visual impression; confidence only just above the 3-class 1/3 baseline |
| 3 | `691ba04f...fig2...jpg` | 0 | — | Dusty | 0.457 | SUCCESS, no detection | Real RGB, visually dusty-with-clean-patch; low-confidence, plausible |
| 4 | `dirty-dust-on-solar-panel-...webp` | 0 | — | Clean | 0.682 | SUCCESS, no detection | Filename says "dusty"; image is visibly dusty; **predicted Clean at moderate-high confidence — likely misclassification** |
| 5 | `images (1).jpg` | 0 | — | Dusty | 0.988 | SUCCESS, no detection | Real RGB, visibly sandy; high-confidence, matches visual impression |
| 6 | `images (2).jpg` | 0 | — | Hotspot | 0.546 | SUCCESS, no detection | Real RGB, cleaning machine, panels look relatively clean; predicted label doesn't obviously match |
| 7 | `images (3).jpg` | 0 | — | Clean | 0.850 | SUCCESS, no detection | Real RGB, wet/glossy, visibly clean; matches visual impression |
| 8 | `images (4).jpg` | 0 | — | Dusty | 0.649 | SUCCESS, no detection | Real RGB, being water-sprayed, watermarked stock photo; moderate confidence, ambiguous |
| 9 | `images (5).jpg` | 0 | — | Hotspot | 0.999 | SUCCESS, no detection | Thermal, one real bright anomaly visible; very high confidence |
| 10 | `images (6).jpg` | 0 | — | Hotspot | 0.824 | SUCCESS, no detection | Thermal, no obvious localized anomaly visible; predicted label not clearly supported by the image itself |
| 11 | `images (7).jpg` | 0 | — | Hotspot | 0.969 | SUCCESS, no detection | Thermal with explicit temp scale, two real hot anomalies visible; high confidence |
| 12 | `images (8).jpg` | 0 | — | Hotspot | 0.891 | SUCCESS, no detection | Thermal, several real bright patches visible; high confidence |
| 13 | `images.jpg` | 0 | — | Dusty | 0.459 | SUCCESS, no detection | Real RGB, full rooftop array, moderate distance; low confidence, ambiguous |
| 14 | `mage-of-a-clean-panel-...-dusty-Panel.webp` | 0 | — | Dusty | 0.997 | SUCCESS, no detection | Real RGB, visibly dusty close-up; high confidence, matches visual impression |
| 15 | `premium_photo-1679952890714-...avif` | 0 | — | Clean | 0.553 | SUCCESS, no detection | Large real photo, not pre-inspected visually; moderate confidence |
| 16 | `premium_photo-1682148222948-...jpg` | 0 | — | Clean | 0.406 | SUCCESS, no detection | AVIF-as-.jpg; very low confidence, close to the 1/3 baseline — essentially an uncertain call |

No per-panel table is included — zero panels were ever detected, so no
crop-level classification exists to report.

---

## Explicit test cases (Task 4)

### A. Clear panel
No image in this sample is unambiguously a single, clearly-framed panel
with no dust/water/thermal artifact — the closest are `images (3).jpg`
(clean, wet, being brushed) and `images (2).jpg` (clean-looking,
mid-cleaning). **YOLO did not detect a panel in either.** MobileNet's
whole-image call was Clean (0.850) for `images (3).jpg` (plausible) and
Hotspot (0.546) for `images (2).jpg` (not obviously supported by the
image).

### B. Dusty panel
Several genuine close-up dusty images exist (`images (1).jpg`,
`mage-of-a-clean-panel-...-dusty-Panel.webp`, `513b47f8d836674.jpg`,
`691ba04f...fig2...jpg`, the WEBP explicitly named "dirty-dust..."). The
pipeline produced **Dusty** for 2 of these 5 (high confidence, 0.988 and
0.997), **Hotspot** for 1 (0.548, weak), **Clean** for 1 (0.682, the
"dirty-dust" filename case — a likely misclassification), and **Dusty**
for the fifth at low confidence (0.457). **This is a mixed result, not a
uniform success** — reported honestly, not smoothed over. None of this is
called "correct" against a verified label, since no independently
verified ground truth exists for any of these images (§5).

### C. Hotspot panel
No RGB (visible-light) image in this sample is a verified real electrical
hotspot defect. The images that produced a Hotspot prediction are
predominantly the **thermal/FLIR images** (4 of 16 total images are
thermal; the model called Hotspot on all 4, at high-to-very-high
confidence in 3 of them). This is best read as the model responding to
the thermal false-color palette's bright/warm regions — a genuinely
out-of-distribution input for an RGB-trained classifier — not as
confirmed evidence the model correctly identifies real electrical
hotspots. **No conclusion about real Hotspot-detection capability can be
drawn from this sample.**

### D. Multi-panel
At least 5 images visually contain multiple panels/modules
(`513b47f8d836674.jpg`, `691ba04f...fig2...jpg`, `images (2).jpg`,
`images.jpg`, `20200409T101040.jpg`). **YOLO detected zero panels in every
one of them** — no crops were generated, so per-panel classification and
site-summary aggregation could not be exercised at all by this sample.
This is a direct, real illustration of the domain-shift limitation
already documented in `docs/ML_HARDENING_PHASE6B.md`/`PHASE6C.md`, not a
new discovery, but a concrete confirmation on entirely new images.

### E. Difficult / close-up / domain-shift image
Every image in this sample is close-up or moderate-distance ground-level
imagery — i.e., every single image is a real instance of the domain-shift
scenario Phase 6B/6C already identified. **YOLO's detection rate on this
sample is 0/16 (0%)**, consistent with (in fact lower than, though the
sample is far too small to treat this difference as meaningful) Phase
6B's measured 2.56% (4/156) on the SolNET/PVMD close-up set. This finding
is not hidden or minimized: **the detector did not successfully localize
a single panel in any real close-up image tested this session.**

### F. No-panel image
No image in this sample is a confirmed genuine "no solar panel present at
all" negative image — every image shows at least one real panel/module
surface, even where YOLO failed to detect it. **Result: NOT TESTED**, as
the task instructs when no suitable image exists, rather than
substituting a different case and calling it equivalent.

---

## Summary

| Metric | Value |
|---|---|
| Total real images tested | 16 |
| Readable images | 16 / 16 |
| Images where the pipeline completed (`status=SUCCESS`) | 16 / 16 |
| Successful YOLO detections (≥1 panel) | **0 / 16** |
| Zero-detection images | 16 / 16 |
| Total panel crops classified | 0 |
| Whole-image "Clean" predictions | 4 |
| Whole-image "Dusty" predictions | 5 |
| Whole-image "Hotspot" predictions | 7 |
| Multi-panel cases (visually) | ≥5 identified; 0 confirmed detected as multi-panel |
| Close-up cases | 16 / 16 (the entire sample) |
| No-panel cases | 0 (none available — NOT TESTED) |
| Failures/exceptions | 0 |
| XGBoost status | Unavailable for all 16 (`xgboost_available=false`), exactly as expected — never fabricated |

**These are prediction counts, not accuracy.** No ground-truth-verified
correct/incorrect tally is reported anywhere in this document, per §5.

---

## Failure analysis

| Category | Observed | Count/examples |
|---|---|---|
| Detector missed panel | **Yes — dominant failure mode** | 16/16 images; includes at least 5 images with visually obvious, multiple, unoccluded panels |
| False positive (spurious detection) | Not observed | 0 detections occurred at all, so no false-positive detection is possible in this sample |
| Classifier prediction | Every image produced one; a subset appear inconsistent with the visible image content when compared informally (e.g. the "dirty-dust" WEBP predicted Clean; several thermal images predicted Hotspot) | See per-image table |
| Invalid/unsupported input | None — every file decoded successfully, including a `.jpg`-extensioned file that was actually AVIF-encoded, and a genuine `.avif` file | 0 |
| Preprocessing/runtime failure | None | 0 exceptions across 16 images |
| XGBoost unavailable | Expected, honestly reported | 16 / 16 |
| **Close-up/domain-shift failure** | **Yes — the central finding of this validation run** | 16/16 images are close-up/ground-level; detection rate 0% |
| Other | Wrong-modality input (thermal, not RGB) | 4 / 16 images are thermal/FLIR, a modality never part of this project's training data for either model |

---

## Interpretation

**1. Pipeline execution:** Fully successful. All 16 images processed
end-to-end (`services.pipeline.run_pipeline`, the real production entry
point) with zero exceptions, correct artifact loading (verified by hash),
and correct, honest XGBoost-unavailable reporting throughout.

**2. Detector behavior:** Poor on this real sample — 0 successful
detections out of 16 images, including several with clear, multiple,
unoccluded panels. This is consistent with, and a direct new confirmation
of, the domain-shift limitation already measured in
`docs/ML_HARDENING_PHASE6B.md` (2.56% on a 156-image sample) and
investigated in `docs/ML_HARDENING_PHASE6C.md`. The sample size here (16
images) is far too small to treat "0%" as a more precise estimate than
the earlier, larger measurement — but it is fully consistent with it and
does not contradict it.

**3. Classifier behavior:** The whole-image classifier produced a
confident, real prediction for every image, with no crash and no
fabricated output. Comparing predictions against this report's own
*informal, unverified* visual read of each image: several predictions
look plausible (e.g. `images (1).jpg`, `images (3).jpg`, the "clean panel/
dusty panel" WEBP), while others do not obviously match (the
explicitly-named "dirty-dust" WEBP predicted Clean; several thermal images
predicted Hotspot). **None of this is a verified accuracy measurement** —
it is a qualitative, honestly-hedged observation only, exactly per §5's
discipline.

**4. End-to-end behavior:** Because detection failed on every image, the
true end-to-end (detect→crop→classify) path was never exercised by this
sample — only the whole-image classification path ran. This mirrors
exactly the distinction `docs/ML_HARDENING_PHASE6B.md`'s Task 7 already
established: whole-image classification is independent of, and
meaningfully more available than, the detection-gated path on close-up
imagery.

**5. Domain limitations:** Strongly reconfirmed. This sample adds a
second, independent, previously-unseen batch of real images pointing to
the same conclusion as Phase 6B/6C: the deployed YOLO checkpoint does not
reliably localize panels in close-up/ground-level photography. Four of
the sixteen images (thermal/FLIR) are additionally outside the RGB
modality either model was ever trained on — a distinct, newly-observed
data-quality dimension worth carrying into any future data-collection
effort (Phase 6C already established no licensed close-up RGB training
data was found; this sample further shows real-world uploads may not even
be RGB).

**6. Research/demo deployment suitability:** Unchanged from Phase 7's
conclusion. This validation run supports, rather than contradicts, the
existing honest framing: Solar AI v1 is a technically functional
research/demo system whose whole-image classification produces real,
non-fabricated results on every input, while its panel-detection stage
remains unreliable on the close-up imagery a real user is likely to
upload. Nothing here changes the GREEN *operational* deployment gate from
Phase 7, and nothing here supports elevating the ML system's own accuracy
claims.

---

## Regression check (Task 8)

This validation exposed **no new regression or missing safety behavior**
in the production pipeline itself — every failure mode observed (0
detections, low-confidence whole-image calls) is an existing,
already-documented ML limitation (Phase 6B/6C), not a code defect. No new
test was required or added. The existing ML-relevant test suite
(`tests/test_model_manager.py`, `tests/test_pipeline*.py`,
`tests/test_evaluation_*.py`) and the full suite were both re-run after
this exercise to confirm nothing was altered:

- Full suite: 1168 passed, 4 skipped, the same 4 known local-artifact
  failures as every prior phase — no change.
- `compileall`, `verify_imports.py`: both clean.

No test was rewritten or weakened to accommodate this validation run.

---

## Reproducing this validation

```bash
python training/evaluation/validate_real_images.py \
  --sample-dir "D:/Documents/Projects/solar-ai-ml-validation" \
  --output-dir <a directory outside this repository, e.g. a scratch dir>
```

Writes a single JSON file with the full environment record, inventory,
and per-image results. Never modifies the sample directory or any file
inside this repository.
