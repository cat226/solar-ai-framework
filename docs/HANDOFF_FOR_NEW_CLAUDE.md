# Solar AI Framework — project handoff and status report

Written 2026-10-05 for a Claude instance (or engineer) taking over this work.
Everything here was verified against the repo, the weight files, the datasets on disk
and the measured results as of that date. Where something is an estimate, an AI
judgement, or untested, it says so. **Read sections 1–3 before touching anything.**

---

## 1. What this project is (one paragraph)

Solar AI inspects solar panels from **ground-level / pole-mounted camera photos** (not
aerial imagery). Given a photo it (1) finds every individual panel, (2) crops each panel,
(3) classifies each crop as `CLEAN`, `DUSTY`, `BIRD_DROP` or `PHYSICAL_DAMAGE`, and (4) records
the result in a local SQLite history, with a best-effort "same physical panel across
visits" tracker. It ships as a Streamlit dashboard (`app.py`, landing page "Ground
Inspection"), a FastAPI ingestion endpoint, and a Raspberry-Pi edge agent that runs the
same model code on the device. The actual deployment target is **the owner's one
rooftop array** — that matters for how results must be judged (section 6).

The legacy **aerial** pipeline (wide-area survey from above, `weights/yolo_solar.pt` +
MobileNet) was fully removed from the app (commit `b1ea748`). Its weights remain on disk.
It is out of scope now; the owner explicitly asked that aerial datasets not be collected.

---

## 2. Rules you must follow (these override your defaults)

### 2.1 Git rules (`AGENTS.md`, verbatim)
1. Never work on main. 2. Work only on the active feature branch. 3. Pull latest main before
starting a new task. 4. Commit only the current task. 5. Never amend old commits unless
instructed. 6. Never force push. 7. Never merge branches. 8. Never resolve merge conflicts
automatically. 9. **Wait for architecture approval before pushing.** 10. Keep commit messages
task-based.

State today: branch `feat/ground-pv-research-detector`, **108 commits ahead of `main`,
never pushed** (remote `origin` = `https://github.com/cat226/solar-ai-framework`). Nothing
has been merged or pushed. Whether/when to push is the owner's decision (rule 9).

### 2.2 The no-fabrication policy (the project's core value)
Never fabricate or fake: model weights, training metrics, evaluation results, hashes,
dataset provenance/licences, or access approvals. Never create placeholder weights to make
a readiness check pass. Never weaken a readiness check (`ModelLoadError`, manifest
verification) to hide a missing artifact. Never use unknown-licence data or bypass access
controls. When blocked, say so plainly and stop at the gate. Distinguish *verified /
inferred / proposed / blocked* in every status report. The existing code enforces this
(`weights_only=True` loading, SHA-256 manifest checks, fail-closed loaders) — keep it.

### 2.3 Honest-evaluation lessons learned the hard way
- **Always measure the currently-active model on the deployment domain (the owner's
  rooftop) before rejecting a challenger on an out-of-domain benchmark.** In Phase 19-B a
  better model was rejected on the public Susan benchmark while the model left active
  scored 27% on the owner's own photos; the owner noticed in real use. Reversed on
  2026-09-29 (section 7).
- A high score on a class-skewed test set can be a collapsed always-predict-majority model
  (the x10-oversample classifier scored 95.8% by predicting CLEAN for everything).
- Stop after one or two well-motivated ablations; this project has several documented
  "stop this direction" results (resolution, augmentation, tiling, v2 detector).
- Ground truth labels in the rooftop data are **AI-assisted, not owner-verified** (4.2).

### 2.4 Data/licence constraints
Susan dataset: CC BY 4.0 *as displayed on Roboflow*, image provenance not independently
verified. Condition and candidate sets: provenance grade C
(`personal-experimental-unverified-provenance`) — **do not redistribute, publish, use
commercially, or claim verified CC BY rights.** The rooftop/first-party photos are the
owner's private images — never upload them anywhere public. `quarantine`d datasets
(unknown licence, `RIGHTS_BLOCKED`) must never be trained on.

---

## 3. Fast orientation

### 3.1 Repo map (what lives where)
| Path | Purpose |
|---|---|
| `app.py` | Streamlit landing page "Ground Inspection" (upload photos → run pipeline → results) |
| `pages/` | Dashboard: 01 Overview, 02 Panel Results, 03 Site Health, 05 Model Status, 06 Limitations, 07 History, 08 Analytics, 09 Alerts, 10 Settings, 11 Panel History, 12 Cameras, 13 Camera Configuration (no page 04) |
| `services/ground_pipeline.py` | **The single shared inference pipeline** (detect → crop → classify). Holds the active weight paths. Imported by `app.py`, the Pi agent and the API. No Streamlit dependency. |
| `services/storage.py` | SQLite persistence (`data/inspections.db`): inspections, panel identities/observations, cameras |
| `api/main.py` | FastAPI: `GET /v1/health`, `POST /v1/ingest` (cameras authenticate with per-camera tokens) |
| `pi_agent/` | Raspberry Pi edge agent (`capture_and_send.py`, `config.example.yaml`) — runs the same pipeline locally, POSTs only JSON results, never raw images |
| `training/detection/ground_pv/` | Ground-PV research library: dynamic multi-panel detection, spatial ordering, panel→condition pipeline, panel identity, dataset builders/auditors, taxonomy, paths |
| `scripts/` | ~50 CLIs: dataset builders, Kaggle notebook generators (`make_*_notebook.py`), evaluators, labeling tools |
| `kaggle/` | Notebook sources + dataset-staging dirs for Kaggle GPU training (bulk staging is git-ignored) |
| `datasets/ground_pv_research/` | Dataset trees; images git-ignored, manifests/reports tracked |
| `weights/` | Checkpoints — **git-ignored** (only `manifest.json`, `.gitkeep`, two `train_summary.json` tracked) |
| `docs/` | One report per phase; start with the ones listed in 3.3 |
| `tests/` | ~75 test files; suite = **1232 passed, 5 skipped** (symlink-privilege/absent-dataset skips) |
| `utils/` | config/secrets, auth gate, logging, security, UI theme |

### 3.2 How to run things (Windows 11; shell is PowerShell/Git-Bash)
```
# interpreter: project venv, Python 3.12.10 (pytest.ini pins 3.12)
.venv\Scripts\python.exe -m pytest -q          # full suite, ~2–5 min, expect 1232 passed / 5 skipped
.venv\Scripts\streamlit.exe run app.py        # app on http://localhost:8501
```
- Auth: single shared password via `APP_ACCESS_PASSWORD` env var or `.streamlit/secrets.toml`;
  **unset = local dev mode (no gate)**. Copy `.streamlit/secrets.toml.example` /
  `.env.example`; the real files are git-ignored.
- **Port 8501 "not available"** almost always means a leftover Streamlit/python process from an
  earlier run. `netstat -ano | findstr :8501` → `taskkill /PID <pid> /F /T`, or use `--server.port 8502`.
- This machine has **no NVIDIA GPU** (local torch is CPU-only). Anything that needs training
  runs on **Kaggle GPU** (section 9). Inference on CPU is fine for the app.
- Smoke-checking the pipeline without the UI:
  `services.ground_pipeline.load_ground_models()`, `load_condition_classifier()`, `run_capture(...)`.

### 3.3 Docs to read, in order
1. `docs/PHASE_16_ARCHITECTURE_REVIEW.md` — why the app is ground-first and panel-first.
2. `docs/ROOFTOP_PANEL_DETECTOR_PHASE19.md` — current panel detector, labeling method, honest ceiling (§6.1).
3. `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md` — current classifier; **§0 documents a reversed decision**; §4a is the superseded original reasoning kept as audit trail.
4. `docs/DYNAMIC_PANEL_DETECTION.md`, `docs/PANEL_CONDITION_PIPELINE.md`, `docs/PANEL_IDENTITY_TRACKING.md` (Phase 17), `docs/RASPBERRY_PI_DEPLOYMENT.md` (Phase 18).
5. `docs/FIRST_PARTY_COLLECTION_GUIDE.md` — how new owner photos must be imported/labeled.
6. `docs/CONDITION_MODEL_*` — the 4-class condition research history.

---

## 4. Models: what is active, what is not

### 4.1 Active models (what the app actually loads — `services/ground_pipeline.py`)
| Role | File in `weights/` | SHA-256 | Notes |
|---|---|---|---|
| Panel detector (YOLO11n, 1 effective class `panel`) | `yolo_solar_rooftop_v1.pt` | `63fbdaad0cb8bb9d98ed09990c89e1e080ca0943437fa95c2afd64c1742e45fb` | Phase 19, trained on Kaggle |
| **PRIMARY** condition classifier (MobileNetV2, 4 classes, alphabetical order `BIRD_DROP, CLEAN, DUSTY, PHYSICAL_DAMAGE`) | `mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth` | `96528bfe5146afadfc83026475e3d0350ba348bd3f5111afa03c9db886662508` | Phase 19-B, promoted 2026-09-29 |
| SECONDARY condition box-detector (YOLO11n, localized cross-check) | `yolo_solar_condition_4class_research_v2.pt` | `2dcf6122c18590f33f3b44ad88e054d409509274193977046ad2076d368de28e` | Phase 15, Susan-derived, **not** domain-adapted |

Class order is a hard contract: a wrong order silently corrupts labels. The classifier
loader asserts the 4-class taxonomy; do not reorder.

### 4.2 Measured performance (all numbers are real; read the caveats)
**Panel detector — held-out, scene-disjoint rooftop test (10 images, 48 labeled panels):**
| Measure | Predecessor (`yolo_solar_ground_research_v1.pt`) | Current |
|---|---|---|
| Panels found @ conf 0.25 (app default) | 6/48 (recall 0.125), 4 of 10 images zero detections | **39/48 (0.812)**, precision 0.696 |
| Kaggle `val()` mAP50 / recall | 0.274 / – | 0.935 / 0.935 |
| Susan test (panel only) | – | mAP50 0.982 (unchanged, no regression) |

Confidence sweep (honest ceiling; no threshold reaches 100%): 0.05 → 46/48 (precision 0.554,
37 FP); 0.10 → 45/48; 0.25 → 39/48; 0.40 → 37/48 (precision 0.787). Remaining misses are all
the same scene type: dense, steep-angle, 10+ panel wide shots where panels overlap in 2D.
The two local measurements (0.935 vs 0.812) differ because of different matching code;
both agree on the conclusion. Report: `datasets/ground_pv_research/rooftop_panel_v1/reports/`.
Visually confirmed on the app: the two "6 panels in frame" photos detect exactly 6 each; a
2×7 wide shot separates into 14 boxes with no merged group boxes.

**Condition classifier — two benchmarks, never conflate them:**
| Checkpoint | Susan-derived test (n=242) | Owner-rooftop held-out crops (n=48) |
|---|---|---|
| Phase 16-B baseline (`mobilenet_condition_4class_research_v1.pth`, was active) | 0.752 | **0.271 (13/48)** — misread CLEAN as DUSTY/PHYSICAL_DAMAGE, sometimes at 0.85–0.93 confidence |
| **Active: rooftop MIX, oversample x1** | 0.731 | **0.938 (45/48)** |
| MIX oversample x10 (rejected) | 0.715 | 0.958 but predicts CLEAN for all 48 — collapsed |

**CRITICAL CAVEAT on the 0.938:** the rooftop held-out set is 46 CLEAN + 2 DUSTY. The number
mostly proves "it says CLEAN on clean panels." On this rooftop there are **zero held-out
examples of BIRD_DROP or PHYSICAL_DAMAGE and only 2 of DUSTY**, so the active classifier's
ability to *detect problems* on this rooftop is essentially **unmeasured**. The only evidence
beyond that is one live check (a photo with a hand-confirmed bird-drop mark now reads
BIRD_DROP at 0.74/0.87 on two panels). It also costs 2.1 points on the Susan benchmark — a
deliberate, disclosed trade.

Box-detector (secondary): mAP50 0.444 / box recall 0.441, image-level agreement 0.727
(Phase 15/16-A.1). Weakness is mostly localization noise.

### 4.3 Models on disk that are NOT active (do not delete; they are evidence)
`yolo_solar_ground_research_v1.pt` (sha `2b26334d…`, frozen predecessor, pinned by
`tests/test_ground_pv_frozen_artifacts_integrity.py` — **never overwrite it**),
`yolo_solar_rooftop_only_v1.pt` (rejected: overfit, Susan mAP50 0.587),
`mobilenet_condition_4class_research_v1.pth` (sha `1614b0a6…`, former classifier),
`..._rooftop_mix_oversample10_v1.pth`, the 4-class box-detector variants
(`v1, v2, v3, v3_960, v4_aug`), the older 5-class research checkpoint, and the aerial /
experimental-domain weights (`yolo_solar.pt` sha `0b586099…` is a **frozen production
artifact** listed in `weights/manifest.json`; also `mobilenet_solar_v1.pth`,
`mobilenet_interim_3class.pth`, `yolo_solar_domain_*`). `weights/manifest.json` is the strict
v1.0.0 production manifest — **do not add research checkpoints to it.**

---

## 5. Data

### 5.1 Where everything is
- **Consolidated copy (ground-level + first-party only): `E:\Solar AI Consolidated Datasets\`** —
  2.17 GB / ~51,900 files, copy verified (counts, bytes, SHA-256 spot-checks). Contains
  `README.md`, `MANIFEST.json`, `SHA256SUMS.txt`. **Read its README first.**
  The `data.yaml` files inside the copies still hold the *original* `D:\…` absolute paths on
  purpose; use the sibling `*.consolidated.yaml` files, or training will silently read the originals.
- Originals: `datasets/ground_pv_research/` in the repo (images git-ignored), the owner's 48
  rooftop photos in `D:\Documents\Solar Test Images\`, the first 6 holdout photos in
  `E:\Solar Real Images\`.
- Project bulk root per the storage policy: `E:\Solar AI Training Images\` (aerial data lives
  here; excluded from the consolidated folder at the owner's request). E: was disconnected for
  part of this work and has a history of dropouts — verify with `SHA256SUMS.txt` after any
  reconnect.
- Excluded on purpose: `E:\Ecotrace Dataset` (different project), `E:\SolarAI_Datasets\quarantine`
  (unknown licence), `...\domain_adaptation_v1` (experimental aerial), `phase12_audit`.
- Private Kaggle datasets under the owner's account `edithstark/`: `solar-ai-ground-pv-research-v1`,
  `…-condition-4class-research-v1`, `…-condition-4class-classifier-v1`,
  `…-beast-candidate-v1`, `solar-ai-rooftop-panel-v1`, `solar-ai-rooftop-condition-topup-v1`.

### 5.2 Datasets and what they train
| Dataset | Use |
|---|---|
| `yolo` (Susan, 3-class normalized, leakage-remediated) | panel-detector lineage; Phase 19 uses its `panel` boxes only |
| `condition_4class_v1` | secondary box-detector |
| `condition_4class_classifier_v1` (2,411 crops, one dominant label per image; 910 images lost a minority annotation — disclosed) | classifier base |
| `rooftop_panel_v1` (46 images, 1 class, scene-disjoint 29/7/10) | current panel detector |
| `rooftop_condition_v1` (242 crops + `labeling_sheet.csv`) | current classifier (blended, train-only, with the base set) |
| `representative_panel_v1` | first-party manifest/raw/annotations (Phase 14B/B.1 tooling) |
| `yolo_beast_v1`, `candidate_*` | Phase 13B candidate (one installation; research only) |

### 5.3 How the owner-rooftop labels were made (read before trusting them)
- **Panel boxes**: MobileSAM automatic masks → color/shape filtering → an AI reviewer
  (Claude) looked at all 48 overlays and chose/added boxes → hand-added boxes snapped to SAM
  masks (47/47 within tolerance). Status = **`annotated_ai_assisted_unverified`** (a status added
  in Phase 19; it never satisfies the `annotated_human_verified` gates). Two images
  (`IMG_5886`, `IMG_5887`) with dense far-field rows were excluded as ambiguous. Far-field
  slivers and glare-blurred distant panels are deliberately **unlabeled**, so recall on tiny
  distant panels is unmeasured.
- **Condition labels**: AI visual review of all 242 crops → 235 CLEAN, 3 BIRD_DROP, 2 DUSTY,
  1 PHYSICAL_DAMAGE, 1 unusable (SKIP). CLEAN means "no visible indication", not a certified
  absence of dust (dust is often invisible at this resolution). The 2 DUSTY were judged real
  because the haze stayed in the same panel-relative position across consecutive video frames
  of the same panel (a sky reflection would shift). None of this is owner-verified.
- Tooling: `scripts/propose_panels_sam.py`, `render_panel_proposal_review.py`,
  `build_panel_labels_from_review.py`, `build_rooftop_panel_dataset.py`,
  `export_rooftop_condition_labeling_sheet.py`, `build_rooftop_condition_mix_package.py`.
  MobileSAM weights (`mobile_sam.pt`, Apache-2.0, ~41 MB) are **not** in the repo; re-download from the
  Ultralytics assets release if you need to re-run labeling. SAM is an annotation aid only and
  must never ship in a model.

---

## 6. Status by area

| Area | Status | Evidence / caveat |
|---|---|---|
| Panel detection | **Working, good not perfect** | 39–46/48 held-out panels depending on threshold; dense wide shots still lose a few |
| Per-panel condition | **Working on clean panels; defect detection thinly evidenced** | see 4.2 caveat; DUSTY has 2 examples total |
| App (`app.py` + dashboard) | Runs; verified end-to-end in a real browser (Playwright) on two real photos (12/12 panels, per-panel reads) | some UI text is stale — see 8.2 |
| Persistent panel identity (Phase 17) | Implemented and unit-tested (17 tests) | geometry matching for a **fixed** camera only; not validated on real multi-visit data |
| Pi ingestion (Phase 18) | API + agent implemented and tested in software | **no real Raspberry Pi/camera was ever available — hardware behavior unbenchmarked** |
| Tests | 1232 passed, 5 skipped | last full run 2026-09-29 |
| Generalization to another rooftop | **Unverified** | everything trained/tested on one installation (held-out scenes are a different session on the same site) |
| Branch | Local only, 108 commits ahead of main | not pushed (rule 9) |

---

## 7. How we got here (decision log, condensed)
- Phases 1–12: aerial pipeline, then discovery that aerial models are blind on close-up ground
  photos; built a separate ground-level research line. Public ground-level panel data of the
  right kind does not exist (Phase 12 audit); Susan (Roboflow) is the base.
- 13B "beast" detector training: Kaggle GPU hung >17 h twice; kept v1. 11 (detector v2):
  severe real-world regression, rejected. 10: tiled inference, rejected.
- 14A/B: dynamic N-panel detection (no fixed 4-panel assumption), first-party data
  collection tooling with strict readiness gates (honest "NOT READY" at 6 images).
- 15: four-class taxonomy (ELECTRICAL_DAMAGE removed, not relabeled); condition box-detector
  baseline and ablations (duration: modest gain; resolution and augmentation: not supported → stop).
- 16: made Ground Inspection the landing page; added MobileNet classifier as primary condition
  signal (box detector kept as secondary); 17: panel identity; 18: Pi ingestion + admin pages.
- **19 (2026-09-18/19)**: owner supplied 48 rooftop photos. Baseline first (old detector: 6/48).
  Retrained the panel detector (MIX of Susan + rooftop ×10 beat rooftop-only, which overfit);
  promoted. Honest-ceiling analysis; owner asked for "100% perfect" — declined as not a
  legitimate target and explained why.
- **19-B**: condition retrain. First decision "NOT ADOPTED" (2026-09-19) based on a 2–4 pt Susan
  regression. **Reversed 2026-09-29** after the owner reported poor real-world behavior and
  `data/inspections.db` showed most panels flagged DUSTY; the old classifier scored 27% on the
  owner's held-out photos. Promoted the x1 retrain, verified live. Lesson recorded in 2.3.
- 2026-10-03: consolidated ground-level datasets to `E:\Solar AI Consolidated Datasets\`.

---

## 8. Open work (priority order) and known defects

### 8.1 Open work
1. **Owner verification of labels.** `rooftop_condition_v1/labeling_sheet.csv` still holds the AI
   placeholder labels; panel boxes are AI-assisted. Anything the owner can confirm or correct from
   physical knowledge of the panels is the single highest-value input. Do not "verify" them yourself.
2. **Real defect evidence.** The condition classifier needs genuinely held-out non-CLEAN examples
   on this rooftop (BIRD_DROP, PHYSICAL_DAMAGE, DUSTY) before any claim about defect detection
   can be made. More retraining on the same ~6 non-CLEAN crops will not fix this (proven).
3. **More photos of dense/wide scenes** are the only honest lever for the remaining panel
   misses; the owner currently has none to give.
4. **Hardware validation** of the Pi agent/API on a real device, and multi-visit validation of
   panel identity — never done.
5. **Different-site generalization** — if another installation is ever targeted, expect to need
   its own labeled photos.
6. **Branch decision** — push / PR / how to split 108 commits is the owner's call (rule 9).
7. Secondary box-detector is still Susan-only (not domain-adapted); the app flags disagreement
   between the two condition engines rather than hiding it.

### 8.2 Known stale / untidy items (verified by grep 2026-10-05; not yet fixed)
- `pages/06_⚠️_Limitations.py` still says the condition classifier is "measured 0.752" and that
  the panel detector "can miss a single large foreground panel" with a "real working range of
  ~1–3 panels" — all pre-Phase-19. `app.py` lines ~54 and ~176 repeat the "~1–3 panels"
  claim, and `pages/02_🔎_Panel_Results.py:81` cites 0.752. Update them to the Phase 19 numbers
  (they were not touched when the other pages were).
- Stray empty dirs: `weights;C/`, `ground-yolo11n-baseline-v1/`. Legacy root files
  (`TASK-*.md`, `TASK-001.patch`, `PATCH_SUMMARY.md`) and the orphaned `pr-body.md`.
- Inconsistent convention: `kaggle/rooftop_panel_kernel/` is committed while other `*_kernel/`
  dirs are git-ignored; `.gitignore` has duplicate entries.
- CRLF/LF warnings on commit are cosmetic.

---

## 9. Training workflow and gotchas

Training = **Kaggle GPU** (owner's account `edithstark`; **credentials are the owner's and are
not transferable** — you cannot run these without them). Pattern used throughout:
`scripts/make_*_notebook.py` generates a notebook + `kernel-metadata.json` →
`kaggle datasets create|version -p <dir> --dir-mode zip` → `kaggle kernels push -p <dir>` →
poll `kaggle kernels status` → fetch outputs → copy to `weights/` (git-ignored) → write the
report/manifest under `datasets/ground_pv_research/<set>/reports/` → document in `docs/`.
- Pinned on Kaggle: `torch==2.7.1`, `torchvision==0.22.1`, `ultralytics==8.4.138`.
- Windows `kaggle` CLI bugs hit here: `kaggle kernels output` crashes writing a log containing
  non-cp1252 characters (use `KaggleApi().kernels_logs('owner/slug')` and write UTF-8 yourself);
  paths with backslashes passed from Git-Bash get mangled (use relative forward-slash paths or
  PowerShell with absolute paths); downloads can fail mid-way (`IncompleteRead`) — retry with `--force`.
- `ImageFolder` raises on an empty class directory; the rooftop top-up loader
  (`FixedMappingImageDataset` in `scripts/make_rooftop_condition_mix_notebook.py`) exists
  because some classes have zero examples in some splits.
- Never evaluate a challenger only on the public benchmark (2.3). Always compare against the
  active model on the same held-out rooftop crops (`datasets/ground_pv_research/rooftop_condition_v1/reports/compare_classifiers_on_rooftop_test.py`).
- Background processes started for testing must be stopped; a leftover Streamlit server causes
  the "Port 8501 is not available" error.
- Do not run `find /` style whole-drive searches in Git-Bash on this machine; they hang.

---

## 10. Handoff logistics — what the next person actually needs

A plain `git clone` will **not** give you a working system:
1. **Code**: the branch is unpushed. The owner must either push it (their approval, rule 9) or
   hand over the folder.
2. **Weights** (git-ignored): copy the three active files in 4.1 (plus the frozen predecessor if
   running the integrity tests) into `weights/`, then check each against the SHA-256 table.
   Without them the app fails closed with `ModelLoadError` — that is by design.
3. **Datasets** (git-ignored): `E:\Solar AI Consolidated Datasets\` (ground-level + first-party).
   Remember the `*.consolidated.yaml` note. The owner's photos are private.
4. **Secrets**: none are needed locally; set `APP_ACCESS_PASSWORD` only if you want the login gate.
5. **Environment**: Python 3.12, `pip install -r requirements.txt` (+ `requirements-dev.txt`);
   `requirements-constraints.txt` pins versions.

### First-hour checklist for the new Claude
1. Read this file, `AGENTS.md`, then `docs/ROOFTOP_CONDITION_MIX_PHASE19B.md` §0.
2. `git status` / `git branch --show-current` (must be the feature branch; never `main`).
3. Verify weight SHA-256s; run `pytest -q` (expect 1232 passed / 5 skipped).
4. Start the app, upload two photos from `E:\Solar AI Consolidated Datasets\2_first_party_photos\solar_test_images_rooftop_48\`
   (e.g. `IMG_5891.jpg`, `IMG_5893.jpg` → 6 panels each) and confirm sensible output.
5. Ask the owner what they want next before changing models. Their stated complaints so far have
   been about real-world behavior; check `data/inspections.db` (tables `ground_inspections`,
   `panel_observations`, …) to see what they actually saw — that is how the 27% problem was found.
6. Fix the stale UI text in 8.2 as a small, separate, task-based commit if the owner agrees.
