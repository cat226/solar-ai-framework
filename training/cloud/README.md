# Cloud Training Orchestration

Lets Claude Code prepare, launch, and monitor Solar AI training jobs on remote
GPU workers while keeping the actual training logic
(`training/classification/`, `training/detection/`) untouched and
provider-independent.

**Currently implemented: Kaggle only.** Hugging Face is a public-read-only
data source today (no write token configured — see below); Google Colab has
no programmatic control path at all in this environment and is not
implemented. This document says exactly what has been verified against the
real platforms versus what is written but not yet exercised for real — do
not assume more than what's stated here.

## Architecture

```text
training/cloud/
├── base/
│   ├── job_spec.py            # TrainingJobSpec — serializable, reproducible job description
│   ├── registry.py            # Append-only JSONL experiment registry
│   └── artifact_validation.py # Checkpoint/artifact integrity checks
├── kaggle/
│   ├── adapter.py             # prepare() / dry_run() / launch() / status() / logs() / outputs()
│   └── cli.py                 # `python -m training.cloud.kaggle.cli {prepare,dry-run,launch}`
└── runs/                      # Locally-staged kernel packages (gitignored, per-machine)

training/experiments/
└── registry.jsonl             # Created on first recorded experiment; committed (no secrets)
```

The training scripts themselves are unchanged — a cloud run just means
`training/detection/train_yolo.py` (or an equivalent thin Kaggle entry
script that calls into the same code) executes on Kaggle's machine instead
of locally. This package only handles getting code/config there and
results back.

## Platform status (audited 2026-09-04)

| Platform | Auth | What works | What doesn't |
|---|---|---|---|
| GitHub | ✅ `gh` CLI logged in | clone/pull/branch/commit/push/PR/CI — everything used throughout this project | — |
| Hugging Face | ❌ no write token (`HF_TOKEN` unset) | Public dataset download (already used for BDAPPV) | Upload/private repos — would need a real token |
| **Kaggle** | ✅ verified (see below) | Auth, dataset listing, kernel package prep/validation | No native cancel; no native resume |
| Google Colab | ❌ no tool exists | Nothing programmatic | Everything — no MCP/API/browser-automation path available here |

## Kaggle authentication

Credentials live at `~/.kaggle/access_token` (a Kaggle personal access
token, not the older `kaggle.json` username+key format — the installed
`kaggle` CLI (2.2.4) supports both). **The file must be plain UTF-8 with no
BOM and no trailing whitespace** — a token saved via Windows PowerShell's
`>` or `Out-File` without `-Encoding utf8` will default to UTF-16 and the
Kaggle library will reject it with a cryptic
`ValueError: Invalid header value` deep in `urllib3`, not an auth error.
Verified working: `kaggle competitions list`, `kaggle datasets list`,
`kaggle kernels init`, `kaggle kernels status <ref>` (against a
known-nonexistent ref, correct permission-denied response). Authenticated
account: `edithstark`.

## Kernel push/execute semantics (important)

**There is no separate "create" or "execute" step on Kaggle.**
`kaggle kernels push` uploads the kernel package *and immediately starts
GPU execution*. This adapter's `launch()` function represents exactly that
— it is documented and named as the irreversible operation, not wrapped in
a friendlier-sounding abstraction. `prepare()` and `dry_run()` are the only
fully local, zero-cost, repeatable operations.

**There is no native cancel/stop command.** The Kaggle kernel CLI's full
subcommand set is `{list, files, get, init, push, pull, output, status,
logs, update, delete, topics}` — confirmed by reading `kaggle kernels
--help` directly. `delete` removes the kernel definition, which is more
destructive than "stop this run" and is not confirmed to actually halt
in-progress GPU execution rather than just deleting the record. If a job
needs to stop, the practical options are: let it finish, let it hit
Kaggle's own execution time limit, or delete the kernel and accept the
ambiguity about whether that halts billing/quota consumption.

**There is no native resume.** Every `push` runs the current code from
scratch. Resuming a training run across kernel launches is an
application-level pattern: the entry script must download the previous
run's checkpoint (via `kaggle kernels output` or a linked dataset) before
starting, using Ultralytics-native (`.pt`, contains optimizer state) or
PyTorch-native (`state_dict`) checkpoint formats — no universal checkpoint
format is invented here.

## Job specification

`TrainingJobSpec` (`base/job_spec.py`) captures everything needed to
reproduce a run: experiment ID, model, git SHA, dataset identity/hash,
class order, all hyperparameters, seed, requested GPU, Python/package
versions. `to_json()` is deterministic (sorted keys) so `spec_hash()` is a
meaningful audit key — two specs with identical content always hash
identically, regardless of construction order. `capture_environment()`
auto-fills the git SHA and package versions from the current environment;
it never fabricates a value it can't determine (raises rather than
recording a placeholder git SHA outside a repo, for instance).

## Experiment registry

`base/registry.py` appends one JSON object per line to
`training/experiments/registry.jsonl` — append-only, so an experiment's
full status history (`prepared` → `launched` → `completed`/`failed`) is
preserved rather than overwritten. `record_experiment()` refuses (raises
`SecretFieldError`) any record whose field names look like they might hold
a credential (`token`, `password`, `secret`, `api_key`, `credential`,
checked recursively) — this is a static-analysis-style guard on field
*names*, not a content scanner, so it won't catch a secret value stored
under an innocuous key; don't rely on it as the only safeguard.

## Local storage policy (2026-09-04)

The machine this project has mostly been developed on has C:/D: drives
running low on space, so `base/storage_paths.py` centralizes where large,
locally-generated Solar AI data goes: the E: drive
(`E:\Solar AI Training Images\`), not the repository's own drive. This
applies to downloaded/prepared datasets, Kaggle dataset staging, kernel
packages, outputs retrieved from Kaggle (checkpoints, result plots, logs),
local training run directories, and anything else of meaningful size — not
to small git-tracked metadata like `training/experiments/registry.jsonl` or
`kernel-metadata.json`, which stay in the repository as before.

- `SOLAR_AI_DATA_ROOT` — the root; overridable via the same-named
  environment variable, otherwise defaults to the E: path **only on
  Windows** (`sys.platform == "win32"`) and is `None` everywhere else —
  deliberately never guesses an `E:`-shaped path on a platform (e.g. the
  Linux container a Kaggle kernel runs in) where that would just create a
  literal `E:` subdirectory.
- `KAGGLE_RUNS_DIR` / `LOCAL_TRAINING_RUNS_DIR` — the two subdirectories
  currently in use (Kaggle kernel packages + retrieved outputs; local
  training run outputs). Both `None` when `SOLAR_AI_DATA_ROOT` is `None`.
- `default_kaggle_package_dir(experiment_id)` — what
  `build_yolo_smoke_package.py` / `build_yolo_full_training_package.py` /
  `build_dataset_mount_diagnostic_package.py` use for `--package-dir` when
  it isn't passed explicitly; falls back to the original
  `training/cloud/runs/<experiment_id>` when `KAGGLE_RUNS_DIR` is `None`.
- `ensure_free_space(path, required_bytes, label=...)` — raises
  `InsufficientSpaceError` rather than letting an operation silently fall
  back to a different (low-space) drive; callers that stage/download
  something large should check this first and report the requirement
  rather than proceeding.

`training/detection/train_yolo.py`'s `--project` default follows the same
convention but does **not** import this module — see the comment at the
top of that file: it runs on Kaggle as a bare `python /path/to/
train_yolo.py` subprocess with no `cwd` set to the repo root, so a
`training.cloud.*` package import would fail there with
`ModuleNotFoundError`. Its default is a small, self-contained,
independently-tested copy of the same env-var/platform logic instead.

Existing audited datasets already on E: (`_raw_downloads/`, `source/`,
`prepared/`, `yolo_source/`, `yolo_prepared/`, `yolo_smoke_dataset/`) keep
their existing names/locations — this policy only adds new subdirectories
for data that didn't have an established home yet.

## Artifact validation

`base/artifact_validation.py` provides reusable checks used both after a
local run and after retrieving a checkpoint from Kaggle:
`validate_file_exists`, `compute_and_check_sha256`,
`validate_torch_checkpoint_integrity` (loads with `weights_only=True`, the
same safe-loading contract as `models/model_manager.py`),
`validate_mobilenet_class_head` (confirms the classifier output size
matches the expected class count — this is what stops an interim/subset
checkpoint from ever being mistaken for the production artifact), and
`validate_ultralytics_checkpoint_integrity` (confirms the YOLO checkpoint
loads and reports exactly one class, "solar panel"). Every check either
genuinely passes, genuinely fails with a specific error, or the caller gets
an exception — nothing here silently reports success on data it couldn't
actually evaluate.

## Dry run and launch approval

```bash
# 1. Prepare (local only, no network call)
python -m training.cloud.kaggle.cli prepare \
    --experiment-id exp-0001 --model yolo_detection \
    --entrypoint <path-to-a-kaggle-entry-script> \
    --package-dir training/cloud/runs/exp-0001 \
    --dataset-source gabrielkasmi/bdappv --gpu

# 2. Validate the prepared package (local only, no network call)
python -m training.cloud.kaggle.cli dry-run --package-dir training/cloud/runs/exp-0001

# 3. Launch — IRREVERSIBLE, starts GPU billing/quota consumption immediately.
#    Refuses to run without --yes.
python -m training.cloud.kaggle.cli launch \
    --package-dir training/cloud/runs/exp-0001 --experiment-id exp-0001 --yes
```

The same protection exists at the Python API level, not just the CLI:
`training.cloud.kaggle.adapter.launch()` raises `LaunchNotConfirmedError`
unless called with `confirm=True` explicitly, and re-validates the package
with `dry_run()` before ever shelling out — a corrupted or incomplete
package cannot be launched even with confirmation.

## Testing

All Kaggle CLI calls are mocked in tests — no test invokes a real kernel or
consumes GPU time. `prepare()`/`dry_run()` are pure local file operations
and are tested for real, without mocking, since they never shell out.

```bash
pytest tests/test_cloud_job_spec.py tests/test_cloud_registry.py \
       tests/test_cloud_artifact_validation.py \
       tests/test_cloud_kaggle_adapter.py tests/test_cloud_kaggle_cli.py
python -m compileall training/cloud
```

`test_cloud_artifact_validation.py` additionally exercises two real local
checkpoints from this machine's own training runs
(`weights/mobilenet_interim_3class.pth`, `weights/yolo_solar_candidate_epoch2.pt`)
when present, skipping cleanly when they're not (they're gitignored and
won't exist in CI or a fresh checkout).

## Cost control

Nothing in this package can spend money by itself — Kaggle kernels run
against the account's normal Kaggle quota, and Kaggle does not bill by
default. **This account's actual GPU quota/hours were not inspected** —
that lives in Kaggle's own account settings page, not exposed via the CLI,
and checking it is a reasonable thing to do before a real launch but wasn't
done as part of this implementation. `launch()`'s `confirm=True`
requirement and the CLI's `--yes` flag exist specifically so that GPU
execution is never triggered by an automated or accidental code path,
free-tier or otherwise.

## Troubleshooting

- **`ValueError: Invalid header value b'Bearer ...'`** — the access token
  file has wrong encoding (see "Kaggle authentication" above). Rewrite it
  as plain UTF-8 with no BOM and no trailing whitespace/newline.
- **`kaggle kernels list --mine` returns "Not found"`** — this is very
  likely just an empty result (a Kaggle account with zero kernels so far),
  not an authentication failure. Cross-check with `kaggle datasets list`,
  which returns real data on any valid token.
- **`LaunchNotConfirmedError`** — this is the safety mechanism working as
  designed. Pass `confirm=True` (API) or `--yes` (CLI) only when you
  actually intend to consume Kaggle GPU time right now.
- **`KaggleCLIError`** — wraps any non-zero-exit `kaggle` CLI invocation;
  `.stdout`/`.stderr` on the exception carry the CLI's own error text.
