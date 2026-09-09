# Phase 13A — dataset storage migration to F: + Phase 12 candidate download

**Status: BLOCKED at gate 1 (§1 "verify F: is available").**

**There is no F: drive on this machine.** No files were copied, moved, deleted,
re-encoded or modified. No candidate was downloaded. No production artifact was
touched. No commit of code/config was made (none was warranted — nothing
migrated).

---

## 1. Drive inspection

### D: — available, healthy

| | |
|---|---|
| volume | `New Volume`, NTFS, **Fixed** disk, HealthStatus **Healthy** |
| capacity | 118.1 GB |
| free (before) | **26.2 GB** |
| free (after) | **26.2 GB** (unchanged — nothing was written or deleted) |

### F: — **NOT PRESENT**

Checked via five independent mechanisms, all consistent:

| check | result |
|---|---|
| `Get-PSDrive -PSProvider FileSystem` (PowerShell) | only **C:** and **D:** |
| `Get-Volume` | only **D:** listed (C: implied) |
| `Get-CimInstance Win32_LogicalDisk` | only **C:** and **D:** |
| `Get-ChildItem 'F:\'` × 3 | all fail: *"Cannot find drive. A drive with the name 'F' does not exist."* |
| `Get-Disk` / `Get-Partition` | **one** physical disk (`SAMSUNG MZALQ256HBJD-00BL2`, 238.5 GB NVMe); its partitions are System + Reserved + **C: (119.3)** + Recovery + **D: (118.1)** — the entire disk. No space or partition for an F:. |
| `mountvol` | 4 volumes: C:, D:, and two System/Recovery volumes with *no* mount points. No F: volume. |
| `net use`, `Get-SmbMapping`, `subst` | no mapped network drives, no SMB shares, no virtual/`subst` drives |
| `Win32_DiskDrive` | one disk only, "Fixed hard disk media" — **no USB / removable / external drive attached** |

A stale in-process `Get-PSDrive` mapping briefly showed a phantom `F:` (183.5 GB)
in one git-bash call, accompanied by `"No medium found"` / `"No such device"`
errors on access — i.e. a **cached mapping for a drive that is no longer
connected** (a previously-attached removable or mapped drive). It is not
mountable and has no backing device.

C: free space: 5.1 GB (context only — not a migration target).

## 2. What would have been migrated (inventory only — nothing moved)

| dataset (`D:\…\solar-ai-framework\datasets\ground_pv_research\`) | size | files | notes |
|---|---:|---:|---|
| `raw/` (Susan `panel_solar_bw945_v1` + audited) | 397 MB | 12,990 | raw immutable Roboflow export |
| `yolo/` (Susan-derived leakage-clean 3-class) | 398 MB | 12,989 | approved v1 training set, all 640×640 |
| `condition/` (S1+S3 unified 5-class) | 233 MB | 5,044 | `images/` 228 MB gitignored |
| `candidate_defects/` (S1,S3,C1,C3,S4,S6 raw) | 556 MB | 11,675 | `raw/` 549 MB gitignored |
| `phase12_audit/samples/` | 206 MB | 4,655 | Phase 12 sampled candidates (gitignored, not retained) |
| `reports/` | 18 MB | 36 | committed text/plots |
| `splits/` | 3.2 MB | 3 | split manifests |
| **total `datasets/`** | **≈ 1.8 GB** | ≈ 47,092 | |
| `D:\solar_ai_training_scratch\Solar Real Images\` (6 owner photos) | 920 KB | 6 | reference only — never trained, never committed |

## 3. Owner reference photos — verified in place, NOT moved

All six SHA-256 values match the Phase 6 `real_world_holdout.md` record:

| file | bytes | SHA-256 (first 24) |
|---|---:|---|
| …15.25.56.jpeg | 158,669 | `583f063090d74ba32783951c…` ✓ |
| …15.25.57 (1).jpeg | 163,303 | `24eb0006e67bfa5f4482ef02…` ✓ |
| …15.25.57.jpeg | 137,436 | `f6a6024376444fe7bcae425c…` ✓ |
| …15.25.58.jpeg | 143,812 | `a8243d233c7be3dba7fc776c…` ✓ |
| …15.25.59 (1).jpeg | 155,230 | `0863acbca61c7ebf88d07b09…` ✓ |
| …15.25.59.jpeg | 171,216 | `7bf163746890f2e245b622a1…` ✓ |

Source deleted: **NO** (correctly — verification target F: does not exist).

## 4. Phase 12 candidate download

**Not attempted.** §8 requires the existing datasets to be "safely copied and
verified" first, and §9 mandates the destination
`F:\SolarAI_Datasets\ground_pv_research\candidate_acquisition\…`. Both depend on
F:. Downloading `solar-panel-detection-bwpkx/solar-panel-detection-ubwtz` to D:
instead would contradict the explicit destination, add ~50 MB to a drive with
only 26 GB free, and defeat the stated purpose of moving research data *off* D:.
Deferred until a real destination drive is available (a decision for the user,
as with the Phase 2 E:→D: fallback).

Roboflow authentication was **not** the blocker — `ROBOFLOW_API_KEY` is present in
the project `.env` and worked in Phase 12.

## 5. Integrity

| artifact | SHA-256 | state |
|---|---|---|
| `weights/yolo_solar.pt` | `0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea` | **unchanged** |
| `weights/yolo_solar_ground_research_v1.pt` | `2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50` | **unchanged** |
| `weights/yolo_solar_ground_research_v2.pt` | `26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3` | **unchanged** |
| `weights/yolo_solar_condition_research_v1.pt` | `b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578` | **unchanged** |
| `weights/mobilenet_solar_v1.pth` | `afccaccfcc309952f7a94d754aaafc22e7e3391416b9518c5f4a8635b1c2682b` | **unchanged** |
| `weights/manifest.json` | `eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363` | **unchanged** |

Datasets: **byte-identical, unmoved.** `models/`, `services/`, `app.py`,
`pages/`, `v1.0.0`, `.git`, source, `.venv`: untouched. No `labels.cache`, no
re-encoding, no annotation edits, no split changes, no dedup.

## 6. Tests

`pytest -q -k "ground_pv or condition"` → **133 passed, 1,331 deselected.** No
failures; no dataset-path resolution problems (everything still resolves against
`D:\…\datasets\`, which was not moved).

## 7. Git

- `git diff` / `git status`: only the pre-existing, unrelated
  `scripts/infer_ground_pv.py` (a Phase 6 change, not touched here). No
  production files.
- **No portable dataset-root configuration change was made** — nothing migrated,
  so none is warranted. (`SOLAR_AI_DATA_ROOT` remains a future option once a
  destination exists.)
- Commit: this report only. Pushed: **NO**. Merged: **NO**. Deployed: **NO**.

## 8. To unblock

1. Attach / mount the intended stable dataset drive so it appears as a real
   volume (`Get-Volume` must list it with real free space).
2. Confirm the drive letter (it may not be `F:` — Windows assigns the next free
   letter; earlier phases saw the same removable drive appear as `E:` then `M:`).
3. Re-run this phase: `robocopy` copy-first → count/size/SHA verify → sample-open
   → delete D: source only on full match → download the Phase 12 candidate to the
   new drive → write its download manifest.

Given the history of this removable drive vanishing mid-operation (Phases 2, 5,
6), a copy-and-verify pass should be treated as non-atomic and re-verifiable, and
**no D: source should be deleted until a second independent verification run
passes.**
