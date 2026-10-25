"""tests/test_evaluation_report_schema_and_artifact_identity.py

Two concerns, both about keeping evaluation reports trustworthy over time:

1. Evaluation report schema: the JSON summaries evaluation scripts produce
   must always carry the fields a report-writer (or a future automated
   comparison) depends on - filename drift or a silently-dropped field
   would make docs/ML_EVALUATION_v1.0.0.md's numbers impossible to
   reproduce or verify against a fresh run.
2. Model artifact identity: the reviewed SHA-256 hashes in
   weights/manifest.json must keep matching the actual real artifacts on
   disk (when present - this repository does not commit model weights, so
   these are skipped, not failed, when the gitignored files aren't
   available locally, e.g. in CI).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.evaluation.common import sha256_file

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEvaluationSummarySchema:
    """Every evaluate_*.py script's summary JSON must carry these fields -
    a static, deterministic check of the code (via inspecting the literal
    dict keys assigned to `summary = {...}`), not a live run.

    The aerial-pipeline evaluation scripts this class used to also cover
    (evaluate_yolo.py, evaluate_mobilenet.py, evaluate_end_to_end.py,
    yolo_threshold_sweep.py) were removed alongside the rest of the aerial
    pipeline — see docs/PHASE_16_ARCHITECTURE_REVIEW.md. leakage_audit.py
    is generic dataset-quality tooling with no dependency on either
    pipeline, so its coverage stays."""

    def test_leakage_audit_summary_has_required_top_level_fields(self):
        import inspect
        import training.evaluation.leakage_audit as module
        source = inspect.getsource(module.main)
        for key in ["exact_duplicates", "near_duplicates", "source_signal",
                    "classification", "clean_test_subset"]:
            assert f'"{key}"' in source, f"leakage_audit.py summary is missing {key!r}"


class TestReleaseManifestArtifactIdentity:
    """weights/manifest.json's reviewed hashes must match the real files -
    this is the actual repository-committed manifest (see
    docs/RELEASE_v1.0.0.md), not a copy - if it ever drifts from the real
    artifacts, scripts/verify_model_artifacts.py would (correctly) start
    failing, and this test would catch the same drift earlier."""

    @pytest.fixture
    def manifest(self):
        path = _REPO_ROOT / "weights" / "manifest.json"
        if not path.is_file():
            pytest.skip("weights/manifest.json not present in this checkout")
        return json.loads(path.read_text())

    def test_manifest_entries_have_64_character_hex_sha256(self, manifest):
        import re
        for entry in manifest["artifacts"]:
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry

    def test_manifest_hashes_match_real_artifacts_when_present(self, manifest):
        found_any = False
        for entry in manifest["artifacts"]:
            artifact_path = _REPO_ROOT / "weights" / entry["path"]
            if not artifact_path.is_file():
                continue  # gitignored real weights - not present in every checkout (e.g. CI)
            found_any = True
            assert sha256_file(artifact_path) == entry["sha256"], (
                f"weights/manifest.json's recorded hash for {entry['path']} does not match "
                "the real file on disk - the manifest is stale or the artifact changed "
                "without updating it."
            )
        if not found_any:
            pytest.skip("no real model artifacts present in this checkout to verify against")
