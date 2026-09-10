"""tests/test_ground_pv_frozen_artifacts_integrity.py — Phase 15B regression
guard: every checkpoint/manifest this project has declared FROZEN must stay
byte-identical to its last known-good SHA-256.

This turns the manual "compute sha256, compare to memory" verification
done at the end of every ground_pv phase into an actual, automated,
repeatable check — so an accidental overwrite (e.g. a training script's
`--output` pointed at the wrong path) is caught by the test suite, not
merely by remembering to check manually.

Skipped entirely on a machine/CI run with no local `weights/` artifacts
(this repo never commits `*.pt`/`manifest.json` — see .gitignore) — this is
a DIFFERENT, more specific skip than the four pre-existing environment
failures in test_adversarial.py / test_runtime_readiness.py, which assume
an EMPTY weights/ directory; this file assumes the OPPOSITE (a machine that
already has the real artifacts) and simply does nothing when that isn't
the case, rather than failing either way.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _REPO_ROOT / "weights"

# Known-good SHA-256 for every artifact this project has declared frozen.
# Update an entry ONLY when a human has deliberately and knowingly replaced
# that specific artifact (e.g. a new production release) — never to make a
# failing test pass without first confirming the change was intentional.
_FROZEN_ARTIFACTS: dict[str, str] = {
    "yolo_solar.pt": "0b58609905b8ffe41465751cb842b6ed0368ca2555927e31fb5cd587af37e8ea",
    "manifest.json": "eabfec951724dff9975e57c9d3a290d272c61853a66b7c958fc3cfe58e907363",
    "yolo_solar_ground_research_v1.pt": "2b26334dc4023ba4c89a80e2350ddc9012358a48cdef713aee86c8e79a1bbb50",
    "yolo_solar_ground_research_v2.pt": "26e4d603de0718b9b20c6bf5c9761ef1b114b6954f280925ea6c08e469357aa3",
    "yolo_solar_condition_research_v1.pt": "b077251a514424481d51b6a57ebcc30f4a607ea7559053cbb3ef7d0de88f9578",
    # Phase 15A's local CPU smoke checkpoint — frozen the moment Phase 15B
    # created a NEW, separately-named checkpoint rather than overwriting it.
    "yolo_solar_condition_4class_research_v1.pt": "08994846e1087b6f485492e646f286ca9be8ea44ef7eb729c6e893fc31a8c7a3",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts_present() -> bool:
    return all((_WEIGHTS_DIR / name).is_file() for name in _FROZEN_ARTIFACTS)


pytestmark = pytest.mark.skipif(
    not _artifacts_present(),
    reason="frozen weights/ artifacts not present on this machine (never committed to git)",
)


@pytest.mark.parametrize("filename,expected_sha256", sorted(_FROZEN_ARTIFACTS.items()))
def test_frozen_artifact_unchanged(filename: str, expected_sha256: str) -> None:
    path = _WEIGHTS_DIR / filename
    actual = _sha256_file(path)
    assert actual == expected_sha256, (
        f"{filename} has changed! expected sha256={expected_sha256}, got {actual}. "
        "This file is declared FROZEN — if this change was deliberate and "
        "reviewed, update _FROZEN_ARTIFACTS explicitly; never silently."
    )


def test_new_research_checkpoint_never_reuses_a_frozen_filename():
    """Phase 15B's own rule: a new checkpoint must get a NEW, clearly
    versioned filename (e.g. _v2), never landing on one of the names
    already declared frozen above."""
    candidate = "yolo_solar_condition_4class_research_v2.pt"
    assert candidate not in _FROZEN_ARTIFACTS
