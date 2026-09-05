"""training/cloud/base/registry.py — Lightweight, file-based experiment registry.

One JSON Lines file (append-only for new experiments; status updates append
a new line for the same experiment_id, with the last line for a given ID
being authoritative). Deliberately not a database - this project's own
guidance is to avoid a heavy MLOps platform when a simple, auditable,
git-trackable file does the job.

No secrets are ever accepted into a record - see _reject_secrets().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DEFAULT_REGISTRY_PATH = Path("training/experiments/registry.jsonl")

# Field-name substrings that must never appear as a top-level key in a record -
# a defence against accidentally logging a credential into a committed file.
_FORBIDDEN_KEY_SUBSTRINGS = ("token", "password", "secret", "api_key", "apikey", "credential")


class SecretFieldError(ValueError):
    """Raised when a record contains a field name that looks like a credential."""


def _reject_secrets(record: dict[str, Any]) -> None:
    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lower = str(k).lower()
                if any(bad in lower for bad in _FORBIDDEN_KEY_SUBSTRINGS):
                    raise SecretFieldError(f"refusing to record field that looks like a credential: {path}{k}")
                _walk(v, f"{path}{k}.")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}].")

    _walk(record)


def record_experiment(record: dict[str, Any], *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Append one experiment record. Raises SecretFieldError if the record
    contains a field name that looks like it might hold a credential."""
    _reject_secrets(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def load_experiments(*, registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Return every record in file order (oldest first). Malformed lines are
    skipped, not fatal - a registry file should degrade gracefully."""
    if not registry_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def get_experiment(experiment_id: str, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> Optional[dict[str, Any]]:
    """Return the most recent record for experiment_id (last line wins), or None."""
    latest: Optional[dict[str, Any]] = None
    for record in load_experiments(registry_path=registry_path):
        if record.get("experiment_id") == experiment_id:
            latest = record
    return latest


def update_experiment_status(
    experiment_id: str,
    status: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    **fields: Any,
) -> None:
    """Append a new record for an existing experiment_id reflecting a status
    change (e.g. queued -> running -> completed/failed). The previous record
    for this ID is not mutated or removed - the registry is append-only, so
    the full history of an experiment's status transitions is preserved."""
    existing = get_experiment(experiment_id, registry_path=registry_path)
    if existing is None:
        raise KeyError(f"no existing experiment record for {experiment_id!r} to update")
    updated = dict(existing)
    updated["status"] = status
    updated.update(fields)
    record_experiment(updated, registry_path=registry_path)
