"""training/cloud/kaggle/entrypoints/render.py — Substitutes CONFIG placeholder
tokens in an entrypoint template before it is packaged for Kaggle.

Kaggle kernels have no environment-variable mechanism (see
entrypoints/yolo_detection.py's module docstring for how this was verified),
so per-experiment configuration is baked into the entry script's source at
build time instead of being read from the Kaggle environment at run time.
This is a plain, explicit string substitution - no templating engine, no
hidden magic - so the rendered file stays a normal, readable Python script.
"""
from __future__ import annotations

import re
from pathlib import Path

_TOKEN_RE = re.compile(r'"__SOLAR_AI_[A-Z_]+__"')


def render_entrypoint(template_path: Path, values: dict[str, str], output_path: Path) -> Path:
    """Render an entrypoint template by substituting each CONFIG value.

    `values` keys must match the CONFIG dict keys in the template (e.g.
    "git_sha", "data_root") - each becomes the token "__SOLAR_AI_<KEY
    UPPERCASED>__" in the source text.

    Raises ValueError if any value is missing, or if the rendered output
    still contains an unsubstituted "__SOLAR_AI_*__" token - this must
    never silently produce a half-rendered script.
    """
    text = template_path.read_text(encoding="utf-8")

    for key, value in values.items():
        token = f"__SOLAR_AI_{key.upper()}__"
        if token not in text:
            raise ValueError(f"template {template_path} has no placeholder for config key {key!r} (expected token {token})")
        text = text.replace(f'"{token}"', repr(str(value)))

    leftover = _TOKEN_RE.findall(text)
    if leftover:
        raise ValueError(f"rendered entrypoint still contains unsubstituted placeholder(s): {leftover}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
