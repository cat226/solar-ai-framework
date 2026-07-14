"""utils/logger.py — Centralized logging factory.

All modules should obtain their logger via :func:`get_logger` rather than
calling ``logging.getLogger`` directly.  The root logger is configured once
(on first import) using the ``logging`` section of ``configs/settings.yaml``.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from utils.config import CFG


def _configure_root_logger() -> None:
    """Apply the settings.yaml logging config to the root logger once."""
    log_cfg: dict = CFG.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    fmt: str = log_cfg.get(
        "format",
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    datefmt: str = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()

    # Only configure once
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
              Defaults to the root logger if *None*.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Module initialised.")
    """
    _configure_root_logger()
    return logging.getLogger(name)
