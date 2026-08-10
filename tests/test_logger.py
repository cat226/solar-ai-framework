"""tests/test_logger.py - Deterministic tests for utils/logger.py.

Covers the centralized logging factory:

A. _configure_root_logger
   - adds StreamHandler when none present
   - uses sys.stdout
   - sets formatter from config
   - sets root level from config
   - idempotent when handler already present

B. get_logger
   - returns named logger
   - configures root on first call
   - returns root logger when name is None

Design rules:
- no external dependencies
- deterministic
- isolated handler manipulation with strict restoration
"""

from __future__ import annotations

import logging
import sys

import pytest

from utils.logger import _configure_root_logger, get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_root_handlers():
    """Remove all handlers from the root logger, returning the original list and level."""
    root = logging.getLogger()
    original = root.handlers[:]
    original_level = root.level
    root.handlers.clear()
    return original, original_level


def _restore_root_handlers(original, original_level):
    """Restore previously saved root handlers and level."""
    root = logging.getLogger()
    root.handlers[:] = original
    root.setLevel(original_level)


# ---------------------------------------------------------------------------
# A. _configure_root_logger
# ---------------------------------------------------------------------------

class TestConfigureRootLogger:
    """Root logger configuration behaviour."""

    def test_adds_stream_handler_when_none_present(self):
        original, original_level = _clear_root_handlers()
        try:
            _configure_root_logger()
            root = logging.getLogger()
            assert len(root.handlers) == 1
            handler = root.handlers[0]
            assert isinstance(handler, logging.StreamHandler)
            assert handler.stream is sys.stdout
        finally:
            _restore_root_handlers(original, original_level)

    def test_sets_formatter_from_config(self):
        original, original_level = _clear_root_handlers()
        try:
            _configure_root_logger()
            root = logging.getLogger()
            handler = root.handlers[0]
            assert isinstance(handler.formatter, logging.Formatter)
            fmt = handler.formatter._fmt
            datefmt = handler.formatter.datefmt
            assert "%(asctime)s" in fmt
            assert "%(levelname)s" in fmt
            assert "%(name)s" in fmt
            assert "%(message)s" in fmt
            assert datefmt == "%Y-%m-%d %H:%M:%S"
        finally:
            _restore_root_handlers(original, original_level)

    def test_sets_root_level_from_config(self):
        original, original_level = _clear_root_handlers()
        try:
            _configure_root_logger()
            root = logging.getLogger()
            assert root.level == logging.INFO
        finally:
            _restore_root_handlers(original, original_level)

    def test_idempotent_when_handler_already_present(self):
        root = logging.getLogger()
        original = root.handlers[:]
        original_level = root.level
        count_before = len(original)

        _configure_root_logger()

        assert len(root.handlers) == count_before
        _restore_root_handlers(original, original_level)

    def test_does_not_override_existing_handler(self):
        original, original_level = _clear_root_handlers()
        try:
            custom_handler = logging.StreamHandler(sys.stderr)
            root = logging.getLogger()
            root.addHandler(custom_handler)

            _configure_root_logger()

            assert len(root.handlers) == 1
            assert root.handlers[0] is custom_handler
        finally:
            _restore_root_handlers(original, original_level)


# ---------------------------------------------------------------------------
# B. get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    """Public logger factory behaviour."""

    def test_returns_named_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_returns_root_logger_when_name_is_none(self):
        logger = get_logger(None)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "root" or logger.name == ""

    def test_configures_root_on_first_call(self):
        original, original_level = _clear_root_handlers()
        try:
            get_logger("some.module")
            root = logging.getLogger()
            assert len(root.handlers) >= 1
        finally:
            _restore_root_handlers(original, original_level)

    def test_repeated_calls_return_same_named_logger(self):
        logger1 = get_logger("my.module")
        logger2 = get_logger("my.module")
        assert logger1 is logger2
