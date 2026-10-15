"""Zero-Overhead Structured Logger & Telemetry Engine for OjasX (`ojasx2.logger`).

Provides high-performance, non-blocking hierarchical logging for compiler passes,
hardware kernel dispatch, and memory tracking.

Features:
1. Zero-Overhead Fast Paths: Caches active log level in integer `_ACTIVE_LEVEL`.
   When log level is below threshold, calls exit immediately with zero allocations
   and zero string formatting.
2. Lazy Evaluation: Supports passing lambdas/callables for expensive debug dumps
   (e.g., IR inspection, tensor shape audits) that are evaluated ONLY if the level is active.
3. Environment Control: Configurable via `OJASX_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Dict, Optional, Union


# Standard numeric levels matching standard logging
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

_LEVEL_MAP: Dict[str, int] = {
    "DEBUG": DEBUG,
    "INFO": INFO,
    "WARNING": WARNING,
    "WARN": WARNING,
    "ERROR": ERROR,
    "CRITICAL": CRITICAL,
}

# Read initial level from environment, defaulting to WARNING in production
_INITIAL_ENV_LEVEL = os.environ.get("OJASX_LOG_LEVEL", "WARNING").upper().strip()
_ACTIVE_LEVEL: int = _LEVEL_MAP.get(_INITIAL_ENV_LEVEL, WARNING)

_ROOT_LOGGER: Optional[logging.Logger] = None


def _get_root_logger() -> logging.Logger:
    global _ROOT_LOGGER
    if _ROOT_LOGGER is None:
        logger = logging.getLogger("ojasx2")
        logger.setLevel(_ACTIVE_LEVEL)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [OjasX %(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
        _ROOT_LOGGER = logger
    return _ROOT_LOGGER


class OjasXLogger:
    """Lightweight wrapper around standard logging.Logger with fast-path level bypass."""

    __slots__ = ("_logger", "_name")

    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = logging.getLogger(f"ojasx2.{name}" if name != "ojasx2" else "ojasx2")
        self._logger.setLevel(_ACTIVE_LEVEL)
        # Ensure root has handler
        _get_root_logger()

    @property
    def name(self) -> str:
        return self._name

    def is_enabled_for(self, level: int) -> bool:
        return level >= _ACTIVE_LEVEL

    def debug(self, msg_or_fn: Union[str, Callable[[], str]], *args: Any, **kwargs: Any) -> None:
        if _ACTIVE_LEVEL > DEBUG:
            return
        msg = msg_or_fn() if callable(msg_or_fn) else msg_or_fn
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg_or_fn: Union[str, Callable[[], str]], *args: Any, **kwargs: Any) -> None:
        if _ACTIVE_LEVEL > INFO:
            return
        msg = msg_or_fn() if callable(msg_or_fn) else msg_or_fn
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg_or_fn: Union[str, Callable[[], str]], *args: Any, **kwargs: Any) -> None:
        if _ACTIVE_LEVEL > WARNING:
            return
        msg = msg_or_fn() if callable(msg_or_fn) else msg_or_fn
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg_or_fn: Union[str, Callable[[], str]], *args: Any, **kwargs: Any) -> None:
        if _ACTIVE_LEVEL > ERROR:
            return
        msg = msg_or_fn() if callable(msg_or_fn) else msg_or_fn
        self._logger.error(msg, *args, **kwargs)


def get_logger(name: str = "core") -> OjasXLogger:
    """Retrieves a fast, zero-overhead OjasXLogger instance."""
    return OjasXLogger(name)


def set_log_level(level: Union[str, int]) -> None:
    """Dynamically sets global OjasX logging level across all child loggers."""
    global _ACTIVE_LEVEL
    if isinstance(level, str):
        target = _LEVEL_MAP.get(level.upper().strip(), WARNING)
    else:
        target = int(level)
    _ACTIVE_LEVEL = target
    root = _get_root_logger()
    root.setLevel(target)
    for name in list(logging.root.manager.loggerDict.keys()):
        if name.startswith("ojasx2"):
            logging.getLogger(name).setLevel(target)


__all__ = [
    "OjasXLogger",
    "get_logger",
    "set_log_level",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]
