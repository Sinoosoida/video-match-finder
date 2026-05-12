"""Tiny logging utility. Writes a timestamped log file you can `tail -f`,
in parallel with the normal tqdm progress bar / Rich console output.

Path: $VMF_LOG_FILE or /tmp/vmf-scan.log. Empty string disables file output.
"""
from __future__ import annotations

import logging
import os
import sys

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger("vmf")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    path = os.environ.get("VMF_LOG_FILE", "/tmp/vmf-scan.log")
    if path:
        try:
            fh = logging.FileHandler(path, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s.%(msecs)03d %(threadName)-14s %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(fh)
        except OSError:
            print(f"[vmf] could not open log file {path}", file=sys.stderr)

    _LOGGER = logger
    return logger


log = get_logger()
