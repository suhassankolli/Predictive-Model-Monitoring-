"""
Structured logging setup for ModelSentinel.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
