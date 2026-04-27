"""Centralized logging configuration.

Routes structured log records from every pipeline stage to ./yt_digest.log
(and the console at INFO+). Idempotent — safe to call multiple times.
"""

import logging
import sys
from pathlib import Path

LOG_FILE = Path("./yt_digest.log")
_FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
