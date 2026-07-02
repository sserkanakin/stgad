"""shared, Vimal-style timestamped logging for the training + eval scripts.

every line is  `YYYY-MM-DD HH:MM:SS [LEVEL] message`  and goes to BOTH stdout (the
slurm .out) and, once a run dir is known, the per-run train.log — so the console
log and the saved log are identical and easy to read/navigate.

usage:
    from src.utils.logging_setup import setup_logging, get_logger
    log = setup_logging(ckpt_dir / "train.log")   # once, in main()
    log.info("hello")
    # in module-level helpers:
    from src.utils.logging_setup import get_logger
    get_logger().info("...")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_NAME = "diffusion"
_FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str = _NAME) -> logging.Logger:
    """the shared logger (configure it once with setup_logging)."""
    return logging.getLogger(name)


def setup_logging(log_path: Optional[str | Path] = None, name: str = _NAME) -> logging.Logger:
    """(re)configure the shared logger: stdout always, + a FileHandler for log_path if given.

    idempotent — clears old handlers so a resumed run doesn't double-print. call once at
    the start of main(); module helpers then just get_logger().info(...)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FMT)
    logger.addHandler(sh)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)          # append: resumes keep one continuous log
        fh.setFormatter(_FMT)
        logger.addHandler(fh)
    return logger
