"""
Shared logging setup for the Dota Analyst backend.

Replaces the dozens of `print(f"[board] ...")` / `print(f"[ml] ...")` /
`print(f"[discovery] ...")` calls scattered across the codebase. Each
module just does:

    from ._logging import get_logger
    log = get_logger(__name__)
    log.info("built board", extra={"series": len(prematch)})

`setup_logging()` should be called once at process startup (the FastAPI
lifespan handler is a natural place). It reads `LOG_LEVEL` and
`LOG_FORMAT` from the environment (see `.env.example`).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class _JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter — one log record per line, no extra deps."""

    # Standard LogRecord attributes we never want in the JSON payload
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger.

    Idempotent by default — calling twice is a no-op unless `force=True`.
    Safe to call from `app.py` lifespan, scripts and tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()

    root = logging.getLogger()
    # Clear any prior handlers (uvicorn configures its own; we keep those
    # for the access log by adding ours at WARNING+ to avoid duplicates)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    if fmt_name == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root.addHandler(handler)
    root.setLevel(_LOG_LEVELS.get(level_name, logging.INFO))

    # Quiet down noisy third-party loggers; let INFO through on ours
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. First call also configures the root logger."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
