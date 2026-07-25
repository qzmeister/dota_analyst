"""
Shared logging setup for the Dota Analyst gateway.

This is a near-copy of `business._logging` so the gateway container
has no runtime dependency on the `business` package. They can
diverge if needed; for now, the contract is "JSON or text, level
from env, idempotent setup".

Replaces the ad-hoc `print(...)` in middleware. Each module does:

    from ._logging import get_logger
    log = get_logger(__name__)
    log.info("request", extra={"path": path})
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
    """Configure the root logger. Idempotent unless `force=True`."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()

    root = logging.getLogger()
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

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
