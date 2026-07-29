"""Structured (JSON) logging configuration.

Uses only the standard library `logging` module with a custom JSON
formatter, deliberately avoiding an extra dependency for something this
simple at Phase 1. Every log line is a single JSON object, which makes
logs easy to parse in any future cloud logging pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a JSON structured formatter.

    Safe to call more than once (e.g. in tests): existing handlers are
    cleared first to avoid duplicate log lines.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
