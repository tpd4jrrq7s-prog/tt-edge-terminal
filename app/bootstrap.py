"""Application bootstrap: loads settings and configures logging.

This is the single place where startup wiring happens. Later phases
(ingestion, persistence, API) will extend `AppContext` and `bootstrap()`
rather than each module configuring its own logging/settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.logging import configure_logging
from config.settings import Settings, get_settings


@dataclass(frozen=True)
class AppContext:
    """Holds initialized, application-wide dependencies."""

    settings: Settings
    logger: logging.Logger


def bootstrap() -> AppContext:
    """Initialize settings and logging, returning a ready-to-use AppContext."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(settings.app_name)
    return AppContext(settings=settings, logger=logger)
