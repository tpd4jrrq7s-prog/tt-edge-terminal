"""Lightweight, fully synchronous polling abstraction for ingestion.

Deliberately does not start background threads, asyncio tasks, or any
kind of loop on its own — callers decide when and how often to invoke
`run_once()`. This keeps the scheduler trivial to test and leaves the
choice of deployment target (cron, asyncio loop, task queue, etc.) to a
later phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ingestion.service import IngestionResult, IngestionService

logger = logging.getLogger(__name__)


@dataclass
class IngestionScheduler:
    """Wraps an `IngestionService` with a configurable, explicit polling interval."""

    service: IngestionService
    poll_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

    def run_once(self) -> IngestionResult:
        """Run a single ingestion pass immediately, without waiting or looping."""
        logger.info(
            "scheduler.run_once poll_interval_seconds=%.1f", self.poll_interval_seconds
        )
        return self.service.run_once()
