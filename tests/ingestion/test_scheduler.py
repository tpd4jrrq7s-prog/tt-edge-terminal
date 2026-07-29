"""Tests for the IngestionScheduler run-once behavior."""

from __future__ import annotations

import pytest

from ingestion.scheduler import IngestionScheduler
from ingestion.service import IngestionService
from ingestion.sources.mock_source import MockTableTennisSource


def test_scheduler_run_once_delegates_to_service():
    service = IngestionService(source=MockTableTennisSource())
    scheduler = IngestionScheduler(service=service, poll_interval_seconds=30)

    result = scheduler.run_once()

    assert len(result.matches) == 3


def test_scheduler_default_poll_interval_is_configurable():
    service = IngestionService(source=MockTableTennisSource())
    scheduler = IngestionScheduler(service=service, poll_interval_seconds=15)

    assert scheduler.poll_interval_seconds == 15


def test_scheduler_rejects_non_positive_interval():
    service = IngestionService(source=MockTableTennisSource())
    with pytest.raises(ValueError):
        IngestionScheduler(service=service, poll_interval_seconds=0)


def test_scheduler_does_not_run_automatically_on_construction(monkeypatch):
    service = IngestionService(source=MockTableTennisSource())
    calls = []
    monkeypatch.setattr(service, "run_once", lambda: calls.append("called"))

    IngestionScheduler(service=service)

    assert calls == []
