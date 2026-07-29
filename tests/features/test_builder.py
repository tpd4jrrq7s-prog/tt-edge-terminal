"""Tests for HistoricalFeatureBuilder orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from features.builder import HistoricalFeatureBuilder
from features.errors import SnapshotLeakageError, TargetMatchNotFoundError
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _finished(mid, day, winner="p1") -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _scheduled(mid, day) -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, status=MatchRecordStatus.SCHEDULED,
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def test_raises_when_target_match_missing():
    repo = InMemoryMatchRepository()
    builder = HistoricalFeatureBuilder(repo)
    with pytest.raises(TargetMatchNotFoundError):
        builder.build("nope", BASE)


def test_builds_snapshot_using_only_prior_matches():
    repo = InMemoryMatchRepository()
    repo.add(_finished("m1", 1))
    repo.add(_finished("m2", 5))
    target = _scheduled("target", 10)
    repo.add(target)

    builder = HistoricalFeatureBuilder(repo)
    snapshot = builder.build("target", target.scheduled_at)

    assert snapshot.player_a_features.observation_count == 2
    assert snapshot.matchup_features.head_to_head_matches == 2
    assert "target" not in snapshot.provenance.player_a_source_match_ids


def test_raises_leakage_error_if_as_of_is_after_target_completion():
    repo = InMemoryMatchRepository()
    target = _finished("target", 1)
    repo.add(target)
    builder = HistoricalFeatureBuilder(repo)
    # as_of well after the target match itself completed: the repository would now
    # return the target match as part of "prior" history for both players.
    with pytest.raises(SnapshotLeakageError):
        builder.build("target", target.completed_at + timedelta(days=1))


def test_default_snapshot_id_is_deterministic():
    repo = InMemoryMatchRepository()
    target = _scheduled("target", 10)
    repo.add(target)
    builder = HistoricalFeatureBuilder(repo)
    first = builder.build("target", target.scheduled_at)
    second = builder.build("target", target.scheduled_at)
    assert first.id == second.id
    assert first == second


def test_custom_id_factory_is_used():
    repo = InMemoryMatchRepository()
    target = _scheduled("target", 10)
    repo.add(target)
    builder = HistoricalFeatureBuilder(repo, id_factory=lambda match_id, as_of: f"custom-{match_id}")
    snapshot = builder.build("target", target.scheduled_at)
    assert snapshot.id == "custom-target"
