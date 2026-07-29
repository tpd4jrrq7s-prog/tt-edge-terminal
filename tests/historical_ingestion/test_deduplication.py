"""Tests for deterministic duplicate detection and conflict classification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.historical_ingestion import HistoricalIngestionSettings
from historical_ingestion.deduplication import (
    DuplicateOutcome,
    compute_match_internal_id,
    compute_odds_internal_id,
    evaluate_match_duplicate,
    evaluate_odds_duplicate,
)
from persistence.in_memory import InMemoryMatchRepository, InMemoryOddsRepository
from persistence.models import HistoricalMatchRecord, HistoricalOddsRecord, HistoricalSetRecord, MatchRecordStatus

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _match_record(**overrides) -> HistoricalMatchRecord:
    defaults = dict(
        id=compute_match_internal_id("mock", "m1"), provider="mock", provider_match_id="m1",
        player_a_id="p1", player_b_id="p2", scheduled_at=AWARE, actual_start_at=AWARE,
        completed_at=AWARE + timedelta(hours=1), status=MatchRecordStatus.FINISHED, winner_id="p1",
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=AWARE, ingested_at=AWARE,
    )
    defaults.update(overrides)
    return HistoricalMatchRecord(**defaults)


def test_new_match_is_inserted():
    repo = InMemoryMatchRepository()
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", AWARE, MatchRecordStatus.FINISHED, "p1",
        [HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)], HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.INSERTED


def test_identical_reimport_is_idempotent():
    repo = InMemoryMatchRepository()
    existing = _match_record()
    repo.add(existing)
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", existing.scheduled_at, existing.status, existing.winner_id,
        existing.sets, HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.SKIPPED_IDEMPOTENT


def test_conflicting_winner_is_rejected():
    repo = InMemoryMatchRepository()
    repo.add(_match_record())
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", AWARE, MatchRecordStatus.FINISHED, "p2",
        [HistoricalSetRecord(set_number=1, player_a_points=7, player_b_points=11)], HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.REJECTED_CONFLICT
    assert "winner" in decision.reasons[0]


def test_scheduled_to_completed_progression_is_merged_safe():
    repo = InMemoryMatchRepository()
    scheduled = HistoricalMatchRecord(
        id=compute_match_internal_id("mock", "m1"), provider="mock", provider_match_id="m1",
        player_a_id="p1", player_b_id="p2", scheduled_at=AWARE, status=MatchRecordStatus.SCHEDULED,
        provider_timestamp=AWARE, ingested_at=AWARE,
    )
    repo.add(scheduled)
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", AWARE, MatchRecordStatus.FINISHED, "p1",
        [HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)], HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.MERGED_SAFE


def test_cross_provider_likely_duplicate_is_quarantined():
    repo = InMemoryMatchRepository()
    repo.add(_match_record(provider="other", provider_match_id="om1", id=compute_match_internal_id("other", "om1")))
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", AWARE, MatchRecordStatus.FINISHED, "p1",
        [HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)], HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.QUARANTINED


def test_distant_cross_provider_match_is_not_flagged():
    repo = InMemoryMatchRepository()
    repo.add(_match_record(provider="other", provider_match_id="om1", id=compute_match_internal_id("other", "om1"), scheduled_at=AWARE - timedelta(days=30)))
    decision = evaluate_match_duplicate(
        repo, "mock", "m1", "p1", "p2", AWARE, MatchRecordStatus.FINISHED, "p1",
        [HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)], HistoricalIngestionSettings(),
    )
    assert decision.outcome is DuplicateOutcome.INSERTED


def _odds_record(**overrides) -> HistoricalOddsRecord:
    defaults = dict(
        id=compute_odds_internal_id("mock", "mock:m1", "Pinnacle", "p1", AWARE),
        match_id="mock:m1", bookmaker="Pinnacle", selection_id="p1", decimal_odds=1.8,
        captured_at=AWARE, provider="mock",
    )
    defaults.update(overrides)
    return HistoricalOddsRecord(**defaults)


def test_new_odds_is_inserted():
    repo = InMemoryOddsRepository()
    decision = evaluate_odds_duplicate(repo, "mock", "mock:m1", "Pinnacle", "p1", 1.8, AWARE)
    assert decision.outcome is DuplicateOutcome.INSERTED


def test_identical_odds_reimport_is_idempotent():
    repo = InMemoryOddsRepository()
    repo.add(_odds_record())
    decision = evaluate_odds_duplicate(repo, "mock", "mock:m1", "Pinnacle", "p1", 1.8, AWARE)
    assert decision.outcome is DuplicateOutcome.SKIPPED_IDEMPOTENT


def test_conflicting_odds_value_is_rejected():
    repo = InMemoryOddsRepository()
    repo.add(_odds_record())
    decision = evaluate_odds_duplicate(repo, "mock", "mock:m1", "Pinnacle", "p1", 2.5, AWARE)
    assert decision.outcome is DuplicateOutcome.REJECTED_CONFLICT


def test_distinct_timestamps_preserve_odds_history():
    repo = InMemoryOddsRepository()
    repo.add(_odds_record())
    decision = evaluate_odds_duplicate(repo, "mock", "mock:m1", "Pinnacle", "p1", 1.9, AWARE + timedelta(minutes=30))
    assert decision.outcome is DuplicateOutcome.INSERTED
