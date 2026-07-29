"""Tests for temporal historical record validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from persistence.models import (
    HistoricalMatchRecord,
    HistoricalOddsRecord,
    HistoricalPlayerRecord,
    HistoricalSetRecord,
    MatchRecordStatus,
)

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAIVE = datetime(2026, 1, 1)


def _finished_match(**overrides) -> HistoricalMatchRecord:
    defaults = dict(
        id="m1",
        provider="mock",
        provider_match_id="1",
        player_a_id="p1",
        player_b_id="p2",
        scheduled_at=AWARE,
        actual_start_at=AWARE,
        completed_at=AWARE,
        status=MatchRecordStatus.FINISHED,
        winner_id="p1",
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=AWARE,
        ingested_at=AWARE,
    )
    defaults.update(overrides)
    return HistoricalMatchRecord(**defaults)


def test_set_record_rejects_tie():
    with pytest.raises(ValidationError):
        HistoricalSetRecord(set_number=1, player_a_points=10, player_b_points=10)


def test_player_record_rejects_naive_ingested_at():
    with pytest.raises(ValidationError):
        HistoricalPlayerRecord(id="p1", name="A", provider="mock", provider_player_id="1", ingested_at=NAIVE)


def test_match_record_rejects_naive_scheduled_at():
    with pytest.raises(ValidationError):
        _finished_match(scheduled_at=NAIVE)


def test_match_record_rejects_same_player_both_sides():
    with pytest.raises(ValidationError):
        _finished_match(player_a_id="p1", player_b_id="p1")


def test_match_record_rejects_winner_not_a_participant():
    with pytest.raises(ValidationError):
        _finished_match(winner_id="someone-else")


def test_finished_match_requires_winner():
    with pytest.raises(ValidationError):
        _finished_match(status=MatchRecordStatus.FINISHED, winner_id=None)


def test_scheduled_match_must_not_have_winner():
    with pytest.raises(ValidationError):
        HistoricalMatchRecord(
            id="m1", provider="mock", provider_match_id="1", player_a_id="p1", player_b_id="p2",
            scheduled_at=AWARE, status=MatchRecordStatus.SCHEDULED, winner_id="p1",
            provider_timestamp=AWARE, ingested_at=AWARE,
        )


def test_scheduled_match_must_not_have_sets_or_start():
    with pytest.raises(ValidationError):
        HistoricalMatchRecord(
            id="m1", provider="mock", provider_match_id="1", player_a_id="p1", player_b_id="p2",
            scheduled_at=AWARE, status=MatchRecordStatus.SCHEDULED,
            sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
            provider_timestamp=AWARE, ingested_at=AWARE,
        )


def test_completion_cannot_precede_start():
    with pytest.raises(ValidationError):
        _finished_match(actual_start_at=AWARE, completed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))


def test_duplicate_set_numbers_rejected():
    with pytest.raises(ValidationError):
        _finished_match(
            sets=[
                HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7),
                HistoricalSetRecord(set_number=1, player_a_points=9, player_b_points=11),
            ]
        )


def test_blank_provider_id_rejected():
    with pytest.raises(ValidationError):
        _finished_match(provider_match_id="")


def test_effective_timestamp_prefers_completion_then_start_then_schedule():
    match = _finished_match()
    assert match.effective_timestamp == match.completed_at

    scheduled_only = HistoricalMatchRecord(
        id="m2", provider="mock", provider_match_id="2", player_a_id="p1", player_b_id="p2",
        scheduled_at=AWARE, status=MatchRecordStatus.SCHEDULED,
        provider_timestamp=AWARE, ingested_at=AWARE,
    )
    assert scheduled_only.effective_timestamp == AWARE


def test_odds_record_rejects_non_decimal_odds():
    with pytest.raises(ValidationError):
        HistoricalOddsRecord(
            id="o1", match_id="m1", bookmaker="Pinnacle", selection_id="p1",
            decimal_odds=1.0, captured_at=AWARE, provider="mock",
        )


def test_odds_record_rejects_naive_captured_at():
    with pytest.raises(ValidationError):
        HistoricalOddsRecord(
            id="o1", match_id="m1", bookmaker="Pinnacle", selection_id="p1",
            decimal_odds=1.8, captured_at=NAIVE, provider="mock",
        )
