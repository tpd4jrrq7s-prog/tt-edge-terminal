"""Tests for the in-memory repository implementations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from persistence.errors import DuplicateRecordError, ProviderMappingConflictError
from persistence.in_memory import InMemoryMatchRepository, InMemoryOddsRepository, InMemoryPlayerRepository
from persistence.models import HistoricalMatchRecord, HistoricalOddsRecord, HistoricalPlayerRecord, MatchRecordStatus

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _player(pid="p1", provider_id="1") -> HistoricalPlayerRecord:
    return HistoricalPlayerRecord(id=pid, name="A", provider="mock", provider_player_id=provider_id, ingested_at=AWARE)


def _match(mid, day, winner=None) -> HistoricalMatchRecord:
    scheduled = AWARE + timedelta(days=day)
    is_scheduled = winner is None
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled,
        actual_start_at=None if is_scheduled else scheduled,
        completed_at=None if is_scheduled else scheduled + timedelta(hours=1),
        status=MatchRecordStatus.SCHEDULED if is_scheduled else MatchRecordStatus.FINISHED,
        winner_id=winner,
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


class TestPlayerRepository:
    def test_add_and_get(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player())
        assert repo.get("p1").name == "A"
        assert repo.count() == 1

    def test_get_missing_returns_none(self):
        repo = InMemoryPlayerRepository()
        assert repo.get("nope") is None

    def test_duplicate_id_rejected(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player())
        with pytest.raises(DuplicateRecordError):
            repo.add(_player())

    def test_provider_mapping_conflict_rejected(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player(pid="p1", provider_id="1"))
        with pytest.raises(ProviderMappingConflictError):
            repo.add(_player(pid="p2", provider_id="1"))

    def test_has_provider_id(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player())
        assert repo.has_provider_id("mock", "1") is True
        assert repo.has_provider_id("mock", "999") is False

    def test_get_returns_defensive_copy(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player())
        fetched = repo.get("p1")
        fetched.name = "MUTATED"
        assert repo.get("p1").name == "A"

    def test_clear(self):
        repo = InMemoryPlayerRepository()
        repo.add(_player())
        repo.clear()
        assert repo.count() == 0
        assert repo.get("p1") is None


class TestMatchRepository:
    def test_add_and_get(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m1", 0, winner="p1"))
        assert repo.get("m1") is not None
        assert repo.count() == 1

    def test_duplicate_id_rejected(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m1", 0, winner="p1"))
        with pytest.raises(DuplicateRecordError):
            repo.add(_match("m1", 1, winner="p1"))

    def test_provider_mapping_conflict_rejected(self):
        repo = InMemoryMatchRepository()
        m1 = _match("m1", 0, winner="p1")
        m2 = m1.model_copy(update={"id": "m2"})  # same provider_match_id as m1
        repo.add(m1)
        with pytest.raises(ProviderMappingConflictError):
            repo.add(m2)

    def test_strict_before_cutoff_excludes_equal_timestamp(self):
        repo = InMemoryMatchRepository()
        match = _match("m1", 0, winner="p1")
        repo.add(match)
        cutoff = match.effective_timestamp
        assert repo.list_player_matches_before("p1", cutoff) == []

    def test_strict_before_cutoff_includes_earlier_matches(self):
        repo = InMemoryMatchRepository()
        match = _match("m1", 0, winner="p1")
        repo.add(match)
        cutoff = match.effective_timestamp + timedelta(seconds=1)
        assert len(repo.list_player_matches_before("p1", cutoff)) == 1

    def test_list_player_matches_deterministic_order(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m3", 3, winner="p1"))
        repo.add(_match("m1", 1, winner="p2"))
        repo.add(_match("m2", 2, winner="p1"))
        cutoff = AWARE + timedelta(days=10)
        matches = repo.list_player_matches_before("p1", cutoff)
        assert [m.id for m in matches] == ["m1", "m2", "m3"]

    def test_head_to_head_only_returns_matches_between_the_pair(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m1", 0, winner="p1"))
        other = _match("m2", 1, winner="p1").model_copy(update={"player_a_id": "p1", "player_b_id": "p3", "provider_match_id": "m2"})
        repo.add(other)
        cutoff = AWARE + timedelta(days=10)
        h2h = repo.list_head_to_head_before("p1", "p2", cutoff)
        assert [m.id for m in h2h] == ["m1"]

    def test_get_returns_defensive_copy(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m1", 0, winner="p1"))
        fetched = repo.get("m1")
        fetched.winner_id = "p2"
        assert repo.get("m1").winner_id == "p1"

    def test_clear(self):
        repo = InMemoryMatchRepository()
        repo.add(_match("m1", 0, winner="p1"))
        repo.clear()
        assert repo.count() == 0


class TestOddsRepository:
    def _odds(self, oid, captured_offset_minutes, decimal_odds=1.8) -> HistoricalOddsRecord:
        return HistoricalOddsRecord(
            id=oid, match_id="m1", bookmaker="Pinnacle", selection_id="p1",
            decimal_odds=decimal_odds, captured_at=AWARE + timedelta(minutes=captured_offset_minutes), provider="mock",
        )

    def test_add_and_get(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 0))
        assert repo.get("o1") is not None

    def test_duplicate_id_rejected(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 0))
        with pytest.raises(DuplicateRecordError):
            repo.add(self._odds("o1", 5))

    def test_exact_duplicate_observation_rejected(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 0))
        with pytest.raises(DuplicateRecordError):
            repo.add(self._odds("o2", 0))  # same match/bookmaker/selection/captured_at

    def test_multiple_distinct_timestamps_preserved(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 0, decimal_odds=1.8))
        repo.add(self._odds("o2", 5, decimal_odds=1.9))
        assert repo.count() == 2

    def test_older_observation_not_overwritten_by_newer(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 0, decimal_odds=1.8))
        repo.add(self._odds("o2", 5, decimal_odds=1.9))
        assert repo.get("o1").decimal_odds == 1.8
        assert repo.get("o2").decimal_odds == 1.9

    def test_list_at_or_before_is_inclusive(self):
        repo = InMemoryOddsRepository()
        odds = self._odds("o1", 0)
        repo.add(odds)
        results = repo.list_for_match_at_or_before("m1", odds.captured_at)
        assert len(results) == 1

    def test_list_excludes_odds_after_cutoff(self):
        repo = InMemoryOddsRepository()
        repo.add(self._odds("o1", 10))
        results = repo.list_for_match_at_or_before("m1", AWARE)
        assert results == []
