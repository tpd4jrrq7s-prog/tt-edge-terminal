"""Tests for HistoricalRankingRecord/HistoricalCompetitionRecord and their in-memory repositories."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from persistence.errors import DuplicateRecordError, ProviderMappingConflictError
from persistence.in_memory import InMemoryCompetitionRepository, InMemoryRankingRepository
from persistence.models import HistoricalCompetitionRecord, HistoricalRankingRecord

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAIVE = datetime(2026, 1, 1)


def _ranking(rid="r1", player="p1", ranking=5, day=0) -> HistoricalRankingRecord:
    return HistoricalRankingRecord(
        id=rid, player_id=player, ranking=ranking, effective_at=AWARE + timedelta(days=day),
        provider="mock", provider_record_id=rid, ingested_at=AWARE + timedelta(days=day),
    )


def test_ranking_record_rejects_naive_effective_at():
    with pytest.raises(ValidationError):
        HistoricalRankingRecord(
            id="r1", player_id="p1", ranking=1, effective_at=NAIVE,
            provider="mock", provider_record_id="r1", ingested_at=AWARE,
        )


def test_ranking_repository_preserves_multiple_observations():
    repo = InMemoryRankingRepository()
    repo.add(_ranking("r1", day=0, ranking=10))
    repo.add(_ranking("r2", day=5, ranking=8))
    assert repo.count() == 2


def test_ranking_repository_latest_before_is_strict():
    repo = InMemoryRankingRepository()
    r = _ranking("r1", day=0, ranking=10)
    repo.add(r)
    assert repo.latest_before("p1", r.effective_at) is None
    assert repo.latest_before("p1", r.effective_at + timedelta(seconds=1)).ranking == 10


def test_ranking_repository_latest_before_picks_most_recent():
    repo = InMemoryRankingRepository()
    repo.add(_ranking("r1", day=0, ranking=10))
    repo.add(_ranking("r2", day=5, ranking=8))
    cutoff = AWARE + timedelta(days=10)
    latest = repo.latest_before("p1", cutoff)
    assert latest.ranking == 8


def test_ranking_repository_duplicate_id_rejected():
    repo = InMemoryRankingRepository()
    repo.add(_ranking("r1"))
    with pytest.raises(DuplicateRecordError):
        repo.add(_ranking("r1"))


def _competition(cid="c1", provider_id="pc1") -> HistoricalCompetitionRecord:
    return HistoricalCompetitionRecord(
        id=cid, provider="mock", provider_competition_id=provider_id, name="Demo Open",
        ingested_at=AWARE,
    )


def test_competition_record_rejects_active_to_before_active_from():
    with pytest.raises(ValidationError):
        HistoricalCompetitionRecord(
            id="c1", provider="mock", provider_competition_id="pc1", name="Demo",
            active_from=AWARE, active_to=AWARE - timedelta(days=1), ingested_at=AWARE,
        )


def test_competition_repository_add_and_get():
    repo = InMemoryCompetitionRepository()
    repo.add(_competition())
    assert repo.get("c1").name == "Demo Open"


def test_competition_repository_provider_mapping_conflict():
    repo = InMemoryCompetitionRepository()
    repo.add(_competition("c1", "pc1"))
    with pytest.raises(ProviderMappingConflictError):
        repo.add(_competition("c2", "pc1"))


def test_competition_repository_has_provider_id():
    repo = InMemoryCompetitionRepository()
    repo.add(_competition())
    assert repo.has_provider_id("mock", "pc1") is True
    assert repo.has_provider_id("mock", "nope") is False
