"""Tests for ranking integration into rolling/matchup features and the feature builder."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from features.builder import HistoricalFeatureBuilder
from features.matchup import build_matchup_features
from features.player import build_player_rolling_features
from persistence.in_memory import InMemoryMatchRepository, InMemoryRankingRepository
from persistence.models import HistoricalMatchRecord, HistoricalRankingRecord, HistoricalSetRecord, MatchRecordStatus

AS_OF = datetime(2026, 2, 1, tzinfo=timezone.utc)


def test_ranking_is_none_by_default():
    features = build_player_rolling_features("p1", [], AS_OF)
    assert features.ranking is None


def test_ranking_is_set_when_supplied():
    features = build_player_rolling_features("p1", [], AS_OF, latest_ranking=7)
    assert features.ranking == 7


def test_matchup_ranking_differential_none_when_either_missing():
    a = build_player_rolling_features("p1", [], AS_OF, latest_ranking=5)
    b = build_player_rolling_features("p2", [], AS_OF)
    matchup = build_matchup_features("p1", "p2", [], a, b, AS_OF)
    assert matchup.ranking_differential is None


def test_matchup_ranking_differential_favors_better_ranked_player_a():
    a = build_player_rolling_features("p1", [], AS_OF, latest_ranking=3)
    b = build_player_rolling_features("p2", [], AS_OF, latest_ranking=50)
    matchup = build_matchup_features("p1", "p2", [], a, b, AS_OF)
    assert matchup.ranking_differential == 47.0


def _scheduled(mid, day) -> HistoricalMatchRecord:
    scheduled = AS_OF + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, status=MatchRecordStatus.SCHEDULED,
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def test_builder_uses_ranking_repository_when_supplied():
    match_repo = InMemoryMatchRepository()
    ranking_repo = InMemoryRankingRepository()
    target = _scheduled("target", 10)
    match_repo.add(target)
    ranking_repo.add(
        HistoricalRankingRecord(
            id="r1", player_id="p1", ranking=2, effective_at=AS_OF, provider="mock",
            provider_record_id="r1", ingested_at=AS_OF,
        )
    )
    builder = HistoricalFeatureBuilder(match_repo, ranking_repository=ranking_repo)
    snapshot = builder.build("target", target.scheduled_at)
    assert snapshot.player_a_features.ranking == 2
    assert snapshot.player_b_features.ranking is None


def test_builder_never_uses_ranking_at_or_after_as_of():
    match_repo = InMemoryMatchRepository()
    ranking_repo = InMemoryRankingRepository()
    target = _scheduled("target", 10)
    match_repo.add(target)
    # A ranking observation effective at/after the cutoff must never leak in.
    ranking_repo.add(
        HistoricalRankingRecord(
            id="r1", player_id="p1", ranking=1, effective_at=target.scheduled_at, provider="mock",
            provider_record_id="r1", ingested_at=target.scheduled_at,
        )
    )
    builder = HistoricalFeatureBuilder(match_repo, ranking_repository=ranking_repo)
    snapshot = builder.build("target", target.scheduled_at)
    assert snapshot.player_a_features.ranking is None


def test_builder_without_ranking_repository_leaves_ranking_none():
    match_repo = InMemoryMatchRepository()
    target = _scheduled("target", 10)
    match_repo.add(target)
    builder = HistoricalFeatureBuilder(match_repo)
    snapshot = builder.build("target", target.scheduled_at)
    assert snapshot.player_a_features.ranking is None
    assert snapshot.player_b_features.ranking is None
