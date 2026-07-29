"""Tests for the data quality engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.analytics import AnalyticsSettings
from domain.match import Match, MatchStatus, SetScore
from domain.odds import Odds
from domain.player import Player
from engine.models import HeadToHeadRecord, HistoricalMatch, IssueSeverity, SetResult
from engine.quality import assess_data_quality

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _match(status=MatchStatus.SCHEDULED, sets=None) -> Match:
    return Match(
        id="m1",
        player_one=Player(id="p1", name="Ma Long"),
        player_two=Player(id="p2", name="Fan Zhendong"),
        status=status,
        sets=sets or [],
    )


def test_perfect_data_scores_highly():
    match = _match()
    odds = [Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.8, player_two_odds=2.1, captured_at=AS_OF)]
    history_one = [
        HistoricalMatch(player_id="p1", opponent_id=f"opp-{i}", played_at=AS_OF, won=True) for i in range(3)
    ]
    history_two = [
        HistoricalMatch(player_id="p2", opponent_id=f"opp-{i}", played_at=AS_OF, won=True) for i in range(3)
    ]
    result = assess_data_quality(match, odds, history_one, history_two, None, AS_OF, AnalyticsSettings())
    assert result.score == 100.0
    assert result.warnings == []


def test_missing_odds_reduces_score_and_flags_warning():
    match = _match()
    result = assess_data_quality(match, [], [], [], None, AS_OF, AnalyticsSettings())
    assert result.odds_available is False
    assert any(w.field == "odds" for w in result.warnings)
    assert result.score < 100.0


def test_stale_odds_flagged_as_not_fresh():
    match = _match()
    old_odds = [
        Odds(
            match_id="m1", bookmaker="Pinnacle", player_one_odds=1.8, player_two_odds=2.1,
            captured_at=AS_OF - timedelta(hours=5),
        )
    ]
    result = assess_data_quality(match, old_odds, [], [], None, AS_OF, AnalyticsSettings())
    assert result.odds_available is True
    assert result.odds_fresh is False


def test_finished_match_without_sets_is_a_warning():
    match = _match(status=MatchStatus.FINISHED, sets=[])
    result = assess_data_quality(match, [], [], [], None, AS_OF, AnalyticsSettings())
    assert any(w.severity == IssueSeverity.WARNING and w.field == "match.sets" for w in result.warnings)


def test_finished_match_with_tied_sets_is_critical():
    match = _match(
        status=MatchStatus.FINISHED,
        sets=[SetScore(player_one_points=11, player_two_points=5), SetScore(player_one_points=5, player_two_points=11)],
    )
    result = assess_data_quality(match, [], [], [], None, AS_OF, AnalyticsSettings())
    assert any(w.severity == IssueSeverity.CRITICAL for w in result.warnings)


def test_below_minimum_sample_size_is_flagged():
    match = _match()
    settings = AnalyticsSettings(min_history_sample_size=5)
    history = [HistoricalMatch(player_id="p1", opponent_id="x", played_at=AS_OF, won=True)]
    result = assess_data_quality(match, [], history, [], None, AS_OF, settings)
    assert any("below the 5-match minimum" in w.detail for w in result.warnings)


def test_duplicate_historical_observations_are_flagged():
    match = _match()
    entry = HistoricalMatch(player_id="p1", opponent_id="x", played_at=AS_OF, won=True)
    result = assess_data_quality(match, [], [entry, entry], [], None, AS_OF, AnalyticsSettings())
    assert any("Duplicate" in w.detail for w in result.warnings)


def test_head_to_head_inconsistent_with_history_is_flagged():
    match = _match()
    history = [HistoricalMatch(player_id="p1", opponent_id="p2", played_at=AS_OF, won=True) for _ in range(3)]
    h2h = HeadToHeadRecord(player_one_wins=1, player_two_wins=0)
    result = assess_data_quality(match, [], history, [], h2h, AS_OF, AnalyticsSettings())
    assert any("Head-to-head" in w.detail for w in result.warnings)


def test_history_sample_sizes_are_reported():
    match = _match()
    history_one = [HistoricalMatch(player_id="p1", opponent_id="x", played_at=AS_OF, won=True)]
    result = assess_data_quality(match, [], history_one, [], None, AS_OF, AnalyticsSettings())
    assert result.history_sample_size_player_one == 1
    assert result.history_sample_size_player_two == 0
