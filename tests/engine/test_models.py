"""Tests for engine input model validation (SetResult, HistoricalMatch, MatchAnalysisRequest)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from domain.match import Match, MatchStatus
from domain.odds import Odds
from domain.player import Player
from engine.models import HistoricalMatch, MatchAnalysisRequest, SetResult


def test_set_result_rejects_tied_points():
    with pytest.raises(ValidationError):
        SetResult(player_points=10, opponent_points=10)


def test_set_result_properties():
    s = SetResult(player_points=11, opponent_points=7)
    assert s.won is True
    assert s.margin == 4
    assert s.total_points == 18


def _historical_match(**overrides) -> HistoricalMatch:
    defaults = dict(
        player_id="p1",
        opponent_id="opp1",
        played_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        won=True,
        sets=[SetResult(player_points=11, opponent_points=7)],
    )
    defaults.update(overrides)
    return HistoricalMatch(**defaults)


def test_historical_match_rejects_same_player_and_opponent():
    with pytest.raises(ValidationError):
        _historical_match(player_id="p1", opponent_id="p1")


def test_historical_match_rejects_sets_contradicting_won_true():
    with pytest.raises(ValidationError):
        _historical_match(won=True, sets=[SetResult(player_points=5, opponent_points=11)])


def test_historical_match_rejects_sets_contradicting_won_false():
    with pytest.raises(ValidationError):
        _historical_match(won=False, sets=[SetResult(player_points=11, opponent_points=5)])


def test_historical_match_walkover_does_not_require_consistent_sets():
    match = _historical_match(won=True, walkover=True, sets=[])
    assert match.sets_won == 0
    assert match.sets_lost == 0


def test_historical_match_derived_properties():
    match = _historical_match(
        won=True,
        sets=[SetResult(player_points=11, opponent_points=7), SetResult(player_points=9, opponent_points=11)],
    )
    assert match.sets_won == 1
    assert match.sets_lost == 1
    assert match.point_margin == (11 - 7) + (9 - 11)


def _match() -> Match:
    return Match(
        id="m1",
        player_one=Player(id="p1", name="Ma Long"),
        player_two=Player(id="p2", name="Fan Zhendong"),
        status=MatchStatus.SCHEDULED,
    )


def test_match_analysis_request_accepts_minimal_input():
    request = MatchAnalysisRequest(match=_match())
    assert request.odds == []
    assert request.player_one_history == []


def test_match_analysis_request_rejects_odds_for_a_different_match():
    bad_odds = Odds(match_id="different-match", bookmaker="Pinnacle", player_one_odds=1.5, player_two_odds=2.5)
    with pytest.raises(ValidationError):
        MatchAnalysisRequest(match=_match(), odds=[bad_odds])


def test_match_analysis_request_rejects_history_for_wrong_player():
    bad_history = [_historical_match(player_id="someone-else")]
    with pytest.raises(ValidationError):
        MatchAnalysisRequest(match=_match(), player_one_history=bad_history)
