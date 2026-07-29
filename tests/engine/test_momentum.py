"""Tests for the momentum engine."""

from __future__ import annotations

from datetime import datetime, timezone

from config.analytics import AnalyticsSettings
from engine.models import HistoricalMatch, MomentumState, PointEvent, SetResult
from engine.momentum import calculate_momentum

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def test_no_data_returns_neutral_scores_and_zero_confidence():
    result = calculate_momentum([], [], [], AS_OF, AnalyticsSettings())
    assert result.state is MomentumState.NO_DATA
    assert result.player_one_score == 50.0
    assert result.player_two_score == 50.0
    assert result.confidence == 0.0


def test_pre_match_state_used_when_only_history_available():
    history = [
        HistoricalMatch(
            player_id="p1", opponent_id="o1", played_at=AS_OF, won=True,
            sets=[SetResult(player_points=11, opponent_points=5)],
        )
    ]
    result = calculate_momentum([], history, [], AS_OF, AnalyticsSettings())
    assert result.state is MomentumState.PRE_MATCH


def test_in_play_state_used_when_point_progression_available():
    events = [
        PointEvent(set_number=1, winner="player_one", player_one_score=1, player_two_score=0),
        PointEvent(set_number=1, winner="player_one", player_one_score=2, player_two_score=0),
    ]
    result = calculate_momentum(events, [], [], AS_OF, AnalyticsSettings())
    assert result.state is MomentumState.IN_PLAY


def test_in_play_momentum_favors_player_on_a_point_streak():
    events = [
        PointEvent(set_number=1, winner="player_two", player_one_score=0, player_two_score=1),
        PointEvent(set_number=1, winner="player_one", player_one_score=1, player_two_score=1),
        PointEvent(set_number=1, winner="player_one", player_one_score=2, player_two_score=1),
        PointEvent(set_number=1, winner="player_one", player_one_score=3, player_two_score=1),
        PointEvent(set_number=1, winner="player_one", player_one_score=4, player_two_score=1),
    ]
    result = calculate_momentum(events, [], [], AS_OF, AnalyticsSettings())
    assert result.player_one_score > result.player_two_score


def test_momentum_scores_are_complementary_in_play():
    events = [
        PointEvent(set_number=1, winner="player_one", player_one_score=1, player_two_score=0),
        PointEvent(set_number=1, winner="player_two", player_one_score=1, player_two_score=1),
    ]
    result = calculate_momentum(events, [], [], AS_OF, AnalyticsSettings())
    assert abs(result.player_one_score + result.player_two_score - 100.0) < 1e-9


def test_short_streak_does_not_produce_high_confidence():
    settings = AnalyticsSettings(momentum_min_points=50)
    events = [
        PointEvent(set_number=1, winner="player_one", player_one_score=1, player_two_score=0),
        PointEvent(set_number=1, winner="player_one", player_one_score=2, player_two_score=0),
    ]
    result = calculate_momentum(events, [], [], AS_OF, settings)
    assert result.confidence < 0.2


def test_many_lead_changes_reduce_confidence():
    settings = AnalyticsSettings(momentum_min_points=1)
    volatile_events = []
    for i in range(20):
        if i % 2 == 0:
            volatile_events.append(
                PointEvent(set_number=1, winner="player_one", player_one_score=i + 1, player_two_score=i)
            )
        else:
            volatile_events.append(
                PointEvent(set_number=1, winner="player_two", player_one_score=i, player_two_score=i + 1)
            )
    stable_events = [
        PointEvent(set_number=1, winner="player_one", player_one_score=i + 1, player_two_score=0)
        for i in range(20)
    ]
    volatile_result = calculate_momentum(volatile_events, [], [], AS_OF, settings)
    stable_result = calculate_momentum(stable_events, [], [], AS_OF, settings)
    assert volatile_result.confidence < stable_result.confidence


def test_momentum_is_deterministic():
    history_one = [
        HistoricalMatch(
            player_id="p1", opponent_id="o1", played_at=AS_OF, won=True,
            sets=[SetResult(player_points=11, opponent_points=5)],
        )
    ]
    result_a = calculate_momentum([], history_one, [], AS_OF, AnalyticsSettings())
    result_b = calculate_momentum([], history_one, [], AS_OF, AnalyticsSettings())
    assert result_a == result_b
