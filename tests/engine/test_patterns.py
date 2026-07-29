"""Tests for the pattern detection engine."""

from __future__ import annotations

from datetime import datetime, timezone

from config.analytics import AnalyticsSettings
from engine.models import HistoricalMatch, PointEvent, SetResult
from engine.patterns import detect_patterns

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _match(sets, won, **overrides) -> HistoricalMatch:
    defaults = dict(player_id="p1", opponent_id="opp", played_at=AS_OF, won=won, sets=sets)
    defaults.update(overrides)
    return HistoricalMatch(**defaults)


def test_no_history_produces_no_patterns():
    result = detect_patterns("p1", "player_one", [], [], AnalyticsSettings())
    assert result == []


def test_slow_starter_detected_when_first_set_usually_lost():
    history = [
        _match([SetResult(player_points=5, opponent_points=11), SetResult(player_points=11, opponent_points=8), SetResult(player_points=11, opponent_points=6)], won=True)
        for _ in range(4)
    ]
    signals = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    slow_starter = next(s for s in signals if s.pattern == "slow_starter")
    assert slow_starter.strength == 1.0
    assert slow_starter.sample_size == 4


def test_frequent_comeback_pattern_requires_at_least_one_first_set_loss():
    always_won_first = [
        _match([SetResult(player_points=11, opponent_points=5), SetResult(player_points=11, opponent_points=6)], won=True)
        for _ in range(3)
    ]
    signals = detect_patterns("p1", "player_one", always_won_first, [], AnalyticsSettings())
    assert not any(s.pattern == "frequent_comeback" for s in signals)


def test_frequent_comeback_pattern_detected():
    history = [
        _match(
            [SetResult(player_points=5, opponent_points=11), SetResult(player_points=11, opponent_points=8), SetResult(player_points=11, opponent_points=6)],
            won=True,
        )
        for _ in range(3)
    ]
    signals = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    comeback = next(s for s in signals if s.pattern == "frequent_comeback")
    assert comeback.strength == 1.0


def test_straight_sets_tendency_pattern():
    history = [
        _match([SetResult(player_points=11, opponent_points=3), SetResult(player_points=11, opponent_points=5)], won=True)
        for _ in range(3)
    ]
    signals = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    straight = next(s for s in signals if s.pattern == "straight_sets_tendency")
    assert straight.strength == 1.0


def test_insufficient_sample_size_produces_low_confidence():
    settings = AnalyticsSettings(min_history_sample_size=10)
    history = [
        _match([SetResult(player_points=11, opponent_points=5), SetResult(player_points=11, opponent_points=6)], won=True)
    ]
    signals = detect_patterns("p1", "player_one", history, [], settings)
    assert all(s.confidence < 1.0 for s in signals)


def test_high_volatility_requires_at_least_two_matches():
    history = [_match([SetResult(player_points=11, opponent_points=5)], won=True)]
    signals = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    assert not any(s.pattern == "high_volatility" for s in signals)


def test_pressure_point_pattern_from_current_match_progression():
    events = [
        PointEvent(set_number=1, winner="player_one", player_one_score=9, player_two_score=9),
        PointEvent(set_number=1, winner="player_one", player_one_score=10, player_two_score=9),
        PointEvent(set_number=1, winner="player_two", player_one_score=10, player_two_score=10),
    ]
    signals = detect_patterns("p1", "player_one", [], events, AnalyticsSettings())
    pressure = next(s for s in signals if s.pattern == "strong_pressure_point_performance")
    assert pressure.sample_size == 3
    assert pressure.strength == 2 / 3


def test_pressure_point_pattern_absent_without_progression():
    signals = detect_patterns("p1", "player_one", [], [], AnalyticsSettings())
    assert not any(s.pattern == "strong_pressure_point_performance" for s in signals)


def test_patterns_are_deterministic():
    history = [
        _match([SetResult(player_points=11, opponent_points=5), SetResult(player_points=8, opponent_points=11), SetResult(player_points=11, opponent_points=9)], won=True)
        for _ in range(3)
    ]
    result_one = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    result_two = detect_patterns("p1", "player_one", history, [], AnalyticsSettings())
    assert result_one == result_two
