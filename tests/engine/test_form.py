"""Tests for the form engine."""

from __future__ import annotations

from datetime import datetime, timezone

from config.analytics import AnalyticsSettings
from engine.form import (
    calculate_form,
    completion_weight,
    match_performance_score,
    opponent_strength_weight,
    recency_weight,
)
from engine.models import HistoricalMatch, SetResult

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _match(**overrides) -> HistoricalMatch:
    defaults = dict(
        player_id="p1",
        opponent_id="opp",
        played_at=AS_OF,
        won=True,
        sets=[SetResult(player_points=11, opponent_points=7)],
    )
    defaults.update(overrides)
    return HistoricalMatch(**defaults)


def test_calculate_form_with_no_history_returns_neutral_and_zero_confidence():
    result = calculate_form([], AS_OF, AnalyticsSettings())
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert result.matches_considered == 0


def test_calculate_form_is_deterministic():
    history = [_match(), _match(won=False, sets=[SetResult(player_points=5, opponent_points=11)])]
    result_one = calculate_form(history, AS_OF, AnalyticsSettings())
    result_two = calculate_form(history, AS_OF, AnalyticsSettings())
    assert result_one == result_two


def test_calculate_form_insufficient_history_has_lower_confidence():
    settings = AnalyticsSettings(min_history_sample_size=10)
    result = calculate_form([_match()], AS_OF, settings)
    assert result.confidence < 1.0


def test_calculate_form_more_history_increases_confidence():
    settings = AnalyticsSettings(min_history_sample_size=5)
    few = calculate_form([_match()], AS_OF, settings)
    many = calculate_form([_match() for _ in range(5)], AS_OF, settings)
    assert many.confidence > few.confidence


def test_match_performance_score_win_beats_loss():
    win = _match(won=True, sets=[SetResult(player_points=11, opponent_points=2)])
    loss = _match(won=False, sets=[SetResult(player_points=2, opponent_points=11)])
    assert match_performance_score(win) > match_performance_score(loss)


def test_match_performance_score_dominant_win_beats_narrow_win():
    dominant = _match(won=True, sets=[SetResult(player_points=11, opponent_points=1)])
    narrow = _match(won=True, sets=[SetResult(player_points=11, opponent_points=9)])
    assert match_performance_score(dominant) >= match_performance_score(narrow)


def test_match_performance_score_walkover_win_is_100():
    walkover = _match(won=True, walkover=True, sets=[])
    assert match_performance_score(walkover) == 100.0


def test_match_performance_score_walkover_loss_is_0():
    walkover = _match(won=False, walkover=True, sets=[])
    assert match_performance_score(walkover) == 0.0


def test_recency_weight_newer_match_weighs_more():
    older = recency_weight(datetime(2026, 1, 1, tzinfo=timezone.utc), AS_OF, half_life_days=90.0)
    newer = recency_weight(datetime(2026, 7, 20, tzinfo=timezone.utc), AS_OF, half_life_days=90.0)
    assert newer > older


def test_recency_weight_at_reference_time_is_one():
    assert recency_weight(AS_OF, AS_OF, half_life_days=90.0) == 1.0


def test_recency_weight_halves_at_half_life():
    half_life = 30.0
    past = datetime(2026, 6, 29, tzinfo=timezone.utc)  # 30 days before AS_OF
    weight = recency_weight(past, AS_OF, half_life_days=half_life)
    assert abs(weight - 0.5) < 1e-6


def test_opponent_strength_weight_none_is_neutral():
    assert opponent_strength_weight(None) == 1.0


def test_opponent_strength_weight_top_ranked_opponent_weighs_more():
    assert opponent_strength_weight(1) > opponent_strength_weight(150)


def test_completion_weight_walkover_and_retired():
    assert completion_weight(_match(walkover=True)) == 0.3
    assert completion_weight(_match(retired=True)) == 0.5
    assert completion_weight(_match()) == 1.0


def test_newer_matches_are_weighted_more_in_form_calculation():
    settings = AnalyticsSettings()
    recent_win = _match(played_at=datetime(2026, 7, 28, tzinfo=timezone.utc), won=True)
    old_loss = _match(
        played_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        won=False,
        sets=[SetResult(player_points=2, opponent_points=11)],
    )
    result = calculate_form([recent_win, old_loss], AS_OF, settings)
    # The recent win should dominate the weighted average, well above neutral.
    assert result.score > 60.0
