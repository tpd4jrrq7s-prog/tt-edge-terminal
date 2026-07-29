"""Tests for player rolling features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config.historical import HistoricalIntelligenceSettings
from features.player import build_player_rolling_features
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

AS_OF = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _match(mid: str, day: int, won: bool, sets: list[tuple[int, int]] | None = None) -> HistoricalMatchRecord:
    scheduled = AS_OF - timedelta(days=day)
    winner = "p1" if won else "p2"
    default_sets = sets or ([(11, 7), (11, 9)] if won else [(7, 11), (9, 11)])
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=i + 1, player_a_points=a, player_b_points=b) for i, (a, b) in enumerate(default_sets)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def test_no_history_returns_empty_windows():
    features = build_player_rolling_features("p1", [], AS_OF, HistoricalIntelligenceSettings())
    assert features.observation_count == 0
    all_time = features.window("all_time")
    assert all_time.matches_played == 0
    assert all_time.win_rate is None
    assert features.result_streak == 0
    assert features.recency_weighted_win_rate is None
    assert features.rest_hours_since_previous_match is None


def test_win_rate_and_counts():
    history = [_match("m1", 10, won=True), _match("m2", 20, won=False)]
    features = build_player_rolling_features("p1", history, AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.matches_played == 2
    assert all_time.wins == 1
    assert all_time.losses == 1
    assert all_time.win_rate == 0.5


def test_windows_select_most_recent_n():
    history = [_match(f"m{i}", i, won=(i % 2 == 0)) for i in range(1, 8)]
    settings = HistoricalIntelligenceSettings(rolling_window_sizes=[3])
    features = build_player_rolling_features("p1", history, AS_OF, settings)
    last_3 = features.window("last_3")
    assert last_3.matches_played == 3
    all_time = features.window("all_time")
    assert all_time.matches_played == 7


def test_set_and_point_rates():
    history = [_match("m1", 1, won=True, sets=[(11, 5), (11, 5)])]
    features = build_player_rolling_features("p1", history, AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.sets_won == 2
    assert all_time.sets_lost == 0
    assert all_time.set_win_rate == 1.0
    assert all_time.points_won == 22
    assert all_time.points_lost == 10
    assert all_time.point_win_rate == pytest.approx(22 / 32)


def test_straight_sets_win_rate():
    straight = _match("m1", 1, won=True, sets=[(11, 5), (11, 5)])
    non_straight = _match("m2", 2, won=True, sets=[(11, 5), (9, 11), (11, 5)])
    features = build_player_rolling_features("p1", [straight, non_straight], AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.straight_sets_win_rate == 0.5
    assert all_time.straight_sets_win_rate_n == 2


def test_deciding_set_features():
    decider_win = _match("m1", 1, won=True, sets=[(11, 5), (7, 11), (11, 9)])
    decider_loss = _match("m2", 2, won=False, sets=[(11, 5), (7, 11), (9, 11)])
    features = build_player_rolling_features("p1", [decider_win, decider_loss], AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.deciding_set_appearance_rate == 1.0
    assert all_time.deciding_set_win_rate == 0.5
    assert all_time.deciding_set_win_rate_n == 2


def test_first_set_and_comeback_features():
    comeback = _match("m1", 1, won=True, sets=[(5, 11), (11, 8), (11, 6)])
    features = build_player_rolling_features("p1", [comeback], AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.first_set_win_rate == 0.0
    assert all_time.comeback_win_rate == 1.0
    assert all_time.comeback_win_rate_n == 1
    assert all_time.loss_rate_after_winning_first_set is None
    assert all_time.loss_rate_after_winning_first_set_n == 0


def test_loss_after_winning_first_set():
    blown_lead = _match("m1", 1, won=False, sets=[(11, 5), (5, 11), (7, 11)])
    features = build_player_rolling_features("p1", [blown_lead], AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.loss_rate_after_winning_first_set == 1.0
    assert all_time.comeback_win_rate is None


def test_rest_and_workload_windows():
    history = [_match("m1", 0.5, won=True), _match("m2", 3, won=True), _match("m3", 10, won=True)]
    # Use days as ints since HistoricalMatchRecord takes timedelta(days=...) with floats fine.
    features = build_player_rolling_features("p1", history, AS_OF, HistoricalIntelligenceSettings())
    assert features.rest_hours_since_previous_match is not None
    assert features.matches_in_previous_7d >= 2


def test_result_streak_win_and_loss():
    win_streak = [_match("m1", 1, won=True), _match("m2", 2, won=True), _match("m3", 3, won=False)]
    features = build_player_rolling_features("p1", win_streak, AS_OF, HistoricalIntelligenceSettings())
    assert features.result_streak == 2

    loss_streak = [_match("m1", 1, won=False), _match("m2", 2, won=False), _match("m3", 3, won=True)]
    features2 = build_player_rolling_features("p1", loss_streak, AS_OF, HistoricalIntelligenceSettings())
    assert features2.result_streak == -2


def test_recency_weighted_win_rate_favors_recent_matches():
    recent_win_old_loss = [
        _match("m1", 1, won=True),
        _match("m2", 300, won=False),
    ]
    features = build_player_rolling_features("p1", recent_win_old_loss, AS_OF, HistoricalIntelligenceSettings())
    assert features.recency_weighted_win_rate > 0.5


def test_volatility_requires_at_least_two_completed_matches():
    single = [_match("m1", 1, won=True)]
    features = build_player_rolling_features("p1", single, AS_OF, HistoricalIntelligenceSettings())
    assert features.volatility_score is None

    multiple = [_match("m1", 1, won=True), _match("m2", 2, won=False)]
    features2 = build_player_rolling_features("p1", multiple, AS_OF, HistoricalIntelligenceSettings())
    assert features2.volatility_score is not None


def test_incomplete_matches_are_counted_and_excluded_from_win_rate():
    cancelled = HistoricalMatchRecord(
        id="m1", provider="mock", provider_match_id="m1", player_a_id="p1", player_b_id="p2",
        scheduled_at=AS_OF - timedelta(days=1), status=MatchRecordStatus.CANCELLED,
        provider_timestamp=AS_OF - timedelta(days=1), ingested_at=AS_OF - timedelta(days=1),
    )
    completed = _match("m2", 2, won=True)
    features = build_player_rolling_features("p1", [cancelled, completed], AS_OF, HistoricalIntelligenceSettings())
    all_time = features.window("all_time")
    assert all_time.matches_played == 2
    assert all_time.incomplete_match_count == 1
    assert all_time.wins == 1
    assert all_time.win_rate == 1.0


def test_features_are_deterministic():
    history = [_match("m1", 1, won=True), _match("m2", 2, won=False)]
    a = build_player_rolling_features("p1", history, AS_OF, HistoricalIntelligenceSettings())
    b = build_player_rolling_features("p1", history, AS_OF, HistoricalIntelligenceSettings())
    assert a == b
