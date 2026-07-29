"""Tests for head-to-head matchup features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config.historical import HistoricalIntelligenceSettings
from features.matchup import build_matchup_features
from features.player import build_player_rolling_features
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

AS_OF = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _match(mid, day, winner, competition_id="comp-1", best_of=5, sets=None) -> HistoricalMatchRecord:
    scheduled = AS_OF - timedelta(days=day)
    default_sets = sets or ([(11, 7), (11, 9)] if winner == "p1" else [(7, 11), (9, 11)])
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        competition_id=competition_id, best_of=best_of,
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=i + 1, player_a_points=a, player_b_points=b) for i, (a, b) in enumerate(default_sets)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _empty_player_features(pid):
    return build_player_rolling_features(pid, [], AS_OF, HistoricalIntelligenceSettings())


def test_no_head_to_head_history_is_all_none_not_fabricated_5050():
    features = build_matchup_features(
        "p1", "p2", [], _empty_player_features("p1"), _empty_player_features("p2"), AS_OF
    )
    assert features.head_to_head_matches == 0
    assert features.player_a_win_rate is None
    assert features.player_b_win_rate is None
    assert features.recent_head_to_head_win_rate_player_a is None


def test_head_to_head_win_rates_are_complementary():
    history = [_match("m1", 1, "p1"), _match("m2", 2, "p2"), _match("m3", 3, "p1")]
    features = build_matchup_features(
        "p1", "p2", history, _empty_player_features("p1"), _empty_player_features("p2"), AS_OF
    )
    assert features.head_to_head_matches == 3
    assert features.player_a_win_rate == pytest.approx(2 / 3)
    assert features.player_b_win_rate == pytest.approx(1 / 3)


def test_days_since_last_meeting():
    history = [_match("m1", 5, "p1")]
    features = build_matchup_features(
        "p1", "p2", history, _empty_player_features("p1"), _empty_player_features("p2"), AS_OF
    )
    # effective_timestamp uses completed_at, which is scheduled_at + 1 hour.
    assert features.days_since_last_meeting == pytest.approx(5.0 - 1 / 24)


def test_deciding_set_h2h():
    history = [_match("m1", 1, "p1", sets=[(11, 5), (7, 11), (11, 9)])]
    features = build_matchup_features(
        "p1", "p2", history, _empty_player_features("p1"), _empty_player_features("p2"), AS_OF
    )
    assert features.deciding_set_h2h_matches == 1
    assert features.deciding_set_h2h_win_rate_player_a == 1.0


def test_competition_specific_h2h_filters_correctly():
    history = [_match("m1", 1, "p1", competition_id="comp-a"), _match("m2", 2, "p2", competition_id="comp-b")]
    features = build_matchup_features(
        "p1", "p2", history, _empty_player_features("p1"), _empty_player_features("p2"), AS_OF,
        target_competition_id="comp-a",
    )
    assert features.competition_specific_h2h_matches == 1
    assert features.competition_specific_h2h_win_rate_player_a == 1.0


def test_format_specific_h2h_filters_correctly():
    history = [_match("m1", 1, "p1", best_of=5), _match("m2", 2, "p2", best_of=3)]
    features = build_matchup_features(
        "p1", "p2", history, _empty_player_features("p1"), _empty_player_features("p2"), AS_OF,
        target_best_of=5,
    )
    assert features.format_specific_h2h_matches == 1
    assert features.format_specific_h2h_win_rate_player_a == 1.0


def test_workload_and_form_differential_use_player_features():
    a_features = build_player_rolling_features("p1", [_match("m1", 1, "p1")], AS_OF, HistoricalIntelligenceSettings())
    b_features = build_player_rolling_features("p2", [], AS_OF, HistoricalIntelligenceSettings())
    matchup = build_matchup_features("p1", "p2", [], a_features, b_features, AS_OF)
    assert matchup.workload_differential == float(a_features.matches_in_previous_7d - b_features.matches_in_previous_7d)
    assert matchup.form_differential is None  # b has no all-time win_rate


def test_matchup_is_deterministic():
    history = [_match("m1", 1, "p1")]
    a_features = _empty_player_features("p1")
    b_features = _empty_player_features("p2")
    first = build_matchup_features("p1", "p2", history, a_features, b_features, AS_OF)
    second = build_matchup_features("p1", "p2", history, a_features, b_features, AS_OF)
    assert first == second
