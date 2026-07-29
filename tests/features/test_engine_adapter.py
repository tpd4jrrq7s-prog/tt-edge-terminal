"""Tests for the FeatureSnapshot -> Phase 2B engine input adapter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from features.engine_adapter import adapt_snapshot_to_engine_inputs
from features.snapshots import build_feature_snapshot
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

AS_OF = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _match(mid, day, winner="p1") -> HistoricalMatchRecord:
    scheduled = AS_OF - timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def test_no_head_to_head_history_yields_zero_record_not_fabricated():
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=[],
    )
    adapted = adapt_snapshot_to_engine_inputs(snapshot)
    assert adapted.head_to_head.player_one_wins == 0
    assert adapted.head_to_head.player_two_wins == 0
    assert adapted.head_to_head.total_matches == 0


def test_head_to_head_counts_are_derived_from_win_rate():
    history = [_match("m1", 1, "p1"), _match("m2", 2, "p2"), _match("m3", 3, "p1")]
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=history,
    )
    adapted = adapt_snapshot_to_engine_inputs(snapshot)
    assert adapted.head_to_head.player_one_wins == 2
    assert adapted.head_to_head.player_two_wins == 1
    assert adapted.head_to_head.last_played_at is not None


def test_context_is_none_when_not_supplied():
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=[],
    )
    adapted = adapt_snapshot_to_engine_inputs(snapshot)
    assert adapted.context is None


def test_context_passes_through_caller_supplied_values():
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=[],
    )
    adapted = adapt_snapshot_to_engine_inputs(snapshot, competition_name="Demo Open", best_of_sets=5)
    assert adapted.context.competition_name == "Demo Open"
    assert adapted.context.best_of_sets == 5


def test_unconvertible_features_are_reported_not_silently_dropped():
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=[],
    )
    adapted = adapt_snapshot_to_engine_inputs(snapshot)
    assert len(adapted.unconvertible_features) > 0
    assert any("volatility" in name for name in adapted.unconvertible_features)
