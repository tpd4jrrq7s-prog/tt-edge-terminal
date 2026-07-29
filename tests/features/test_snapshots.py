"""Tests for leakage-safe FeatureSnapshot assembly."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config.historical import HistoricalIntelligenceSettings
from features.errors import SnapshotLeakageError
from features.snapshots import build_feature_snapshot, compute_fingerprint
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


def test_build_feature_snapshot_basic_shape():
    history = [_match("m1", 5)]
    snapshot = build_feature_snapshot(
        snapshot_id="snap-1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=history, player_b_history=[], head_to_head_history=history,
    )
    assert snapshot.id == "snap-1"
    assert snapshot.target_match_id == "target"
    assert snapshot.player_a_features.observation_count == 1
    assert snapshot.provenance.player_a_observation_count == 1
    assert snapshot.provenance.head_to_head_observation_count == 1


def test_target_match_in_own_history_raises_leakage_error():
    leaking_history = [_match("target", 5)]  # same ID as the target match itself
    with pytest.raises(SnapshotLeakageError):
        build_feature_snapshot(
            snapshot_id="snap-1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
            player_a_history=leaking_history, player_b_history=[], head_to_head_history=[],
        )


def test_source_match_at_or_after_as_of_raises_leakage_error():
    future_match = _match("m1", -5)  # 5 days AFTER as_of
    with pytest.raises(SnapshotLeakageError):
        build_feature_snapshot(
            snapshot_id="snap-1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
            player_a_history=[future_match], player_b_history=[], head_to_head_history=[],
        )


def test_missing_history_produces_no_history_warnings_and_lower_quality():
    with_history = build_feature_snapshot(
        snapshot_id="s1", target_match_id="t1", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[_match("m1", 1), _match("m2", 2), _match("m3", 3)],
        player_b_history=[_match("m4", 1), _match("m5", 2), _match("m6", 3)],
        head_to_head_history=[],
    )
    no_history = build_feature_snapshot(
        snapshot_id="s2", target_match_id="t2", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=[], player_b_history=[], head_to_head_history=[],
    )
    assert no_history.provenance.data_quality_score < with_history.provenance.data_quality_score
    assert "no head-to-head history available" in no_history.provenance.warnings


def test_fingerprint_is_deterministic():
    assert compute_fingerprint(["a", "b", "c"]) == compute_fingerprint(["a", "b", "c"])
    assert compute_fingerprint(["a", "b"]) != compute_fingerprint(["a", "c"])


def test_snapshot_input_fingerprint_deterministic_for_identical_inputs():
    history = [_match("m1", 5)]
    first = build_feature_snapshot(
        snapshot_id="s1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=history, player_b_history=[], head_to_head_history=[],
    )
    second = build_feature_snapshot(
        snapshot_id="s1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=history, player_b_history=[], head_to_head_history=[],
    )
    assert first.provenance.input_fingerprint == second.provenance.input_fingerprint
    assert first == second


def test_rebuild_with_same_inputs_is_equivalent():
    history = [_match("m1", 5), _match("m2", 10, winner="p2")]
    settings = HistoricalIntelligenceSettings()
    first = build_feature_snapshot(
        snapshot_id="s1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=history, player_b_history=[], head_to_head_history=history, settings=settings,
    )
    second = build_feature_snapshot(
        snapshot_id="s1", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=AS_OF,
        player_a_history=history, player_b_history=[], head_to_head_history=history, settings=settings,
    )
    assert first == second
