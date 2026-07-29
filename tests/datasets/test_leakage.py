"""Tests for explicit leakage detection — deliberately inject leaking inputs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from datasets.builder import DatasetBuilder
from datasets.errors import LeakageViolation
from datasets.leakage import (
    check_no_forbidden_target_fields,
    check_provider_mapping_consistency,
    check_snapshot_matches_own_example,
    check_source_timestamps_before_cutoff,
    check_target_excluded_from_history,
    run_dataset_leakage_report,
)
from datasets.models import DatasetSplit
from datasets.splits import chronological_holdout_split
from features.builder import HistoricalFeatureBuilder
from features.snapshots import build_feature_snapshot
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _finished(mid, day, winner="p1") -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _examples(n=8):
    repo = InMemoryMatchRepository()
    matches = [_finished(f"m{i}", i, winner=("p1" if i % 2 == 0 else "p2")) for i in range(n)]
    repo.add_many(matches)
    feature_builder = HistoricalFeatureBuilder(repo)
    builder = DatasetBuilder(repo, feature_builder)
    examples, _ = builder.build([m.id for m in matches])
    return repo, examples


def test_clean_dataset_passes_full_report():
    repo, examples = _examples()
    plan = chronological_holdout_split(examples)
    report = run_dataset_leakage_report(examples, repo, plan)
    assert report.passed


def test_future_match_in_history_is_rejected():
    as_of = BASE + timedelta(days=1)
    future_match = _finished("future", 5)  # after as_of
    with pytest.raises(Exception):
        build_feature_snapshot(
            snapshot_id="s", target_match_id="t", player_a_id="p1", player_b_id="p2", as_of=as_of,
            player_a_history=[future_match], player_b_history=[], head_to_head_history=[],
        )


def test_target_inclusion_is_detected_by_check_function():
    as_of = BASE + timedelta(days=10)
    leaking = _finished("target", 1)
    snapshot = build_feature_snapshot(
        snapshot_id="s", target_match_id="target", player_a_id="p1", player_b_id="p2", as_of=as_of,
        player_a_history=[leaking.model_copy(update={"id": "other"})], player_b_history=[], head_to_head_history=[],
    )
    # Manually corrupt provenance to simulate a leaked target inclusion, bypassing the builder.
    corrupted = snapshot.model_copy(
        update={"provenance": snapshot.provenance.model_copy(update={"player_a_source_match_ids": ["target"]})}
    )
    result = check_target_excluded_from_history(corrupted)
    assert result.passed is False


def test_source_timestamp_at_or_after_cutoff_is_detected():
    repo, examples = _examples()
    # Corrupt one example's provenance to claim it used a match that is at/after its own as_of.
    example = examples[0]
    late_match_id = examples[-1].target_match_id
    corrupted_snapshot = example.features.model_copy(
        update={
            "provenance": example.features.provenance.model_copy(
                update={"player_a_source_match_ids": [*example.features.provenance.player_a_source_match_ids, late_match_id]}
            )
        }
    )
    result = check_source_timestamps_before_cutoff(corrupted_snapshot, repo)
    assert result.passed is False


def test_snapshot_example_mismatch_is_detected():
    repo, examples = _examples()
    example = examples[0]
    other_example = examples[1]
    swapped = example.model_copy(update={"features": other_example.features})
    result = check_snapshot_matches_own_example(swapped)
    assert result.passed is False


def test_duplicate_train_test_across_splits_is_rejected():
    from datasets.leakage import check_no_duplicate_examples_across_splits

    splits = [
        DatasetSplit(label="train", example_ids=["ex-1", "ex-2"], start=BASE, end=BASE + timedelta(days=1)),
        DatasetSplit(label="test", example_ids=["ex-2", "ex-3"], start=BASE + timedelta(days=2), end=BASE + timedelta(days=3)),
    ]
    result = check_no_duplicate_examples_across_splits(splits)
    assert result.passed is False
    assert "ex-2" in result.detail


def test_nonchronological_split_boundaries_rejected():
    from datasets.leakage import check_chronological_split_boundaries

    splits = [
        DatasetSplit(label="train", example_ids=["ex-1"], start=BASE + timedelta(days=5), end=BASE + timedelta(days=10)),
        DatasetSplit(label="test", example_ids=["ex-2"], start=BASE, end=BASE + timedelta(days=1)),
    ]
    result = check_chronological_split_boundaries(splits)
    assert result.passed is False


def test_provider_mapping_conflict_detected():
    repo, examples = _examples(2)
    matches = [repo.get(e.target_match_id) for e in examples]
    conflicting = matches[1].model_copy(update={"provider_match_id": matches[0].provider_match_id})
    result = check_provider_mapping_consistency([matches[0], conflicting])
    assert result.passed is False


def test_forbidden_target_field_detection_on_clean_snapshot():
    repo, examples = _examples()
    result = check_no_forbidden_target_fields(examples[0].features)
    assert result.passed is True


def test_assert_no_leakage_raises_on_failed_report():
    from datasets.leakage import LeakageReport, assert_no_leakage
    from datasets.leakage import LeakageCheckResult

    bad_report = LeakageReport(checks=[LeakageCheckResult(name="fake_check", passed=False, detail="intentional failure")])
    with pytest.raises(LeakageViolation):
        assert_no_leakage(bad_report)
