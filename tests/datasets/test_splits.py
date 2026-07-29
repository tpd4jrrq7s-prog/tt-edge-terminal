"""Tests for chronological dataset splitting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config.historical import HistoricalIntelligenceSettings
from datasets.builder import DatasetBuilder
from datasets.errors import InsufficientDataForSplitError
from datasets.splits import chronological_holdout_split, rolling_window_splits, walk_forward_splits
from features.builder import HistoricalFeatureBuilder
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


def _examples(n):
    repo = InMemoryMatchRepository()
    matches = [_finished(f"m{i}", i, winner=("p1" if i % 2 == 0 else "p2")) for i in range(n)]
    repo.add_many(matches)
    feature_builder = HistoricalFeatureBuilder(repo)
    builder = DatasetBuilder(repo, feature_builder)
    examples, _ = builder.build([m.id for m in matches])
    return examples


def test_holdout_split_is_chronological_train_before_test():
    examples = _examples(10)
    plan = chronological_holdout_split(examples)
    train, validation, test = plan.splits
    assert train.end <= validation.start
    assert validation.end <= test.start


def test_holdout_split_has_no_id_overlap():
    examples = _examples(10)
    plan = chronological_holdout_split(examples)
    all_ids = [eid for split in plan.splits for eid in split.example_ids]
    assert len(all_ids) == len(set(all_ids))


def test_holdout_split_respects_configured_ratios_roughly():
    examples = _examples(10)
    settings = HistoricalIntelligenceSettings(split_train_ratio=0.6, split_validation_ratio=0.2, split_test_ratio=0.2)
    plan = chronological_holdout_split(examples, settings)
    assert plan.splits[0].size == 6


def test_holdout_split_too_few_examples_raises():
    with pytest.raises(InsufficientDataForSplitError):
        chronological_holdout_split(_examples(2))


def test_walk_forward_folds_expand_training_window():
    examples = _examples(12)
    plan = walk_forward_splits(examples, n_folds=3)
    sizes = [fold.train.size for fold in plan.folds]
    assert sizes == sorted(sizes)  # non-decreasing: expanding window


def test_walk_forward_test_segments_never_overlap():
    examples = _examples(12)
    plan = walk_forward_splits(examples, n_folds=3)
    seen = set()
    for fold in plan.folds:
        overlap = seen & set(fold.test.example_ids)
        assert not overlap
        seen.update(fold.test.example_ids)


def test_walk_forward_insufficient_data_raises():
    with pytest.raises(InsufficientDataForSplitError):
        walk_forward_splits(_examples(2), n_folds=5)


def test_rolling_window_splits_are_fixed_size():
    examples = _examples(12)
    plan = rolling_window_splits(examples, window_size=4, step=2)
    for fold in plan.folds:
        assert fold.train.size == 4


def test_rolling_window_insufficient_data_raises():
    with pytest.raises(InsufficientDataForSplitError):
        rolling_window_splits(_examples(2), window_size=5, step=1)


def test_splits_are_deterministic():
    examples = _examples(10)
    plan_one = chronological_holdout_split(examples)
    plan_two = chronological_holdout_split(examples)
    assert plan_one == plan_two
