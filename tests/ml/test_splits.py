"""Tests for chronological FeatureMatrix row splitting. No random default split exists."""

from __future__ import annotations

import pytest

from ml.errors import InsufficientDataError
from ml.splits import chronological_split, expanding_window_split, rolling_window_split, walk_forward_split
from tests.ml._fixtures import synthetic_matrix


def _matrix(n: int):
    return synthetic_matrix([[float(i)] for i in range(n)])


def test_chronological_split_partitions_are_disjoint_and_cover_all_rows():
    matrix = _matrix(10)
    plan = chronological_split(matrix, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
    all_indices = plan.train_indices + plan.validation_indices + plan.test_indices
    assert sorted(all_indices) == list(range(10))
    assert len(set(all_indices)) == 10


def test_chronological_split_respects_time_order():
    matrix = _matrix(10)
    plan = chronological_split(matrix)
    assert max(plan.train_indices) < min(plan.validation_indices)
    assert max(plan.validation_indices) < min(plan.test_indices)


def test_chronological_split_respects_ratios_roughly():
    matrix = _matrix(10)
    plan = chronological_split(matrix, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
    assert len(plan.train_indices) == 6
    assert len(plan.validation_indices) == 2
    assert len(plan.test_indices) == 2


def test_chronological_split_rejects_ratios_not_summing_to_one():
    matrix = _matrix(10)
    with pytest.raises(ValueError):
        chronological_split(matrix, train_ratio=0.5, validation_ratio=0.2, test_ratio=0.2)


def test_chronological_split_too_few_rows_raises():
    with pytest.raises(InsufficientDataError):
        chronological_split(_matrix(2))


def test_chronological_split_is_deterministic():
    matrix = _matrix(10)
    plan_one = chronological_split(matrix)
    plan_two = chronological_split(matrix)
    assert plan_one == plan_two


def test_walk_forward_folds_expand_training_window():
    matrix = _matrix(12)
    plan = walk_forward_split(matrix, n_folds=3)
    sizes = [len(fold.train_indices) for fold in plan.folds]
    assert sizes == sorted(sizes)


def test_walk_forward_test_segments_never_overlap():
    matrix = _matrix(12)
    plan = walk_forward_split(matrix, n_folds=3)
    seen: set[int] = set()
    for fold in plan.folds:
        overlap = seen & set(fold.test_indices)
        assert not overlap
        seen.update(fold.test_indices)


def test_walk_forward_insufficient_data_raises():
    with pytest.raises(InsufficientDataError):
        walk_forward_split(_matrix(2), n_folds=5)


def test_expanding_window_split_matches_walk_forward_folds():
    matrix = _matrix(12)
    walk_forward = walk_forward_split(matrix, n_folds=3)
    expanding = expanding_window_split(matrix, n_folds=3)
    assert expanding.strategy == "expanding_window"
    assert expanding.folds == walk_forward.folds


def test_rolling_window_splits_are_fixed_size():
    matrix = _matrix(12)
    plan = rolling_window_split(matrix, window_size=4, step=2)
    for fold in plan.folds:
        assert len(fold.train_indices) == 4


def test_rolling_window_train_start_advances_by_step():
    matrix = _matrix(12)
    plan = rolling_window_split(matrix, window_size=4, step=2)
    starts = [fold.train_indices[0] for fold in plan.folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_rolling_window_insufficient_data_raises():
    with pytest.raises(InsufficientDataError):
        rolling_window_split(_matrix(2), window_size=5, step=1)


def test_no_random_split_function_exists():
    import ml.splits as splits_module

    names = dir(splits_module)
    assert not any("random" in name.lower() for name in names)
