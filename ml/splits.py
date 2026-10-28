"""Time-based splitting for FeatureMatrix rows.

Mirrors `datasets.splits`' chronological philosophy at the row-index
level (the currency the ml layer actually trains on) — **no random
default split is implemented**. Every strategy here is chronological by
construction, since a match-outcome model must only ever be validated
on rows that occur after everything it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ml.errors import InsufficientDataError
from ml.models import FeatureMatrix


@dataclass(frozen=True)
class MLSplitFold:
    fold_index: int
    train_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class MLSplitPlan:
    strategy: str
    train_indices: list[int] = field(default_factory=list)
    validation_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)
    folds: list[MLSplitFold] = field(default_factory=list)


def _chronological_order(matrix: FeatureMatrix) -> list[int]:
    indexed = list(enumerate(matrix.as_of_values()))
    indexed.sort(key=lambda pair: (pair[1], pair[0]))
    return [i for i, _ in indexed]


def chronological_split(
    matrix: FeatureMatrix, train_ratio: float = 0.6, validation_ratio: float = 0.2, test_ratio: float = 0.2
) -> MLSplitPlan:
    """A single train -> validation -> test split, in chronological order."""
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/validation/test ratios must sum to 1.0 (got {total:.6f})")

    order = _chronological_order(matrix)
    n = len(order)
    if n < 3:
        raise InsufficientDataError(f"Need at least 3 rows for a chronological split, got {n}")

    train_end = max(1, min(int(round(n * train_ratio)), n - 2))
    val_end = max(train_end + 1, min(train_end + int(round(n * validation_ratio)), n - 1))

    return MLSplitPlan(
        strategy="chronological",
        train_indices=order[:train_end],
        validation_indices=order[train_end:val_end],
        test_indices=order[val_end:],
    )


def walk_forward_split(matrix: FeatureMatrix, n_folds: int) -> MLSplitPlan:
    """Expanding-window walk-forward folds: each fold's train set grows to include all prior folds."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    order = _chronological_order(matrix)
    n = len(order)
    fold_size = n // (n_folds + 1)
    if fold_size < 1:
        raise InsufficientDataError(f"Not enough rows ({n}) to build {n_folds} walk-forward fold(s)")

    folds: list[MLSplitFold] = []
    for fold_index in range(n_folds):
        train_end = fold_size * (fold_index + 1)
        test_end = min(fold_size * (fold_index + 2), n)
        test_indices = order[train_end:test_end]
        if not test_indices:
            break
        folds.append(MLSplitFold(fold_index=fold_index, train_indices=order[:train_end], test_indices=test_indices))
    return MLSplitPlan(strategy="walk_forward", folds=folds)


def expanding_window_split(matrix: FeatureMatrix, n_folds: int) -> MLSplitPlan:
    """Alias for `walk_forward_split`: an expanding window IS the walk-forward strategy applied here."""
    plan = walk_forward_split(matrix, n_folds)
    return MLSplitPlan(strategy="expanding_window", folds=plan.folds)


def rolling_window_split(matrix: FeatureMatrix, window_size: int, step: int) -> MLSplitPlan:
    """Fixed-size (non-expanding) rolling training window, advancing by `step` each fold."""
    if window_size < 1 or step < 1:
        raise ValueError("window_size and step must both be >= 1")
    order = _chronological_order(matrix)
    n = len(order)
    if n < window_size + 1:
        raise InsufficientDataError(f"Not enough rows ({n}) for a rolling window of size {window_size}")

    folds: list[MLSplitFold] = []
    fold_index = 0
    train_start = 0
    while True:
        train_end = train_start + window_size
        test_end = train_end + step
        if train_end >= n:
            break
        test_indices = order[train_end:min(test_end, n)]
        if not test_indices:
            break
        folds.append(
            MLSplitFold(fold_index=fold_index, train_indices=order[train_start:train_end], test_indices=test_indices)
        )
        train_start += step
        fold_index += 1

    if not folds:
        raise InsufficientDataError(f"Not enough rows ({n}) to build any rolling-window fold of size {window_size}")
    return MLSplitPlan(strategy="rolling_window", folds=folds)
