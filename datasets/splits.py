"""Time-based dataset splitting.

Random train/test splitting is deliberately not implemented as an
option here — every split strategy below is chronological by
construction, since a table-tennis prediction model must only ever be
validated on data that occurs after everything it was trained on.
"""

from __future__ import annotations

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from datasets.errors import InsufficientDataForSplitError
from datasets.models import DatasetSplit, SplitFold, SplitPlan, TrainingExample


def _ordered(examples: list[TrainingExample]) -> list[TrainingExample]:
    return sorted(examples, key=lambda e: (e.as_of, e.id))


def _split_from(label: str, items: list[TrainingExample]) -> DatasetSplit:
    return DatasetSplit(
        label=label,
        example_ids=[e.id for e in items],
        start=items[0].as_of if items else None,
        end=items[-1].as_of if items else None,
    )


def chronological_holdout_split(
    examples: list[TrainingExample], settings: HistoricalIntelligenceSettings | None = None
) -> SplitPlan:
    """A single train -> validation -> test split, in chronological order, by configured ratios."""
    settings = settings or get_historical_intelligence_settings()
    ordered = _ordered(examples)
    n = len(ordered)
    if n < 3:
        raise InsufficientDataForSplitError(f"Need at least 3 examples for a chronological holdout split, got {n}")

    train_end = max(1, min(int(round(n * settings.split_train_ratio)), n - 2))
    val_end = max(train_end + 1, min(train_end + int(round(n * settings.split_validation_ratio)), n - 1))

    train, validation, test = ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]
    return SplitPlan(
        strategy="chronological_holdout",
        splits=[_split_from("train", train), _split_from("validation", validation), _split_from("test", test)],
    )


def walk_forward_splits(examples: list[TrainingExample], n_folds: int) -> SplitPlan:
    """Expanding-window walk-forward folds: each fold's train set grows to include all prior folds."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    ordered = _ordered(examples)
    n = len(ordered)
    fold_size = n // (n_folds + 1)
    if fold_size < 1:
        raise InsufficientDataForSplitError(
            f"Not enough examples ({n}) to build {n_folds} walk-forward fold(s)"
        )

    folds = []
    for fold_index in range(n_folds):
        train_end = fold_size * (fold_index + 1)
        test_end = min(fold_size * (fold_index + 2), n)
        train_items = ordered[:train_end]
        test_items = ordered[train_end:test_end]
        if not test_items:
            break
        folds.append(
            SplitFold(
                fold_index=fold_index,
                train=_split_from(f"train_fold_{fold_index}", train_items),
                test=_split_from(f"test_fold_{fold_index}", test_items),
            )
        )

    return SplitPlan(strategy="walk_forward", folds=folds)


def rolling_window_splits(examples: list[TrainingExample], window_size: int, step: int) -> SplitPlan:
    """Fixed-size rolling (non-expanding) training window folds, advancing by `step` each fold."""
    if window_size < 1 or step < 1:
        raise ValueError("window_size and step must both be >= 1")
    ordered = _ordered(examples)
    n = len(ordered)
    if n < window_size + 1:
        raise InsufficientDataForSplitError(
            f"Not enough examples ({n}) for a rolling window of size {window_size}"
        )

    folds = []
    fold_index = 0
    train_start = 0
    while True:
        train_end = train_start + window_size
        test_end = train_end + step
        if train_end >= n:
            break
        train_items = ordered[train_start:train_end]
        test_items = ordered[train_end:min(test_end, n)]
        if not test_items:
            break
        folds.append(
            SplitFold(
                fold_index=fold_index,
                train=_split_from(f"train_fold_{fold_index}", train_items),
                test=_split_from(f"test_fold_{fold_index}", test_items),
            )
        )
        train_start += step
        fold_index += 1

    if not folds:
        raise InsufficientDataForSplitError(
            f"Not enough examples ({n}) to build any rolling-window fold of size {window_size}"
        )

    return SplitPlan(strategy="rolling_window", folds=folds)
