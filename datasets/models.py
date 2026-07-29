"""Typed contracts for training examples, dataset manifests, and time-based splits."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from features.models import FeatureSnapshot


class TrainingExample(BaseModel):
    """One labeled training example: a leakage-safe pre-match snapshot plus the known outcome.

    The label (`player_a_won`) is derived strictly from the target
    match's recorded result — never from anything inside `features`,
    which was built to exclude the target match entirely.
    """

    id: str = Field(..., min_length=1)
    target_match_id: str = Field(..., min_length=1)
    as_of: datetime
    player_a_id: str
    player_b_id: str
    player_a_won: int = Field(..., ge=0, le=1)
    features: FeatureSnapshot
    feature_schema_version: str
    builder_version: str
    provenance_fingerprint: str


class LabelDistribution(BaseModel):
    player_a_wins: int = Field(default=0, ge=0)
    player_b_wins: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.player_a_wins + self.player_b_wins


class DatasetManifest(BaseModel):
    """Metadata describing one built dataset — enough to audit or reproduce it."""

    dataset_id: str = Field(..., min_length=1)
    created_at: datetime
    feature_schema_version: str
    builder_version: str
    source_match_count: int = Field(..., ge=0)
    training_example_count: int = Field(..., ge=0)
    skipped_count: int = Field(..., ge=0)
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    repository_fingerprint: str
    label_distribution: LabelDistribution
    warnings: list[str] = Field(default_factory=list)


class DatasetSplit(BaseModel):
    """One chronological slice of a dataset (e.g. train, validation, test, or one fold's train/test)."""

    label: str
    example_ids: list[str] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None

    @property
    def size(self) -> int:
        return len(self.example_ids)


class SplitFold(BaseModel):
    """One walk-forward/rolling-window fold: a train segment and a later test segment.

    Unlike a flat holdout split, folds are expected to share training
    examples across folds (an expanding or rolling window) — only
    train/test *within the same fold* must be disjoint and chronological.
    """

    fold_index: int = Field(..., ge=0)
    train: DatasetSplit
    test: DatasetSplit

    @model_validator(mode="after")
    def _train_precedes_test_within_fold(self) -> "SplitFold":
        overlap = set(self.train.example_ids) & set(self.test.example_ids)
        if overlap:
            raise ValueError(f"fold {self.fold_index}: train/test overlap on IDs {sorted(overlap)}")
        if self.train.end is not None and self.test.start is not None and self.test.start < self.train.end:
            raise ValueError(f"fold {self.fold_index}: test split starts before train split ends")
        return self


class SplitPlan(BaseModel):
    """A complete split plan for one dataset — either flat holdout `splits`, or CV `folds`."""

    strategy: str
    splits: list[DatasetSplit] = Field(default_factory=list)
    folds: list[SplitFold] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_flat_splits_are_chronological_and_disjoint(self) -> "SplitPlan":
        seen_ids: set[str] = set()
        previous_end: datetime | None = None
        for split in self.splits:
            overlap = seen_ids.intersection(split.example_ids)
            if overlap:
                raise ValueError(f"Split {split.label!r} overlaps with an earlier split on IDs: {sorted(overlap)}")
            seen_ids.update(split.example_ids)
            if previous_end is not None and split.start is not None and split.start < previous_end:
                raise ValueError(
                    f"Split {split.label!r} starts before the previous split ends "
                    f"({split.start.isoformat()} < {previous_end.isoformat()})"
                )
            if split.end is not None:
                previous_end = split.end
        return self

    @model_validator(mode="after")
    def _validate_folds_progress_chronologically(self) -> "SplitPlan":
        previous_test_end: datetime | None = None
        for fold in self.folds:
            if previous_test_end is not None and fold.test.start is not None and fold.test.start < previous_test_end:
                raise ValueError(f"fold {fold.fold_index}: test segment does not progress forward in time")
            if fold.test.end is not None:
                previous_test_end = fold.test.end
        return self
