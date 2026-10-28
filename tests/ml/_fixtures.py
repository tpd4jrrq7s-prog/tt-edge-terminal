"""Shared test builders for the `ml` package's test suite.

Two flavors: `build_examples`/`build_matrix` go through the real
`features`/`datasets` pipeline (for feature-matrix/target tests that
need real snapshot semantics); `synthetic_matrix` builds a `FeatureMatrix`
directly from `ml.models` (for split/preprocessing/model tests that only
need controlled numeric values, independent of feature extraction).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from datasets.builder import DatasetBuilder
from datasets.models import TrainingExample
from features.builder import HistoricalFeatureBuilder
from ml.feature_matrix import build_feature_matrix
from ml.models import FeatureKind, FeatureMatrix, FeatureRow, FeatureSchema
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_match(mid: str, day: float, winner: str, best_of: int | None = 5) -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    sets = (
        [HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)]
        if winner == "p1"
        else [HistoricalSetRecord(set_number=1, player_a_points=7, player_b_points=11)]
    )
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, best_of=best_of, winner_id=winner, sets=sets,
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def build_examples(n: int, best_of: int | None = 5) -> list[TrainingExample]:
    """`n` completed matches between the same two players, alternating winners 2-wins/1-loss."""
    repo = InMemoryMatchRepository()
    matches = [make_match(f"m{i}", i, winner=("p1" if i % 3 != 0 else "p2"), best_of=best_of) for i in range(n)]
    repo.add_many(matches)
    feature_builder = HistoricalFeatureBuilder(repo)
    builder = DatasetBuilder(repo, feature_builder)
    examples, _ = builder.build([m.id for m in matches])
    return examples


def build_matrix(n: int, best_of: int | None = 5) -> tuple[FeatureMatrix, list[TrainingExample]]:
    examples = build_examples(n, best_of=best_of)
    pairs = [(e.id, e.features) for e in examples]
    return build_feature_matrix(pairs), examples


def synthetic_matrix(
    rows_values: list[list[float]],
    missing: list[list[bool]] | None = None,
    start: datetime | None = None,
    step_days: float = 1.0,
) -> FeatureMatrix:
    """A small, fully-controlled FeatureMatrix independent of real feature extraction."""
    n_features = len(rows_values[0])
    schema = FeatureSchema(
        version="synthetic-1.0", feature_names=[f"f{i}" for i in range(n_features)],
        feature_kinds=[FeatureKind.FLOAT] * n_features,
    )
    base = start or BASE
    mask = missing or [[False] * n_features for _ in rows_values]
    rows = [
        FeatureRow(source_id=f"r{i}", as_of=base + timedelta(days=i * step_days), values=values, missing_mask=mask[i])
        for i, values in enumerate(rows_values)
    ]
    return FeatureMatrix(schema=schema, rows=rows)
