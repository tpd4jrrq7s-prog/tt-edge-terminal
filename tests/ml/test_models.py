"""Tests for the typed public contracts in ml.models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ml.models import FeatureKind, FeatureMatrix, FeatureRow, FeatureSchema, TargetKind, TargetVector

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_feature_schema_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        FeatureSchema(version="1.0.0", feature_names=["a", "b"], feature_kinds=[FeatureKind.FLOAT])


def test_feature_schema_rejects_duplicate_names():
    with pytest.raises(ValidationError):
        FeatureSchema(version="1.0.0", feature_names=["a", "a"], feature_kinds=[FeatureKind.FLOAT, FeatureKind.FLOAT])


def test_feature_schema_size():
    schema = FeatureSchema(version="1.0.0", feature_names=["a", "b", "c"], feature_kinds=[FeatureKind.FLOAT] * 3)
    assert schema.size == 3


def test_feature_row_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        FeatureRow(source_id="r1", as_of=BASE, values=[1.0, 2.0], missing_mask=[False])


def test_feature_matrix_constructs_via_schema_alias():
    schema = FeatureSchema(version="1.0.0", feature_names=["a"], feature_kinds=[FeatureKind.FLOAT])
    row = FeatureRow(source_id="r1", as_of=BASE, values=[1.0], missing_mask=[False])
    matrix = FeatureMatrix(schema=schema, rows=[row])
    assert matrix.schema_.version == "1.0.0"
    assert matrix.n_rows == 1
    assert matrix.n_features == 1


def test_feature_matrix_select_rows_preserves_schema():
    schema = FeatureSchema(version="1.0.0", feature_names=["a"], feature_kinds=[FeatureKind.FLOAT])
    rows = [FeatureRow(source_id=f"r{i}", as_of=BASE, values=[float(i)], missing_mask=[False]) for i in range(3)]
    matrix = FeatureMatrix(schema=schema, rows=rows)

    subset = matrix.select_rows([0, 2])

    assert subset.n_rows == 2
    assert subset.schema_ == matrix.schema_
    assert subset.source_ids() == ["r0", "r2"]


def test_feature_matrix_source_ids_and_as_of_values():
    schema = FeatureSchema(version="1.0.0", feature_names=["a"], feature_kinds=[FeatureKind.FLOAT])
    rows = [FeatureRow(source_id=f"r{i}", as_of=BASE, values=[0.0], missing_mask=[False]) for i in range(2)]
    matrix = FeatureMatrix(schema=schema, rows=rows)
    assert matrix.source_ids() == ["r0", "r1"]
    assert matrix.as_of_values() == [BASE, BASE]


def test_target_vector_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        TargetVector(kind=TargetKind.PLAYER_A_WON, values=[1.0, 0.0], source_ids=["a"])


def test_target_vector_size():
    vector = TargetVector(kind=TargetKind.PLAYER_A_WON, values=[1.0, 0.0], source_ids=["a", "b"])
    assert vector.size == 2
