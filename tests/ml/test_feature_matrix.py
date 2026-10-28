"""Tests for converting FeatureSnapshots into a stable, versioned FeatureMatrix."""

from __future__ import annotations

from ml.feature_matrix import FEATURE_DEFINITIONS, FEATURE_SCHEMA_VERSION, build_feature_matrix, build_feature_schema
from tests.ml._fixtures import build_matrix


def test_schema_version_is_stable_constant():
    schema = build_feature_schema()
    assert schema.version == FEATURE_SCHEMA_VERSION
    assert schema.version == "1.0.0"


def test_feature_names_are_unique_and_match_definitions():
    schema = build_feature_schema()
    assert schema.feature_names == [d.name for d in FEATURE_DEFINITIONS]
    assert len(set(schema.feature_names)) == len(schema.feature_names)


def test_feature_ordering_is_stable_across_builds():
    schema_one = build_feature_schema()
    schema_two = build_feature_schema()
    assert schema_one.feature_names == schema_two.feature_names
    assert schema_one.feature_kinds == schema_two.feature_kinds


def test_matrix_rows_preserve_input_order_and_ids():
    matrix, examples = build_matrix(5)
    assert matrix.source_ids() == [e.id for e in examples]


def test_every_row_matches_schema_width():
    matrix, _ = build_matrix(5)
    for row in matrix.rows:
        assert len(row.values) == matrix.n_features
        assert len(row.missing_mask) == matrix.n_features


def test_no_dictionaries_exposed_on_rows():
    matrix, _ = build_matrix(3)
    for row in matrix.rows:
        assert isinstance(row.values, list)
        assert isinstance(row.missing_mask, list)
        assert all(isinstance(v, float) for v in row.values)
        assert all(isinstance(m, bool) for m in row.missing_mask)


def test_first_match_has_no_prior_history_so_windows_are_missing():
    matrix, examples = build_matrix(5)
    schema = matrix.schema_
    first_row = matrix.rows[0]
    window_index = schema.feature_names.index("player_a.last_5.win_rate")
    assert first_row.missing_mask[window_index] is True
    assert first_row.values[window_index] == 0.0


def test_later_match_has_observed_history():
    matrix, examples = build_matrix(5)
    schema = matrix.schema_
    last_row = matrix.rows[-1]
    window_index = schema.feature_names.index("player_a.all_time.win_rate")
    assert last_row.missing_mask[window_index] is False


def test_build_feature_matrix_is_deterministic_for_same_input():
    _, examples = build_matrix(4)
    pairs = [(e.id, e.features) for e in examples]
    matrix_one = build_feature_matrix(pairs)
    matrix_two = build_feature_matrix(pairs)
    assert matrix_one == matrix_two


def test_matrix_n_features_matches_definition_count():
    matrix, _ = build_matrix(3)
    assert matrix.n_features == len(FEATURE_DEFINITIONS)
