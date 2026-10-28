"""Tests for leakage-safe preprocessing: imputation, scaling, normalization."""

from __future__ import annotations

import numpy as np
import pytest

from ml.errors import NotFittedError
from ml.preprocessing import FittedPreprocessor, MissingValueStrategy, PreprocessingConfig, ScalingStrategy
from tests.ml._fixtures import synthetic_matrix


def test_transform_before_fit_raises():
    matrix = synthetic_matrix([[1.0], [2.0], [3.0]])
    preprocessor = FittedPreprocessor()
    with pytest.raises(NotFittedError):
        preprocessor.transform(matrix)


def test_output_width_always_matches_schema_even_with_all_missing_train_column():
    # Column 1 is entirely missing in the training rows (indices 0-1), but observed later.
    matrix = synthetic_matrix(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 9.0], [4.0, 9.0]],
        missing=[[False, True], [False, True], [False, False], [False, False]],
    )
    preprocessor = FittedPreprocessor()
    preprocessor.fit(matrix, train_indices=[0, 1])
    transformed = preprocessor.transform(matrix)
    assert transformed.shape == (4, 2)


def test_fit_uses_only_training_indices_not_full_matrix():
    # Median imputation: if fit saw all 5 rows the median of column 0 would be 3.0;
    # fit on only the first two rows [1.0, 2.0] must use their median (1.5), not 3.0.
    matrix = synthetic_matrix(
        [[1.0], [2.0], [np.nan], [4.0], [5.0]],
        missing=[[False], [False], [True], [False], [False]],
    )
    config = PreprocessingConfig(missing_value_strategy=MissingValueStrategy.MEDIAN, scaling_strategy=ScalingStrategy.NONE)
    preprocessor = FittedPreprocessor(config)
    preprocessor.fit(matrix, train_indices=[0, 1])
    transformed = preprocessor.transform(matrix, indices=[2])
    assert transformed[0, 0] == pytest.approx(1.5)


def test_constant_missing_value_strategy_fills_configured_value():
    matrix = synthetic_matrix([[1.0], [2.0]], missing=[[False], [True]])
    config = PreprocessingConfig(
        missing_value_strategy=MissingValueStrategy.CONSTANT, constant_fill_value=-1.0,
        scaling_strategy=ScalingStrategy.NONE,
    )
    preprocessor = FittedPreprocessor(config)
    preprocessor.fit(matrix, train_indices=[0, 1])
    transformed = preprocessor.transform(matrix)
    assert transformed[1, 0] == pytest.approx(-1.0)


def test_standard_scaling_normalizes_training_distribution():
    values = [[float(i)] for i in range(10)]
    matrix = synthetic_matrix(values)
    config = PreprocessingConfig(scaling_strategy=ScalingStrategy.STANDARD)
    preprocessor = FittedPreprocessor(config)
    preprocessor.fit(matrix, train_indices=list(range(10)))
    transformed = preprocessor.transform(matrix)
    assert transformed.mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed.std() == pytest.approx(1.0, abs=1e-9)


def test_scaling_none_leaves_imputed_values_unscaled():
    matrix = synthetic_matrix([[1.0], [2.0], [3.0]])
    config = PreprocessingConfig(scaling_strategy=ScalingStrategy.NONE)
    preprocessor = FittedPreprocessor(config)
    preprocessor.fit(matrix, train_indices=[0, 1, 2])
    transformed = preprocessor.transform(matrix)
    assert transformed.flatten().tolist() == [1.0, 2.0, 3.0]


def test_normalize_scales_each_row_to_unit_norm():
    matrix = synthetic_matrix([[3.0, 4.0], [1.0, 0.0]])
    config = PreprocessingConfig(scaling_strategy=ScalingStrategy.NONE, normalize=True)
    preprocessor = FittedPreprocessor(config)
    preprocessor.fit(matrix, train_indices=[0, 1])
    transformed = preprocessor.transform(matrix)
    norms = np.linalg.norm(transformed, axis=1)
    assert norms == pytest.approx([1.0, 1.0])


def test_is_fitted_flag():
    matrix = synthetic_matrix([[1.0], [2.0]])
    preprocessor = FittedPreprocessor()
    assert preprocessor.is_fitted is False
    preprocessor.fit(matrix, train_indices=[0, 1])
    assert preprocessor.is_fitted is True


def test_categorical_features_placeholder_defaults_empty():
    config = PreprocessingConfig()
    assert config.categorical_features == []
