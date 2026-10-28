"""Leakage-safe preprocessing: missing-value imputation, optional scaling/normalization.

**No leakage**: `FittedPreprocessor.fit()` must be called only with the
*training* row indices of a `FeatureMatrix`. All imputation/scaling
statistics (means, medians, standard deviations, min/max) are computed
exclusively from those rows; `.transform()` then applies the same
fitted statistics to any rows (train, validation, or test) without
recomputing anything from them.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from ml.errors import NotFittedError
from ml.models import FeatureMatrix, FeatureRow


class MissingValueStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    CONSTANT = "constant"


class ScalingStrategy(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    MIN_MAX = "min_max"


class PreprocessingConfig(BaseModel):
    """Configuration for `FittedPreprocessor`. Categorical encoding is a placeholder for
    a future phase — no categorical features exist in the Phase 5A feature schema yet."""

    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    constant_fill_value: float = 0.0
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    normalize: bool = False
    categorical_features: list[str] = Field(default_factory=list)


class FittedPreprocessor:
    """A preprocessor fitted on a specific set of training rows. Never re-fit on val/test data."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self._config = config or PreprocessingConfig()
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | MinMaxScaler | None = None
        self._normalizer: Normalizer | None = None
        self._schema_version: str | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _to_array(self, matrix: FeatureMatrix, indices: list[int]) -> np.ndarray:
        rows: list[FeatureRow] = [matrix.rows[i] for i in indices]
        array = np.array([row.values for row in rows], dtype=float)
        mask = np.array([row.missing_mask for row in rows], dtype=bool)
        array = np.where(mask, np.nan, array)
        return array

    def fit(self, matrix: FeatureMatrix, train_indices: list[int]) -> "FittedPreprocessor":
        """Fit imputation/scaling statistics using only `train_indices` of `matrix`."""
        array = self._to_array(matrix, train_indices)

        strategy = self._config.missing_value_strategy
        if strategy is MissingValueStrategy.CONSTANT:
            self._imputer = SimpleImputer(
                strategy="constant", fill_value=self._config.constant_fill_value, keep_empty_features=True
            )
        else:
            self._imputer = SimpleImputer(strategy=strategy.value, keep_empty_features=True)
        imputed = self._imputer.fit_transform(array)

        if self._config.scaling_strategy is ScalingStrategy.STANDARD:
            self._scaler = StandardScaler().fit(imputed)
        elif self._config.scaling_strategy is ScalingStrategy.MIN_MAX:
            self._scaler = MinMaxScaler().fit(imputed)
        else:
            self._scaler = None

        if self._config.normalize:
            self._normalizer = Normalizer().fit(imputed)
        else:
            self._normalizer = None

        self._schema_version = matrix.schema_.version
        self._fitted = True
        return self

    def transform(self, matrix: FeatureMatrix, indices: list[int] | None = None) -> np.ndarray:
        """Apply the already-fitted imputation/scaling to `indices` (default: all rows)."""
        if not self._fitted or self._imputer is None:
            raise NotFittedError("FittedPreprocessor.transform() called before fit()")

        row_indices = indices if indices is not None else list(range(matrix.n_rows))
        array = self._to_array(matrix, row_indices)
        transformed = self._imputer.transform(array)
        if self._scaler is not None:
            transformed = self._scaler.transform(transformed)
        if self._normalizer is not None:
            transformed = self._normalizer.transform(transformed)
        return transformed
