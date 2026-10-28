"""Baseline model interfaces: Logistic Regression and Dummy Classifier only.

Gradient boosting and any other model family are deliberately not
implemented in this phase — Phase 5A is infrastructure, not a
production model.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


class BaselineModelKind(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    DUMMY_MOST_FREQUENT = "dummy_most_frequent"
    DUMMY_STRATIFIED = "dummy_stratified"
    DUMMY_UNIFORM = "dummy_uniform"


_DUMMY_STRATEGY_BY_KIND = {
    BaselineModelKind.DUMMY_MOST_FREQUENT: "most_frequent",
    BaselineModelKind.DUMMY_STRATIFIED: "stratified",
    BaselineModelKind.DUMMY_UNIFORM: "uniform",
}


@runtime_checkable
class BaselineModel(Protocol):
    """The minimal fitted-classifier interface the rest of `ml` depends on."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def build_baseline_model(
    kind: BaselineModelKind,
    hyperparameters: dict[str, float | int | str | bool] | None = None,
    random_state: int = 0,
) -> BaselineModel:
    """Construct an unfitted baseline model. `random_state` is fixed by default for determinism."""
    params = dict(hyperparameters or {})

    if kind is BaselineModelKind.LOGISTIC_REGRESSION:
        params.setdefault("random_state", random_state)
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)

    strategy = _DUMMY_STRATEGY_BY_KIND.get(kind)
    if strategy is None:
        raise ValueError(f"Unsupported baseline model kind: {kind!r}")
    params.setdefault("random_state", random_state)
    return DummyClassifier(strategy=strategy, **params)
