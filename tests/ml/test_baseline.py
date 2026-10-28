"""Tests for baseline model interfaces: LogisticRegression and DummyClassifier only."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ml.baseline import BaselineModelKind, build_baseline_model

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
Y = np.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "kind",
    [
        BaselineModelKind.LOGISTIC_REGRESSION,
        BaselineModelKind.DUMMY_MOST_FREQUENT,
        BaselineModelKind.DUMMY_STRATIFIED,
        BaselineModelKind.DUMMY_UNIFORM,
    ],
)
def test_fit_predict_predict_proba_shapes(kind):
    model = build_baseline_model(kind)
    model.fit(X, Y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    assert predictions.shape == (6,)
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_logistic_regression_is_constructed():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    assert isinstance(model, LogisticRegression)


def test_logistic_regression_default_hyperparameters_are_deterministic():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    assert model.get_params()["random_state"] == 0
    assert model.get_params()["max_iter"] == 1000


def test_hyperparameters_can_be_overridden():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION, hyperparameters={"C": 0.5})
    assert model.get_params()["C"] == 0.5


def test_dummy_classifiers_use_expected_strategy():
    most_frequent = build_baseline_model(BaselineModelKind.DUMMY_MOST_FREQUENT)
    stratified = build_baseline_model(BaselineModelKind.DUMMY_STRATIFIED)
    uniform = build_baseline_model(BaselineModelKind.DUMMY_UNIFORM)
    assert isinstance(most_frequent, DummyClassifier) and most_frequent.strategy == "most_frequent"
    assert isinstance(stratified, DummyClassifier) and stratified.strategy == "stratified"
    assert isinstance(uniform, DummyClassifier) and uniform.strategy == "uniform"


def test_same_random_state_gives_reproducible_fit():
    model_one = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model_two = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model_one.fit(X, Y)
    model_two.fit(X, Y)
    assert np.allclose(model_one.coef_, model_two.coef_)
