"""Tests for the evaluate_model orchestration API."""

from __future__ import annotations

import numpy as np

from ml.baseline import BaselineModelKind, build_baseline_model
from ml.evaluation import evaluate_model

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
Y = np.array([0, 0, 0, 1, 1, 1])


def test_evaluate_model_shapes_and_alignment():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X, Y)
    source_ids = [f"row-{i}" for i in range(len(Y))]

    result = evaluate_model(model, X, Y, source_ids)

    assert len(result.predicted_probabilities) == len(Y)
    assert len(result.predicted_labels) == len(Y)
    assert result.source_ids == source_ids
    assert result.metrics.sample_size == len(Y)


def test_predicted_labels_are_thresholded_at_half():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X, Y)
    result = evaluate_model(model, X, Y, [f"r{i}" for i in range(len(Y))])
    for probability, label in zip(result.predicted_probabilities, result.predicted_labels):
        assert label == (1 if probability >= 0.5 else 0)


def test_calibration_curve_bins_do_not_exceed_requested_bins():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X, Y)
    result = evaluate_model(model, X, Y, [f"r{i}" for i in range(len(Y))], n_bins=5)
    assert result.calibration_curve.n_bins == 5
    assert len(result.calibration_curve.bins) <= 5
