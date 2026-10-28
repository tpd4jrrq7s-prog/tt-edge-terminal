"""Orchestrates evaluating a fitted model against a (typically held-out) split."""

from __future__ import annotations

import numpy as np

from ml.baseline import BaselineModel
from ml.metrics import compute_calibration_curve, evaluate_classifier
from ml.models import EvaluationResult


def evaluate_model(
    model: BaselineModel, X: np.ndarray, y: np.ndarray, source_ids: list[str], n_bins: int = 10
) -> EvaluationResult:
    """Predict with `model` on `X`, then compute classification metrics and a calibration curve."""
    y_proba = np.asarray(model.predict_proba(X))[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluate_classifier(np.asarray(y), y_pred, y_proba)
    curve = compute_calibration_curve(np.asarray(y), y_proba, n_bins=n_bins)

    return EvaluationResult(
        metrics=metrics,
        calibration_curve=curve,
        predicted_probabilities=[float(p) for p in y_proba],
        predicted_labels=[int(p) for p in y_pred],
        source_ids=list(source_ids),
    )
