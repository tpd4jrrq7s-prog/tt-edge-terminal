"""Standard binary-classification evaluation metrics and calibration curves.

Thin, typed wrappers over `sklearn.metrics` — the formulas themselves
are exactly scikit-learn's well-tested implementations, not
reimplemented here.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    log_loss as sk_log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.models import CalibrationBin, CalibrationCurve, ClassificationMetrics, ConfusionMatrix


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ClassificationMetrics:
    """Compute accuracy/precision/recall/F1/ROC-AUC/log-loss/Brier score/confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    labels = np.unique(y_true)
    roc_auc: float | None
    if len(labels) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        sample_size=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        log_loss=float(sk_log_loss(y_true, y_proba, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_true, y_proba)),
        confusion_matrix=ConfusionMatrix(
            true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp)
        ),
    )


def compute_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> CalibrationCurve:
    """Bin predicted probabilities and compare against observed positive rates per bin."""
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_proba, bin_edges[1:-1], right=True), 0, n_bins - 1)
    counts_by_bin = np.bincount(bin_indices, minlength=n_bins)
    occupied_bins = sorted(set(bin_indices.tolist()))

    bins = [
        CalibrationBin(
            bin_index=bin_index,
            mean_predicted_probability=float(mean_predicted[i]),
            fraction_of_positives=float(fraction_of_positives[i]),
            count=int(counts_by_bin[bin_index]),
        )
        for i, bin_index in enumerate(occupied_bins)
    ]
    return CalibrationCurve(n_bins=n_bins, bins=bins)
