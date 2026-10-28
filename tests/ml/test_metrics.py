"""Tests for classification metrics and calibration curve computation."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import compute_calibration_curve, evaluate_classifier


def test_perfect_predictions_score_maximally():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_confusion_matrix_counts_are_correct():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.6, 0.4, 0.8])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    cm = metrics.confusion_matrix
    assert (cm.true_negative, cm.false_positive, cm.false_negative, cm.true_positive) == (1, 1, 1, 1)


def test_roc_auc_is_none_for_single_class_truth():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.4])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics.roc_auc is None


def test_sample_size_matches_input_length():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics.sample_size == 3


def test_log_loss_and_brier_score_are_bounded_nonnegative():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.2, 0.7, 0.55, 0.9])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics.log_loss >= 0.0
    assert 0.0 <= metrics.brier_score <= 1.0


def test_calibration_curve_bin_counts_sum_to_sample_size():
    rng_like_probs = np.linspace(0.0, 1.0, 20)
    y_true = np.array([1 if p >= 0.5 else 0 for p in rng_like_probs])
    curve = compute_calibration_curve(y_true, rng_like_probs, n_bins=10)
    assert sum(b.count for b in curve.bins) == 20
    assert curve.n_bins == 10


def test_calibration_curve_handles_gaps_between_populated_bins():
    # Probabilities land only in bins 0, 4/5, and 9 out of 10 -- some bins are empty.
    y_proba = np.array([0.02, 0.03, 0.45, 0.55, 0.97, 0.98])
    y_true = np.array([0, 0, 1, 0, 1, 1])
    curve = compute_calibration_curve(y_true, y_proba, n_bins=10)
    assert sum(b.count for b in curve.bins) == 6
    bin_indices = [b.bin_index for b in curve.bins]
    assert bin_indices == sorted(bin_indices)
    assert all(0.0 <= b.fraction_of_positives <= 1.0 for b in curve.bins)
    assert all(0.0 <= b.mean_predicted_probability <= 1.0 for b in curve.bins)


def test_precision_recall_zero_division_is_safe():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
