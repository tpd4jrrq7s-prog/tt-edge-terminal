"""Tests for post-hoc probability calibration: Platt scaling and isotonic regression."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ml.baseline import BaselineModelKind, build_baseline_model
from ml.calibration import CalibrationMethod, calibrate_model

X_TRAIN = np.array([[float(i)] for i in range(20)])
Y_TRAIN = np.array([0] * 10 + [1] * 10)
X_CALIBRATION = np.array([[float(i)] for i in range(20, 30)])
Y_CALIBRATION = np.array([0] * 5 + [1] * 5)


def _fitted_model():
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X_TRAIN, Y_TRAIN)
    return model


def test_platt_calibration_returns_fitted_wrapper():
    model = _fitted_model()
    calibrated = calibrate_model(model, X_CALIBRATION, Y_CALIBRATION, CalibrationMethod.PLATT)
    assert isinstance(calibrated, CalibratedClassifierCV)
    probabilities = calibrated.predict_proba(X_CALIBRATION)
    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_isotonic_calibration_returns_fitted_wrapper():
    model = _fitted_model()
    calibrated = calibrate_model(model, X_CALIBRATION, Y_CALIBRATION, CalibrationMethod.ISOTONIC)
    probabilities = calibrated.predict_proba(X_CALIBRATION)
    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibration_does_not_mutate_the_base_model():
    model = _fitted_model()
    original_coef = model.coef_.copy()
    calibrate_model(model, X_CALIBRATION, Y_CALIBRATION, CalibrationMethod.PLATT)
    assert np.array_equal(model.coef_, original_coef)


def test_calibration_is_fit_on_the_calibration_set_not_the_training_set():
    model = _fitted_model()
    platt = calibrate_model(model, X_CALIBRATION, Y_CALIBRATION, CalibrationMethod.PLATT)
    # A model calibrated on a *different* calibration set should generally produce
    # different probabilities than one calibrated on another set with different labels.
    alternate_calibration_labels = np.array([1] * 5 + [0] * 5)
    platt_alternate = calibrate_model(model, X_CALIBRATION, alternate_calibration_labels, CalibrationMethod.PLATT)
    assert not np.allclose(
        platt.predict_proba(X_CALIBRATION), platt_alternate.predict_proba(X_CALIBRATION)
    )
