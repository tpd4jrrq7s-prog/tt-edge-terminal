"""Post-hoc probability calibration: Platt scaling and isotonic regression.

Both wrap an already-fitted classifier with `sklearn.calibration.CalibratedClassifierCV`
via `FrozenEstimator` — calibration is fit on a separate calibration
set, never on the data the base model itself was trained on, to avoid
overfitting the calibration curve.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ml.baseline import BaselineModel


class CalibrationMethod(str, Enum):
    PLATT = "platt"
    ISOTONIC = "isotonic"


_SKLEARN_METHOD = {
    CalibrationMethod.PLATT: "sigmoid",
    CalibrationMethod.ISOTONIC: "isotonic",
}


def calibrate_model(
    fitted_model: BaselineModel, X_calibration: np.ndarray, y_calibration: np.ndarray, method: CalibrationMethod
) -> CalibratedClassifierCV:
    """Wrap a fitted model with post-hoc probability calibration fit on a separate set."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=_SKLEARN_METHOD[method])
    calibrated.fit(X_calibration, y_calibration)
    return calibrated
