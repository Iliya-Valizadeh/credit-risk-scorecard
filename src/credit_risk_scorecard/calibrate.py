"""Post-hoc calibration: map the model's scores to PDs that mean what they say.

The LightGBM model is trained with scale_pos_weight, which tells it to treat each
defaulter as ~11 applicants. That helps ranking but inflates every predicted PD.
Both calibrators here are fitted on a calibration split the model never saw, then
judged on a separate test split.

- Platt scaling: a logistic regression on the log-odds of the raw score. Two parameters,
  so it can only shift and stretch the curve.
- Isotonic regression: a step function that only has to be non-decreasing. More
  flexible, needs more data, and can overfit a small calibration set.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, raw, y):
        self._lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw), y)
        return self

    def predict(self, raw):
        return self._lr.predict_proba(_logit(raw))[:, 1]


class IsotonicCalibrator:
    def fit(self, raw, y):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(raw, dtype=float), y)
        return self

    def predict(self, raw):
        return self._iso.predict(np.asarray(raw, dtype=float))


class Calibrator(Protocol):
    """Shared shape of the two calibrators below, for type-checking CALIBRATORS."""

    def fit(self, raw, y) -> Calibrator: ...
    def predict(self, raw): ...


CALIBRATORS: dict[str, type[Calibrator]] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}
