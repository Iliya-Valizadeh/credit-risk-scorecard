"""Illustrative expected-loss view at a decision threshold.

Expected loss per loan = PD x LGD x EAD.
  PD  = probability of default (from the model)
  LGD = loss given default, the share of the exposure lost when a borrower defaults
  EAD = exposure at default, the amount owed at the time of default

ILLUSTRATIVE ONLY. The dataset has no recovery or balance data, so LGD and EAD are
assumptions, not estimates:
  - LGD is a flat 45%, the supervisory value for senior unsecured exposures under the
    Basel foundation IRB approach.
  - EAD is the full credit amount (AMT_CREDIT), ignoring any repayment before default.
  - TARGET means "had payment difficulties", which is broader than default.
"""
from __future__ import annotations
import numpy as np

LGD = 0.45


def expected_loss_table(y_true, ead, raw_pd, calibrated_pd, threshold, lgd=LGD) -> dict:
    """Loss on the applicants the model would approve (raw score below threshold).

    Compares the loss each set of PDs predicts with the loss the observed outcomes imply,
    under the same LGD and EAD assumptions.
    """
    y_true = np.asarray(y_true, dtype=float)
    ead = np.asarray(ead, dtype=float)
    approved = np.asarray(raw_pd) < threshold
    realised = float((y_true[approved] * lgd * ead[approved]).sum())
    return {
        "threshold": threshold,
        "lgd": lgd,
        "approved_share": float(approved.mean()),
        "approved_exposure": float(ead[approved].sum()),
        "el_raw_pd": float((np.asarray(raw_pd)[approved] * lgd * ead[approved]).sum()),
        "el_calibrated_pd": float((np.asarray(calibrated_pd)[approved] * lgd * ead[approved]).sum()),
        "loss_implied_by_outcomes": realised,
    }
