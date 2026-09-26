"""Demo for `make demo` (ADR 0001, decision 4).

Runs the real training and scoring pipeline (credit_risk_scorecard.model) on synthetic
applicants, not the Kaggle data. No download, no API key, no network access. It writes
nothing to reports/ or models/, so it can never overwrite a real number.

Run:  python -m credit_risk_scorecard.demo
"""

from __future__ import annotations

import numpy as np

from . import features as F
from . import model as M
from .synthetic import make_applicants

N_SCORE = 5


def main() -> int:
    print(
        "Demo only: the applicants below are synthetic (made up), not the Home "
        "Credit data. These numbers are not the project's results; see "
        "reports/metrics.json for those."
    )
    print()

    train_df = make_applicants(n=3000, seed=1)
    artifacts: dict = {}
    m = M.run(train_df, n_boot=200, figures=False, save=False, artifacts=artifacts)

    print(
        f"Trained on {m['data']['train']:,} synthetic rows, calibrated on "
        f"{m['data']['calibration']:,}, tested on {m['data']['test']:,}."
    )
    print(
        f"  Logistic regression ROC-AUC on synthetic test data: "
        f"{m['models']['logreg']['roc_auc']:.3f}"
    )
    print(
        f"  LightGBM ROC-AUC on synthetic test data:            "
        f"{m['models']['lightgbm']['roc_auc']:.3f}"
    )
    threshold = artifacts["threshold"]
    print(
        f"  Decline threshold (flags the riskiest "
        f"{m['operating_point']['target_flag_rate']:.0%}): {threshold:.3f}"
    )
    print()

    applicants = make_applicants(n=N_SCORE, seed=99)
    pre, lgbm = artifacts["pre"], artifacts["lgbm"]
    X = F.clean(applicants).drop(columns=["TARGET"])
    proba = lgbm.predict_proba(pre.transform(X).astype(np.float32))[:, 1]
    print(
        f"Scoring {N_SCORE} synthetic applicants with the trained pipeline "
        "(raw LightGBM score, not calibrated):"
    )
    print(f"{'SK_ID_CURR':<12}{'raw PD':>10}{'decision':>12}")
    for sk_id, p in zip(applicants["SK_ID_CURR"], proba, strict=True):
        decision = "decline" if p >= threshold else "approve"
        print(f"{sk_id:<12}{p:>10.3f}{decision:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
