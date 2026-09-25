"""A basic group check: does the decision treat groups differently at the threshold?

For each group, at a fixed threshold on the model score:
  approval_rate   share of applicants approved
  tpr             share of actual defaulters the model flags (true positive rate)
  fpr             share of non-defaulters the model flags, i.e. good customers declined
  observed_rate   actual default rate in the group
  mean_pd         mean calibrated PD in the group (compare with observed_rate)

What this shows: whether approval and error rates differ by group on this data.
What it does not show: why. A gap can come from a real difference in default rates,
from proxies for the group in other features, or from the group column itself, which
is a model input here. It is not a legal assessment and says nothing about groups that
are not in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25  # the youngest applicant in the data is 20
    return pd.cut(
        years, bins=[0, 30, 45, 60, 200], right=False, labels=["20-29", "30-44", "45-59", "60+"]
    ).astype(str)


def group_table(groups, y_true, raw_score, calibrated_pd, threshold, min_n=100) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "y": np.asarray(y_true, dtype=int),
            "flag": (np.asarray(raw_score) >= threshold).astype(int),
            "pd": np.asarray(calibrated_pd, dtype=float),
        }
    )
    rows = []
    for g, d in df.groupby("group"):
        if len(d) < min_n:
            continue
        pos, neg = d[d.y == 1], d[d.y == 0]
        rows.append(
            {
                "group": g,
                "n": len(d),
                "observed_rate": d.y.mean(),
                "mean_pd": d.pd.mean(),
                "approval_rate": 1 - d.flag.mean(),
                "tpr": pos.flag.mean() if len(pos) else np.nan,
                "fpr": neg.flag.mean() if len(neg) else np.nan,
            }
        )
    return pd.DataFrame(rows)
