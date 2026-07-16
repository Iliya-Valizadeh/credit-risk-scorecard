"""Evaluation: AUC, PR-AUC, threshold analysis, calibration curve."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve


def headline(y_true, proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def threshold_table(y_true, proba, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Precision/recall/flag-rate at each decision threshold.

    'flag_rate' = share of applicants predicted high-risk (declined). The bank trades
    recall of true defaulters against how many good customers get declined.
    """
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "flag_rate": pred.mean(),
        })
    return pd.DataFrame(rows).round(4)


def plot_calibration(y_true, proba, path=None, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration"); ax.legend()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=120)
    return fig
