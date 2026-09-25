"""Evaluation: AUC, PR-AUC, threshold analysis, calibration metrics, bootstrap intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve  # noqa: E402


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


def brier(y_true, proba) -> float:
    """Mean squared gap between predicted PD and the 0/1 outcome. Lower is better."""
    return float(brier_score_loss(y_true, proba))


def ece(y_true, proba, n_bins=10) -> float:
    """Expected calibration error with equal-count bins.

    Sort applicants by predicted PD, cut them into n_bins groups of equal size, and take
    the size-weighted average of |mean predicted PD - observed default rate| per group.
    Equal-count bins are used because most PDs are small, so equal-width bins would put
    nearly everyone in the first two bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) * abs(proba[idx].mean() - y_true[idx].mean())
    return float(total / len(proba))


def bootstrap_ci(y_true, scores: dict, metric=roc_auc_score, n_boot=1000, seed=42,
                 diff: tuple[str, str] | None = None) -> dict:
    """95% percentile intervals from resampling the test rows with replacement.

    `scores` maps a model name to its scores on the same rows. Every model is scored on
    the same resampled rows, so `diff=(a, b)` gives a paired interval for
    metric(a) - metric(b).
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {name: [] for name in scores}
    if diff:
        draws["diff"] = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y_true[i].min() == y_true[i].max():  # a one-class resample has no AUC
            continue
        vals = {name: metric(y_true[i], np.asarray(s)[i]) for name, s in scores.items()}
        for name, v in vals.items():
            draws[name].append(v)
        if diff:
            draws["diff"].append(vals[diff[0]] - vals[diff[1]])
    out = {}
    for name, d in draws.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {"lo": float(lo), "hi": float(hi)}
    return out


def plot_calibration(y_true, probas: dict, path=None, n_bins=10):
    """Reliability curve for one or more sets of predicted PDs on the same rows."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    top = 0.0
    for label, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label, markersize=4)
        top = max(top, mean_pred.max(), frac_pos.max())
    top = min(1.0, top * 1.08)
    ax.plot([0, top], [0, top], "--", color="grey", label="perfect")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted PD (decile)"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration on the test split"); ax.legend()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
    return fig
