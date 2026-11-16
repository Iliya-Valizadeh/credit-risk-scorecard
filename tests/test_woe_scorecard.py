"""WoE scorecard: bins are monotone, points add up, no leakage from test, PSI is sane."""

from typing import Any

import numpy as np
import pandas as pd

from credit_risk_scorecard import model as M
from credit_risk_scorecard import woe_scorecard as W


def _fit(
    applicants: pd.DataFrame,
) -> tuple[W.WoEScorecard, pd.DataFrame, pd.DataFrame, pd.Series]:
    (X_tr, y_tr), _, (X_te, y_te) = M.split(applicants)
    return W.WoEScorecard().fit(X_tr, y_tr), X_tr, X_te, y_te


def test_numeric_bins_are_monotone(applicants: pd.DataFrame) -> None:
    sc, *_ = _fit(applicants)
    for fb in sc.bins.values():
        if fb.kind != "numeric":
            continue
        n_main = len(fb.edges) + 1
        rates = fb.bad_rates[:n_main]
        assert np.all(np.diff(rates) >= -1e-12) or np.all(np.diff(rates) <= 1e-12), fb.name


def test_selection_and_signs(applicants: pd.DataFrame) -> None:
    sc, *_ = _fit(applicants)
    assert sc.selected
    assert all(sc.bins[c].iv >= W.MIN_IV for c in sc.selected)
    assert all(v < W.MIN_IV for v in sc.dropped.values())
    assert sc.lr is not None
    assert sc.lr.coef_.min() > 0


def test_points_sum_to_score_and_scale(applicants: pd.DataFrame) -> None:
    sc, X_tr, X_te, _ = _fit(applicants)
    p = np.clip(sc.predict_pd(X_te), 1e-9, 1 - 1e-9)
    expected = W.OFFSET - W.FACTOR * np.log(p / (1 - p))
    assert np.allclose(sc.score(X_te), expected, atol=1e-6)
    # 600 points is 50:1 odds, and 20 more points double the odds.
    assert np.isclose(W.OFFSET + W.FACTOR * np.log(50), 600)
    assert np.isclose(W.FACTOR * np.log(2), 20)
    table = sc.points_table()
    assert {"feature", "bin", "woe", "points"} <= set(table.columns)


def test_unseen_and_missing_values_do_not_crash(applicants: pd.DataFrame) -> None:
    sc, _, X_te, _ = _fit(applicants)
    X = X_te.copy().head(50)
    for c in sc.selected:
        X.loc[X.index[0], c] = np.nan
    cat = [c for c in sc.selected if sc.bins[c].kind == "categorical"]
    if cat:
        X[cat[0]] = X[cat[0]].astype("object")
        X.loc[X.index[1], cat[0]] = "never seen before"
    assert np.isfinite(sc.predict_pd(X)).all()


def test_psi() -> None:
    same = np.array([0.25, 0.25, 0.5])
    assert W.psi(same, same) == 0
    assert W.psi(same, np.array([0.5, 0.25, 0.25])) > 0.1
    s = np.random.default_rng(0).normal(size=5000)
    assert W.score_psi(s, s) < 1e-9
    assert W.score_psi(s, s + 2) > 0.25


def test_approved_bad_rate() -> None:
    y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    pd_hat = np.linspace(0.01, 0.9, 10)
    r = W.approved_bad_rate(y, pd_hat, np.quantile(pd_hat, 0.8))
    assert r["approved_share"] == 0.8
    assert r["bad_rate_approved"] == 0.125
    assert r["bad_rate_declined"] == 1.0


def test_compare_runs_end_to_end(applicants: pd.DataFrame) -> None:
    art: dict[str, Any] = {}
    M.run(applicants, n_boot=10, figures=False, save=False, artifacts=art)
    m, table, iv = W.compare(art, n_boot=20)
    assert 0.5 < m["roc_auc"]["scorecard"] <= 1
    assert m["psi_train_vs_test"]["score"] >= 0
    assert isinstance(table, pd.DataFrame) and len(table) > 0
    assert iv["kept"].sum() == m["n_kept"]
