"""WoE scorecard: bins are monotone, points add up, no leakage from test, PSI is sane."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk_scorecard import model as M
from credit_risk_scorecard import woe_scorecard as W
from credit_risk_scorecard.synthetic import make_applicants


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


def test_tiny_numeric_bins_are_pooled_into_a_neighbour() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    # 2% of rows sit below a big block of ties, so the first quantile bin is under 3%.
    x = np.concatenate([rng.uniform(0, 4, 80), np.full(800, 5.0), rng.uniform(6, 10, n - 880)])
    y = (rng.random(n) < 0.1 + 0.05 * (x > 5)).astype(float)
    fb = W._fit_numeric("x", pd.Series(x), y)
    assert fb.counts.sum() == n
    assert (fb.counts >= W.MIN_BIN_SHARE * n).all()


def test_few_missing_values_fold_into_a_bin_and_many_get_their_own() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(0, 10, n)
    y = (rng.random(n) < 0.05 + 0.02 * x).astype(float)
    few = x.copy()
    few[:20] = np.nan  # 0.5%: under the 3% cut-off
    fb_few = W._fit_numeric("x", pd.Series(few), y)
    assert 0 <= fb_few.missing_bin < len(fb_few.woe)
    assert "MISSING" not in fb_few.labels
    many = x.copy()
    many[:400] = np.nan  # 10%
    fb_many = W._fit_numeric("x", pd.Series(many), y)
    assert fb_many.missing_bin == len(fb_many.woe) - 1
    assert fb_many.labels[-1] == "MISSING"
    # A missing value maps to that bin's WoE, and an all-neutral value to 0.
    out = fb_many.transform(pd.Series([np.nan]))
    assert out[0] == pytest.approx(fb_many.woe[fb_many.missing_bin])


def test_categorical_rare_levels_pool_into_other_and_unseen_levels_use_it() -> None:
    rng = np.random.default_rng(2)
    n = 5000
    lvl = rng.choice(["A", "B"], n).astype(object)
    lvl[:10] = "RARE"  # 0.2% of rows
    x = pd.Series(lvl, dtype="object")
    x[10:400] = np.nan
    y = ((x == "A") & (rng.random(n) < 0.3) | (x != "A") & (rng.random(n) < 0.05)).to_numpy(float)
    fb = W._fit_categorical("c", x, y)
    assert set(fb.labels) == {"A", "B", "MISSING", "OTHER"}
    other = fb.labels.index("OTHER")
    assert fb.levels["RARE"] == other and fb.unseen_bin == other
    idx = fb.bin_index(pd.Series(["A", "RARE", "never seen", None]))
    assert idx[1] == other and idx[2] == other
    assert idx[3] == fb.labels.index("MISSING")
    assert fb.iv > 0


def test_categorical_without_rare_levels_treats_unseen_values_as_neutral() -> None:
    x = pd.Series(["A", "B"] * 500)
    y = np.array([1.0, 0.0] * 500)
    fb = W._fit_categorical("c", x, y)
    assert fb.unseen_bin == -1
    assert fb.transform(pd.Series(["Z"]))[0] == 0.0


def test_a_feature_with_a_negative_coefficient_is_dropped() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    b = rng.normal(size=n)
    a = b + rng.normal(size=n)
    # Each feature predicts the outcome alone, but together b gets a negative weight.
    p = 1 / (1 + np.exp(-(-2 + 0.9 * (2 * a - b))))
    y = (rng.random(n) < p).astype(int)
    sc = W.WoEScorecard(min_iv=0.0).fit(pd.DataFrame({"a": a, "b": b}), y)
    assert sc.sign_dropped == ["b"]
    assert sc.selected == ["a"]
    assert sc.lr is not None and sc.lr.coef_.min() > 0


def test_main_writes_the_scorecard_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    df = make_applicants(n=2000, seed=11)
    monkeypatch.setattr(M, "load_frame", lambda from_db=False: df)
    monkeypatch.setattr(M, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["woe_scorecard", "--sample", "1500", "--n-boot", "10"])
    W.main()
    m = json.loads((tmp_path / "woe_metrics.json").read_text(encoding="utf-8"))
    assert m["data"]["sample"] == 1500
    assert 0.5 < m["roc_auc"]["scorecard"] <= 1
    table = pd.read_csv(tmp_path / "scorecard_table.csv")
    assert {"feature", "bin", "points"} <= set(table.columns)
    iv = pd.read_csv(tmp_path / "woe_iv_psi.csv")
    assert len(iv) == m["n_candidate_features"]
    assert "random_state=" in capsys.readouterr().out
