"""WoE-binned logistic regression scorecard: the bank-standard baseline.

Run:  python -m credit_risk_scorecard.woe_scorecard             # full data
      python -m credit_risk_scorecard.woe_scorecard --sample 20000

Steps:
1. Bin every feature on the train split only. Numeric features get up to 20 quantile
   bins, then adjacent bins are pooled until the bad rate is monotone in the feature
   (pool-adjacent-violators, increasing or decreasing, whichever gives the higher IV).
   Categorical features get one bin per level, with rare levels pooled into OTHER.
   Missing values get their own bin when there are enough of them.
2. Compute Weight of Evidence (WoE) and Information Value (IV) per feature. Features with
   IV below MIN_IV are dropped.
3. Fit a logistic regression on the WoE values (train only).
4. Scale to points: 600 points at 50:1 good:bad odds, 20 points to double the odds.
5. Compare with LightGBM on the same train / calibration / test split.

WoE here is ln(share of bads / share of goods), so a higher WoE means a riskier bin and
the fitted coefficients should be positive.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import evaluate as E
from . import features as F
from . import model as M
from .config import RANDOM_STATE

MIN_IV = 0.02
MAX_BINS = 20
MIN_BIN_SHARE = 0.03  # numeric bins smaller than this share of non-missing rows are pooled
MIN_LEVEL_SHARE = 0.01  # categorical levels rarer than this go to OTHER
SMOOTH = 0.5  # added to bad and good counts so no bin has an infinite WoE
PDO = 20.0
BASE_SCORE = 600.0
BASE_ODDS = 50.0  # good:bad
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
DECLINE_RATE = M.FLAG_RATE


def _woe(bad: np.ndarray, good: np.ndarray, total_bad: float, total_good: float) -> np.ndarray:
    return np.log(((bad + SMOOTH) / total_bad) / ((good + SMOOTH) / total_good))


def _iv(bad: np.ndarray, good: np.ndarray, total_bad: float, total_good: float) -> float:
    w = _woe(bad, good, total_bad, total_good)
    return float(np.sum((bad / total_bad - good / total_good) * w))


def _pav(bad: np.ndarray, tot: np.ndarray, increasing: bool) -> list[list[float]]:
    """Pool adjacent bins until bad rate is monotone. Returns [bad, tot, last_bin] groups."""
    stack: list[list[float]] = []
    for i in range(len(bad)):
        stack.append([float(bad[i]), float(tot[i]), i])
        while len(stack) >= 2:
            r0 = stack[-2][0] / stack[-2][1]
            r1 = stack[-1][0] / stack[-1][1]
            if (r0 > r1) if increasing else (r0 < r1):
                b, t, last = stack.pop()
                stack[-1][0] += b
                stack[-1][1] += t
                stack[-1][2] = last
            else:
                break
    return stack


@dataclass
class FeatureBins:
    name: str
    kind: str  # "numeric" or "categorical"
    labels: list[str]
    woe: np.ndarray
    counts: np.ndarray
    bad_rates: np.ndarray
    iv: float
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    levels: dict[str, int] = field(default_factory=dict)  # categorical level -> bin
    missing_bin: int = -1
    unseen_bin: int = -1

    def bin_index(self, x: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(self.edges, v, side="right")
            idx = np.where(np.isnan(v), self.missing_bin, idx)
            return idx.astype(int)
        s = x.astype("object").where(x.notna(), "MISSING").astype(str)
        return np.asarray(s.map(self.levels).fillna(self.unseen_bin).to_numpy(dtype=int))

    def transform(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        woe = np.append(self.woe, 0.0)  # a trailing neutral value for "no bin"
        return woe[np.where(idx < 0, len(self.woe), idx)]


def _fit_numeric(name: str, x: pd.Series, y: np.ndarray) -> FeatureBins:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    miss = np.isnan(v)
    total_bad, total_good = float(y.sum()), float((1 - y).sum())
    xv, yv = v[~miss], y[~miss]
    qs = np.quantile(xv, np.linspace(0, 1, MAX_BINS + 1)[1:-1])
    edges = np.unique(qs)
    idx = np.searchsorted(edges, xv, side="right")
    nb = len(edges) + 1
    bad = np.bincount(idx, weights=yv, minlength=nb)
    tot = np.bincount(idx, minlength=nb).astype(float)
    keep = tot > 0
    # Pool tiny bins into their smaller neighbour before the monotone pass.
    groups = [[bad[i], tot[i], i] for i in range(nb) if keep[i]]
    min_n = MIN_BIN_SHARE * len(xv)
    while len(groups) > 1:
        sizes = [g[1] for g in groups]
        j = int(np.argmin(sizes))
        if sizes[j] >= min_n:
            break
        k = (
            j - 1
            if j == len(groups) - 1
            else (j + 1 if j == 0 else (j - 1 if sizes[j - 1] <= sizes[j + 1] else j + 1))
        )
        lo, hi = sorted((j, k))
        groups[lo] = [groups[lo][0] + groups[hi][0], groups[lo][1] + groups[hi][1], groups[hi][2]]
        del groups[hi]
    bad_g = np.array([g[0] for g in groups])
    tot_g = np.array([g[1] for g in groups])
    best = None
    for inc in (True, False):
        pooled = _pav(bad_g, tot_g, inc)
        b = np.array([p[0] for p in pooled])
        t = np.array([p[1] for p in pooled])
        iv = _iv(b, t - b, total_bad, total_good)
        if best is None or iv > best[0]:
            best = (iv, pooled)
    pooled = best[1]  # type: ignore[index]
    last_bins = [int(groups[int(p[2])][2]) for p in pooled]
    cuts = np.array([edges[b] for b in last_bins[:-1]], dtype=float)
    b = np.array([p[0] for p in pooled])
    t = np.array([p[1] for p in pooled])
    n_miss = int(miss.sum())
    missing_bin = -1
    if n_miss > 0:
        mb, mt = float(y[miss].sum()), float(n_miss)
        if n_miss >= MIN_BIN_SHARE * len(v):
            b, t = np.append(b, mb), np.append(t, mt)
            missing_bin = len(b) - 1
        else:  # too few missing rows for their own bin: fold into the nearest bad rate
            j = int(np.argmin(np.abs(b / t - mb / mt)))
            b[j] += mb
            t[j] += mt
            missing_bin = j
    good = t - b
    woe = _woe(b, good, total_bad, total_good)
    iv = _iv(b, good, total_bad, total_good)
    labels = []
    for i in range(len(cuts) + 1):
        if i == 0:
            labels.append(f"< {cuts[0]:.4g}" if len(cuts) else "all")
        elif i == len(cuts):
            labels.append(f">= {cuts[-1]:.4g}")
        else:
            labels.append(f"{cuts[i - 1]:.4g} to < {cuts[i]:.4g}")
    n_main = len(cuts) + 1
    if missing_bin >= n_main:
        labels.append("MISSING")
    return FeatureBins(
        name, "numeric", labels, woe, t, b / t, iv, edges=cuts, missing_bin=missing_bin
    )


def _fit_categorical(name: str, x: pd.Series, y: np.ndarray) -> FeatureBins:
    s = x.astype("object").where(x.notna(), "MISSING").astype(str)
    share = s.value_counts(normalize=True)
    rare = set(share[share < MIN_LEVEL_SHARE].index) - {"MISSING"}
    s = s.where(~s.isin(rare), "OTHER")
    total_bad, total_good = float(y.sum()), float((1 - y).sum())
    df = pd.DataFrame({"lvl": s.to_numpy(), "y": y})
    g = df.groupby("lvl")["y"].agg(["sum", "count"])
    b, t = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    good = t - b
    levels = {lvl: i for i, lvl in enumerate(g.index)}
    for r in rare:
        levels[r] = levels.get("OTHER", -1)
    return FeatureBins(
        name,
        "categorical",
        list(g.index),
        _woe(b, good, total_bad, total_good),
        t,
        b / t,
        _iv(b, good, total_bad, total_good),
        levels=levels,
        unseen_bin=levels.get("OTHER", -1),
    )


class WoEScorecard:
    """Fit on train only. `fit` bins, filters on IV and fits the logistic regression."""

    def __init__(self, min_iv: float = MIN_IV, C: float = 1.0):
        self.min_iv = min_iv
        self.C = C
        self.bins: dict[str, FeatureBins] = {}
        self.selected: list[str] = []
        self.dropped: dict[str, float] = {}
        self.lr: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> WoEScorecard:
        yv = np.asarray(y, dtype=float)
        numeric, categorical = F.split_columns(X)
        for c in numeric:
            self.bins[c] = _fit_numeric(c, X[c], yv)
        for c in categorical:
            self.bins[c] = _fit_categorical(c, X[c], yv)
        self.selected = sorted(
            (c for c, b in self.bins.items() if b.iv >= self.min_iv),
            key=lambda c: -self.bins[c].iv,
        )
        self.dropped = {c: b.iv for c, b in self.bins.items() if b.iv < self.min_iv}
        # Refit, dropping the feature with the most negative coefficient each round, until
        # every coefficient is positive. A negative coefficient on a WoE column means the
        # feature's direction flipped against its own bins, which a scorecard cannot show.
        self.sign_dropped: list[str] = []
        W = self.woe_matrix(X)
        while True:
            self.lr = LogisticRegression(C=self.C, max_iter=1000).fit(W, yv)
            coef = self.lr.coef_[0]
            if coef.min() > 0 or len(self.selected) == 1:
                break
            j = int(np.argmin(coef))
            self.sign_dropped.append(self.selected.pop(j))
            W = np.delete(W, j, axis=1)
        return self

    def woe_matrix(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.bins[c].transform(X[c]) for c in self.selected])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        assert self.lr is not None
        return np.asarray(self.lr.predict_proba(self.woe_matrix(X))[:, 1])

    def points_table(self) -> pd.DataFrame:
        """One row per feature bin: the points it adds. Score = sum of one row per feature."""
        assert self.lr is not None
        n = len(self.selected)
        b0 = float(self.lr.intercept_[0])
        rows = []
        for c, coef in zip(self.selected, self.lr.coef_[0], strict=True):
            fb = self.bins[c]
            pts = -FACTOR * (coef * fb.woe + b0 / n) + OFFSET / n
            for i, lab in enumerate(fb.labels):
                rows.append(
                    {
                        "feature": c,
                        "bin": lab,
                        "n_train": int(fb.counts[i]),
                        "bad_rate": float(fb.bad_rates[i]),
                        "woe": float(fb.woe[i]),
                        "coef": float(coef),
                        "points": float(pts[i]),
                    }
                )
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points for each row, from the bins. Equals OFFSET - FACTOR * logit(PD)."""
        assert self.lr is not None
        n = len(self.selected)
        b0 = float(self.lr.intercept_[0])
        total = np.zeros(len(X))
        for c, coef in zip(self.selected, self.lr.coef_[0], strict=True):
            total += -FACTOR * (coef * self.bins[c].transform(X[c]) + b0 / n) + OFFSET / n
        return total


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index from two arrays of bin shares."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(sc: WoEScorecard, X_ref: pd.DataFrame, X_new: pd.DataFrame) -> dict[str, float]:
    out = {}
    for c in sc.selected:
        fb = sc.bins[c]
        k = len(fb.woe)
        ref = np.bincount(fb.bin_index(X_ref[c])[fb.bin_index(X_ref[c]) >= 0], minlength=k)
        new = np.bincount(fb.bin_index(X_new[c])[fb.bin_index(X_new[c]) >= 0], minlength=k)
        out[c] = psi(ref / ref.sum(), new / new.sum())
    return out


def score_psi(s_ref: np.ndarray, s_new: np.ndarray, n_bins: int = 10) -> float:
    """PSI of the score, using decile cuts taken from the reference (train) scores."""
    cuts = np.unique(np.quantile(s_ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    ref = np.bincount(np.searchsorted(cuts, s_ref, side="right"), minlength=len(cuts) + 1)
    new = np.bincount(np.searchsorted(cuts, s_new, side="right"), minlength=len(cuts) + 1)
    return psi(ref / ref.sum(), new / new.sum())


def approved_bad_rate(y: ArrayLike, pd_hat: ArrayLike, threshold: float) -> dict[str, float]:
    y = np.asarray(y)
    approved = np.asarray(pd_hat) < threshold
    return {
        "decline_rate": float(1 - approved.mean()),
        "approved_share": float(approved.mean()),
        "bad_rate_approved": float(y[approved].mean()),
        "bad_rate_declined": float(y[~approved].mean()),
    }


def compare(
    art: dict[str, Any], n_boot: int = 1000
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit the scorecard on the same split `model.run` used and compare on test.

    `art` is the dict `model.run(..., artifacts=art)` fills in. Returns metrics, the points
    table, and a per-feature IV / PSI table.
    """
    X_tr, X_cal, X_te = art["X_tr"], art["X_cal"], art["X_te"]
    y_tr, y_te = art["y_tr"], art["y_te"]
    sc = WoEScorecard().fit(X_tr, y_tr)
    p_sc, p_sc_cal = sc.predict_pd(X_te), sc.predict_pd(X_cal)
    p_gb, p_gb_iso = art["p_gb"], art["pd_cal"]

    scores = {"scorecard": p_sc, "lightgbm": p_gb}
    boot = E.bootstrap_ci(y_te, scores, roc_auc_score, n_boot, diff=("lightgbm", "scorecard"))
    thr_sc = float(np.quantile(p_sc_cal, 1 - DECLINE_RATE))
    s_tr, s_te = sc.score(X_tr), sc.score(X_te)
    fpsi = feature_psi(sc, X_tr, X_te)
    coef = dict(zip(sc.selected, sc.lr.coef_[0], strict=True))  # type: ignore[union-attr]
    iv_table = pd.DataFrame(
        {
            "feature": list(sc.bins),
            "iv": [sc.bins[c].iv for c in sc.bins],
            "kept": [c in sc.selected for c in sc.bins],
            "n_bins": [len(sc.bins[c].woe) for c in sc.bins],
            "psi_train_vs_test": [fpsi.get(c, np.nan) for c in sc.bins],
        }
    ).sort_values("iv", ascending=False)
    m: dict[str, Any] = {
        "n_candidate_features": len(sc.bins),
        "n_kept": len(sc.selected),
        "n_dropped_iv_below_min": len(sc.dropped),
        "min_iv": MIN_IV,
        "n_dropped_for_negative_coefficient": len(sc.sign_dropped),
        "n_negative_coefficients": int(sum(v <= 0 for v in coef.values())),
        "top_iv": [(c, round(sc.bins[c].iv, 4)) for c in sc.selected[:10]],
        "points": {"pdo": PDO, "base_score": BASE_SCORE, "base_odds_good_to_bad": BASE_ODDS},
        "roc_auc": {
            "scorecard": float(roc_auc_score(y_te, p_sc)),
            "lightgbm": float(roc_auc_score(y_te, p_gb)),
            "ci_scorecard": boot["scorecard"],
            "ci_lightgbm": boot["lightgbm"],
            "gap_lightgbm_minus_scorecard_ci": boot["diff"],
            "n_boot": n_boot,
        },
        "calibration": {
            "scorecard": {"brier": E.brier(y_te, p_sc), "ece": E.ece(y_te, p_sc)},
            "lightgbm_raw": {"brier": E.brier(y_te, p_gb), "ece": E.ece(y_te, p_gb)},
            "lightgbm_isotonic": {"brier": E.brier(y_te, p_gb_iso), "ece": E.ece(y_te, p_gb_iso)},
            "mean_pd_scorecard": float(p_sc.mean()),
            "observed_default_rate_test": float(np.mean(y_te)),
        },
        "approved_bad_rate_at_20pct_decline": {
            "scorecard": approved_bad_rate(y_te, p_sc, thr_sc),
            "lightgbm": approved_bad_rate(y_te, p_gb, art["threshold"]),
            "all_applicants": float(np.mean(y_te)),
        },
        "psi_train_vs_test": {
            "score": score_psi(s_tr, s_te),
            "max_feature": max(fpsi.values()),
            "max_feature_name": max(fpsi, key=fpsi.get),  # type: ignore[arg-type]
            "n_features_above_0.10": int(sum(v > 0.10 for v in fpsi.values())),
            "n_features_above_0.25": int(sum(v > 0.25 for v in fpsi.values())),
        },
    }
    m["calibration"]["scorecard"]["note"] = "unweighted logistic fit, no extra calibrator"
    return m, sc.points_table(), iv_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    art: dict[str, Any] = {}
    base = M.run(
        M.load_frame(), sample=a.sample, n_boot=a.n_boot, figures=False, save=False, artifacts=art
    )
    m, table, iv_table = compare(art, a.n_boot)
    m["data"] = base["data"]
    M.REPORTS_DIR.mkdir(exist_ok=True)
    (M.REPORTS_DIR / "woe_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    table.round(4).to_csv(M.REPORTS_DIR / "scorecard_table.csv", index=False)
    iv_table.round(4).to_csv(M.REPORTS_DIR / "woe_iv_psi.csv", index=False)
    print(json.dumps(m, indent=2))
    print(f"random_state={RANDOM_STATE}")


if __name__ == "__main__":
    main()
