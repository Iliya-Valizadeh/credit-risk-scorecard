"""The logistic regression convergence check on synthetic data."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk_scorecard import features as F
from credit_risk_scorecard import logreg_check as L
from credit_risk_scorecard import model as M
from credit_risk_scorecard.synthetic import make_applicants


def test_old_preprocessor_does_not_centre(applicants: pd.DataFrame) -> None:
    X = F.clean(applicants).drop(columns=["TARGET"])
    Xt = L.old_preprocessor(X).fit_transform(X)
    numeric, _ = F.split_columns(X)
    assert Xt.shape[0] == len(X)
    # Scaled without centring: the numeric columns keep a non-zero mean.
    dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert abs(dense[:, : len(numeric)].mean(axis=0)).max() > 0.01


def test_fit_and_score_reports_the_fit(applicants: pd.DataFrame) -> None:
    (X_tr, y_tr), _, (X_te, y_te) = M.split(applicants)
    r = L.fit_and_score(
        F.build_preprocessor(X_tr),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X_tr,
        y_tr,
        X_te,
        y_te,
        "float32",
    )
    assert r["matrix"] == "dense float32"
    assert r["n_iter"] <= r["max_iter"]
    assert 0.5 < r["roc_auc"] <= 1
    assert 0 < r["pr_auc"] <= 1


def test_main_writes_the_markdown_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    df = make_applicants(n=1500, seed=3)
    monkeypatch.setattr(M, "load_frame", lambda from_db=False: df)
    monkeypatch.setattr(M, "REPORTS_DIR", tmp_path)
    L.main()
    text = (tmp_path / "logreg_convergence.md").read_text(encoding="utf-8")
    assert "# Logistic regression: convergence before and after" in text
    assert "| Old: no centring, saga, max_iter=1000 |" in text
    assert "ROC-AUC change (new minus old)" in capsys.readouterr().out
