"""The gender check on synthetic data. Reports go to a temporary folder."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from credit_risk_scorecard import gender_check as G
from credit_risk_scorecard import model as M
from credit_risk_scorecard.synthetic import make_applicants


def test_gender_rows_returns_women_then_men(
    trained: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    m, _ = trained
    f, mm = G.gender_rows(m)
    assert (f["group"], mm["group"]) == ("F", "M")
    assert set(G.GROUP_COLS) <= set(f)


def test_proxy_table_columns_and_ordering(
    trained: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, art = trained
    proxies, most_corr = G.proxy_table(art, top_shap=5, min_corr=0.0)
    assert {"feature", "corr_with_male", "mean_abs_shap", "shap_rank"} <= set(most_corr.columns)
    assert len(most_corr) <= 10
    assert (proxies["shap_rank"] <= 5).all()
    assert proxies["shap_rank"].is_monotonic_increasing
    assert (most_corr["corr_with_male"].abs() <= 1 + 1e-9).all()


def test_gender_predictability_is_an_auc(
    trained: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, art = trained
    auc = G.gender_predictability(art)
    assert 0.0 <= auc <= 1.0


def test_main_writes_the_report_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    df: pd.DataFrame = make_applicants(n=2000, seed=5)
    monkeypatch.setattr(M, "load_frame", lambda from_db=False: df)
    monkeypatch.setattr(M, "REPORTS_DIR", tmp_path)
    G.main()
    md = (tmp_path / "gender_check.md").read_text(encoding="utf-8")
    assert "# Taking CODE_GENDER out of the model" in md
    out = json.loads((tmp_path / "gender_check.json").read_text(encoding="utf-8"))
    assert {"with_gender", "without_gender", "gender_predictability_auc"} <= set(out)
    assert "Gap (men minus women)" in capsys.readouterr().out
