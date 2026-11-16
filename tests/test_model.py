"""model.py outside the happy path: sampling, the template row, saved files and the CLI.

Every file the code writes goes to a temporary folder, so nothing under reports/ or
models/ is touched.
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_risk_scorecard import features as F
from credit_risk_scorecard import model as M
from credit_risk_scorecard.synthetic import make_applicants


def test_split_with_a_sample_keeps_the_class_mix(applicants: pd.DataFrame) -> None:
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = M.split(applicants, sample=1000)
    assert len(X_tr) + len(X_cal) + len(X_te) == 1000
    assert (len(X_tr), len(X_cal), len(X_te)) == (600, 200, 200)
    assert abs(y_te.mean() - applicants["TARGET"].mean()) < 0.03


def test_baseline_and_boosted_models_fit_and_predict(applicants: pd.DataFrame) -> None:
    (X_tr, y_tr), _, _ = M.split(applicants, sample=600)
    Xt = F.build_preprocessor(X_tr).fit_transform(X_tr).astype(np.float32)
    lr = M.train_baseline(Xt, y_tr, max_iter=500)
    assert lr.predict_proba(Xt).shape == (len(y_tr), 2)
    gb = M.train_boosted(Xt, y_tr)
    p = gb.predict_proba(Xt)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_template_row_uses_medians_and_modes() -> None:
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 10.0],
            "cat": ["a", "b", "b"],
            "empty_cat": pd.Series([None, None, None], dtype="object"),
        }
    )
    row = M.template_row(df)
    assert row["num"] == 2.0
    assert row["cat"] == "b"
    assert row["empty_cat"] is None


def test_save_artifacts_writes_a_loadable_pipeline(
    trained: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, art = trained
    monkeypatch.setattr(M, "MODELS_DIR", tmp_path / "models")
    M.save_artifacts(art["pre"], art["lgbm"], "calibrator", art["X_tr"], 0.5)
    saved = joblib.load(tmp_path / "models" / "pipeline.joblib")
    assert saved["threshold"] == 0.5
    assert saved["columns"] == list(art["X_tr"].columns)
    assert saved["calibrator"] == "calibrator"
    assert set(saved["template"]) == set(art["X_tr"].columns)
    assert "[saved]" in capsys.readouterr().out


def test_write_reports_and_results_markdown(
    trained: tuple[dict[str, Any], dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m, _ = trained
    monkeypatch.setattr(M, "REPORTS_DIR", tmp_path / "reports")
    M.write_reports(m)
    back = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert back["data"]["rows_used"] == m["data"]["rows_used"]
    md = (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert md == M.results_markdown(m)
    for heading in ("## Ranking", "## Calibration", "## Threshold table", "## Operating point"):
        assert heading in md
    assert "## Top 10 features" not in md  # this run made no figures, so no SHAP list


def test_results_markdown_lists_shap_features_when_present(
    trained: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    m, _ = trained
    with_shap = {**m, "shap_top10": [("EXT_SOURCE_2", 0.5), ("DAYS_BIRTH", 0.25)]}
    md = M.results_markdown(with_shap)
    assert "1. `EXT_SOURCE_2` (0.500)" in md and "2. `DAYS_BIRTH` (0.250)" in md


def test_main_runs_the_cli_and_writes_figures_models_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    df = make_applicants(n=2000, seed=7)
    monkeypatch.setattr(M, "load_frame", lambda from_db=False: df)
    monkeypatch.setattr(M, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(M, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(M, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(sys, "argv", ["model", "--sample", "1500", "--n-boot", "10"])
    M.main()
    assert (tmp_path / "models" / "pipeline.joblib").exists()
    assert (tmp_path / "figures" / "calibration.png").exists()
    assert (tmp_path / "figures" / "shap_summary.png").exists()
    m = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert m["data"]["sample"] == 1500
    assert len(m["shap_top10"]) == 10
    assert "## Top 10 features" in (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert "# Results" in capsys.readouterr().out
