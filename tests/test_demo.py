"""`make demo` runs offline, on synthetic data, and writes nothing under reports/ or
models/ (ADR 0001, decision 4)."""

from credit_risk_scorecard import demo


def test_demo_runs_and_scores_synthetic_applicants(capsys, tmp_path, monkeypatch):
    reports_before = set((demo.M.REPORTS_DIR).glob("*")) if demo.M.REPORTS_DIR.exists() else set()
    models_before = set((demo.M.MODELS_DIR).glob("*")) if demo.M.MODELS_DIR.exists() else set()

    assert demo.main() == 0

    out = capsys.readouterr().out
    assert "synthetic" in out.lower()
    assert "not the project's results" in out.lower()
    assert "approve" in out.lower() or "decline" in out.lower()

    reports_after = set((demo.M.REPORTS_DIR).glob("*")) if demo.M.REPORTS_DIR.exists() else set()
    models_after = set((demo.M.MODELS_DIR).glob("*")) if demo.M.MODELS_DIR.exists() else set()
    assert reports_after == reports_before
    assert models_after == models_before
