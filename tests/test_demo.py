"""`make demo` runs offline, on synthetic data, and writes nothing under reports/ or
models/ (ADR 0001, decision 4)."""

from pathlib import Path

import pytest

from credit_risk_scorecard import demo
from credit_risk_scorecard import model as M


def test_demo_runs_and_scores_synthetic_applicants(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports_before = set((M.REPORTS_DIR).glob("*")) if M.REPORTS_DIR.exists() else set()
    models_before = set((M.MODELS_DIR).glob("*")) if M.MODELS_DIR.exists() else set()

    assert demo.main() == 0

    out = capsys.readouterr().out
    assert "synthetic" in out.lower()
    assert "not the project's results" in out.lower()
    assert "approve" in out.lower() or "decline" in out.lower()

    reports_after = set((M.REPORTS_DIR).glob("*")) if M.REPORTS_DIR.exists() else set()
    models_after = set((M.MODELS_DIR).glob("*")) if M.MODELS_DIR.exists() else set()
    assert reports_after == reports_before
    assert models_after == models_before
