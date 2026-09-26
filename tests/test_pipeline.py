"""End-to-end run on synthetic data: every section of the report gets filled."""

from credit_risk_scorecard import model as M


def test_run_end_to_end(applicants):
    m = M.run(applicants, n_boot=20, figures=False, save=False)
    assert m["data"]["rows_used"] == len(applicants)
    assert m["logreg_convergence"]["converged"]
    for mdl in ("logreg", "lightgbm"):
        assert 0.5 < m["models"][mdl]["roc_auc"] <= 1
    # Calibrating on held-out data should bring the mean PD close to the observed rate.
    obs = m["calibration"]["observed_default_rate_test"]
    assert abs(m["calibration"]["isotonic"]["mean_pd"] - obs) < abs(
        m["calibration"]["raw"]["mean_pd"] - obs
    )
    assert {r["group"] for r in m["fairness"]["gender"]} == {"F", "M"}
    assert m["data"]["excluded_from_model"] == ["CODE_GENDER"]
    md = M.results_markdown(m)
    assert "## Calibration" in md and "ILLUSTRATIVE" in md
