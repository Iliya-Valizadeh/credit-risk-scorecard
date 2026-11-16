"""End-to-end run on synthetic data: every section of the report gets filled."""

import pandas as pd

from credit_risk_scorecard import model as M


def test_run_end_to_end(applicants: pd.DataFrame) -> None:
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


def test_group_table_leaves_rates_empty_when_a_group_has_no_defaulters() -> None:
    import numpy as np

    from credit_risk_scorecard import fairness

    groups = np.array(["a"] * 120 + ["b"] * 120)
    y = np.array([0, 1] * 60 + [0] * 120)  # group b has no defaulters
    raw = np.linspace(0, 1, 240)
    t = fairness.group_table(groups, y, raw, raw, threshold=0.5, min_n=100).set_index("group")
    assert np.isnan(t.loc["b", "tpr"]) and not np.isnan(t.loc["a", "tpr"])
    # Groups under min_n are dropped.
    small = fairness.group_table(groups, y, raw, raw, threshold=0.5, min_n=200)
    assert small.empty
