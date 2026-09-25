import numpy as np
import pandas as pd

from credit_risk_scorecard import features as F


def test_days_employed_sentinel_becomes_missing():
    df = pd.DataFrame({"DAYS_EMPLOYED": [-100, 365243, -5]})
    out = F.clean(df)
    assert out["DAYS_EMPLOYED"].isna().tolist() == [False, True, False]


def test_ratios_are_computed_and_divide_by_zero_is_missing():
    df = pd.DataFrame(
        {"AMT_CREDIT": [100.0, 50.0], "AMT_INCOME_TOTAL": [50.0, 0.0], "AMT_ANNUITY": [10.0, 5.0]}
    )
    out = F.clean(df)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == 2.0
    assert out.loc[0, "ANNUITY_CREDIT_RATIO"] == 0.1
    assert np.isnan(out.loc[1, "CREDIT_INCOME_RATIO"])  # inf -> NaN
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


def test_clean_does_not_modify_input():
    df = pd.DataFrame({"DAYS_EMPLOYED": [365243]})
    F.clean(df)
    assert df.loc[0, "DAYS_EMPLOYED"] == 365243


def test_preprocessor_centres_numeric_columns(applicants):
    X = F.clean(applicants).drop(columns=["TARGET"])
    pre = F.build_preprocessor(X)
    Xt = pre.fit_transform(X)
    numeric, _ = F.split_columns(X)
    means = Xt[:, : len(numeric)].mean(axis=0)
    assert np.allclose(means, 0, atol=1e-6)
    assert not np.isnan(Xt).any()
