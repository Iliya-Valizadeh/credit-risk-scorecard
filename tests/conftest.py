"""Synthetic applicants with the same column names as application_train.csv.

The tests never need the Kaggle data, so they run in CI.
"""
import numpy as np
import pandas as pd
import pytest


def make_applicants(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ext2 = rng.uniform(0, 1, n)
    income = rng.lognormal(11.8, 0.5, n)
    credit = income * rng.uniform(1, 6, n)
    annuity = credit * rng.uniform(0.03, 0.08, n)
    days_birth = -rng.integers(21 * 365, 69 * 365, n)
    days_employed = -rng.integers(0, 40 * 365, n)
    days_employed[rng.random(n) < 0.15] = 365243          # the sentinel
    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    edu = rng.choice(["Secondary", "Higher education"], n)
    logit = -2.6 - 2.5 * (ext2 - 0.5) + 0.3 * (gender == "M")
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "TARGET": target,
        "CODE_GENDER": gender,
        "NAME_EDUCATION_TYPE": edu,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_2": ext2,
    })
    df.loc[rng.random(n) < 0.1, "EXT_SOURCE_2"] = np.nan
    return df


@pytest.fixture
def applicants():
    return make_applicants()
