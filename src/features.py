"""Feature engineering + preprocessing for the credit-risk scorecard.

Decisions here are driven by the Sprint 1 EDA:
- DAYS_EMPLOYED has a sentinel value 365243 that means "not employed / unknown" -> NaN.
- EXT_SOURCE_1/2/3 are the strongest raw separators -> keep, impute gently.
- Heavy missingness in some building-info columns is left to the imputer (baseline);
  informative-missingness flags are a future improvement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DAYS_EMPLOYED_SENTINEL = 365243
ID_COL = "SK_ID_CURR"
TARGET = "TARGET"
# Kept in the data for the group check in src/fairness.py, but never given to the model.
# Canadian human rights law bars discrimination on sex in services, credit included.
NOT_MODEL_INPUTS = ("CODE_GENDER",)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning + a few engineered ratio features."""
    out = df.copy()
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    # Engineered ratios (guard against divide-by-zero -> inf -> NaN).
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def split_columns(X: pd.DataFrame, exclude=NOT_MODEL_INPUTS) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for the model: no id column, and none of
    the columns in `exclude`."""
    skip = {ID_COL, *exclude}
    numeric = [c for c in X.select_dtypes(include=[np.number]).columns if c not in skip]
    # Everything that is not numeric. Works for object and pandas 3 "str" columns alike.
    categorical = [c for c in X.columns if c not in numeric and c not in skip]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, exclude=NOT_MODEL_INPUTS) -> ColumnTransformer:
    """Impute + standardise numerics, impute + one-hot categoricals. Dense output.

    The first version used StandardScaler(with_mean=False) to keep the matrix sparse.
    That divides by the standard deviation but does not centre. Together with the saga
    solver, the logistic regression hit its iteration cap. Centring plus lbfgs converges.
    src/logreg_check.py compares the setups; ROC-AUC barely moves.
    Centring needs a dense matrix, which fits in memory at this size.
    Columns not listed here (the id and anything in `exclude`) are dropped.
    """
    numeric, categorical = split_columns(X, exclude)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], sparse_threshold=0.0)
