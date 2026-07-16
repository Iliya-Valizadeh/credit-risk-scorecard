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


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols), excluding the id column."""
    numeric = [c for c in X.select_dtypes(include=[np.number]).columns if c != ID_COL]
    categorical = [c for c in X.select_dtypes(include=["object", "category"]).columns]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot categoricals. Sparse-friendly."""
    numeric, categorical = split_columns(X)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),  # keep sparse-compatible
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
