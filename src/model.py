"""Train baseline (logistic regression) + boosted (LightGBM), and persist artifacts.

Run:  python -m src.model
Produces models/pipeline.joblib for the Streamlit demo.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE, TARGET, ROOT, DATA_DIR, db_url
from . import features as F
from . import evaluate as E

MODELS_DIR = ROOT / "models"


def load_frame():
    try:
        from sqlalchemy import create_engine
        return pd.read_sql("SELECT * FROM application_train", create_engine(db_url()))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] DB read failed ({exc}); reading CSV")
        return pd.read_csv(DATA_DIR / "application_train.csv")


def prepare(df):
    df = F.clean(df)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pre = F.build_preprocessor(Xtr)
    Xtr_t = pre.fit_transform(Xtr)
    Xva_t = pre.transform(Xva)
    return pre, Xtr, Xtr_t, Xva_t, ytr, yva


def train_baseline(Xtr, ytr):
    clf = LogisticRegression(solver="saga", max_iter=1000, n_jobs=-1,
                             class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf


def train_boosted(Xtr, ytr):
    import lightgbm as lgb
    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    clf = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


def template_row(X_raw: pd.DataFrame) -> dict:
    """A 'typical applicant' — medians for numerics, modes for categoricals.
    The Streamlit app overlays user inputs on this so it can score from a few fields."""
    row = {}
    for c in X_raw.columns:
        s = X_raw[c]
        if s.dtype.kind in "biufc":
            row[c] = float(s.median())
        else:
            row[c] = s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else None
    return row


def save_artifacts(pre, model, X_raw, threshold=0.5):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"preprocessor": pre, "model": model,
         "template": template_row(X_raw), "columns": list(X_raw.columns),
         "threshold": threshold},
        MODELS_DIR / "pipeline.joblib",
    )
    print(f"[saved] {MODELS_DIR/'pipeline.joblib'}")


def main():
    pre, Xtr_raw, Xtr, Xva, ytr, yva = prepare(load_frame())
    for name, trainer in [("logreg", train_baseline), ("lightgbm", train_boosted)]:
        clf = trainer(Xtr, ytr)
        proba = clf.predict_proba(Xva)[:, 1]
        print(f"\n=== {name} ===")
        print(E.headline(yva, proba))
        print(E.threshold_table(yva, proba).to_string(index=False))
        if name == "lightgbm":
            save_artifacts(pre, clf, Xtr_raw, threshold=0.5)


if __name__ == "__main__":
    main()
