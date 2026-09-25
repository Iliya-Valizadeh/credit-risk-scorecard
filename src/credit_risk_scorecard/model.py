"""Train, calibrate and evaluate the scorecard, and write every number the README quotes.

Run:  python -m credit_risk_scorecard.model             # full data, all 307,511 rows
      python -m credit_risk_scorecard.model --sample 20000   # quick stratified sample
      python -m credit_risk_scorecard.model --from-db        # read application_train from PG

Writes reports/metrics.json, reports/results.md, reports/figures/*.png and
models/pipeline.joblib (used by the Streamlit demo).

Split: 60% train / 20% calibration / 20% test, stratified on TARGET. The models only see
train. The calibrators only see calibration. Every reported number is on test.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import evaluate as E
from . import fairness
from . import features as F
from .calibrate import CALIBRATORS
from .config import DATA_DIR, FIGURES_DIR, RANDOM_STATE, ROOT, TARGET, db_url
from .expected_loss import expected_loss_table

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FLAG_RATE = 0.20  # illustrative policy: decline the riskiest 20% of applicants
DOWNSTREAM_CALIBRATOR = "isotonic"  # chosen before looking at test results; see README


def load_frame(from_db: bool = False) -> pd.DataFrame:
    if from_db:
        from sqlalchemy import create_engine

        return pd.read_sql("SELECT * FROM application_train", create_engine(db_url()))
    return pd.read_csv(DATA_DIR / "application_train.csv")


def split(df: pd.DataFrame, sample: int | None = None):
    """Clean, optionally sample, and split 60/20/20 into train / calibration / test."""
    if sample:
        df, _ = train_test_split(
            df, train_size=sample, stratify=df[TARGET], random_state=RANDOM_STATE
        )
    df = F.clean(df)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE
    )
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def train_baseline(Xtr, ytr, max_iter=2000):
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf


def train_boosted(Xtr, ytr):
    import lightgbm as lgb

    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    # The first version also passed subsample=0.8, which LightGBM ignores unless
    # subsample_freq > 0. It was removed so the config says what the model does.
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=31,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(Xtr, ytr)
    return clf


def template_row(X_raw: pd.DataFrame) -> dict:
    """A 'typical applicant': medians for numerics, modes for categoricals.
    The Streamlit app overlays user inputs on this so it can score from a few fields."""
    row: dict[str, Any] = {}
    for c in X_raw.columns:
        s = X_raw[c]
        if s.dtype.kind in "biufc":
            row[c] = float(s.median())
        else:
            mode = s.mode(dropna=True)
            row[c] = mode.iloc[0] if not mode.empty else None
    return row


def save_artifacts(pre, model, calibrator, X_raw, threshold):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "preprocessor": pre,
            "model": model,
            "calibrator": calibrator,
            "template": template_row(X_raw),
            "columns": list(X_raw.columns),
            "threshold": threshold,
        },
        MODELS_DIR / "pipeline.joblib",
    )
    print(f"[saved] {MODELS_DIR / 'pipeline.joblib'}")


def run(
    df: pd.DataFrame,
    sample: int | None = None,
    n_boot: int = 1000,
    figures: bool = True,
    save: bool = True,
    exclude=F.NOT_MODEL_INPUTS,
    artifacts: dict | None = None,
) -> dict:
    """Train, calibrate and evaluate. `exclude` lists columns kept out of the model
    (they stay in the data for the group check). Pass a dict as `artifacts` to get the
    fitted objects and test matrices back, for follow-up analysis."""
    t0 = time.time()
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = split(df, sample)
    pre = F.build_preprocessor(X_tr, exclude)
    Xtr = pre.fit_transform(X_tr).astype(np.float32)
    Xcal = pre.transform(X_cal).astype(np.float32)
    Xte = pre.transform(X_te).astype(np.float32)
    print(
        f"[data] train {len(y_tr):,} / calibration {len(y_cal):,} / test {len(y_te):,} rows, "
        f"{Xtr.shape[1]} features after encoding"
    )

    logreg = train_baseline(Xtr, y_tr)
    lgbm = train_boosted(Xtr, y_tr)
    p_lr = logreg.predict_proba(Xte)[:, 1]
    p_gb = lgbm.predict_proba(Xte)[:, 1]
    p_gb_cal = lgbm.predict_proba(Xcal)[:, 1]

    m: dict[str, Any] = {
        "data": {
            "rows_used": int(len(y_tr) + len(y_cal) + len(y_te)),
            "train": int(len(y_tr)),
            "calibration": int(len(y_cal)),
            "test": int(len(y_te)),
            "default_rate": float(pd.concat([y_tr, y_cal, y_te]).mean()),
            "n_features_encoded": int(Xtr.shape[1]),
            "excluded_from_model": list(exclude),
            "sample": sample,
        },
        "logreg_convergence": {
            "n_iter": int(logreg.n_iter_[0]),
            "max_iter": int(logreg.max_iter),
            "converged": bool(logreg.n_iter_[0] < logreg.max_iter),
        },
        "models": {"logreg": E.headline(y_te, p_lr), "lightgbm": E.headline(y_te, p_gb)},
    }

    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = {"logreg": p_lr, "lightgbm": p_gb}
    m["bootstrap"] = {
        "n_boot": n_boot,
        "roc_auc": E.bootstrap_ci(y_te, scores, roc_auc_score, n_boot, diff=("lightgbm", "logreg")),
        "pr_auc": E.bootstrap_ci(
            y_te, scores, average_precision_score, n_boot, diff=("lightgbm", "logreg")
        ),
    }

    # Calibration: fit on the calibration split, judge on test.
    calibrated = {"raw": p_gb}
    fitted = {}
    for name, cls in CALIBRATORS.items():
        fitted[name] = cls().fit(p_gb_cal, y_cal)
        calibrated[name] = fitted[name].predict(p_gb)
    m["calibration"] = {
        name: {
            "brier": E.brier(y_te, p),
            "ece": E.ece(y_te, p),
            "mean_pd": float(np.mean(p)),
            **E.headline(y_te, p),
        }
        for name, p in calibrated.items()
    }
    m["calibration"]["observed_default_rate_test"] = float(y_te.mean())
    m["calibration"]["logreg_raw"] = {
        "brier": E.brier(y_te, p_lr),
        "ece": E.ece(y_te, p_lr),
        "mean_pd": float(np.mean(p_lr)),
    }
    pd_down = calibrated[DOWNSTREAM_CALIBRATOR]

    # The operating threshold is set on the calibration split, so the test split is not
    # used to choose it: the raw score above which FLAG_RATE of applicants fall.
    threshold = float(np.quantile(p_gb_cal, 1 - FLAG_RATE))
    op = E.threshold_table(y_te, p_gb, thresholds=(threshold,)).iloc[0]
    m["operating_point"] = {
        "target_flag_rate": FLAG_RATE,
        "threshold": threshold,
        "precision": float(op["precision"]),
        "recall": float(op["recall"]),
        "flag_rate_test": float(op["flag_rate"]),
    }
    m["threshold_table"] = E.threshold_table(y_te, p_gb).to_dict(orient="records")
    m["expected_loss"] = expected_loss_table(y_te, X_te["AMT_CREDIT"], p_gb, pd_down, threshold)

    gender = fairness.group_table(X_te["CODE_GENDER"], y_te, p_gb, pd_down, threshold)
    age = fairness.group_table(
        fairness.age_band(X_te["DAYS_BIRTH"]), y_te, p_gb, pd_down, threshold
    )
    m["fairness"] = {
        "threshold": threshold,
        "gender": gender.to_dict(orient="records"),
        "age_band": age.to_dict(orient="records"),
        "gender_rows_below_min_n": int((~X_te["CODE_GENDER"].isin(gender["group"])).sum()),
    }

    if figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        E.plot_calibration(
            y_te,
            {
                "LightGBM raw": p_gb,
                "Platt": calibrated["platt"],
                "isotonic": calibrated["isotonic"],
            },
            path=FIGURES_DIR / "calibration.png",
        )
        from . import explain

        names = list(pre.get_feature_names_out())
        _, sv, Xs = explain.shap_values_for(lgbm, Xte, sample=2000, seed=RANDOM_STATE)
        explain.save_summary(sv, Xs, names, FIGURES_DIR / "shap_summary.png")
        m["shap_top10"] = explain.top_features(sv, names, 10)

    m["runtime_seconds"] = round(time.time() - t0, 1)
    if artifacts is not None:
        artifacts.update(
            pre=pre,
            lgbm=lgbm,
            X_tr=X_tr,
            X_te=X_te,
            Xtr=Xtr,
            Xte=Xte,
            y_tr=y_tr,
            y_te=y_te,
            p_gb=p_gb,
            pd_cal=pd_down,
            threshold=threshold,
        )
    if save:
        save_artifacts(pre, lgbm, fitted[DOWNSTREAM_CALIBRATOR], X_tr, threshold)
    return m


def write_reports(m: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (REPORTS_DIR / "results.md").write_text(results_markdown(m), encoding="utf-8")
    print(f"[saved] {REPORTS_DIR / 'metrics.json'}, {REPORTS_DIR / 'results.md'}")


def results_markdown(m: dict) -> str:
    d, b = m["data"], m["bootstrap"]
    ci = lambda k, mdl: f"{b[k][mdl]['lo']:.3f} to {b[k][mdl]['hi']:.3f}"  # noqa: E731
    L = [
        "# Results",
        "",
        "Generated by `python -m credit_risk_scorecard.model`. Do not edit by hand.",
        "",
        f"Rows used: {d['rows_used']:,} (train {d['train']:,} / calibration "
        f"{d['calibration']:,} / test {d['test']:,}). Default rate {d['default_rate']:.2%}. "
        f"{d['n_features_encoded']} features after one-hot encoding. Kept out of the model "
        f"(used only for the group check): {', '.join(d['excluded_from_model']) or 'none'}.",
        "",
        "## Ranking (test split, 95% bootstrap intervals)",
        "",
        "| Model | ROC-AUC | 95% CI | PR-AUC | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for mdl, label in [("logreg", "Logistic regression"), ("lightgbm", "LightGBM")]:
        h = m["models"][mdl]
        L.append(
            f"| {label} | {h['roc_auc']:.3f} | {ci('roc_auc', mdl)} | "
            f"{h['pr_auc']:.3f} | {ci('pr_auc', mdl)} |"
        )
    L += [
        "",
        f"LightGBM minus logistic regression, paired: ROC-AUC {ci('roc_auc', 'diff')}, "
        f"PR-AUC {ci('pr_auc', 'diff')}.",
        "",
        f"Logistic regression converged: {m['logreg_convergence']['converged']} "
        f"({m['logreg_convergence']['n_iter']} of {m['logreg_convergence']['max_iter']} "
        "iterations).",
        "",
        "## Calibration (LightGBM, test split)",
        "",
        f"Observed default rate on test: {m['calibration']['observed_default_rate_test']:.2%}.",
        "",
        "| PDs | Mean PD | Brier | ECE | ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    for name in ["raw", "platt", "isotonic"]:
        c = m["calibration"][name]
        L.append(
            f"| {name} | {c['mean_pd']:.3f} | {c['brier']:.4f} | {c['ece']:.4f} | "
            f"{c['roc_auc']:.3f} |"
        )
    lr = m["calibration"]["logreg_raw"]
    L += [
        "",
        f"Logistic regression (class-weighted, uncalibrated), for reference: mean PD "
        f"{lr['mean_pd']:.3f}, Brier {lr['brier']:.4f}, ECE {lr['ece']:.4f}.",
        "",
        "## Threshold table (raw LightGBM score, test split)",
        "",
        "| Threshold | Precision | Recall | Flag rate |",
        "|---|---|---|---|",
    ]
    for r in m["threshold_table"]:
        L.append(
            f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['flag_rate']:.3f} |"
        )
    op = m["operating_point"]
    L += [
        "",
        "## Operating point",
        "",
        f"Policy: decline the riskiest {op['target_flag_rate']:.0%}. The raw-score threshold "
        f"that does this on the calibration split is {op['threshold']:.3f}. On test it flags "
        f"{op['flag_rate_test']:.1%} of applicants, catches {op['recall']:.1%} of defaulters "
        f"(recall), and {op['precision']:.1%} of those flagged did default (precision).",
    ]
    el = m["expected_loss"]
    L += [
        "",
        f"## Expected loss at threshold {el['threshold']:.3f} (ILLUSTRATIVE)",
        "",
        f"LGD {el['lgd']:.0%}, EAD = AMT_CREDIT. Approved share {el['approved_share']:.1%}.",
        "",
        "| Source | Loss on approved loans |",
        "|---|---|",
        f"| Raw PDs | {el['el_raw_pd']:,.0f} |",
        f"| Isotonic-calibrated PDs | {el['el_calibrated_pd']:,.0f} |",
        f"| Implied by observed outcomes | {el['loss_implied_by_outcomes']:,.0f} |",
        "",
        f"## Group check at threshold {m['fairness']['threshold']:.3f}",
        "",
    ]
    for key, title in [("gender", "CODE_GENDER"), ("age_band", "Age band")]:
        L += [
            f"### {title}",
            "",
            "| Group | n | Observed default rate | Mean calibrated PD | Approval rate "
            "| TPR | FPR |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in m["fairness"][key]:
            L.append(
                f"| {r['group']} | {r['n']:,} | {r['observed_rate']:.3f} | "
                f"{r['mean_pd']:.3f} | {r['approval_rate']:.3f} | {r['tpr']:.3f} | "
                f"{r['fpr']:.3f} |"
            )
        L.append("")
    if "shap_top10" in m:
        L += ["## Top 10 features by mean |SHAP| (LightGBM, 2,000 test rows)", ""]
        L += [f"{i}. `{f}` ({v:.3f})" for i, (f, v) in enumerate(m["shap_top10"], 1)]
        L.append("")
    L.append(f"Run time: {m['runtime_seconds']} s.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    m = run(load_frame(a.from_db), sample=a.sample, n_boot=a.n_boot)
    write_reports(m)
    print(results_markdown(m))


if __name__ == "__main__":
    main()
