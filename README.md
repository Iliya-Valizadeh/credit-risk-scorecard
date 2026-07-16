# Credit-Risk Scorecard

A credit default risk model on the Home Credit Default Risk dataset, built the way a
bank's model-risk team would expect: SQL feature pipeline in PostgreSQL, a calibrated
and explainable classifier, threshold analysis tied to a business decision, and a small
scoring demo.

> **Status:** In progress. This README is the target spec; sections marked _(pending)_
> are filled in as each weekly gate ships. No number appears here that hasn't been
> produced by code in this repo.

---

## Problem

Lenders approve or decline credit applications under uncertainty about repayment. A
scorecard estimates each applicant's probability of default (PD) so the lender can set a
decision threshold that balances approvals against expected losses. The task here is
binary classification: predict `TARGET` (1 = client had payment difficulties) from
application and bureau data.

## Data

- **Source:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle).
- **Primary table:** `application_train.csv` (~307k rows, 122 columns).
- **Supporting tables:** bureau, previous_application, installments (for engineered features).
- Raw CSVs are **not** committed (see `.gitignore`). Download instructions in [`data/README.md`](data/README.md).

## Method

1. **Load** raw CSVs into PostgreSQL (`src/data_load.py`).
2. **Feature engineering in SQL** — joins/aggregate features across bureau and previous-application tables (`sql/features.sql`).
3. **pandas feature engineering** — ratios, missingness flags, encoding (`src/features.py`).
4. **Baseline model** — logistic regression, the interpretable scorecard reference (`src/model.py`).
5. **Gradient boosting** — LightGBM, tuned (`src/model.py`).
6. **Evaluation** — AUC, precision/recall at chosen operating thresholds, calibration curve (`src/evaluate.py`).
7. **Explainability** — SHAP global + local (`src/explain.py`).
8. **Model-risk write-up** — why calibration and explainability matter to a bank ([`reports/model_risk_writeup.md`](reports/model_risk_writeup.md)).
9. **Scoring demo** — Streamlit app that scores a single applicant (`app/streamlit_app.py`).

## Results

Trained on a 17,000-row sample of the application data (all application-level features),
validated on a held-out 20% split. Base default rate 7.85%. Full-data training should
move these figures modestly; the pipeline is identical.

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic regression (baseline) | 0.734 | 0.197 | interpretable scorecard reference; did not fully converge at max_iter, so a small headroom remains |
| LightGBM | **0.736** | **0.212** | class-imbalance handled via `scale_pos_weight` |

**Threshold analysis (LightGBM, validation set).** The bank trades recall of true
defaulters against how many applicants get flagged (declined):

| Threshold | Precision | Recall | Flag rate |
|-----------|-----------|--------|-----------|
| 0.30 | 0.155 | 0.663 | 0.335 |
| 0.40 | 0.178 | 0.513 | 0.226 |
| 0.50 | 0.199 | 0.371 | 0.146 |
| 0.60 | 0.240 | 0.251 | 0.082 |
| 0.70 | 0.274 | 0.116 | 0.033 |

Illustrative operating point: **0.40** catches ~51% of defaulters while flagging ~23% of
applicants — the exact cut a lender sets depends on the relative cost of a missed default
versus a declined good customer.

## Limitations

Trained on a 17k-row sample, not the full 307k (headline metrics are therefore
approximate); the logistic baseline hit its iteration cap without fully converging; no
fairness audit across protected attributes has been done; and the split is random, not
temporal, so drift is untested. The model should not be used for pricing, collections, or
any real lending decision without full-data retraining and independent validation.

---

## Repo structure

\`\`\`
credit-risk-scorecard/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/                 # raw data lives here locally, not committed
│   └── README.md
├── sql/
│   └── features.sql
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_load.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── explain.py
├── notebooks/
│   └── 01_eda.ipynb
├── app/
│   └── streamlit_app.py
└── reports/
    ├── model_risk_writeup.md
    └── figures/
\`\`\`

## Setup

\`\`\`bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your PostgreSQL credentials
\`\`\`

Then follow [`data/README.md`](data/README.md) to download the dataset, and run:

\`\`\`bash
python -m src.data_load
\`\`\`

## Author

Iliya Valizadeh — BSc Data Science, York University.
GitHub: [Iliya-Valizadeh](https://github.com/Iliya-Valizadeh)
