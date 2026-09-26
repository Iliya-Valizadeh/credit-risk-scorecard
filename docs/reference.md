# Reference

Facts to look up: make targets, scripts, files and settings.

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Installs the pinned packages with `uv` |
| `make lint` | Runs ruff and mypy |
| `make test` | Runs pytest with a coverage report for `src/` |
| `make eval` | Runs `model`, then `gender_check`, then `logreg_check`; needs the Kaggle data |
| `make demo` | Runs the real pipeline on synthetic applicants, with no download or key needed |
| `make check-docs` | Runs the claims, readability, AI-writing signs, link and README checks |
| `make all` | Runs every step above, in order |

## Scripts in `src/credit_risk_scorecard/`

| Module | Run as | What it does |
|---|---|---|
| `features.py` | (imported, not run directly) | Cleans `DAYS_EMPLOYED`, adds the three ratio columns, and the preprocessing pipeline (imputation, standardisation, one-hot encoding) |
| `model.py` | `python -m credit_risk_scorecard.model` | Splits the data, trains logistic regression and LightGBM, calibrates, sets the decline threshold, and writes `reports/metrics.json` and `reports/results.md` |
| `calibrate.py` | (imported, not run directly) | Fits [Platt scaling](glossary.md#platt-scaling) and [isotonic regression](glossary.md#isotonic-regression) on the [calibration](glossary.md#calibration) split |
| `evaluate.py` | (imported, not run directly) | [ROC-AUC](glossary.md#roc-auc), [PR-AUC](glossary.md#pr-auc), the [bootstrap](glossary.md#bootstrap) confidence intervals, the threshold table, the [Brier score](glossary.md#brier-score) and [ECE](glossary.md#ece) |
| `expected_loss.py` | (imported, not run directly) | The illustrative expected-loss table: PD times [LGD](glossary.md#lgd) times exposure |
| `fairness.py` | (imported, not run directly) | Approval rate, recall and false-positive rate by group, at a given threshold |
| `gender_check.py` | `python -m credit_risk_scorecard.gender_check` | Trains the model with and without `CODE_GENDER` and compares the decisions; writes `reports/gender_check.md` and `.json` |
| `logreg_check.py` | `python -m credit_risk_scorecard.logreg_check` | Compares the logistic regression's convergence before and after centring; writes `reports/logreg_convergence.md` |
| `explain.py` | (imported, not run directly) | [SHAP](glossary.md#shap) values for LightGBM |
| `demo.py` | `python -m credit_risk_scorecard.demo` | Runs the real pipeline on synthetic applicants, for `make demo` |
| `synthetic.py` | (imported, not run directly) | Makes the synthetic applicants used by `demo.py` and the tests |
| `data_load.py` | `python -m credit_risk_scorecard.data_load` | Loads the CSV into PostgreSQL, for `model.py --from-db` |
| `config.py` | (imported, not run directly) | Paths, the random seed, and the database connection string |

`model.py` also takes `--sample N` for a quick run on a stratified sample, and
`--from-db` to read from PostgreSQL instead of the CSV.

## Reports in `reports/`

| File | Written by | Holds |
|---|---|---|
| `metrics.json` | `model.py` | Every number this project reports, in one machine-readable file |
| `results.md` | `model.py` | The same numbers, as the Markdown the README quotes |
| `gender_check.md`, `gender_check.json` | `gender_check.py` | The with/without-`CODE_GENDER` comparison |
| `logreg_convergence.md` | `logreg_check.py` | The logistic regression's convergence, before and after centring |
| `figures/calibration.png`, `figures/shap_summary.png` | `model.py` | The two charts in the README |
| `model_risk_writeup.md` | written by hand | A one-page summary for a model-risk reviewer |

## Key settings

| Setting | Value | Where |
|---|---|---|
| Random seed | 42 | `config.py`, `RANDOM_STATE` |
| Split | 60% train, 20% calibration, 20% test, stratified on the target | `model.py`, `split()` |
| [Decline rate](glossary.md#decline-rate) (illustrative policy) | 20% | `model.py`, `FLAG_RATE` |
| Bootstrap resamples | 1,000 | `model.py`, `n_boot` default |
| LightGBM | 600 trees, learning rate 0.02, 31 leaves, `scale_pos_weight` for the class imbalance | `model.py` |
| Calibrator used for every reported PD | Isotonic regression | `model.py`, `DOWNSTREAM_CALIBRATOR` |
| LGD (illustrative) | 45% | `expected_loss.py`, `LGD` |

## Other files

| Path | What it holds |
|---|---|
| `app/streamlit_app.py` | The decision-view demo app; needs `models/pipeline.joblib` |
| `data/README.md` | How to download the Kaggle data |
| `notebooks/01_eda.ipynb` | Early exploration; not used to make the reported numbers |
| `sql/features.sql` | One commented example; not used yet (see [what's weak](whats_weak.md)) |
| `tests/` | Unit and data tests, run on synthetic data with the real column names |
