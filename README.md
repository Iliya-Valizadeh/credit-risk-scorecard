# Credit-risk scorecard

A credit default model on the public Home Credit Default Risk data: a logistic regression
baseline and a LightGBM model, trained on all 307,511 applications.

The part I care about is what happens after the model ranks people. I weighted the
classes to deal with the imbalance, and that broke the probabilities. The model said the
average applicant had a 40% chance of default when the real rate is 8%. This repo
measures that and fixes it with a calibration step fitted on held-out data. It also works
out what the difference means for expected loss, and checks how declines fall across
gender and age. Gender is not a model input; it is kept only for that check.

Every number below comes from `python -m src.model` (about 3 minutes), which writes
[reports/results.md](reports/results.md). Running it twice gives identical numbers. The
limitations section is worth reading before the results.

## Data

The data is [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
from Kaggle. I use `application_train.csv`: 307,511 applications and 122 columns.
`TARGET` = 1 means the client had payment difficulties, which is broader than a formal
default, but I call it default below for short. 8.07% of applicants have it. The bureau
and previous-application tables aren't used yet (see "What's not built"). The raw CSVs
aren't committed; download steps are in [data/README.md](data/README.md).

## Method

1. The data is split 60% train, 20% calibration and 20% test, stratified on the target.
   The models only see train, the calibration step only sees calibration, and every
   number is on test.
2. In `src/features.py`, the `DAYS_EMPLOYED` value 365243 (it means "not employed")
   becomes missing, and three ratios are added: credit/income, annuity/income and
   annuity/credit. Numbers get median imputation and standardisation, categories get
   one-hot encoding, and `CODE_GENDER` is left out. That gives 244 columns.
3. `src/model.py` trains a class-weighted logistic regression and a LightGBM model with
   `scale_pos_weight`, so each defaulter counts about 11.4 times. Settings are fixed; there
   is no tuning.
4. `src/calibrate.py` fits Platt scaling and isotonic regression on the calibration split.
5. `src/evaluate.py` reports ROC-AUC, PR-AUC, 95% bootstrap intervals (1,000 resamples of
   the test set), the Brier score and expected calibration error.
6. The decision view has a threshold table, an operating point, an illustrative
   expected-loss check (`src/expected_loss.py`) and a group check (`src/fairness.py`,
   `src/gender_check.py`).
7. `src/explain.py` computes SHAP values for LightGBM.

## Results

### Ranking

| Model | ROC-AUC | 95% interval | PR-AUC | 95% interval |
|---|---|---|---|---|
| Logistic regression | 0.752 | 0.745 to 0.759 | 0.227 | 0.216 to 0.238 |
| LightGBM | 0.769 | 0.762 to 0.775 | 0.254 | 0.242 to 0.265 |

Both models were scored on the same resampled test sets. The gap (LightGBM minus
logistic) is 0.013 to 0.020 for ROC-AUC and 0.021 to 0.033 for PR-AUC. Both intervals
exclude zero, so LightGBM ranks applicants better, by a small margin.

With `CODE_GENDER` as an input the two models scored 0.770 and 0.755. An earlier version
of this README reported 0.736 and 0.734 from a 17,000-row sample. The code that drew that
sample was never in the repo, so those numbers couldn't be reproduced.

### Convergence of the logistic regression

The first setup (features scaled but not centred, saga solver) hit its 1,000-iteration
cap. The current one (centred, lbfgs) converges in 227 iterations. ROC-AUC is between
0.752 and 0.753 in all three runs in
[reports/logreg_convergence.md](reports/logreg_convergence.md), so the non-convergence
wasn't costing anything in ranking. The old setup also converges, in 148 iterations, when
the matrix is float32, so I can't say the missing centring was the cause.

### Calibration

A bank uses the predicted probability of default (PD) itself for provisions and
pricing, so the PD has to mean what it says.

| LightGBM PDs on test | Mean PD | Brier score | ECE | ROC-AUC |
|---|---|---|---|---|
| Raw (class-weighted) | 0.395 | 0.1851 | 0.3145 | 0.769 |
| Platt scaling | 0.080 | 0.0673 | 0.0032 | 0.769 |
| Isotonic regression | 0.080 | 0.0673 | 0.0027 | 0.768 |

The observed default rate on test is 8.07%. The Brier score is the mean squared gap
between the PD and the 0/1 outcome. ECE (expected calibration error) sorts applicants
into 10 equal-sized groups by PD and averages the gap between predicted and actual
default rates. Calibration cuts the Brier score by 64% and the ECE by 99%, and the
ranking barely changes.

![Calibration before and after](reports/figures/calibration.png)

Platt and isotonic come out the same here. I chose isotonic before looking at test
results, because the calibration split is large (61,502 rows), and it is used for
everything below.

### Operating point

| Raw-score threshold | Precision | Recall | Share declined |
|---|---|---|---|
| 0.30 | 0.118 | 0.897 | 0.614 |
| 0.40 | 0.142 | 0.800 | 0.455 |
| 0.50 | 0.175 | 0.685 | 0.316 |
| 0.60 | 0.220 | 0.532 | 0.196 |
| 0.70 | 0.281 | 0.341 | 0.098 |

The illustrative policy is to decline the riskiest 20% of applicants. The threshold for
that (0.600) is set on the calibration split, not on test. On test it declines 19.5% of
applicants and catches 53.2% of the people who later had payment difficulties. 22.0% of
those declined did.

### Expected loss at that threshold (illustrative)

Expected loss = PD × LGD × exposure. The data has no recovery amounts, so LGD (loss given
default) is set to 45% and exposure to the full credit amount. Both are assumed values;
neither is estimated from the data.

| Summed over approved applicants | Loss |
|---|---|
| Using raw PDs | 4,312,985,709 |
| Using calibrated PDs | 642,168,648 |
| Implied by what actually happened | 633,364,511 |

With the raw PDs, the expected loss is almost 7 times what the outcomes imply. With
calibrated PDs it is within 2%. That is what the broken probabilities would cost if
anyone used the raw scores for provisioning.

### Gender

`CODE_GENDER` used to be a model input, and it was the 8th most important feature by
SHAP. It is now left out of the model and kept in the data only for this check.
`python -m src.gender_check` trains the model with and without it on the same split and
compares the decisions at each model's own 20% decline threshold
([reports/gender_check.md](reports/gender_check.md)).

| Model | ROC-AUC | Group | Approval rate | Defaulters declined (TPR) | Non-defaulters declined (FPR) |
|---|---|---|---|---|---|
| With gender | 0.770 | Women | 0.845 | 0.475 | 0.131 |
| With gender | 0.770 | Men | 0.723 | 0.617 | 0.239 |
| Without gender | 0.769 | Women | 0.832 | 0.492 | 0.144 |
| Without gender | 0.769 | Men | 0.752 | 0.583 | 0.209 |

Dropping the column cost 0.002 of ROC-AUC (0.7703 to 0.7687) and narrowed the gaps. Men were approved 12.2
points less often than women; now it is 7.9 points. Among applicants who had no payment
difficulties, men were declined 10.8 points more often than women; now it is 6.6.

A gap remains, and part of it comes through other inputs. From the model's remaining
inputs, a logistic regression can tell men from women with ROC-AUC 0.876. Three inputs
are among the model's 20 most important and also correlated with gender:
`EXT_SOURCE_1` (SHAP rank 3, correlation with being male −0.20), `DAYS_BIRTH` (rank 9,
+0.15, as men in this data are younger) and `FLAG_OWN_CAR` (rank 20, +0.34 for owning a
car). Men also had payment difficulties more often in this data (10.2% against 7.0%), so
some gap would show up even if the model knew nothing about gender. This check can't
separate the two.

Dropping the column also shifted calibration by group. Calibrated PDs are now above the
observed rate for women (7.4% predicted, 7.0% observed) and below it for men (9.2% and
10.2%). With the column, both were within 0.3 points.

### Age

At the same threshold, without gender:

| Age | n | Observed default rate | Mean calibrated PD | Approval rate | TPR | FPR |
|---|---|---|---|---|---|---|
| 20-29 | 8,975 | 0.115 | 0.116 | 0.650 | 0.679 | 0.307 |
| 30-44 | 24,664 | 0.090 | 0.088 | 0.774 | 0.578 | 0.191 |
| 45-59 | 20,778 | 0.066 | 0.066 | 0.860 | 0.429 | 0.120 |
| 60+ | 7,086 | 0.048 | 0.047 | 0.946 | 0.202 | 0.047 |

Age enters the model directly through `DAYS_BIRTH`. Applicants aged 20 to 29 who had no
payment difficulties are declined 30.7% of the time, against 4.7% for those over 60.
Calibration holds within every age band. Age is also a prohibited ground of
discrimination under the Canadian Human Rights Act. I've left `DAYS_BIRTH` in and
reported its effect; whether a lender could use it is a legal question this repo doesn't
answer.

### What the model relies on

The top features by mean absolute SHAP value (2,000 test rows) are the three external
credit scores (`EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`), then
`ANNUITY_CREDIT_RATIO` (one of the ratios I added), goods price, employment length and
annuity. The full list is in [reports/results.md](reports/results.md).

![SHAP summary](reports/figures/shap_summary.png)

## What's not built

There are no SQL features. `src/data_load.py` loads the CSVs into PostgreSQL, and
`python -m src.model --from-db` can train from there, but `sql/features.sql` holds only
one commented example and the bureau and previous-application tables aren't used. Earlier
versions of this README said there was a SQL feature pipeline. There isn't yet.

Missing values are imputed without a flag. A "this field was missing" column would
probably help, since missingness in credit data often carries signal.

LightGBM runs with fixed settings, with no tuning.

## Limitations

- The split is random, not by time, so drift over time is untested.
- The target is "payment difficulties", not a formal default definition.
- The expected-loss view rests on an assumed LGD and exposure.
- Gender is out of the model, but other inputs still carry it, and age is used directly.
  The group check covers only gender and four age bands.
- Home Credit's applicants aren't a Canadian bank's book.

The model shouldn't be used for pricing, collections or any real lending decision. More
in [MODEL_CARD.md](MODEL_CARD.md) and [reports/model_risk_writeup.md](reports/model_risk_writeup.md).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put application_train.csv in data/ (see data/README.md)
python -m src.model            # everything in this README; about 3 minutes
python -m src.gender_check     # with and without CODE_GENDER; about 6 minutes
python -m src.logreg_check     # the convergence comparison; slower (saga)
streamlit run app/streamlit_app.py
```

For PostgreSQL, copy `.env.example` to `.env`, fill in credentials, run
`python -m src.data_load`, then `python -m src.model --from-db`.

Tests (`pytest`) and linting (`ruff`) run in GitHub Actions on every push. The tests use
synthetic applicants, so they don't need the Kaggle data:
`pip install -r requirements-dev.txt && pytest`.

## Repo structure

```
credit-risk-scorecard/
├── README.md
├── MODEL_CARD.md
├── requirements.txt / requirements-dev.txt
├── data/README.md            # download steps; raw data is not committed
├── sql/features.sql          # placeholder, see "What's not built"
├── src/
│   ├── model.py              # split, train, calibrate, evaluate, write reports
│   ├── features.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── expected_loss.py
│   ├── fairness.py
│   ├── gender_check.py
│   ├── explain.py
│   ├── logreg_check.py
│   ├── data_load.py
│   └── config.py
├── tests/
├── notebooks/01_eda.ipynb
├── app/streamlit_app.py
├── docs/archive/SPRINT1.md   # early working notes
└── reports/
    ├── results.md            # generated
    ├── metrics.json          # generated
    ├── gender_check.md       # generated
    ├── logreg_convergence.md # generated
    ├── model_risk_writeup.md
    └── figures/
```
