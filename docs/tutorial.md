# Tutorial

This page walks you through one full run of credit-risk-scorecard, from a fresh clone
to a scored applicant. You need [uv](https://docs.astral.sh/uv/) and `make`. `uv` can
install the right Python version for you; the version is pinned in `pyproject.toml`.

## Step 1: get the code

```bash
git clone https://github.com/Iliya-Valizadeh/credit-risk-scorecard.git
cd credit-risk-scorecard
```

## Step 2: install

```bash
make setup
```

This runs `uv sync`, which installs the pinned packages from `uv.lock` into a local
`.venv`. No account or key is needed for this step.

## Step 3: run the demo

```bash
make demo
```

This trains the real pipeline (the same code that scores real applicants) on made-up
applicants, so you can see it work without the Kaggle data. It prints a line saying the
data is synthetic, then a few numbers and a small table:

```text
Demo only: the applicants below are synthetic (made up), not the Home Credit data.
[data] train 1,800 / calibration 600 / test 600 rows, 11 features after encoding
Trained on 1,800 synthetic rows, calibrated on 600, tested on 600.
  Logistic regression ROC-AUC on synthetic test data: 0.621
  LightGBM ROC-AUC on synthetic test data:            0.541
  Decline threshold (flags the riskiest 20%): 0.108

Scoring 5 synthetic applicants with the trained pipeline (raw LightGBM score, not calibrated):
SK_ID_CURR      raw PD    decision
100000           0.003     approve
...
```

The [ROC-AUC](glossary.md#roc-auc) numbers here are near chance, because the synthetic
data has no real signal in it. They say nothing about the real model; the real numbers
are in [reports/metrics.json](../reports/metrics.json) and the README. The last five
lines are the [decline rate](glossary.md#decline-rate) policy applied to five made-up
applicants: each one's raw score against the threshold that flags the riskiest 20% of
the [calibration](glossary.md#calibration) rows.

## Step 4 (optional): run the real evaluation

This step needs Kaggle's Home Credit data, which is not committed to this repo. See
[data/README.md](../data/README.md) for how to download `application_train.csv`.

```bash
make eval
```

This runs the model, then the gender check, then the logistic-regression convergence
check, in that order, and writes every file under `reports/` that the README quotes.
It takes a few minutes; the model step alone is about 3 minutes.

## Step 5 (optional): the decision-view app

```bash
uv run streamlit run app/streamlit_app.py
```

The app reads `models/pipeline.joblib`, which `make eval` writes. It is not committed,
so this step needs step 4 first.

## Next

- [How-to: run the checks before you push](how-to/run-the-checks.md)
- [How-to: reproduce the results](how-to/reproduce-the-results.md)
- [Reference](reference.md) for every make target and script
- [Explanation](explanation.md) for why the pipeline is built this way
