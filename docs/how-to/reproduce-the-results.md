# How to reproduce the results

Every number in the README and the model card comes from these three commands, run in
this order, against the real Home Credit data.

## 1. Get the data

Download `application_train.csv` from Kaggle's Home Credit competition into `data/`.
See [data/README.md](../../data/README.md) for the exact steps. The file is not
committed to this repo, so this step can't be skipped.

## 2. Run the evaluation

```bash
make eval
```

This runs, in order:

```bash
uv run python -m credit_risk_scorecard.model         # about 3 minutes
uv run python -m credit_risk_scorecard.gender_check   # about 6 minutes
uv run python -m credit_risk_scorecard.logreg_check    # slower; it fits the old `saga` setup too
```

`model.py` writes `reports/metrics.json`, `reports/results.md`,
`reports/figures/*.png` and `models/pipeline.joblib`. `gender_check.py` and
`logreg_check.py` each write their own Markdown and JSON (or Markdown-only) report.
Running `make eval` twice gives identical numbers, because every random step in the
code uses the same fixed seed.

## 3. Check the numbers didn't move

```bash
git diff reports/
```

If a number changed and you didn't change the code, look at `uv.lock` first: a newer
patch version of a package underneath scikit-learn or LightGBM can shift results by a
small amount even with the same seed.

## 4. See the decision view

```bash
uv run streamlit run app/streamlit_app.py
```

This needs `models/pipeline.joblib` from step 2. It is not committed, so a fresh clone
has to run `make eval` before the app will start.

## Training from PostgreSQL instead of the CSV

Copy `.env.example` to `.env`, fill in your database credentials, load the data with
`uv run python -m credit_risk_scorecard.data_load`, then run
`uv run python -m credit_risk_scorecard.model --from-db`. This is an alternate data
path; the committed numbers come from the CSV.

## Keeping `tools/` in sync with the template

This repo's checks came from `ds-project-standard` via `copier copy`, with the worked
example excluded. To pull in later fixes to `tools/` without bringing the example back:

```bash
uv run copier update --trust --vcs-ref v1.0 \
  --exclude 'src/**' --exclude 'tests/**' --exclude 'reports/**'
```
