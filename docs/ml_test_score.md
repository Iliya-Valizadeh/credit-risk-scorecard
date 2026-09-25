# ML Test Score self-assessment

This page scores this repo against the ML Test Score, a checklist from
[Breck et al. (Google, 2017)][breck]. It uses the same scoring rule as the
`ds-project-standard` template (its ADR 0004). No test gets a point that the code and
its CI runs do not earn.

## In plain words

Google wrote a list of 28 tests for a machine learning system. <!-- not-a-claim -->
A system should pass them before people rely on it.
This page checks this repo against that list. The repo's total score is zero, the
lowest level. Its tests of the data and the code are fair. But nothing runs as a live
service, so nothing is watched after training.

## How the scoring works

The paper gives each test a score:

- none: the test is not done
- half a point: the test is done by hand, and the result is written down
- one point: the test is automated and runs again on every change

A test earns one point only after CI (continuous integration, the checks GitHub runs on
every push) has run it and it passed. A test that does not apply still scores none, as
in the paper.

A section's score is the sum of its seven tests. The overall score is the lowest of the
four section scores, so one weak section pulls the whole system down. The paper reads an
overall score of zero as closer to a research project than a production system.

## What is scored

The system is the credit-risk pipeline in `src/credit_risk_scorecard/`. It cleans the
application data, splits it into train, calibration and test parts, trains logistic
regression and LightGBM, calibrates the scores, sets a decline threshold and checks
results by group. The real run needs Kaggle's data, which is not in the repo, so it
runs only by hand with `make eval`.

The tests in `tests/` use synthetic applicants with the real column names, so CI can
run them. State on 2026-09-25: CI on GitHub ran these tests on commit `c36fc93` of
branch `phase-b-retrofit`, and they passed ([run][ci-run]). That run failed at a later
step, the docs checks, which are not scored here. CI runs the tests again on every
push. A test that CI runs earns one point below. A check that needs the
real data earns half a point at most.

## Summary

| Section | Score out of 7 |
|---|---|
| Features and data | two points |
| Model development | one and a half points |
| Infrastructure | two and a half points |
| Monitoring | none |
| Overall (the lowest section) | 0 |

In total, three tests earn one point each and six earn half a point. The other tests
earn nothing.

## Tests for features and data

| # | Test | Score | Evidence or gap |
|---|---|---|---|
| 1 | Feature expectations are captured in a schema | none | No schema lists each column's type or allowed range. `clean()` handles one known bad value in `DAYS_EMPLOYED`, but nothing checks the rest |
| 2 | All features are beneficial | none | SHAP ranks the top inputs in `reports/results.md`, but no test removes an input to see if the model gets worse |
| 3 | No feature's cost is too much | none | Not measured |
| 4 | Features adhere to meta-level requirements | half | The one written rule is that `CODE_GENDER` is not a model input ([ADR 0005](decisions/0005-drop-code-gender-as-a-model-input.md)). `tests/test_leakage.py` checks it in CI. Only half, because no rule covers age, which enters directly through `DAYS_BIRTH`, or the inputs that carry gender |
| 5 | The data pipeline has appropriate privacy controls | half | The data describes real (anonymised) applicants. `.gitignore` keeps the raw data and the model file out of git, and [ADR 0001](decisions/0001-adopting-the-house-standard.md) (decision 7) writes down why. This is done by hand, and nothing tests it. The template's `.gitignore` would have dropped these rules, and `d7129c6` put them back by hand |
| 6 | New features can be added quickly | none | Not measured |
| 7 | All input feature code is tested | one point | `tests/test_features.py` and `tests/test_leakage.py` cover every line of `features.py`: the bad-value rule, the ratios, the centring and the checks that the target and row id never reach the model. CI runs them. `data_load.py`, the optional loader into PostgreSQL, has no tests, but the model reads the CSV by default |

## Tests for model development

| # | Test | Score | Evidence or gap |
|---|---|---|---|
| 1 | Model specs (the code that defines a model) are reviewed and checked in | none | The model code is in git, but no second person has reviewed it |
| 2 | Offline and online metrics correlate | none | There is no online use |
| 3 | All hyperparameters have been tuned | none | LightGBM and logistic regression run with fixed settings. There is no tuning code |
| 4 | The impact of model staleness is known | none | The data has no application date, so the split is random and staleness was not studied |
| 5 | A simpler model is not better | half | `model.py` scores logistic regression next to LightGBM on the same test rows and reports the gap with a paired [bootstrap](glossary.md#bootstrap) interval in `reports/metrics.json`. This runs by hand on the real data. The CI test only checks that both models rank better than chance on synthetic data |
| 6 | Model quality is sufficient on important data slices | half | `fairness.py` reports approval rate, error rates and calibration by gender and by four age bands in `reports/results.md`. This runs by hand. No slice has a pass or fail bar |
| 7 | The model is tested for considerations of inclusion | half | `gender_check.py` trains the model with and without `CODE_GENDER`, compares decisions for women and men, and measures how well other inputs predict gender (`reports/gender_check.md`). This runs by hand, and it covers gender and age only |

## Tests for infrastructure

| # | Test | Score | Evidence or gap |
|---|---|---|---|
| 1 | Training is reproducible | half | Every random step uses a fixed seed, and CI installs packages from the lockfile. The full run was done twice by hand with identical numbers (`db53b26`), and again after the retrofit, where every counted number matched (`6fee45f`). No automated check reruns it, because CI has no data |
| 2 | Model specs are unit tested | one point | `tests/test_evaluate.py` checks that both calibrators pull inflated scores back down, that the Brier score and calibration error are right on small cases, and that raising the threshold never declines more applicants. `tests/test_pipeline.py` checks that the logistic regression converges. CI runs them |
| 3 | The ML pipeline is integration tested | one point | `tests/test_pipeline.py` runs `model.run` from raw synthetic rows through the split, both models, calibration, the threshold and the group check to the results text. `tests/test_demo.py` runs the demo. CI runs both |
| 4 | Model quality is validated before serving | none | There is no serving step. The Streamlit app loads whatever `models/pipeline.joblib` holds, with no quality gate |
| 5 | The model is debuggable | none | SHAP values can be computed for any row, but no tool follows one applicant through the model step by step |
| 6 | Models are canaried before serving | none | There is no serving step |
| 7 | Serving models can be rolled back | none | There is no serving step, and the model file is not versioned |

## Monitoring tests

| # | Test | Score | Evidence or gap |
|---|---|---|---|
| 1 | Dependency changes result in notification | none | Versions are pinned in `uv.lock`, but nothing reports new releases |
| 2 | Data invariants hold for inputs | none | No check runs on incoming data |
| 3 | Training and serving are not skewed | none | There is no serving step. The app reuses the saved preprocessor, but nothing compares its inputs with the training data |
| 4 | Models are not too stale | none | There is no serving step |
| 5 | Models are numerically stable | none | Nothing checks predictions for missing or infinite values |
| 6 | Computing performance has not regressed | none | `reports/metrics.json` records run time, but nothing compares it between runs |
| 7 | Prediction quality has not regressed | none | There is no served data. No automated check compares a fresh run with `reports/metrics.json` |

[breck]: https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
[ci-run]: https://github.com/Iliya-Valizadeh/credit-risk-scorecard/actions/runs/36195995781
