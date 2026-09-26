# Evaluation plan (written after the results)

This file was written on 2026-09-25 from the git history. It is not a plan made in
advance. The results already existed when it was written. It records how the model is
judged, and for each choice it names the commit that first fixed it. It also says
whether that commit came before or after the first commit that reported results.

The first commit that reported results is `d2d9e9c` (2026-07-16). Its README gave [ROC-AUC](glossary.md#roc-auc)
and [PR-AUC](glossary.md#pr-auc) for both models on a small sample of the data. So no choice below was made
before any result was seen. The house standard asks for a plan committed before the
results, and this repo does not meet that rule. See
[ADR 0001](decisions/0001-adopting-the-house-standard.md), decision 9.

## Question

How well can a model built from the application form rank applicants by their chance of
payment difficulties? And can its scores be read as probabilities at a [decline rate](glossary.md#decline-rate) a
lender might use?

## Data and split

The data is `application_train.csv` from Kaggle's Home Credit competition. The target is
`TARGET`, which marks an applicant who later had payment difficulties.

| Choice | First fixed in | Before or after the first results |
|---|---|---|
| Hold out part of the data and report on it | `d2d9e9c`: one stratified split with `test_size=0.2`, so a fifth of the rows held out | Same commit |
| Use all rows, not a sample | `952b03b` | After |
| Three parts: train (60%), [calibration](glossary.md#calibration) (20%) and test (20%), each stratified on the target | `952b03b` | After |
| The same fixed seed for every split | `d2d9e9c` (`RANDOM_STATE` in `config.py`) | Same commit |

The models see only the train part. The calibrators and the decline threshold see only
the calibration part. Every reported number comes from the test part. The test
`tests/test_leakage.py::test_splits_do_not_share_applicants` checks that no applicant is
in two parts.

The split is random, not by time. The table has no application date, only day counts
relative to each application (such as `DAYS_BIRTH`). So a split by time is not possible
from this table alone. [What's weak](whats_weak.md) ranks this first.

## Main metrics

| Metric | What it measures | First fixed in | Before or after |
|---|---|---|---|
| ROC-AUC | How well the scores rank defaulters above non-defaulters | `d2d9e9c` | Same commit |
| PR-AUC | Ranking quality with a focus on the rare defaulters | `d2d9e9c` | Same commit |
| Precision, recall and decline rate at a threshold | What a decision rule does | `d2d9e9c` | Same commit |
| [Brier score](glossary.md#brier-score) and expected calibration error ([ECE](glossary.md#ece)) | Whether a predicted PD matches the observed default rate | `c790239` | After |
| A 95% [bootstrap](glossary.md#bootstrap) [confidence interval](glossary.md#confidence-interval) for ROC-AUC and PR-AUC, and for the gap between the two models on the same resampled rows | How much each number could move with a different test set | Function in `c790239`, wired in with 1,000 resamples in `952b03b` | After |

No single number was named as the one that decides the project. The README leads with
LightGBM's test ROC-AUC and its gap over logistic regression.

## Baseline

The [baseline](glossary.md#baseline) is a class-weighted logistic regression. It has
been there since `d2d9e9c`, where the README called it "the interpretable scorecard
reference". Since `952b03b`, the claim that LightGBM ranks better rests on the paired
gap, whose interval lies above zero. That rule was not written down before. See
[ADR 0002](decisions/0002-lightgbm-scores-logistic-regression-is-the-baseline.md).

The logistic regression did not converge in `d2d9e9c`. Commit `cbf4ad2` centred the
numeric columns so that it converges. `reports/logreg_convergence.md` compares the
setups before and after.

## Operating point

| Choice | First fixed in | Before or after |
|---|---|---|
| A fixed cut of 0.40 on the raw LightGBM score, called illustrative <!-- not-a-claim --> | `d2d9e9c` (README text), `952b03b` (in code) | Same commit, then after |
| Decline the riskiest 20%, with the threshold set on the calibration part | `7f4f511` | After |

The message of `7f4f511` says why the fixed cut was dropped:
`On full data the old 0.40 raw-score cut declines 45% of applicants`. So this choice was
made after full-data results were seen. The 20% rate itself is a policy choice, not a
result. See [ADR 0004](decisions/0004-operating-point-is-a-20-percent-decline-rate.md).

## Calibration

| Choice | First fixed in | Before or after |
|---|---|---|
| Fit [Platt scaling](glossary.md#platt-scaling) and [isotonic regression](glossary.md#isotonic-regression) on a held-out calibration part | `a0ac986` | After |
| Use isotonic regression for every number that needs a PD | `952b03b` | After |

The code comment in `952b03b` says isotonic was chosen before looking at test results.
The history can't confirm or refute this. The choice and the first code that scored
calibration on the test part arrived in the same commit. See
[ADR 0003](decisions/0003-held-out-calibration-with-isotonic-regression.md).

## What counts as success

No bar for success was set in advance. None is added here, because a bar picked after
the results would not test anything.

## Choices made after full-data results were seen

These choices changed the model or the reports after the full-data numbers were known.
Each one could, in principle, have been steered by those numbers.

- The operating point moved from a fixed cut to a 20% decline rate (`7f4f511`). The
  threshold is set on the calibration part, so the test part did not pick it.
- `CODE_GENDER` was dropped as a model input (`ef4b74b`, 2026-09-25). The full-data
  results with the group check were committed about three hours earlier, in `db53b26`. The
  with-and-without comparison (`04bf471`) came after the drop. See
  [ADR 0005](decisions/0005-drop-code-gender-as-a-model-input.md).
- Every number was re-run on the same test part after each change (`b9e96c3`,
  `b97517e`, `4c419ce`, `6fee45f`). The test part has been looked at many times, so it
  is not a clean holdout any more.

## What would change the conclusion

- A split by time. If a later period is harder to predict, ROC-AUC falls, and the PDs
  drift away from the observed default rate.
- A stronger baseline, such as a scorecard with binned inputs. It could narrow the gap
  between LightGBM and logistic regression.
- A different random split. The intervals cover resampling of the test rows only. They
  do not cover a different train part or a different seed.
- Other groups. The group check covers gender and four age bands only.

## Changes to this plan

None yet.
