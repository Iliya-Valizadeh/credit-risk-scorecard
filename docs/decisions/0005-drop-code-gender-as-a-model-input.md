# 0005: Drop CODE_GENDER as a model input

Date: 2026-09-25, recording a choice made on 2026-09-25 in `ef4b74b`. Status: accepted.

This record was written after the fact, from the git history.

## Context

The data has a `CODE_GENDER` column. Until `ef4b74b`, both models used it as an input.
The first full-data results, committed in `db53b26`, included a group check by gender
(added in `d3cfcfe`). That check reports approval rates and error rates for women and
men at the decline threshold. In the same results, the encoded column `CODE_GENDER_F`
was the eighth most important input by mean absolute [SHAP](../glossary.md#shap) value (`reports/metrics.json`
at `db53b26`, key `shap_top10`).

## Options

- Keep it as an input. The model may rank a little better, but its decisions then
  depend on gender directly.
- Drop it from the data. The model can no longer use it, but the group check can no
  longer measure outcomes by gender either.
- Drop it as a model input, and keep it in the data only for the group check.

## Decision

The last option. The preprocessor no longer passes `CODE_GENDER` to either model, and
`src/credit_risk_scorecard/fairness.py` still reports results by gender. `model.run`
takes an `exclude` argument, so the old setup can still be rebuilt for comparison. The
test `test_gender_is_not_a_model_input_but_stays_for_the_group_check` in
`tests/test_leakage.py` enforces this on every CI run.

The commit message states what changed, but not why. The history holds no written
reason for the choice. The drop came after the full-data results and the gender group
check had been seen.

## Consequences

Commit `04bf471` added `gender_check.py`, which trains the model with and without the
column on the same split. Its results are in `reports/gender_check.md`:

- The ranking barely changed. The drop cost a small amount of [ROC-AUC](../glossary.md#roc-auc).
- The gaps between women and men narrowed but did not close.
- The other inputs still predict gender well. Some of them are among the model's most
  important inputs, so gender still reaches the model through them.
- [Calibration](../glossary.md#calibration) by gender got worse. With the column, predicted and observed default
  rates were close for both groups. Without it, the PD is too high for women and too
  low for men.

Dropping the column is therefore not a fairness fix. It removes the direct use of
gender, and it trades a small loss in ranking and a worse fit by group for that. Age
enters the model directly through `DAYS_BIRTH`. No record decides whether that is
allowed. [What's weak](../whats_weak.md) lists both points.
