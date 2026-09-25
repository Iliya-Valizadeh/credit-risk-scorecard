# 0002: LightGBM scores, logistic regression is the baseline

Date: 2026-09-25, recording a choice first made on 2026-07-16 in `d2d9e9c`. Status:
accepted.

This record was written after the fact, from the git history. The reasons below are
the ones the code and the commit history show. Where the history gives no reason, this
record says so.

## Context

A lender needs a score for each applicant. The usual tool is a logistic regression
scorecard, because each input's effect can be read from one coefficient. Gradient
boosted trees, such as LightGBM, often rank better on tabular data, but no single
number explains them.

The first commit, `d2d9e9c`, trained both. Its README called the logistic regression
"the interpretable scorecard reference". `model.py` saved only LightGBM to
`models/pipeline.joblib`, so LightGBM was the model that the app scored with.

## Options

- Logistic regression only. It is easy to explain and is the standard in credit
  scoring. It gives no sense of how much ranking is lost for that clarity.
- LightGBM only. It gives the best ranking of the options tried, but no reference point
  to show what the extra complexity buys.
- Both. LightGBM scores applicants, and logistic regression is the
  [baseline](../glossary.md#baseline) it must beat.

## Decision

Both, with LightGBM as the scoring model. The history records the baseline's role in
the README text of `d2d9e9c`. It does not record a reason for picking LightGBM beyond
its higher test numbers.

The evidence at the time was weak. The first run used a sample of the data, the gap
between the models was small, and it had no interval. The logistic regression had also
stopped at its iteration limit without converging. Later commits made the comparison
fair:

- `cbf4ad2` centred the numeric columns so the logistic regression converges.
  `reports/logreg_convergence.md` shows the setups before and after.
- `c790239` added a paired [bootstrap](../glossary.md#bootstrap) interval for the gap.
  Both models are scored on the same resampled test rows.
- `952b03b` ran both models on all rows. On the test part, LightGBM's ROC-AUC is higher
  by 0.013 to 0.020 (95% interval, `reports/metrics.json`, key `bootstrap.roc_auc.diff`).

## Consequences

- LightGBM needs a separate tool to explain it. SHAP plots were added in `3b07ccd`.
- LightGBM is trained with `scale_pos_weight`, so its raw scores are too high to read as
  probabilities. [ADR 0003](0003-held-out-calibration-with-isotonic-regression.md)
  covers the fix.
- LightGBM runs with fixed settings. The first README called it "tuned", but no tuning
  code ever existed. Commit `7f4fa7c` removed that word, and the README now says there
  is no tuning.
- The baseline is a plain logistic regression on standardised inputs. It is not a
  scorecard with binned inputs, which is what banks use. A stronger baseline might
  narrow the gap. [What's weak](../whats_weak.md) lists this.
