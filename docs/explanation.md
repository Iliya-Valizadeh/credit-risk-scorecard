# Explanation

Background and reasons. This page says why the pipeline works the way it does. The
[tutorial](tutorial.md) and [how-to guides](how-to/) say what to type.

## Why two models

A lender's usual tool is a logistic regression scorecard, because each input's effect
can be read from one coefficient. Gradient-boosted trees such as LightGBM often rank
better on tabular data, but no single number explains them. This project keeps both:
logistic regression as the [baseline](glossary.md#baseline) a more complex model has to
beat, and LightGBM as the model that is actually scored. See
[ADR 0002](decisions/0002-lightgbm-scores-logistic-regression-is-the-baseline.md) for
how that choice was reconstructed from the git history, since it was never written down
at the time.

## Why class weighting, and why that needs calibration

Only 8.07% of applicants in this data have the target. `scale_pos_weight` in
LightGBM, and class weighting in logistic regression, make each defaulter count for
more during training. This is a common way to handle a rare target. This project never
trained a model without it, so it can't say how much, if at all, the weighting helped
the ranking. What is certain is the cost: the raw score overstates every applicant's
risk. The [calibration](glossary.md#calibration) step exists only to undo that side
effect. It uses data the models never trained on, so the fix isn't fitted on the same
rows it is judged on.

## Why isotonic regression over Platt scaling

Both are fitted on the calibration split, and both scored the same here: Brier 0.0673
for each. [ADR 0003](decisions/0003-held-out-calibration-with-isotonic-regression.md)
records the choice of [isotonic regression](glossary.md#isotonic-regression). Isotonic
regression makes fewer assumptions about the shape of the miscalibration than
[Platt scaling](glossary.md#platt-scaling), which fits one logistic curve, but it can
map many different scores to the same probability, which is part of why [PR-AUC](glossary.md#pr-auc)
on the calibrated PDs (0.244) is a little lower than on the raw scores (0.254). Ranking
is otherwise unaffected, since decisions use the raw score.

## Why a decline rate, not a fixed score cutoff

An earlier version of this project used a fixed cutoff on the raw score. Once the
model ran on the full data, that fixed cutoff declined far more applicants than
intended, because the class weighting had pushed every score up. Declining a fixed
share of applicants, instead of a fixed score, keeps the policy's meaning stable even
if the model or its weighting changes later.
[ADR 0004](decisions/0004-operating-point-is-a-20-percent-decline-rate.md) has the
detail, including that no cost analysis backs the particular 20% chosen.

## Why gender is out of the model, and age is still in

`CODE_GENDER` was an input, and [SHAP](glossary.md#shap) ranked it 8th by importance. Removing it narrowed,
but did not close, the gap in decline rates between men and women, because other inputs
still carry gender information. `DAYS_BIRTH` (age) stayed in, because there was no
similar proxy check for it and no rule yet says whether a lender could use it.
[ADR 0005](decisions/0005-drop-code-gender-as-a-model-input.md) records that decision
and its limits.

## Choices and their costs

Every choice above trades something away:

- The [baseline](glossary.md#baseline) is a plain logistic regression, not a scorecard
  with binned inputs, which is closer to what banks use. A stronger baseline could
  narrow the gap this project reports between the two models.
- Isotonic regression's step-function shape is what makes PR-AUC on the calibrated PDs
  lower than on the raw scores, described above.
- The 20% [decline rate](glossary.md#decline-rate) is a policy choice with no cost
  analysis behind it, so it says
  nothing about what rate a real lender should use.
- Removing `CODE_GENDER` protects against one direct channel, but the remaining gap
  through other inputs is not addressed by this project.

The full, ranked list of weaknesses, including ones not covered here, is in
[what's weak](whats_weak.md).
