# Glossary

Each technical term has a heading and one plain sentence. Link a term to its heading
the first time it appears in each file. `make check-docs` checks this.

## Baseline

A simple method, such as always guessing the most common answer, that a model must
beat to be worth using.

## Bootstrap

A way to see how much a number could change: draw many new samples from the test rows,
with repeats allowed, and compute the number again on each one.

## Brier score

The average squared gap between a predicted probability and the actual outcome (0 or
1). Lower is better, and zero is perfect.

## Calibration

Adjusting a model's scores so that a predicted probability matches how often the
outcome actually happens.

## Confidence interval

A range around a measured number that shows how much the number could move if the
test were run again on new data.

## Decline rate

The share of applicants a lending policy declines at a chosen score cutoff.

## ECE

Expected calibration error: sort applicants into equal-sized groups by predicted
probability, then average the gap between each group's predicted rate and its actual
rate.

## Expected loss

What a lender expects to lose on a loan: the chance of default, times the loss given
default, times the amount at risk.

## Isotonic regression

A way to calibrate scores into probabilities. It fits a curve that only ever goes up,
on held-out data, rather than assuming any particular shape.

## LGD

Loss given default: the share of the amount owed that a lender expects to lose if a
borrower defaults.

## Platt scaling

A way to calibrate scores into probabilities by fitting one logistic curve.

## PR-AUC

The area under the precision-recall curve. It measures how well a model ranks the rare
positive cases, averaged over every possible score cutoff.

## PSI

Population stability index: how much a feature's or a score's distribution has shifted
between two time periods.

## ROC-AUC

The area under the ROC curve: the chance that the model ranks a random defaulter above
a random non-defaulter. 0.5 is no better than a coin flip; 1.0 is perfect. <!-- not-a-claim -->

## SHAP

A method that splits one prediction into the amount each input added or took away, so
the prediction can be explained input by input.
