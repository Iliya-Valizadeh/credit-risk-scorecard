# 0003: Held-out calibration with isotonic regression

Date: 2026-09-25, recording choices first made on 2026-09-24 in `a0ac986` and
`952b03b`. Status: accepted.

This record was written after the fact, from the git history.

## Context

LightGBM is trained with `scale_pos_weight`, so each defaulter counts about as much as
eleven other applicants. It is meant to help ranking on a rare target, but no model
without it was trained to check that. What it does do is push every score up. The README in
`3b07ccd` already said the raw scores over-predict default and would need Platt or
isotonic recalibration before anyone used them as a probability of default (PD).

A PD is needed for the expected-loss view and for comparing predicted and observed
default rates by group.

## Options

The history shows these options:

- Keep the raw scores. This is what the repo did until `a0ac986` added the
  calibrators. Until `2a75872`, the app showed the raw score as a PD, although it was
  too high to read that way.
- [Platt scaling](../glossary.md#platt-scaling): a logistic regression on the log-odds of the raw score. It has two
  parameters, so it can only shift and stretch the curve.
- [Isotonic regression](../glossary.md#isotonic-regression): a step function that only has to go up. It is more flexible,
  but it needs more data and can overfit a small [calibration](../glossary.md#calibration) set. The docstring of
  `calibrate.py` gives both descriptions.

Both calibrators must be fitted on rows that neither the model nor the test uses.
Commit `952b03b` made that possible. It split the data into train, calibration and test
parts, so the calibrators never see a training row or a test row.

## Decision

Fit both calibrators on the calibration part, report both on the test part, and use
isotonic regression wherever a PD is needed. The choice of isotonic is set in
`DOWNSTREAM_CALIBRATOR` in `model.py`, first in `952b03b`.

The reason given in the code comment and the README is that isotonic was chosen before
looking at test results, because the calibration part is large (61,502 rows). The
history can't confirm the "before" part. The constant and the first code that scored
calibration on the test part came in the same commit.

## Consequences

- On the test part, the two calibrators come out almost the same
  (`reports/metrics.json`, key `calibration`). So the choice between them matters little
  here.
- Isotonic regression maps many raw scores to the same value, and ties lower the
  ranking a little. LightGBM's [PR-AUC](../glossary.md#pr-auc) is 0.254 on raw scores and 0.244 after isotonic.
  So the decline threshold is set on the raw score
  ([ADR 0004](0004-operating-point-is-a-20-percent-decline-rate.md)), and the
  calibrated PD is used only where a probability is needed.
- The calibration part costs 20% of the rows that could have gone to training.
- The calibrators are fitted once, on one period. A PD that is right today can drift if
  the default rate changes. The split is not by time, so this is untested.
