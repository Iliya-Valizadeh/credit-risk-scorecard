# 0004: The operating point is a 20% decline rate

Date: 2026-09-25, recording a choice first made on 2026-09-24 in `7f4f511`. Status:
accepted.

This record was written after the fact, from the git history.

## Context

A score becomes a decision only at a threshold: applicants above it are declined. The
threshold sets how many defaulters are caught and how many good applicants are turned
away.

The first README (`d2d9e9c`) showed a table of thresholds on the raw LightGBM score and
called a cut of 0.40 an illustrative operating point. It also said the right cut
depends on the cost of a missed default against the cost of declining a good customer.
Commit `952b03b` put the 0.40 cut into the code.

Once the model ran on all rows, that cut declined far more applicants than intended.
The message of `7f4f511` says:
`On full data the old 0.40 raw-score cut declines 45% of applicants, because the class weighting pushes every score up.`

## Options

- Keep a fixed cut on the raw score. It is simple, but its meaning shifts whenever the
  model or its class weighting changes, as the full-data run showed.
- Set the cut from the costs of each kind of error. This is what a lender would do, but
  the data has no recovery or pricing figures, so the costs would be made up.
- Pick a decline rate as the policy, and find the score that gives that rate. The
  commit message calls this "easier to reason about".

For the last option, the threshold can be found on the test part or on the calibration
part. Finding it on the test part would let the test rows shape the rule they judge.

## Decision

Decline the riskiest 20% of applicants. The threshold is the raw LightGBM score that
declines 20% of the calibration part (`FLAG_RATE` in `model.py`). The test part then
shows what that threshold does: its decline rate, the share of defaulters caught
(recall) and the share of declines who did default (precision).

The history does not say why 20% and not another rate. The code calls it an
"illustrative policy". It is a policy choice, not a result, and it was made after the
full-data results were seen.

## Consequences

- The threshold stays tied to a decline rate that a reader can picture, even if the
  model changes.
- The test decline rate can differ a little from 20%, because the threshold comes from
  another set of rows. `reports/metrics.json` records both (key `operating_point`).
- The group check, the expected-loss view and the app all use this one threshold.
  Commit `2a75872` made the app read the saved threshold.
- No cost analysis supports the 20% rate. A lender would set it from its own costs and
  risk appetite, and the group results would change with it.
