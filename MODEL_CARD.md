# Model card: credit-risk scorecard (LightGBM + isotonic calibration)

All figures are from `python -m src.model` on the test split (61,503 applicants), as
recorded in [reports/results.md](reports/results.md).

## Intended use

- Estimate the probability that a consumer credit applicant will have payment
  difficulties, to support an approve/decline decision.
- A portfolio project on public data. Not for real lending, pricing or collections.
- If used as decision support, a person makes the decision. The model doesn't.

## Data

- Home Credit Default Risk, `application_train.csv`: 307,511 applications, 122 columns.
- Target: `TARGET` = 1 if the client had payment difficulties (8.07% of applicants).
- Split: 60% train, 20% calibration, 20% test, stratified, random seed 42. The split is
  random, not by date.
- Known data issue handled: `DAYS_EMPLOYED` = 365243 means "not employed" and is
  treated as missing.
- Not used: the bureau, previous-application and payment-history tables.

## Model

- 247 inputs after encoding: application fields, three ratios, one-hot categories.
  Includes `CODE_GENDER` and age (`DAYS_BIRTH`).
- LightGBM, 600 trees, learning rate 0.02, 31 leaves, `scale_pos_weight` ≈ 11.4.
- The raw score is mapped to a PD by isotonic regression, fitted on the calibration split.
- Decision rule (illustrative): decline when the raw score is 0.599 or above. That
  declines the riskiest 20% of the calibration split.

## Metrics (test split)

| Metric | Value |
|---|---|
| ROC-AUC | 0.770 (95% bootstrap interval 0.764 to 0.777) |
| PR-AUC | 0.256 (0.244 to 0.268) |
| Logistic regression baseline ROC-AUC | 0.755 (0.748 to 0.761) |
| At the decision rule | 19.7% declined, recall 53.7%, precision 22.0% |

## Calibration

| PDs | Mean PD | Brier | ECE |
|---|---|---|---|
| Raw | 0.394 | 0.1846 | 0.3134 |
| Isotonic (used) | 0.080 | 0.0672 | 0.0033 |
| Observed default rate | 0.0807 | | |

The raw scores overstate risk about five-fold on average because of the class
weighting. Only the calibrated PDs should be read as probabilities. Isotonic regression
maps nearby scores to the same PD, so PR-AUC on calibrated PDs is slightly lower (0.247)
than on raw scores. Decisions use the raw score, so they are unaffected.

## Group check (at the decision rule)

Calibrated PDs match observed default rates within each gender and age group (within
0.003 for every group). Error rates differ. Non-defaulting men are declined at 23.9% vs
13.1% for non-defaulting women, and non-defaulting applicants aged 20-29 at 31.2% vs 5.0%
for those 60+. Gender is a model input. See the README for what this check does and
doesn't show.

## Limitations

- No out-of-time test. Performance on later applicants is unknown.
- The target is broader than a regulatory default definition.
- Uses gender and age directly. It would need a fairness and legal review, and very
  likely removal of the gender input, before any real use.
- The expected-loss view uses an assumed LGD (45%) and exposure (full credit amount).
- No hyperparameter tuning. No bureau or behavioural data.
- Home Credit's applicants may not resemble any particular Canadian lender's customers.

## Monitoring and when to retrain

What I would monitor, and the triggers I would start from:

- **Input drift:** population stability index (PSI) on each of the top 10 SHAP features
  and on the score. Investigate above 0.10, retrain above 0.25.
- **Calibration:** the mean calibrated PD against the observed default rate, as each
  cohort's outcomes mature. Refit the calibrator if the gap exceeds 1 percentage point
  for two consecutive periods.
- **Ranking:** ROC-AUC on each matured cohort. Retrain if it falls below the lower end
  of the test interval (0.764).
- **Group error rates:** re-run the group check each period. Escalate if a gap widens.
- **Scheduled:** retrain at least once a year even without a trigger, and after any
  change to the product, the approval policy or the data sources.

These thresholds are common starting points, not values validated for this model.
