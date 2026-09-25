# Model card: credit-risk scorecard (LightGBM with isotonic calibration)

All figures are from `python -m src.model` on the test split (61,503 applicants), as
recorded in [reports/results.md](reports/results.md), except the with/without-gender
comparison, which is from `python -m src.gender_check`
([reports/gender_check.md](reports/gender_check.md)).

## Intended use

The model estimates the probability that a consumer credit applicant will have payment
difficulties, to support an approve or decline decision. It is a portfolio project on
public data and is not for real lending, pricing or collections. If it were used as
decision support, a person would make the decision.

## Data

Home Credit Default Risk, `application_train.csv`: 307,511 applications and 122 columns.
The target, `TARGET` = 1, means the client had payment difficulties (8.07% of
applicants). The split is 60% train, 20% calibration and 20% test, stratified, with
random seed 42. It is random, not by date. `DAYS_EMPLOYED` = 365243 means "not employed"
and is treated as missing. The bureau, previous-application and payment-history tables
are not used.

## Model

The model has 244 inputs after encoding: application fields, three ratios and one-hot
categories. `CODE_GENDER` is not an input; it stays in the data only for the group
check. Age is an input, through `DAYS_BIRTH`.

It is LightGBM with 600 trees, learning rate 0.02, 31 leaves and `scale_pos_weight` of
about 11.4. Isotonic regression, fitted on the calibration split, turns the raw score
into a PD. The illustrative decision rule declines an applicant when the raw score is
0.600 or above, which declines the riskiest 20% of the calibration split.

## Metrics (test split)

| Metric | Value |
|---|---|
| ROC-AUC | 0.769 (95% bootstrap interval 0.762 to 0.775) |
| PR-AUC | 0.254 (0.242 to 0.265) |
| Logistic regression baseline ROC-AUC | 0.752 (0.745 to 0.759) |
| At the decision rule | 19.5% declined, recall 53.2%, precision 22.0% |

## Calibration

| PDs | Mean PD | Brier | ECE |
|---|---|---|---|
| Raw | 0.395 | 0.1851 | 0.3145 |
| Isotonic (used) | 0.080 | 0.0673 | 0.0027 |
| Observed default rate | 0.0807 | | |

Because of the class weighting, the raw scores overstate risk about five-fold on
average. Only the calibrated PDs should be read as probabilities. Isotonic regression
maps nearby scores to the same PD, so PR-AUC on calibrated PDs is slightly lower (0.244)
than on raw scores. Decisions use the raw score, so they are unaffected.

## Group check (at the decision rule)

By gender, non-defaulting men are declined at 20.9% and non-defaulting women at 14.4%.
With gender as an input those figures were 23.9% and 13.1%. The other inputs still
predict gender with ROC-AUC 0.876, and `EXT_SOURCE_1`, `DAYS_BIRTH` and `FLAG_OWN_CAR`
are the important inputs most correlated with it. Calibrated PDs are 0.4 points above the
observed rate for women and 1.0 point below it for men.

By age, non-defaulting applicants aged 20 to 29 are declined at 30.7%, against 4.7% for
those 60 and over. Calibration holds within each age band. The README explains what the
check does and doesn't show.

## Limitations

- There is no out-of-time test, so performance on later applicants is unknown.
- The target is broader than a regulatory default definition.
- Gender is out of the model but still carried by other inputs, and age is used
  directly. Both would need a fairness and legal review before any real use.
- The expected-loss view uses an assumed LGD (45%) and exposure (full credit amount).
- There is no hyperparameter tuning and no bureau or behavioural data.
- Home Credit's applicants may not resemble any particular Canadian lender's customers.

## Monitoring and when to retrain

These are the checks I would run and the triggers I would start from. They are common
starting points, not values validated for this model.

- Population stability index (PSI) on each of the top 10 SHAP features and on the
  score. Investigate above 0.10 and retrain above 0.25.
- The mean calibrated PD against the observed default rate, as each cohort's outcomes
  mature. Refit the calibrator if the gap exceeds 1 percentage point for two periods in a
  row.
- ROC-AUC on each matured cohort. Retrain if it falls below 0.762, the lower end of the
  test interval.
- The gender and age group check, each period. Escalate if a gap widens.
- A scheduled retrain at least once a year, and after any change to the product, the
  approval policy or the data sources.
