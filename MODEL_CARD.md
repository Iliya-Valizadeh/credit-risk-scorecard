# Model card: credit-risk scorecard (LightGBM with isotonic calibration)

Short form of the model card from Mitchell et al., "Model Cards for Model Reporting"
(https://arxiv.org/abs/1810.03993). Every number here has a row in
[CLAIMS.md](CLAIMS.md). All figures are from
`python -m credit_risk_scorecard.model` on the test split (61,503 applicants), as
recorded in [reports/results.md](reports/results.md), except the with/without-gender
comparison, which is from `python -m credit_risk_scorecard.gender_check`
([reports/gender_check.md](reports/gender_check.md)).

## Model details

A LightGBM classifier, calibrated with [isotonic regression](docs/glossary.md#isotonic-regression),
trained on Kaggle's Home Credit Default Risk data. It is a portfolio project built by
one person (Iliya Valizadeh), not a fielded product, and has no license beyond this
repo's [LICENSE](LICENSE).

## Intended use

The model estimates the probability that a consumer credit applicant will have payment
difficulties, to support an approve or decline decision. It is a portfolio project on
public data and is not for real lending, pricing or collections. If it were used as
decision support, a person would make the final decision.

## Factors

The group check covers gender (`CODE_GENDER`, women and men) and age in four bands (20
to 29, 30 to 44, 45 to 59, 60 and over), <!-- not-a-claim -->
because these are the groups the data
supports. Sex and age are also both prohibited grounds of discrimination under the
Canadian Human Rights Act, so a Canadian lender would likely check them.
Gender is not a model input; it stays in the data only for this check. Age is an input,
through `DAYS_BIRTH`.

## Metrics

[ROC-AUC](docs/glossary.md#roc-auc) and [PR-AUC](docs/glossary.md#pr-auc) for ranking,
with 95% [bootstrap](docs/glossary.md#bootstrap) [confidence intervals](docs/glossary.md#confidence-interval);
the [Brier score](docs/glossary.md#brier-score) and [ECE](docs/glossary.md#ece) for
[calibration](docs/glossary.md#calibration); and precision, recall and the
[decline rate](docs/glossary.md#decline-rate)
at the chosen threshold, since a PD only matters through the decision it leads to. See
[the evaluation plan](docs/eval_plan.md) for why these and not others.

## Evaluation data

Home Credit Default Risk, `application_train.csv`: 307,511 applications and 122 columns. <!-- not-a-claim -->
The target, `TARGET` = 1, means the client had payment difficulties (8.07% of
applicants). The split is 60% train, 20% calibration and 20% test, stratified, with
random seed 42. It is random, not by date. Every number in this card is on the test
split.

## Training data

The same source file, train split only (184,506 rows). `DAYS_EMPLOYED` = 365243 means
"not employed" and is treated as missing. The bureau, previous-application and
payment-history tables are not used. The model has 244 inputs after encoding:
application fields, three engineered ratios and one-hot categories.

It is LightGBM with 600 trees, learning rate 0.02, 31 leaves and `scale_pos_weight` set
so that each defaulter counts several times as much as each non-defaulter in training,
to counteract the rare target. Isotonic regression, fitted on the calibration split,
turns the raw score into a PD. The illustrative decision rule declines an applicant when the raw score is 0.600
or above, which declines the riskiest 20% of the calibration split.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.769 (95% bootstrap interval 0.762 to 0.775) |
| PR-AUC | 0.254 (0.242 to 0.265) |
| Logistic regression [baseline](docs/glossary.md#baseline) ROC-AUC | 0.752 (0.745 to 0.759) |
| At the decision rule | 19.5% declined, recall 53.2%, precision 22.0% |

### Calibration

| PDs | Mean PD | Brier | ECE |
|---|---|---|---|
| Raw | 0.395 | 0.1851 | 0.3145 |
| Isotonic (used) | 0.080 | 0.0673 | 0.0027 |
| Observed default rate | 0.0807 | | |

Because of the class weighting, the raw scores overstate risk about five-fold on
average. Only the calibrated PDs should be read as probabilities. Isotonic regression
maps nearby scores to the same PD, so PR-AUC on calibrated PDs is slightly lower (0.244)
than on raw scores. Decisions use the raw score, so they are unaffected.

### By gender and age (at the decision rule)

By gender, non-defaulting men are declined at 20.9% and non-defaulting women at 14.4%.
With gender as an input those figures were 23.9% and 13.1%. The other inputs still
predict gender with ROC-AUC 0.876, and `EXT_SOURCE_1`, `DAYS_BIRTH` and `FLAG_OWN_CAR`
are the important inputs most correlated with it. Calibrated PDs are 0.4 points above
the observed rate for women and 1.0 point below it for men. <!-- not-a-claim -->

By age, non-defaulting applicants aged 20 to 29 <!-- not-a-claim --> are declined at
30.7%, against 4.7% for those 60 and over. In each age band, the mean calibrated PD is
close to the observed default rate. None of these group figures has an interval. The README explains what the
check does and doesn't show.

## Ethical considerations

A wrong decline costs a creditworthy applicant a loan; a wrong approval costs the
lender an [expected loss](docs/glossary.md#expected-loss) it didn't price for. Both errors are measured above, and both
differ by group. Removing `CODE_GENDER` narrowed but did not close the gap between men
and women, because other inputs still carry gender information (see the README's
"Gender" section). `DAYS_BIRTH` is used directly, and age is a prohibited ground of
discrimination under the Canadian Human Rights Act; whether a lender could use it here
is a legal question this project does not answer. Before any real use, both would need
a fairness and legal review.

## Caveats and recommendations

- There is no out-of-time test, so performance on later applicants is unknown.
- The target is broader than a regulatory default definition.
- The [expected loss](docs/glossary.md#expected-loss) view uses an assumed
  [LGD](docs/glossary.md#lgd) (45%) and exposure (full credit amount).
- There is no hyperparameter tuning and no bureau or behavioural data.
- Home Credit's applicants may not resemble any particular Canadian lender's customers.

The full, ranked list of weaknesses is in [docs/whats_weak.md](docs/whats_weak.md).

### Monitoring and when to retrain

These are the checks I would run and the triggers I would start from. They are common
starting points, not values validated for this model.

- [PSI](docs/glossary.md#psi) on each of the top 10 [SHAP](docs/glossary.md#shap)
  features and on the score.
  Investigate above a PSI of 0.10 and retrain above 0.25. <!-- not-a-claim -->
- The mean calibrated PD against the observed default rate, as each cohort's outcomes
  mature. Refit the calibrator if the gap exceeds 1 percentage point for two periods in
  a row. <!-- not-a-claim -->
- ROC-AUC on each matured cohort. Retrain if it falls below 0.762, the lower end of the
  test interval.
- The gender and age group check, each period. Escalate if a gap widens.
- A scheduled retrain at least once a year, and after any change to the product, the
  approval policy or the data sources. <!-- not-a-claim -->
