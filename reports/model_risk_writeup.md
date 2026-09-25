# Model-risk write-up: credit-risk scorecard

*A one-page note in the terms a bank's model-risk team uses. Every number is from
`python -m src.model` and is recorded in [results.md](results.md).*

## 1. Purpose and use
The model estimates an applicant's probability of default (PD) on a consumer credit
product, to support an approve/decline decision at a chosen threshold. It is decision
support, **not** automated final adjudication. It must not be used for pricing or
collections without separate validation.

## 2. Data and representativeness
Trained on the Home Credit Default Risk application data: all 307,511 applicants, split
60/20/20 into train, calibration and test. 8.07% of applicants have the target (payment
difficulties), a class imbalance of about 1:11 that drives the choices below. A known
data-quality issue is handled explicitly: `DAYS_EMPLOYED` holds a sentinel value (365243)
for applicants not in employment, and it is recoded to missing rather than read as a
1,000-year job. Limitation: this is a single-snapshot public dataset with a random split.
Drift over time, and how well this population matches any specific lender's book, are
untested.

## 3. Performance
Defaults are rare, so accuracy is meaningless: predicting "no default" for everyone
scores about 92%. The model is judged on ranking and on its behaviour at the operating
point.

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|
| Logistic regression | 0.755 (0.748 to 0.761) | 0.229 (0.219 to 0.240) |
| LightGBM | 0.770 (0.764 to 0.777) | 0.256 (0.244 to 0.268) |

The logistic regression is the interpretable reference. The boosted model is the
candidate, and its advantage holds on a paired bootstrap (ROC-AUC gap 0.012 to 0.019).
The illustrative policy declines the riskiest 20%: it catches 53.7% of defaulters at
22.0% precision.

## 4. Calibration
A bank needs more than the ranking. The PD has to mean what it says, because it feeds
expected-loss provisioning and risk-based pricing. The class weighting used in training
(`scale_pos_weight`) inflates every PD: the raw mean PD is 0.394 against an observed
rate of 0.081. Isotonic regression fitted on the calibration split brings the mean PD to
0.080, the Brier score from 0.185 to 0.067, and the expected calibration error from
0.313 to 0.003, with the ranking unchanged. Under an assumed 45% LGD, the loss implied
by raw PDs on approved loans is almost 7 times the loss implied by actual outcomes.
Calibrated PDs come within 3%.

## 5. Explainability
Two obligations: explain the model globally, and explain any individual decline (adverse
action). Globally, SHAP ranks the three external credit scores first (`EXT_SOURCE_3`,
`EXT_SOURCE_2`, `EXT_SOURCE_1`), followed by the engineered annuity-to-credit ratio
(`reports/figures/shap_summary.png`). Per-applicant SHAP values would let an adjudicator
state the reasons a given application scored as it did. That isn't built into the demo
app yet.

## 6. Fairness
Calibration holds within gender and age groups, but error rates don't match. Non-defaulting
men are declined at 23.9% vs 13.1% for non-defaulting women, and the gap by age is wider.
Gender is a direct input and ranks 8th by SHAP. Before any real use, this needs a fairness
and legal review, and the gender input would very likely have to go, with a check for
proxies afterwards.

## 7. Limitations and monitoring
The model shouldn't be trusted outside the population it was trained on. It assumes a
stable relationship between features and default, and it has had no out-of-time test.
In production it would need PD-calibration monitoring, population-stability (PSI)
checks on inputs, group error-rate monitoring, and a retrain trigger when any of them
drifts past tolerance. [MODEL_CARD.md](../MODEL_CARD.md) sets out proposed thresholds.
