# Model-risk write-up: credit-risk scorecard

A one-page note in the terms a bank's model-risk team uses. Every number is from
`python -m credit_risk_scorecard.model` ([results.md](results.md)) or
`python -m credit_risk_scorecard.gender_check` ([gender_check.md](gender_check.md)).

## 1. Purpose and use
The model estimates an applicant's probability of default (PD) on a consumer credit
product. It is a portfolio project on public data and is not for real lending, pricing
or collections. If it were used, it would be decision support for an approve or decline
decision at a chosen threshold, and a person would make the final decision.

## 2. Data and representativeness
The model is trained on the Home Credit Default Risk application data: all 307,511
applicants, split 60/20/20 into train, calibration and test. 8.07% of applicants have
the target (payment difficulties), a class imbalance of about 1:11 that drives the
choices below. `DAYS_EMPLOYED` holds a sentinel value (365243) for applicants not in
employment; it is recoded to missing rather than read as a 1,000-year job. This is a
single-snapshot public dataset with a random split, so drift over time, and how well
this population matches any specific lender's book, are untested.

## 3. Performance
Defaults are rare, so accuracy is meaningless: predicting "no default" for everyone
scores about 92%. The model is judged on ranking and on its behaviour at the operating
point.

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|
| Logistic regression | 0.752 (0.745 to 0.759) | 0.227 (0.216 to 0.238) |
| LightGBM | 0.769 (0.762 to 0.775) | 0.254 (0.242 to 0.265) |

The logistic regression is the interpretable reference and the boosted model is the
candidate. Its advantage holds on a paired bootstrap (ROC-AUC gap 0.013 to 0.020). The
illustrative policy declines the riskiest 20%. It catches 53.2% of defaulters at 22.0%
precision.

## 4. Calibration
The PD has to mean what it says, because it feeds expected-loss provisioning and
risk-based pricing. The class weighting used in training (`scale_pos_weight`) inflates
every PD: the raw mean PD is 0.395 against an observed rate of 0.081. Isotonic regression
fitted on the calibration split brings the mean PD to 0.080, the Brier score from 0.185
to 0.067, and the expected calibration error from 0.315 to 0.003. Under an assumed 45%
LGD, the loss implied by raw PDs on approved loans is almost 7 times the loss implied by
actual outcomes. With calibrated PDs it is within 2%, which is expected, since the
calibrator was fitted on data drawn the same way as the test part.

## 5. Explainability
The model has to be explained globally, and any individual decline has to be explained
too (adverse action). Globally, SHAP ranks the three external credit scores first
(`EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`), then the engineered annuity-to-credit
ratio (`reports/figures/shap_summary.png`). Per-applicant SHAP values would let an
adjudicator state the reasons a given application scored as it did. The demo app doesn't
show them yet.

## 6. Fairness
`CODE_GENDER` was an input and ranked 8th by SHAP. It has been removed from the model and
is kept only to measure outcomes by gender. Removing it narrowed the gaps. Non-defaulting
men were declined at 23.9% against 13.1% for non-defaulting women; now it is 20.9%
against 14.4%. It did not remove them. The other inputs predict gender with ROC-AUC
0.876, and `EXT_SOURCE_1`, `DAYS_BIRTH` and `FLAG_OWN_CAR` are both important to the
model and correlated with gender. Calibration by gender also got worse: the PD is now 0.4
points too high for women and 1.0 point too low for men. Age is used directly, and
non-defaulting applicants aged 20 to 29 are declined at 30.7% against 4.7% for those 60
and over. Before any real use, both would need a fairness and legal review.

## 7. Limitations and monitoring
The model shouldn't be trusted outside the population it was trained on. It assumes a
stable relationship between features and default, and it has had no out-of-time test.
In production it would need PD-calibration monitoring, population-stability (PSI)
checks on inputs, group error-rate monitoring, and a retrain trigger when any of them
drifts past tolerance. [MODEL_CARD.md](../MODEL_CARD.md) sets out proposed thresholds.
