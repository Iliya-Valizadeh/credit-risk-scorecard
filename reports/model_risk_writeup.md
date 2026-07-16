# Model-Risk Write-up — Credit-Risk Scorecard

*One-page note in the terms a bank's model-risk function uses. Numbers in [brackets] are
filled from an actual training run — none are asserted before they are measured.*

## 1. Purpose and use
The model estimates an applicant's probability of default (PD) on a consumer credit
product, to support an approve/decline decision at a chosen threshold. It is a
decision-support scorecard, **not** an automated final adjudication, and must not be used
for pricing or collections without separate validation.

## 2. Data and representativeness
Trained on the Home Credit Default Risk application data (a 17,000-row sample of the ~307k applicants, all application-level fields). Target prevalence is 7.85% default — a ~1:12 class imbalance, which
drives every downstream choice below. Known data-quality issue handled explicitly:
`DAYS_EMPLOYED` carries a sentinel value (365243) for the not-employed population
(present in the full data; recoded wherever it appears), recoded to missing rather than treated as a real duration. Limitation:
the data is a single-snapshot public dataset, so temporal drift and this population's
match to any specific lender's book are untested.

## 3. Performance
Because defaults are rare, accuracy is meaningless (predicting "no default" for everyone
scores ~92%). The model is judged on ranking and on behaviour at the operating point:

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic regression (baseline) | 0.734 | 0.197 |
| LightGBM | 0.736 | 0.212 |

The logistic baseline exists as an interpretable reference; the boosted model is the
candidate. At an illustrative 0.40 threshold, the model flags ~23% of applicants and catches ~51% of
true defaulters (recall) at ~18% precision.

## 4. Calibration
A bank does not just need the *ranking* of applicants — it needs the PD to mean what it
says, because predicted PD feeds expected-loss provisioning and risk-based pricing. A
model can rank well yet be poorly calibrated. We therefore report a reliability curve
(`reports/figures/calibration.png`): predicted PD deciles versus observed default rate.
Deviation from the diagonal indicates where PDs are over- or under-stated and where a
post-hoc calibration step (e.g. isotonic/Platt) would be required before the scores drive
financial math.

## 5. Explainability
Two obligations: explain the model globally, and explain any individual decline
(adverse-action / regulatory requirement). Global drivers come from SHAP
(`reports/figures/shap_summary.png`); consistent with EDA, the external credit scores
`EXT_SOURCE_2/3` are expected to dominate, alongside the engineered credit-to-income
ratios. Local explanations (per-applicant SHAP) let an adjudicator state the specific
reasons a given application scored as it did.

## 6. Limitations and monitoring
The model should not be trusted outside the population it was trained on; it has no
fairness audit across protected attributes (a required step before any real deployment);
and it assumes a stationary relationship between features and default. In production it
would need PD-calibration monitoring, population-stability (PSI) checks on inputs, and a
retrain trigger when either drifts beyond tolerance.
