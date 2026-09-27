# Model card

## Intended use

This project is an educational demonstration of supervised credit-default modeling, calibrated risk probabilities, explainable features, threshold policies, fairness auditing, and human review.

## Prohibited use

Do not use this model to approve, decline, price, limit, or otherwise make real decisions about a person’s access to credit. Do not treat the score as a bureau score or as a measure of a person’s worth.

## Training data

The primary model uses the UCI Default of Credit Card Clients dataset. It contains historical Taiwan credit-card customer records and a next-month default target. The data is not representative of every geography, product, population, or underwriting context.

## Inputs and target

Inputs are credit limit, historical repayment status, bill amounts, payment amounts, and engineered behavioral features. Sex, education, marital status, and age are excluded from prediction and retained only for audit reporting. The target is recorded default in the following month.

## Model and evaluation

The pipeline compares calibrated logistic regression and calibrated histogram gradient boosting. It reports ROC-AUC, PR-AUC, log loss, Brier score, precision, recall, specificity, confusion matrices, calibration, threshold outcomes, and group fairness metrics. Exact metrics are generated in `outputs/reports/training_report.json` for each training run.

## Limitations and risks

Historical correlation is not causal evidence. Features may encode proxies for protected characteristics. Calibration and fairness can change across time, geography, product, and population. The project does not include identity verification, affordability assessment, fraud detection, regulatory controls, or production monitoring.

## Explainability

The API returns directional feature reason codes and structured contributions. Linear candidates use signed coefficient contributions; nonlinear tree candidates use deterministic local feature-ablation effects against the fitted imputer baseline. These are technical model explanations, not causal claims, formal adverse-action notices, or legal conclusions.

## Monitoring and retraining

`POST /monitoring/drift` calculates PSI and missingness drift. It recommends investigation and never retrains automatically. Retraining requires a human review of data quality, temporal validation, fairness, calibration, and policy impact.
