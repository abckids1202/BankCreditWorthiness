# Explainable Credit Approval Simulator

An educational ML-engineering project that estimates near-term credit-default risk and maps it to `approve`, `manual_review`, or `decline`. It is not a bank, does not assess a person's worth, and must not be used for real lending decisions.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/train.py
pytest
uvicorn credit_simulator.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API. In another terminal run `streamlit run dashboard.py` for the dashboard. Training downloads the UCI Default of Credit Card Clients dataset into `data/raw/` and writes model files to `artifacts/` plus reports to `outputs/`.

## Docker

Train once on the host, then run `docker compose up --build`. The API is on port 8000 and the dashboard is on port 8501. The dashboard defaults to `http://api:8000` inside Compose.

## Design

The target is the UCI `default.payment.next.month` field. `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are retained for audit reporting but excluded from model features. Logistic regression and histogram gradient boosting are compared using ROC-AUC, PR-AUC, and Brier score; the better validation candidate is selected. Explanations are reason codes derived from the selected estimator, not legal adverse-action notices.

Training also creates utilization, repayment-delay, payment-ratio, balance-trend, and account-stability features. It saves an educational 300–850 score alongside the raw default probability. Detailed artifacts are generated under `outputs/reports/`, including `training_report.json`, `threshold_analysis.csv`, calibration and risk-distribution plots, target-balance plots, and feature summaries.

The policy uses configurable thresholds: low risk is approved, high risk is declined, and the middle band goes to human review. Inputs outside the expected distribution should also be reviewed. Removing sensitive fields does not prove that a model is fair, so the training report includes group-level approval and error-rate summaries. The numeric score is a presentation layer based on calibrated probability and configurable odds parameters in `configs/default.yaml`; it is not a real-world bureau score.

## API example

After training, use the Swagger UI or send JSON to `POST /predict`. The request contains `LIMIT_BAL`, six repayment-status fields (`PAY_0`, `PAY_2`–`PAY_6`), six bill fields, and six payment fields. The response contains the modeled probability, credit score, risk band, decision thresholds, rationale, model version, structured explanations, reason codes, and warnings.

Borderline cases can be persisted for educational human review with `POST /review-cases`, listed with `GET /review-cases`, inspected with `GET /review-cases/{case_id}`, and updated with `PATCH /review-cases/{case_id}`. SQLite stores the original automatic decision separately from the reviewer decision; reviewer outcomes are not allowed to overwrite the model output.

## Limitations

The source data is historical and geographically limited. It is not representative of all borrowers, contains proxy-sensitive information, and does not establish causal relationships. There is no identity verification, fraud detection, affordability assessment, regulatory compliance layer, or production monitoring in this prototype.

