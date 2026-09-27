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

Training also creates utilization, repayment-delay, payment-ratio, balance-trend, and account-stability features. It saves an educational 300–850 score alongside the raw default probability. Detailed artifacts are generated under `outputs/reports/`, including `training_report.json` with reproducible bootstrap 95% intervals for held-out ROC-AUC, PR-AUC, and Brier score, `threshold_analysis.csv`, `approval_rate_analysis.csv`, `global_feature_importance.csv`, calibration, risk-distribution, feature-distribution, confusion-matrix, and threshold-comparison plots, target-balance plots, and feature summaries.

Each primary training run gets a reproducible version such as `0.3.0+8c1b78222395`. The fingerprint is derived from the dataset hash, selected model, and training configuration, so changing any of those inputs creates a distinguishable artifact version. The full training configuration is also stored in `artifacts/metadata.json` and `outputs/reports/training_report.json` for experiment reconstruction.

The threshold CSV evaluates precision, recall, false-positive rate, and false-negative rate at each tested decline threshold, alongside approval/review/decline population rates, default rates by decision group, and configurable expected cost. The approval-rate report separately evaluates default precision and recall when approving the lowest-risk 50%, 70%, 80%, and 90% of the held-out population; these are diagnostic operating points, not recommended lending cutoffs.

Supported dataset adapters can be profiled with:

```powershell
python scripts/profile_dataset.py --dataset uci_default
python scripts/profile_dataset.py --dataset german_credit
python scripts/profile_dataset.py --dataset give_me_some_credit
```

The UCI Default adapter downloads automatically. The German Credit adapter downloads the UCI archive automatically. The Give Me Some Credit adapter expects `cs-training.csv` downloaded from Kaggle at `data/raw/cs-training.csv`; it does not attempt to bypass Kaggle access controls.

The generalized trainer can train alternate tabular schemas into separate experiment directories:

```powershell
python scripts/train.py --dataset german_credit --output-dir artifacts/german_credit
python scripts/train.py --dataset give_me_some_credit --output-dir artifacts/give_me_some_credit
```

Those alternate artifacts are experiment outputs and are not wired into the current credit-card `/predict` contract, which intentionally remains schema-safe. The German Credit path has been verified end-to-end; Give Me Some Credit requires its Kaggle CSV first.

Alternate training artifacts also receive reproducible fingerprints based on the dataset contents, schema, target, and random seed, allowing experiments to be compared without treating generated model binaries as source code.

After an alternate model is trained, it can be scored through a dataset-specific endpoint using a feature map:

```text
POST /predict/german_credit
POST /predict/give_me_some_credit
```

The endpoint validates that the feature map exactly matches the trained schema and verifies the alternate model checksum before inference. Alternate responses are educational experiment outputs and include a warning that their thresholds and explanations are not yet specialized to that dataset.

Policy experiments can be run without changing the configured automatic policy through `POST /policy/simulate`. Submit a list of probabilities, approval/decline thresholds, and optionally known outcomes; the response reports approval, review, and decline rates plus observed default rates when labels are supplied. This is a research simulator, not a live policy override.

Input drift can be checked with `POST /monitoring/drift`. Submit reference records, current records, and optionally a feature list. The endpoint calculates PSI and missingness deltas, classifies each feature as `ok`, `warning`, or `critical`, and recommends investigation. It never retrains automatically.

The primary model artifact also stores training-distribution statistics for engineered features. At prediction time, inputs exceeding the configured `out_of_distribution_z` threshold are routed to `manual_review` with a warning instead of being silently treated as ordinary applicants.

The policy uses configurable thresholds: low risk is approved, high risk is declined, and the middle band goes to human review. Configuration loading validates threshold ordering, risk-band ordering, missingness limits, and out-of-distribution settings; runtime policy evaluation also rejects non-finite probabilities. Inputs outside the expected distribution should also be reviewed. Removing sensitive fields does not prove that a model is fair, so the training report includes group-level approval and error-rate summaries. The numeric score is a presentation layer based on calibrated probability and configurable odds parameters in `configs/default.yaml`; it is not a real-world bureau score.

The training report also includes detailed fairness comparisons by audit group: approval-rate differences and ratios, equal-opportunity differences, false-positive and false-negative-rate differences, calibration differences, average predicted risk, and average educational score. It also recomputes group metrics at several threshold pairs to show threshold sensitivity; this is diagnostic evidence, not a justification for group-specific policy. The default policy remains uniform across groups.

Project governance and design documentation is available in `docs/`: `model_card.md`, `data_card.md`, `decision_policy.md`, `fairness_report.md`, and `architecture.md`. GitHub Actions runs the test suite on pushes and pull requests. Docker Compose uses `/ready` for its API healthcheck, so the dashboard waits for valid model artifacts rather than merely a running process.

The API also exposes `GET /ready` for artifact readiness separately from `GET /health` process liveness. Readiness verifies that the serialized model, prediction interface, required metadata, and recorded model SHA-256 checksum agree; invalid, corrupt, or tampered artifacts return `503` rather than being treated as ready. Every response includes an `X-Request-ID`; clients may provide their own ID for tracing, and the API logs method, path, status, and duration.

For non-local use, set `CREDIT_API_KEY` in the service environment. Protected endpoints then require the matching `X-API-Key` header. Health, readiness, and API documentation routes remain public for operational checks. Leave the variable unset for the default local-development workflow.

An optional in-memory limiter can be enabled with `CREDIT_RATE_LIMIT_PER_MINUTE=60`. It limits non-public routes per API key (or client address when no key is configured), returns HTTP 429 with `Retry-After`, and leaves health/readiness/docs routes available for operational checks. This is suitable for local demonstrations only; distributed production deployments need a shared, persistent limiter.

Every training run registers its artifact metadata in the local generated `artifacts/model_registry.json`. Use `GET /models` to inspect available dataset/model versions, dataset hashes, artifact fingerprints, and test metrics. Corrupt or non-list registry files are treated as empty rather than being trusted. The registry is local experiment metadata and is intentionally not committed with generated model files.

Prediction events are logged to a local generated SQLite database without storing raw applicant inputs. Use `GET /prediction-stats` for aggregate counts by dataset, model version, and automatic decision. This is intended for educational observability and should be replaced with a governed retention system before any real deployment.

The dashboard is organized into tabs for applicant scoring, threshold simulation, model evidence, the human-review queue, and drift monitoring. Model evidence includes the candidate metrics, fairness report, calibration data, global permutation feature importance, feature distributions, confusion matrix, risk distributions, and threshold comparison table/plot. Threshold simulation and drift monitoring are explicitly labeled as research/diagnostic tools and do not mutate the automatic policy or retrain a model.

## API example

After training, use the Swagger UI or send JSON to `POST /predict`. The request contains `LIMIT_BAL`, six repayment-status fields (`PAY_0`, `PAY_2`–`PAY_6`), six bill fields, and six payment fields. The response contains the modeled probability, credit score, risk band, decision thresholds, rationale, model version, structured explanations, reason codes, and warnings.

For repeatable portfolio/demo scoring, `POST /predict/batch` accepts 1–1,000 applicants using the same schema and returns one validated prediction per applicant. Each prediction uses the same model, feature engineering, policy, explanations, and privacy-conscious event logging as single-applicant scoring.

Borderline cases can be persisted for educational human review with `POST /review-cases`, listed with `GET /review-cases`, inspected with `GET /review-cases/{case_id}`, and updated with `PATCH /review-cases/{case_id}`. SQLite stores the original automatic decision separately from the reviewer decision; reviewer outcomes are not allowed to overwrite the model output.

## Limitations

The source data is historical and geographically limited. It is not representative of all borrowers, contains proxy-sensitive information, and does not establish causal relationships. There is no identity verification, fraud detection, affordability assessment, regulatory compliance layer, or production monitoring in this prototype.

