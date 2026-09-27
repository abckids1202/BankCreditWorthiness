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

The target is the UCI `default.payment.next.month` field. `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are retained for audit reporting but excluded from model features. Logistic regression and histogram gradient boosting are compared in calibrated and uncalibrated forms using ROC-AUC, PR-AUC, and Brier score; the better calibrated validation candidate is selected. Linear explanations use signed coefficient contributions, while tree explanations use local feature-ablation effects; neither is a legal adverse-action notice.

Training also creates utilization, repayment-delay, payment-ratio, balance-trend, account-stability, and missingness-indicator features. Missingness flags are computed from raw decision-time fields before imputation, so the model can distinguish an observed zero from an unavailable value. It saves an educational 300–850 score alongside the raw default probability. Detailed artifacts are generated under `outputs/reports/`, including `training_report.json` with reproducible bootstrap 95% intervals for held-out ROC-AUC, PR-AUC, and Brier score, `model_comparison.csv` comparing the majority baseline with uncalibrated and calibrated logistic-regression and gradient-boosting candidates (including training/inference timing), `threshold_analysis.csv`, `approval_rate_analysis.csv`, `global_feature_importance.csv`, calibration, risk-distribution, feature-distribution, confusion-matrix, and threshold-comparison plots, target-balance plots, and feature summaries. `report_manifest.json` lists every generated artifact for the run, and the feature-engineering section records each engineered field's interpretation and decision-time leakage review.

Each primary training run gets a reproducible version such as `0.3.0+8c1b78222395`. The fingerprint is derived from the dataset hash, selected model, and training configuration, so changing any of those inputs creates a distinguishable artifact version. The full training configuration is also stored in `artifacts/metadata.json` and `outputs/reports/training_report.json` for experiment reconstruction.

To run a reproducible experiment with a separate policy/model configuration, copy `configs/default.yaml`, edit the copy, and pass it explicitly: `python scripts/train.py --config configs/my_experiment.yaml --output-dir artifacts/my_experiment`. The resulting fingerprint records the supplied configuration.

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

Those alternate artifacts are experiment outputs and are not wired into the current credit-card `/predict` contract, which intentionally remains schema-safe. Protected fields in alternate adapters are retained for audit metadata but excluded from `feature_columns` before fitting. Alternate training validates the feature schema, binary/non-null target, protected-field separation, and target-leakage guard before fitting. Adapters that declare `metadata["time_column"]` are trained with chronological train/validation/test partitions rather than random shuffling; datasets without an application-time field use stratified random splitting and report that strategy explicitly. The German Credit path has been verified end-to-end; Give Me Some Credit requires its Kaggle CSV first.

Alternate training artifacts also receive reproducible fingerprints based on the dataset contents, schema, target, and random seed, allowing experiments to be compared without treating generated model binaries as source code. Each alternate output directory also receives a `training_report.json` with class balance, missingness/schema quality, numeric/categorical feature summaries, validation and held-out metrics, split strategy, source metadata, and training timestamp. Random-split experiments use development/validation/test partitions; adapters that declare an application-time field use chronological train/validation/test partitions and also report a separate random-split benchmark for comparison. The chronological model remains the serving artifact; the random benchmark is diagnostic only.

After an alternate model is trained, it can be scored through a dataset-specific endpoint using a feature map:

```text
POST /predict/german_credit
POST /predict/give_me_some_credit
```

The endpoint validates that the feature map exactly matches the trained schema and verifies the alternate model checksum and required manifest fields before inference. Alternate responses are educational experiment outputs and include a warning that their thresholds and explanations are not yet specialized to that dataset.

The active business policy can be inspected with typed `GET /policy/current`, which reports its version, decision thresholds, and risk bands independently of model metadata. Policy experiments can be run without changing the configured automatic policy through `POST /policy/simulate`. Submit a list of probabilities, approval/decline thresholds, and optionally known outcomes; the response reports approval, review, and decline rates plus observed default rates, confusion/error rates, and configurable expected cost when labels are supplied. An optional `audit_groups` map can be supplied with the same-length group labels to calculate group fairness metrics; those labels are audit-only and never enter model inference. Costs represent the educational assumptions for approving a default, declining a non-default, and performing manual review; they are not financial forecasts. This is a research simulator, not a live policy override.

Input drift can be checked with `POST /monitoring/drift`. Submit reference records, current records, and optionally a feature list. The endpoint calculates PSI and missingness deltas, classifies each feature as `ok`, `warning`, or `critical` using both signals, reports the severity reasons and thresholds used, counts current values outside the reference range, rejects explicitly requested features missing from either dataset, and recommends investigation. Aggregate diagnostic summaries (never raw records) are retained in local SQLite with the serving model, policy, and experiment versions and can be read with `GET /monitoring/drift/history`. Prediction/output drift can be checked separately with `POST /monitoring/predictions`, which audits probability PSI and approve/review/decline rate changes using aggregate lists only. If observed 0/1 outcomes are available, the same report adds default-rate drift, Brier scores, and calibration-error drift. Fairness summaries can be compared with aggregate-only `POST /monitoring/fairness`; it compares nested group metrics, reports warning/critical deltas, persists an aggregate history, and exposes it through `GET /monitoring/fairness/history`. Warning and critical thresholds can be supplied per diagnostic request and must be ordered. Neither endpoint retrains automatically.

The primary model artifact also stores training-distribution statistics for engineered features. At prediction time, inputs exceeding the configured `out_of_distribution_z` threshold are routed to `manual_review` with a warning instead of being silently treated as ordinary applicants.

The policy uses configurable thresholds: low risk is approved, high risk is declined, and the middle band goes to human review. Configuration loading validates threshold ordering, risk-band ordering, missingness limits, out-of-distribution settings, split sizes, score bounds, and decision costs; runtime policy evaluation also rejects non-finite probabilities. Predictions expose separate `model_version` and `policy_version` values so a policy change can be audited without implying that the model was retrained. Inputs outside the expected distribution should also be reviewed. Removing sensitive fields does not prove that a model is fair, so the training report includes group-level approval and error-rate summaries. The numeric score is a presentation layer based on calibrated probability and configurable odds parameters in `configs/default.yaml`; it is not a real-world bureau score.

The training report also includes detailed fairness comparisons by audit group: approval-rate differences and ratios, equal-opportunity differences, false-positive and false-negative-rate differences, binned calibration-error differences, average predicted risk, and average educational score. It writes `fairness_threshold_sensitivity.png` to show approval-rate disparity across uniform threshold simulations and recomputes group metrics at several threshold pairs; this is diagnostic evidence, not a justification for group-specific policy. The default policy remains uniform across groups.

Project governance and design documentation is available in `docs/`: `model_card.md`, `data_card.md`, `decision_policy.md`, `fairness_report.md`, and `architecture.md`. GitHub Actions downloads/trains the primary educational model, runs the test suite, validates Compose configuration, and builds the Docker image on pushes and pull requests because generated artifacts are intentionally ignored. Docker Compose uses `/ready` for its API healthcheck, so the dashboard waits for valid model artifacts rather than merely a running process.

The API also exposes `GET /ready` for artifact readiness separately from `GET /health` process liveness. Readiness verifies the serialized model, prediction interface, required policy/training metadata, and recorded model SHA-256 checksum; invalid, incomplete, corrupt, or tampered artifacts return `503` rather than being treated as ready. A ready response identifies the serving model, policy version, artifact fingerprint, and training-configuration hash. Every response includes an `X-Request-ID`; clients may provide a bounded alphanumeric tracing ID, while malformed or oversized IDs are replaced with a UUID. The API logs method, path, status, and duration.

For non-local use, set `CREDIT_API_KEY` in the service environment (Compose forwards it from the host environment or `.env` file to both API and dashboard). The dashboard adds the key to its internal API requests without displaying it. Protected endpoints then require the matching `X-API-Key` header, compared in constant time. When rate limiting is enabled, responses include `X-RateLimit-Limit` and `X-RateLimit-Remaining`, while rejected requests include `Retry-After`. Health, readiness, and API documentation routes remain public for operational checks. Leave the variable unset for the default local-development workflow.

An optional in-memory limiter can be enabled with `CREDIT_RATE_LIMIT_PER_MINUTE=60`; Compose forwards this setting alongside the API key. It limits non-public routes per API key (or client address when no key is configured), returns HTTP 429 with `Retry-After`, and leaves health/readiness/docs routes available for operational checks. This is suitable for local demonstrations only; distributed production deployments need a shared, persistent limiter.

Every training run registers its artifact metadata in the local generated `artifacts/model_registry.json`. Each run also creates an immutable snapshot under `artifacts/versions/<fingerprint>/` (or the corresponding alternate-dataset artifact directory). Artifacts record an experiment ID, schema version `1.0`, a training-configuration SHA-256 fingerprint, and runtime library versions; `/ready` rejects artifacts with an incomplete or unknown manifest. `GET /models` exposes the experiment ID and configuration fingerprint alongside dataset/model versions, dataset hashes, artifact fingerprints, test metrics, artifact availability, and checksum validity. Corrupt or non-list registry files are treated as empty rather than being trusted. The registry is local experiment metadata and is intentionally not committed with generated model files; snapshots provide a recovery/rollback foundation but are not promoted automatically.

To explicitly promote a verified snapshot—or roll back by selecting an older version—run `python scripts/promote_model.py --version <model_version>`. This copies only a checksum-valid snapshot into the serving directory and marks registry status; training and inference never promote models automatically.

Prediction events are logged to a local generated SQLite database without storing raw applicant inputs. Use `GET /prediction-stats` for aggregate counts by dataset, dataset version, experiment ID, model version, policy version, and automatic decision. Primary and alternate-dataset inference paths use the same privacy-conscious logging shape, with migration-safe defaults for older databases. This is intended for educational observability and should be replaced with a governed retention system before any real deployment.

The dashboard is organized into eight tabs: applicant scoring, score explanation, threshold simulation, model metrics, feature distributions, fairness analysis, the human-review queue, and monitoring. The threshold simulator loads the active policy version and thresholds from the API before allowing research overrides. Model metrics includes candidate metrics, calibration data, global permutation feature importance, confusion matrix, risk distributions, threshold comparison data, and a read-only model registry view; the dedicated feature and fairness tabs make those reports easier to inspect. Threshold simulation and drift monitoring are explicitly labeled as research/diagnostic tools and do not mutate the automatic policy or retrain a model.

## API example

After training, use the Swagger UI or send JSON to `POST /predict`. The request contains `LIMIT_BAL`, six repayment-status fields (`PAY_0`, `PAY_2`–`PAY_6`), six bill fields, and six payment fields. The response contains the modeled probability as both the legacy `risk_probability` field and the explicit `default_probability` field, plus credit score, risk band, decision thresholds, rationale, experiment ID, model version, dataset version (the training-data SHA-256), structured explanations, reason codes, and warnings. Review cases retain the same experiment ID so a later reviewer can identify the exact training run behind the automatic recommendation.

Copy-paste demo commands using synthetic data:

```powershell
python scripts/train.py
uvicorn credit_simulator.api:app --reload
Invoke-RestMethod http://127.0.0.1:8000/ready
$body = Get-Content examples/sample_applicant.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
.\examples\requests.ps1
```

Training prints a JSON summary containing the dataset, experiment ID, model version, policy version, artifact fingerprint, output directory, and held-out test metrics. `GET /ready` exposes the same experiment ID for the currently serving artifact. The same summary format is used for alternate dataset experiments. CI also runs the dependency-free `python scripts/static_check.py` syntax gate before training and tests.

The same sample payload is available at `examples/sample_applicant.json`; it contains no real personal information.

For repeatable portfolio/demo scoring, `POST /predict/batch` accepts 1–1,000 applicants using the same schema and returns one validated prediction per applicant. Each prediction uses the same model, feature engineering, policy, explanations, and privacy-conscious event logging as single-applicant scoring.

Borderline cases can be persisted for educational human review with `POST /review-cases` only when the automatic recommendation is `manual_review`, then listed with `GET /review-cases`, inspected with `GET /review-cases/{case_id}`, and updated with `PATCH /review-cases/{case_id}`. These endpoints use typed response schemas documented in OpenAPI. Persisted cases retain the exact decision thresholds, dataset version, fairness-warning list, and data-quality-warning list used at scoring time. `GET /review-cases/{case_id}/history` exposes an append-only audit trail of case creation and reviewer updates, including the supplied reviewer identifier. This identifier is for prototype attribution only; it is not authentication. SQLite stores the original automatic decision separately from the reviewer decision; reviewer outcomes are not allowed to overwrite the model output.
Reviewer outcomes are constrained to `approved`, `declined`, `needs_more_information`, or `escalated`, and notes are limited to 5,000 characters.

## Limitations

The source data is historical and geographically limited. It is not representative of all borrowers, contains proxy-sensitive information, and does not establish causal relationships. There is no identity verification, fraud detection, affordability assessment, regulatory compliance layer, or production monitoring in this prototype.

