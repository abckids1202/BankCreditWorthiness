# Architecture

```text
Dataset adapter
      |
      v
Validation -> feature engineering -> preprocessing -> calibrated model
      |                                      |
      v                                      v
Reports / fairness / drift             probability
                                             |
                              score + risk band + policy
                                             |
                         FastAPI -> dashboard / review cases
```

The primary UCI credit-card model has a dedicated request schema. Alternate dataset adapters train separate experiment artifacts and use dataset-specific prediction endpoints. SQLite stores educational review cases. Generated artifacts are intentionally ignored by Git and can be recreated with the training commands.

Container liveness is exposed at `/health`; artifact readiness is exposed at `/ready`. Docker Compose uses `/ready` for its API healthcheck, preventing the dashboard from starting against missing, corrupt, or checksum-mismatched model artifacts.
