# Decision policy

The model estimates a default probability. The policy translates that probability into a recommendation; it does not change the model probability.

Default thresholds are configured in `configs/default.yaml`:

```text
probability <= 0.20  -> approve
0.20–0.45            -> manual_review
probability >= 0.45  -> decline
```

Inputs flagged as extreme or invalid are routed to review. A human reviewer must not overwrite the original automatic recommendation; the review outcome is stored separately.

`POST /policy/simulate` is for research only. It can compare thresholds and resulting population rates but cannot modify the live policy.

Threshold changes should be evaluated for cost, approval rate, review workload, default rate among approvals, calibration, and group fairness before adoption.
