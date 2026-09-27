# Fairness report guide

Training retains selected audit attributes outside the model feature matrix. The report compares groups on default rate, approval/review/decline rate, true-positive rate, true-negative rate, false-positive rate, false-negative rate, precision, recall, calibration error, average predicted risk, and average educational score.

Comparisons use a deterministic reference group and include approval-rate difference/ratio, equal-opportunity difference, false-positive-rate difference, false-negative-rate difference, and calibration-error difference.

These are diagnostic statistics, not proof of fairness. Group disparities may reflect data quality, historical inequity, proxy variables, label construction, or sampling differences. Do not introduce group-specific thresholds by default.
