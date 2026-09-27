from __future__ import annotations

import numpy as np
import pandas as pd


MONTHS = [1, 2, 3, 4, 5, 6]
PAY_COLUMNS = [f"PAY_{month}" if month != 1 else "PAY_0" for month in MONTHS]
BILL_COLUMNS = [f"BILL_AMT{month}" for month in MONTHS]
PAY_AMT_COLUMNS = [f"PAY_AMT{month}" for month in MONTHS]


def _slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1], dtype=float)
    x = x - x.mean()
    denominator = float((x * x).sum())
    centered = values - values.mean(axis=1, keepdims=True)
    return (centered * x).sum(axis=1) / denominator


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, decision-time features to raw credit-card fields.

    Columns are ordered recent-to-oldest in the UCI dataset. All calculations use
    the six historical months available at scoring time and do not use the target.
    """
    result = frame.copy()
    missing = [column for column in ["LIMIT_BAL", *PAY_COLUMNS, *BILL_COLUMNS, *PAY_AMT_COLUMNS] if column not in result]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    bills = result[BILL_COLUMNS].to_numpy(dtype=float)
    payments = result[PAY_AMT_COLUMNS].to_numpy(dtype=float)
    statuses = result[PAY_COLUMNS].to_numpy(dtype=float)
    limit = result["LIMIT_BAL"].to_numpy(dtype=float)
    safe_limit = np.where(limit > 0, limit, np.nan)
    safe_bills = np.where(np.abs(bills) > 1e-9, np.abs(bills), np.nan)
    utilization = np.abs(bills) / safe_limit[:, None]
    payment_ratio = payments / safe_bills
    payment_ratio = np.nan_to_num(payment_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    late = np.maximum(statuses, 0)
    result["current_utilization"] = utilization[:, 0]
    result["average_utilization"] = np.nanmean(utilization, axis=1)
    result["maximum_utilization"] = np.nanmax(utilization, axis=1)
    result["utilization_volatility"] = np.nanstd(utilization, axis=1)
    result["utilization_trend"] = _slope(np.nan_to_num(utilization, nan=0.0))
    result["months_above_80pct_utilization"] = (np.nan_to_num(utilization, nan=0.0) > 0.80).sum(axis=1)
    result["late_payment_count"] = (statuses > 0).sum(axis=1)
    result["maximum_late_payment"] = late.max(axis=1)
    result["most_recent_late_payment"] = late[:, 0]
    result["average_payment_delay"] = late.mean(axis=1)
    result["recent_three_month_delay"] = late[:, :3].mean(axis=1)
    result["payment_delay_trend"] = _slope(late)
    result["consecutive_late_months"] = np.array([next((i for i, value in enumerate(row) if value <= 0), len(row)) for row in late])
    result["average_payment_to_bill_ratio"] = payment_ratio.mean(axis=1)
    result["minimum_payment_to_bill_ratio"] = payment_ratio.min(axis=1)
    result["months_paying_less_than_bill"] = (payments < np.abs(bills)).sum(axis=1)
    result["payment_trend"] = _slope(payments)
    result["payment_volatility"] = payments.std(axis=1)
    result["average_bill_amount"] = np.abs(bills).mean(axis=1)
    result["maximum_bill_amount"] = np.abs(bills).max(axis=1)
    result["bill_amount_trend"] = _slope(np.abs(bills))
    result["balance_volatility"] = np.abs(bills).std(axis=1)
    result["current_balance_relative_to_average"] = np.abs(bills[:, 0]) / np.maximum(np.abs(bills).mean(axis=1), 1.0)
    result["active_month_count"] = (np.abs(bills) > 0).sum(axis=1)
    result["zero_payment_month_count"] = (payments <= 0).sum(axis=1)
    result["zero_balance_month_count"] = (np.abs(bills) <= 0).sum(axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def engineered_feature_descriptions() -> dict[str, str]:
    return {
        "current_utilization": "Most recent bill balance divided by credit limit",
        "average_utilization": "Average absolute utilization over six months",
        "maximum_utilization": "Maximum absolute utilization over six months",
        "utilization_volatility": "Standard deviation of utilization over six months",
        "utilization_trend": "Linear trend in utilization from recent to older records",
        "months_above_80pct_utilization": "Number of months above 80% utilization",
        "late_payment_count": "Number of months with a positive late-payment status",
        "maximum_late_payment": "Worst observed late-payment severity",
        "most_recent_late_payment": "Most recent late-payment severity",
        "average_payment_delay": "Average positive payment-delay severity",
        "recent_three_month_delay": "Average payment-delay severity in the three most recent months",
        "payment_delay_trend": "Linear trend in payment-delay severity",
        "consecutive_late_months": "Recent consecutive months with late-payment status",
        "average_payment_to_bill_ratio": "Average payment amount divided by absolute bill amount",
        "minimum_payment_to_bill_ratio": "Minimum payment-to-bill ratio",
        "months_paying_less_than_bill": "Number of months where payment was below absolute bill amount",
        "payment_trend": "Linear trend in payment amount",
        "payment_volatility": "Standard deviation of payment amount",
        "average_bill_amount": "Average absolute bill amount",
        "maximum_bill_amount": "Maximum absolute bill amount",
        "bill_amount_trend": "Linear trend in absolute bill amount",
        "balance_volatility": "Standard deviation of absolute bill amount",
        "current_balance_relative_to_average": "Most recent balance relative to six-month average",
        "active_month_count": "Number of months with a non-zero bill",
        "zero_payment_month_count": "Number of months with no payment",
        "zero_balance_month_count": "Number of months with zero balance",
    }
