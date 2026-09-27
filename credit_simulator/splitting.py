from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    time_column: str


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> TemporalSplits:
    """Split chronologically so later observations never enter training.

    The source frame is copied and sorted by the declared time column. Ties
    retain their original order through a stable sort. No random shuffling is
    performed; this is intended for application-time or cohort data.
    """
    if time_column not in frame.columns:
        raise ValueError(f"Temporal split column is missing: {time_column}")
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than 1")
    if len(frame) < 3:
        raise ValueError("Temporal split requires at least three rows")
    parsed = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"Temporal split column contains invalid or missing timestamps: {time_column}")
    ordered = frame.assign(_temporal_order=parsed).sort_values("_temporal_order", kind="mergesort").drop(columns="_temporal_order")
    test_count = max(1, int(round(len(ordered) * test_fraction)))
    validation_count = max(1, int(round(len(ordered) * validation_fraction))) if validation_fraction else 0
    if test_count + validation_count >= len(ordered):
        raise ValueError("Temporal split fractions leave no training rows")
    train_end = len(ordered) - validation_count - test_count
    validation_end = len(ordered) - test_count
    return TemporalSplits(ordered.iloc[:train_end].copy(), ordered.iloc[train_end:validation_end].copy(), ordered.iloc[validation_end:].copy(), time_column)
