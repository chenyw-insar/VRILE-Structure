"""Preprocessing functions for sea‑ice extent time series.

This module contains helper functions to interpolate daily extent
records, compute the ΔSIE time series, apply leap day and year range
filters, compute month masks, identify monthly mask boundary days,
and calculate daily and monthly climatologies of ΔSIE and its
percentiles.  It also provides a helper to compute the mean‑removed
ΔSIE series.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from .config import DetectionConfig
from .months import parse_months


def interpolate_daily_extent(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate a sparse extent series to a complete daily series.

    The input DataFrame must have columns ``date`` (Timestamp) and
    ``extent``.  Missing days are filled by time interpolation on the
    extent series.  The returned DataFrame has a continuous daily
    ``date`` index and an ``extent`` column.
    """
    out = df.set_index("date").sort_index()
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full_index)
    out.index.name = "date"
    out["extent"] = out["extent"].interpolate(method="time", limit_direction="both")
    return out.reset_index()


def compute_delta_sie(df: pd.DataFrame, delta_days: int = 3) -> pd.DataFrame:
    """Compute the ΔSIE time series.

    For ``delta_days=3`` this reproduces the definition ΔSIE(n)
    = SIE(n+1) − SIE(n−2) by computing ``extent[t+2] − extent[t−1]`` on
    a zero‑based array.  Boundary values at the start and end of the
    array are filled using nearest‑neighbour extrapolation.
    """
    if delta_days < 1 or delta_days > 3:
        raise ValueError("delta_days must be 1, 2, or 3 to match the original workflow.")
    out = df.copy()
    x = out["extent"].to_numpy(dtype=float)
    dx = np.empty_like(x, dtype=float)
    if delta_days == 3:
        dx[2:-1] = x[3:] - x[:-3]
        dx[:2] = dx[2]
        dx[-1] = dx[-2]
    elif delta_days == 2:
        dx[1:-1] = x[2:] - x[:-2]
        dx[0] = dx[1]
        dx[-1] = dx[-2]
    else:
        dx[:-1] = x[1:] - x[:-1]
        dx[-1] = dx[-2]
    out["delta_sie"] = dx
    out["year"] = out.date.dt.year
    out["month"] = out.date.dt.month
    out["day"] = out.date.dt.day
    return out


def remove_leap_days_and_limit_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Drop leap days and restrict the year range."""
    mask = (df.year >= start_year) & (df.year <= end_year) & ~(
        (df.month == 2) & (df.day == 29)
    )
    return df.loc[mask].reset_index(drop=True)


def is_month_boundary_artifact_date(dates: pd.Series) -> pd.Series:
    """Return True for last two days and first day of each month."""
    dt = pd.to_datetime(dates)
    days_in_month = dt.dt.days_in_month
    return (dt.dt.day == 1) | (dt.dt.day >= days_in_month - 1)


def months_mask(df: pd.DataFrame, months: str | tuple[int, ...] | list[int]) -> pd.Series:
    """Return a boolean mask selecting rows whose month is requested."""
    m = df["month"]
    selected = parse_months(months)
    if set(selected) == set(range(1, 13)):
        return pd.Series(True, index=df.index)
    return m.isin(selected)


def daily_delta_climatology(df: pd.DataFrame, config: DetectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily ΔSIE climatology and monthly percentile thresholds."""
    cl = df[(df.year >= config.climatology_start_year) & (df.year <= config.climatology_end_year)].copy()
    if config.exclude_month_boundary:
        cl = cl.loc[~is_month_boundary_artifact_date(cl["date"])]
    # daily climatology grouped by calendar day
    daily = (
        cl.groupby(["month", "day"], as_index=False)
        .agg(
            extent_climo=("extent", "mean"),
            delta_climo=("delta_sie", "mean"),
            delta_p05=("delta_sie", lambda x: np.nanpercentile(x, 5)),
            delta_p95=("delta_sie", lambda x: np.nanpercentile(x, 95)),
            n=("delta_sie", "count"),
        )
    )
    # monthly percentiles
    monthly = (
        cl.groupby("month", as_index=False)
        .agg(
            delta_month_p05=("delta_sie", lambda x: np.nanpercentile(x, 5)),
            delta_month_p10=("delta_sie", lambda x: np.nanpercentile(x, 10)),
            delta_month_p95=("delta_sie", lambda x: np.nanpercentile(x, 95)),
            delta_month_custom=("delta_sie", lambda x: np.nanpercentile(x, config.percentile)),
            n=("delta_sie", "count"),
        )
    )
    return daily, monthly


def _mean_removed_series(df: pd.DataFrame, daily_climo: pd.DataFrame, delta_days: int) -> pd.Series:
    """Compute the mean‑removed ΔSIE series."""
    key = daily_climo.set_index(["month", "day"])["extent_climo"]
    vals: list[float] = []
    for m, d, e in zip(df.month, df.day, df.extent):
        vals.append(e - key.get((m, d), np.nan))
    anomaly_extent = pd.Series(vals, index=df.index, dtype=float).interpolate(limit_direction="both")
    tmp = pd.DataFrame({"date": df.date, "extent": anomaly_extent})
    return compute_delta_sie(tmp, delta_days=delta_days)["delta_sie"]
