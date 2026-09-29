"""High‑level functions for detecting VRILE dates from sea‑ice extent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import DetectionConfig, VRILEMethod
from .io import read_extent_csv, write_event_table
from .preprocessing import (
    interpolate_daily_extent,
    compute_delta_sie,
    remove_leap_days_and_limit_years,
    months_mask,
    is_month_boundary_artifact_date,
    daily_delta_climatology,
    _mean_removed_series,
)
from .filtering import highpass_butterworth
from .utils import deduplicate_adjacent


def detect_vriles(
    extent_csv: str | Path,
    config: DetectionConfig,
    method: VRILEMethod = "both",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Detect VRILE dates from a sea‑ice extent CSV.

    Parameters
    ----------
    extent_csv : str or Path
        Path to a CSV file containing columns ``year``, ``month``,
        ``day`` and ``extent``.  See :func:`vrile.io.read_extent_csv`.
    config : DetectionConfig
        Parameter bundle controlling the detection.  See
        :class:`vrile.config.DetectionConfig`.
    method : {"mean_removed", "butterworth", "both"}
        Which detection method(s) to apply.  When ``both`` is
        specified the union of the two sets of events is returned.

    Returns
    -------
    events : DataFrame
        Table of detected VRILEs.  Contains at least the columns
        ``date``, ``method``, ``vrile_value``, ``raw_delta_sie``,
        ``threshold``, ``extent``, ``year``, ``month``, ``day``.
    diagnostics : dict
        Dictionary with keys ``daily``, ``daily_climatology`` and
        ``monthly_climatology`` containing intermediate time series
        used in the detection.
    """
    raw = read_extent_csv(extent_csv)
    daily = interpolate_daily_extent(raw)
    daily = compute_delta_sie(daily, config.delta_days)
    daily = remove_leap_days_and_limit_years(daily, config.start_year, config.end_year)
    daily_climo, monthly_climo = daily_delta_climatology(daily, config)

    candidates: list[pd.DataFrame] = []
    base_mask = months_mask(daily, config.months)
    if config.exclude_month_boundary:
        base_mask &= ~is_month_boundary_artifact_date(daily["date"])

    # Butterworth method
    if method in ("butterworth", "both"):
        bw = highpass_butterworth(
            daily["delta_sie"].to_numpy(),
            config.butterworth_cutoff_days,
            config.butterworth_order,
        )
        threshold = np.nanpercentile(bw, config.percentile)
        daily["delta_sie_butterworth"] = bw
        m = base_mask & (daily["delta_sie_butterworth"] <= threshold)
        ev = daily.loc[m, ["date", "extent", "delta_sie", "year", "month", "day"]].copy()
        ev["method"] = "butterworth"
        ev["vrile_value"] = daily.loc[m, "delta_sie_butterworth"].to_numpy()
        ev["raw_delta_sie"] = ev["delta_sie"]
        ev["threshold"] = threshold
        candidates.append(ev)

    # Mean‑removed method
    if method in ("mean_removed", "both"):
        mr = _mean_removed_series(daily, daily_climo, config.delta_days)
        daily["delta_sie_mean_removed"] = mr
        mr_threshold = np.nanpercentile(daily.loc[base_mask, "delta_sie_mean_removed"], config.percentile)
        m = base_mask & (daily["delta_sie_mean_removed"] <= mr_threshold)
        ev = daily.loc[m, ["date", "extent", "delta_sie", "year", "month", "day"]].copy()
        ev["method"] = "mean_removed"
        ev["vrile_value"] = daily.loc[m, "delta_sie_mean_removed"].to_numpy()
        ev["raw_delta_sie"] = ev["delta_sie"]
        ev["threshold"] = mr_threshold
        candidates.append(ev)

    if candidates:
        events = pd.concat(candidates, ignore_index=True)
        events = events.sort_values(["date", "method"]).reset_index(drop=True)
        events = deduplicate_adjacent(events, config.unique_gap_days)
    else:
        events = pd.DataFrame(
            columns=["date", "method", "vrile_value", "raw_delta_sie", "threshold", "extent", "year", "month", "day"]
        )

    return events, {"daily": daily, "daily_climatology": daily_climo, "monthly_climatology": monthly_climo}
