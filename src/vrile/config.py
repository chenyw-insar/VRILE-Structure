"""Configuration dataclasses for VRILE detection.

The :class:`DetectionConfig` encapsulates the tunable parameters used
throughout the detection workflow.  It intentionally mirrors the
settings described in Cavallo et al. 2025【21†L1-L11】 to make the
implementation self‑documenting.  Default values correspond to those
used in the published analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .months import parse_months

VRILEMethod = Literal["mean_removed", "butterworth", "both"]


@dataclass(frozen=True, init=False)
class DetectionConfig:
    """Parameter container for VRILE detection.

    Attributes
    ----------
    start_year, end_year:
        The inclusive range of calendar years over which to perform
        detection.  All days outside this range are ignored.
    climatology_start_year, climatology_end_year:
        The inclusive range of years used to compute the daily and
        monthly climatological means and percentiles of ΔSIE.
        These must lie inside ``[start_year, end_year]``.
    delta_days:
        Number of days to span when computing ΔSIE.  The original
        algorithm uses 3 days (i.e. ΔSIE(n) = SIE(n+1) − SIE(n−2)).  A
        value of 2 or 1 may be used to experiment with shorter
        differences, but values outside of {1, 2, 3} will raise an
        exception.
    percentile:
        The lower percentile threshold used to define extreme ΔSIE
        events.  For example, a value of 5.0 selects the bottom
        5 percent of ΔSIE values.
    butterworth_cutoff_days, butterworth_order:
        Parameters for the high‑pass Butterworth filter.  See
        :func:`vrile.filtering.highpass_butterworth` for details.
    unique_gap_days:
        When multiple VRILEs occur on consecutive days, they can be
        consolidated into a single event.  Events separated by no more than
        ``unique_gap_days`` form a contiguous block, and the final sorted
        row/date in each block is retained.
    months:
        Month numbers to include in the detection.  Labels such as
        ``jja`` or ``annual`` are accepted by the constructor and stored
        as explicit month numbers.
    exclude_month_boundary:
        If True, skip the last two days and the first day of each
        calendar month.  The NSIDC Sea Ice Index applies a monthly
        ocean/weather mask which introduces artificial jumps on the
        first of each month; these are excluded by default.
    """

    start_year: int = 1989
    end_year: int = 2023
    climatology_start_year: int = 1990
    climatology_end_year: int = 2018
    delta_days: int = 3
    percentile: float = 5.0
    butterworth_cutoff_days: float = 18.0
    butterworth_order: int = 12
    unique_gap_days: int = 1
    months: tuple[int, ...] = tuple(range(1, 13))
    exclude_month_boundary: bool = True

    def __init__(
        self,
        start_year: int = 1989,
        end_year: int = 2023,
        climatology_start_year: int = 1990,
        climatology_end_year: int = 2018,
        delta_days: int = 3,
        percentile: float = 5.0,
        butterworth_cutoff_days: float = 18.0,
        butterworth_order: int = 12,
        unique_gap_days: int = 1,
        months: str | tuple[int, ...] | list[int] | None = "annual",
        exclude_month_boundary: bool = True,
    ) -> None:
        object.__setattr__(self, "start_year", start_year)
        object.__setattr__(self, "end_year", end_year)
        object.__setattr__(self, "climatology_start_year", climatology_start_year)
        object.__setattr__(self, "climatology_end_year", climatology_end_year)
        object.__setattr__(self, "delta_days", delta_days)
        object.__setattr__(self, "percentile", percentile)
        object.__setattr__(self, "butterworth_cutoff_days", butterworth_cutoff_days)
        object.__setattr__(self, "butterworth_order", butterworth_order)
        object.__setattr__(self, "unique_gap_days", unique_gap_days)
        object.__setattr__(self, "months", parse_months(months))
        object.__setattr__(self, "exclude_month_boundary", exclude_month_boundary)
