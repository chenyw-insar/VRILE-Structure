"""Miscellaneous utilities for VRILE detection."""

from __future__ import annotations

import pandas as pd


def deduplicate_adjacent(events: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Merge events that occur within a given number of days.

    This helper treats successive VRILEs that are separated by less
    than or equal to ``gap_days`` as belonging to the same physical
    phenomenon.  The active production rule retains the final sorted row/date
    in each contiguous block; it does not select the most negative value.

    Parameters
    ----------
    events : DataFrame
        Event table sorted by date.
    gap_days : int
        Maximum number of days between events to merge.

    Returns
    -------
    DataFrame
        Deduplicated event table.
    """
    if events.empty or gap_days <= 0:
        return events.reset_index(drop=True)
    keep_rows: list = []
    group: list = [events.iloc[0]]
    for _, row in events.iloc[1:].iterrows():
        prev_date = pd.to_datetime(group[-1]["date"])
        this_date = pd.to_datetime(row["date"])
        if (this_date - prev_date).days <= gap_days:
            group.append(row)
        else:
            # Published "Unique VRILEs" use the final day of an adjacent series.
            keep_rows.append(group[-1])
            group = [row]
    # handle the final group
    keep_rows.append(group[-1])
    return pd.DataFrame(keep_rows).reset_index(drop=True)
