"""Signal‑processing helpers for VRILE analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def highpass_butterworth(values: np.ndarray, cutoff_days: float = 18.0, order: int = 12) -> np.ndarray:
    """Apply a Butterworth high‑pass filter to a daily time series.

    Parameters
    ----------
    values : array‑like
        One‑dimensional array of daily values.  Missing values are
        linearly interpolated before filtering.
    cutoff_days : float
        Cutoff period in days.  Frequencies corresponding to
        variations longer than this are suppressed.
    order : int
        Filter order.  Higher orders yield a steeper transition but
        require a longer padding sequence.

    Returns
    -------
    ndarray
        Filtered series of the same length as ``values``.

    Raises
    ------
    ValueError
        If the input contains too few finite values or is too short
        relative to the filter order.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        good = np.isfinite(values)
        if good.sum() < order * 3:
            raise ValueError("Too few finite values for Butterworth filtering.")
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    nyquist = 0.5  # cycles/day for daily sampling
    cutoff = (1.0 / cutoff_days) / nyquist
    b, a = butter(order, cutoff, btype="highpass")
    padlen = 3 * max(len(a), len(b))
    if len(values) <= padlen:
        raise ValueError(f"Time series too short for filtfilt: len={len(values)}, required>{padlen}.")
    return filtfilt(b, a, values)