"""VRILE analysis package.

This package exposes high‑level functions for detecting very rapid
ice‑loss events (VRILEs) from sea‑ice extent records and for
locating their associated sea‑ice concentration loss objects.  It is
organised into several modules:

* :mod:`vrile.config` – dataclasses for configuration and defaults.
* :mod:`vrile.io` – input/output helpers for CSV and NetCDF data.
* :mod:`vrile.regions` – definitions of Arctic sub‑regions.
* :mod:`vrile.preprocessing` – helper functions for preparing SIE time
  series and climatologies.
* :mod:`vrile.filtering` – signal‑processing routines, including the
  Butterworth high‑pass filter.
* :mod:`vrile.detection` – high‑level driver for detecting VRILE
  dates.
* :mod:`vrile.plotting` – simple Matplotlib plotting helpers.
* :mod:`vrile.utils` – miscellaneous helper functions.

The top‑level namespace re‑exports the most commonly used classes
and functions for convenience.
"""

from .config import DetectionConfig
from .detection import detect_vriles
from .io import read_extent_csv, write_event_table, open_sic_dataset, locate_event_object, locate_events_from_table
from .regions import Region, parse_region, list_regions
from .plotting import plot_daily_delta, plot_event_counts

__all__ = [
    "DetectionConfig",
    "detect_vriles",
    "read_extent_csv",
    "write_event_table",
    "open_sic_dataset",
    "locate_event_object",
    "locate_events_from_table",
    "Region",
    "parse_region",
    "list_regions",
    "plot_daily_delta",
    "plot_event_counts",
]