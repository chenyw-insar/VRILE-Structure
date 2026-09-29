"""Input/output utilities for VRILE analysis.

This module provides helper functions to read the NSIDC Sea Ice Index
CSV, write event tables, and open NetCDF files containing sea‑ice
concentration (SIC).  It also implements the object‑finding logic
used to locate the largest connected region of SIC loss around each
detected VRILE date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union
import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from pyproj import CRS, Transformer

from .regions import NSIDC_0780_REGION_IDS, Region, parse_region


def read_extent_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a NSIDC sea‑ice extent CSV file.

    The official NSIDC file for the Northern Hemisphere is called
    ``N_seaice_extent_daily_v3.0.csv`` and contains two header lines
    followed by columns ``year, month, day, extent`` (with extent in
    million square kilometres).  This function tolerates both the
    two‑header‑line format and a normal one‑line header; it also
    accepts arbitrary capitalisation and ordering of the four required
    columns.

    Returns a DataFrame with columns ``date`` (pandas Timestamp) and
    ``extent`` (float).  Rows with missing extent values are dropped.
    """
    path = Path(path)
    try:
        # Attempt to read the two‑line header format used by NSIDC
        raw = pd.read_csv(path, skiprows=2, header=None)
        if raw.shape[1] < 4:
            raise ValueError
        df = raw.iloc[:, :4].copy()
        df.columns = ["year", "month", "day", "extent"]
    except Exception:
        df = pd.read_csv(path)
        lower = {c.lower().strip(): c for c in df.columns}
        # If a 'date' column exists we parse it directly
        if "date" in lower and "extent" in lower:
            dcol = lower["date"]
            ecol = lower["extent"]
            df2 = df[[dcol, ecol]].copy()
            df2.columns = ["date", "extent"]
            df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
            df2["extent"] = pd.to_numeric(df2["extent"], errors="coerce")
            df2 = df2.dropna(subset=["date", "extent"])
            return df2.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        # Otherwise look for the traditional year/month/day/extent columns
        required = ["year", "month", "day", "extent"]
        missing = [r for r in required if r not in lower]
        if missing:
            raise ValueError(f"Could not identify columns {missing} in {path}")
        df = df[[lower["year"], lower["month"], lower["day"], lower["extent"]]].copy()
        df.columns = required
    df["date"] = pd.to_datetime(
        dict(year=df.year.astype(int), month=df.month.astype(int), day=df.day.astype(int)), errors="coerce"
    )
    df = df[["date", "extent"]].sort_values("date").drop_duplicates("date")
    df["extent"] = pd.to_numeric(df["extent"], errors="coerce")
    return df.dropna(subset=["date", "extent"]).reset_index(drop=True)


def write_event_table(events: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a VRILE event table to CSV.

    The input DataFrame must contain at least the columns ``date`` and
    ``method``.  The date column is converted to ISO format strings.
    Missing directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = events.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _guess_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise ValueError(f"Could not find any coordinate/variable among: {', '.join(candidates)}")


def _guess_sic_var(ds: xr.Dataset, user_var: str | None = None) -> str:
    if user_var:
        if user_var not in ds:
            raise ValueError(f"SIC variable '{user_var}' not found. Available variables: {list(ds.data_vars)}")
        return user_var
    preferred = [
        "cdr_seaice_conc",
        "seaice_conc_cdr",
        "sic",
        "ice_conc",
        "concentration",
        "sea_ice_concentration",
    ]
    for v in preferred:
        if v in ds:
            return v
    # Fallback: first data variable with a time dimension
    for v in ds.data_vars:
        if "time" in ds[v].dims:
            return v
    raise ValueError("Could not infer SIC variable. Pass --sic-var explicitly.")


def projected_grid_geometry(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, Transformer, CRS]:
    """Return native x/y meshes and the inverse transform from CF metadata."""
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Projected grid requires x/y coordinates")
    if "crs" not in ds:
        raise ValueError("Projected grid requires a CF crs variable")

    crs_var = ds["crs"]
    attrs = crs_var.attrs
    crs_text = attrs.get("spatial_ref") or attrs.get("crs_wkt") or attrs.get("proj4text")
    if crs_text:
        src_crs = CRS.from_user_input(crs_text)
    else:
        src_crs = CRS.from_cf(attrs)
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    x2, y2 = np.meshgrid(ds["x"].values, ds["y"].values)
    return np.asarray(x2, dtype=np.float64), np.asarray(y2, dtype=np.float64), transformer, src_crs


def _add_projected_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Add 2-D lat/lon coordinates for NSIDC polar stereographic grids."""
    if "lat" in ds.coords and "lon" in ds.coords:
        return ds
    if "lat" in ds.variables and "lon" in ds.variables:
        return ds
    if "x" not in ds.coords or "y" not in ds.coords or "crs" not in ds:
        return ds
    x2, y2, transformer, _ = projected_grid_geometry(ds)
    lon2, lat2 = transformer.transform(x2, y2)
    return ds.assign_coords(
        lon=(("y", "x"), np.asarray(lon2, dtype=np.float64)),
        lat=(("y", "x"), np.asarray(lat2, dtype=np.float64)),
    )


def open_sic_dataset(paths: Union[str, Path, List[Union[str, Path]]], sic_var: str | None = None) -> Tuple[xr.Dataset, str, str, str]:
    """Open one or more gridded SIC NetCDF files and guess key names.

    ``paths`` may be a glob pattern, a directory, a single file, or a list
    of files.  The returned tuple is ``(dataset, variable_name,
    latitude_name, longitude_name)``.  If a coordinate name is not one
    of ``time``, ``lat`` or ``lon``, it is renamed to the canonical name.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if any(ch in str(paths) for ch in "*?[]"):
            files = sorted(glob.glob(str(paths)))
        elif p.is_dir():
            files = sorted(str(x) for x in p.glob("*.nc"))
        else:
            files = [str(p)]
    else:
        files = [str(Path(x)) for x in paths]
    if not files:
        raise FileNotFoundError(f"No SIC NetCDF files found for {paths}")
    if len(files) > 1:
        datasets = [xr.open_dataset(f) for f in files]
        ds = xr.concat(datasets, dim="time").sortby("time")
    else:
        ds = xr.open_dataset(files[0])
    ds = _add_projected_lat_lon(ds)
    var = _guess_sic_var(ds, sic_var)
    time_name = _guess_coord_name(ds, ["time", "tdim"])
    lat_name = _guess_coord_name(ds, ["lat", "latitude", "yc", "ygrid", "nav_lat"])
    lon_name = _guess_coord_name(ds, ["lon", "longitude", "xc", "xgrid", "nav_lon"])
    # rename to canonical names where necessary
    rename_map = {}
    if time_name != "time":
        rename_map[time_name] = "time"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds, var, "lat", "lon"


def _select_by_date(da: xr.DataArray, date: pd.Timestamp) -> xr.DataArray:
    """Select the nearest time slice for the given date."""
    try:
        return da.sel(time=date, method="nearest")
    except Exception as exc:
        raise ValueError(f"Could not select SIC date {date.date()} from dataset time coordinate.") from exc


def _point_region_memberships(lon: float, lat: float, regions: Iterable[str]) -> tuple[str, str]:
    memberships = [name for name in regions if parse_region(name).contains(np.asarray([lon]), np.asarray([lat]))[0]]
    return (memberships[0], ";".join(memberships)) if memberships else ("", "")


def _object_region_assignment(obj: np.ndarray, weights: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> dict[str, float | str]:
    official_regions = list(NSIDC_0780_REGION_IDS)
    total_cells = int(np.count_nonzero(obj))
    total_weight = float(np.nansum(np.where(obj, weights, 0.0)))
    memberships: list[str] = []
    stats: list[tuple[str, float, int]] = []
    for name in official_regions:
        region_mask = parse_region(name).contains(lon2, lat2)
        overlap = obj & region_mask
        cells = int(np.count_nonzero(overlap))
        if cells <= 0:
            continue
        loss_weight = float(np.nansum(np.where(overlap, weights, 0.0)))
        memberships.append(name)
        stats.append((name, loss_weight, cells))
    if not stats:
        return {
            "object_primary_region": "",
            "object_region_memberships": "",
            "object_region_assignment_method": "unassigned_no_region_overlap",
            "object_region_weight_fraction": np.nan,
            "object_region_cell_fraction": np.nan,
        }
    order_index = {name: idx for idx, name in enumerate(official_regions)}
    primary, primary_weight, primary_cells = max(
        stats,
        key=lambda item: (item[1], item[2], -order_index[item[0]]),
    )
    return {
        "object_primary_region": primary,
        "object_region_memberships": ";".join(memberships),
        "object_region_assignment_method": "object_loss_weighted_mask",
        "object_region_weight_fraction": primary_weight / total_weight if total_weight > 0 else np.nan,
        "object_region_cell_fraction": primary_cells / total_cells if total_cells > 0 else np.nan,
    }


def _max_loss_location(obj: np.ndarray, arr: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> tuple[float, float]:
    values = np.where(obj, arr, np.nan)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    iy, ix = np.unravel_index(int(np.nanargmin(values)), values.shape)
    return float(lon2[iy, ix]), float(lat2[iy, ix])


def locate_event_object(
    sic_paths: Union[str, Path, List[Union[str, Path]]],
    event_date: Union[str, pd.Timestamp],
    region: Union[str, Region] = "pan_arctic",
    sic_var: str | None = None,
    window_days: int = 5,
    loss_threshold: float = -0.1,
    min_object_cells: int = 4,
) -> dict:
    """Locate the largest connected SIC‑loss object for a single VRILE date.

    Parameters
    ----------
    sic_paths : str or Path or list
        NetCDF file(s), directory, or glob pattern for daily SIC.  See
        :func:`open_sic_dataset` for details.
    event_date : str or pandas.Timestamp
        End date of the SIC‑change window.  A `window_days`‑day
        difference is computed ending on this date.
    region : str or Region
        Either the name of a built‑in region (e.g. ``barents_sea``) or a
        comma‑separated bounding box ``lon_min,lon_max,lat_min,lat_max``.
    sic_var : str, optional
        Name of the SIC variable to use.  If omitted, a suitable
        variable is guessed.
    window_days : int
        Number of days in the difference window.  The published
        workflow uses a five‑day window to capture the change leading
        up to the VRILE date.
    loss_threshold : float
        Only consider SIC differences less than this threshold as
        “loss”.  Use 0 for all negative values, or a negative value to
        require a minimum magnitude of loss (e.g. −0.15).

    Returns
    -------
    dict
        A dictionary describing the located object.  If no object is
        found, the ``found`` key will be False and an explanatory
        message will be included.
    """
    reg = parse_region(region) if isinstance(region, str) else region
    event_date = pd.Timestamp(event_date)
    start_date = event_date - pd.Timedelta(days=window_days)
    ds, var, lat_name, lon_name = open_sic_dataset(sic_paths, sic_var=sic_var)
    da = ds[var]
    end = _select_by_date(da, event_date).squeeze(drop=True)
    start = _select_by_date(da, start_date).squeeze(drop=True)
    diff = end - start
    lat = ds[lat_name]
    lon = ds[lon_name]
    # broadcast 1‑D coordinates to a grid if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon2, lat2 = np.meshgrid(lon.values, lat.values)
    else:
        lat2 = lat.values
        lon2 = lon.values
    region_mask = reg.contains(lon2, lat2)
    arr = np.asarray(diff.values, dtype=float)
    arr = np.squeeze(arr)
    valid_loss = np.isfinite(arr) & region_mask & (arr < loss_threshold)
    valid_loss = ndimage.binary_closing(valid_loss, structure=np.ones((3, 3), dtype=bool), iterations=1)
    valid_loss = valid_loss & np.isfinite(arr) & region_mask
    labels, nlab = ndimage.label(valid_loss)
    if nlab == 0:
        return {
            "date": event_date.strftime("%Y-%m-%d"),
            "region": reg.name,
            "found": False,
            "message": "No connected SIC‑loss object found under the requested threshold/region.",
        }
    sizes = ndimage.sum(np.ones_like(arr, dtype=float), labels, index=np.arange(1, nlab + 1))
    valid_labels = np.where(sizes > min_object_cells)[0] + 1
    if len(valid_labels) == 0:
        return {
            "date": event_date.strftime("%Y-%m-%d"),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "region": reg.name,
            "found": False,
            "message": "No connected SIC-loss object exceeded the minimum size threshold.",
        }
    largest_label = int(valid_labels[np.argmax(sizes[valid_labels - 1])])
    obj = labels == largest_label
    weights = np.where(obj, -np.minimum(arr, 0.0), 0.0)
    wsum = np.nansum(weights)
    if wsum <= 0:
        center_lat = float(np.nanmean(lat2[obj]))
        center_lon = float(np.nanmean(lon2[obj]))
    else:
        center_lat = float(np.nansum(lat2 * weights) / wsum)
        center_lon = float(np.nansum(lon2 * weights) / wsum)
    center_region, center_memberships = _point_region_memberships(center_lon, center_lat, NSIDC_0780_REGION_IDS)
    max_loss_lon, max_loss_lat = _max_loss_location(obj, arr, lon2, lat2)
    object_region = _object_region_assignment(obj, weights, lon2, lat2)
    min_change = float(np.nanmin(np.where(obj, arr, np.nan)))
    mean_change = float(np.nanmean(np.where(obj, arr, np.nan)))
    return {
        "date": event_date.strftime("%Y-%m-%d"),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "region": reg.name,
        "found": True,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_region": center_region,
        "center_region_memberships": center_memberships,
        "max_loss_lat": max_loss_lat,
        "max_loss_lon": max_loss_lon,
        **object_region,
        "n_grid_cells": int(sizes[largest_label - 1]),
        "min_sic_change": min_change,
        "mean_sic_change": mean_change,
        "sic_variable": var,
        "window_days": window_days,
        "loss_threshold": loss_threshold,
        "min_object_cells": min_object_cells,
    }


def locate_events_from_table(
    events_csv: Union[str, Path],
    sic_paths: Union[str, Path, List[Union[str, Path]]],
    output_csv: Union[str, Path],
    region: str = "pan_arctic",
    sic_var: str | None = None,
    window_days: int = 5,
    loss_threshold: float = -0.1,
    min_object_cells: int = 4,
) -> pd.DataFrame:
    """Locate SIC‑loss objects for all events in a table.

    Reads ``events_csv`` (as produced by :func:`vrile.detection.detect_vriles`),
    loops over the dates and applies :func:`locate_event_object`.
    Writes the resulting table to ``output_csv`` and returns it as a
    DataFrame.
    """
    events = pd.read_csv(events_csv, parse_dates=["date"])
    rows = []
    for d in events["date"]:
        rows.append(
            locate_event_object(
                sic_paths,
                d,
                region=region,
                sic_var=sic_var,
                window_days=window_days,
                loss_threshold=loss_threshold,
                min_object_cells=min_object_cells,
            )
        )
    out = pd.DataFrame(rows)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
