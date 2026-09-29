#!/usr/bin/env python
"""Derive simple Arctic cyclone tracks from ERA5 mean sea-level pressure.

This script is designed as a reproducible first-pass cyclone diagnostic for
the VRILE mechanism experiment. It uses the ERA5 ``msl`` field already
downloaded for this toolbox, detects daily local sea-level-pressure minima,
links nearby minima across adjacent days, and writes track tables under:

``data/processed/cyclone_tracks/era5_derived/``

The method is intentionally conservative and transparent. It is not a
replacement for a peer-reviewed cyclone tracking package, but it keeps the
cyclone diagnostics on the same ERA5 grid, time range, and preprocessing used
by the rest of the VRILE experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vrile.regions import Region, parse_region


EARTH_RADIUS_KM = 6371.0


@dataclass
class Candidate:
    date: pd.Timestamp
    lat: float
    lon: float
    center_pressure_hpa: float
    depth_hpa: float
    grid_i: int
    grid_j: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", default="all", help="'all', 'jja', 'summer', or comma-separated month numbers.")
    parser.add_argument("--input-glob", default="data/raw/era5/single_levels/*.nc")
    parser.add_argument('--out_dir', default="data/processed/cyclone_tracks/era5_derived")
    parser.add_argument(
        "--metric-region",
        default="barents_sea",
        help="Official NSIDC region used only for the auxiliary daily nearest-cyclone metrics table.",
    )
    parser.add_argument(
        "--detection-region",
        default="-60,120,55,90",
        help=(
            "Cyclone-center search domain as a region name or lon_min,lon_max,lat_min,lat_max. "
            "Default focuses on the North Atlantic-Barents sector for the phase-2 experiment."
        ),
    )
    parser.add_argument("--variable", default="msl", help="ERA5 mean sea-level pressure variable name.")
    parser.add_argument("--lat-min", type=float, default=50.0, help="Minimum latitude for cyclone-center detection.")
    parser.add_argument("--max-center-pressure-hpa", type=float, default=1015.0)
    parser.add_argument("--min-depth-hpa", type=float, default=1.0, help="Minimum surrounding-minus-center pressure contrast.")
    parser.add_argument("--depth-radius-grid", type=int, default=4, help="Grid offset used for a simple pressure-depth proxy.")
    parser.add_argument("--min-separation-km", type=float, default=400.0, help="Suppress duplicate centers closer than this distance on the same day.")
    parser.add_argument("--max-centers-per-day", type=int, default=40)
    parser.add_argument("--max-link-distance-km", type=float, default=1000.0, help="Maximum day-to-day distance for linking centers into tracks.")
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--metrics-radius-km", type=float, default=1000.0, help="Radius for regional daily cyclone counts.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_months(value: str) -> list[int]:
    key = value.strip().lower()
    if key in {"all", "annual", "year"}:
        return list(range(1, 13))
    if key == "jja":
        return [6, 7, 8]
    if key == "summer":
        return [6, 7, 8, 9]
    months = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    bad = [month for month in months if month < 1 or month > 12]
    if bad:
        raise ValueError(f"Invalid month(s): {bad}")
    return months


def find_time_name(ds: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError("Input ERA5 file has no valid_time/time coordinate.")


def normalize_lon180(lon: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def region_center(region: Region) -> tuple[float, float]:
    lon_min = region.lon_min % 360.0
    lon_max = region.lon_max % 360.0
    if abs(region.lon_max - region.lon_min) >= 360.0:
        lon = 0.0
    elif lon_min <= lon_max:
        lon = (lon_min + lon_max) / 2.0
    else:
        lon = ((lon_min + lon_max + 360.0) / 2.0) % 360.0
    lat = (region.lat_min + region.lat_max) / 2.0
    return float(normalize_lon180(lon)), lat


def shifted_2d(array: np.ndarray, di: int, dj: int, fill_value: float) -> np.ndarray:
    out = np.full(array.shape, fill_value, dtype=float)
    src_i0 = max(0, -di)
    src_i1 = array.shape[0] - max(0, di)
    dst_i0 = max(0, di)
    dst_i1 = array.shape[0] - max(0, -di)
    src_j0 = max(0, -dj)
    src_j1 = array.shape[1] - max(0, dj)
    dst_j0 = max(0, dj)
    dst_j1 = array.shape[1] - max(0, -dj)
    if src_i0 < src_i1 and src_j0 < src_j1:
        out[dst_i0:dst_i1, dst_j0:dst_j1] = array[src_i0:src_i1, src_j0:src_j1]
    return out


def surrounding_mean(field_hpa: np.ndarray, radius: int) -> np.ndarray:
    offsets = [
        (-radius, 0),
        (radius, 0),
        (0, -radius),
        (0, radius),
        (-radius, -radius),
        (-radius, radius),
        (radius, -radius),
        (radius, radius),
    ]
    total = np.zeros_like(field_hpa, dtype=float)
    count = np.zeros_like(field_hpa, dtype=float)
    for di, dj in offsets:
        shifted = shifted_2d(field_hpa, di, dj, np.nan)
        valid = np.isfinite(shifted)
        total += np.where(valid, shifted, 0.0)
        count += valid
    return total / np.where(count == 0, np.nan, count)


def local_minima_mask(field_hpa: np.ndarray) -> np.ndarray:
    mask = np.isfinite(field_hpa)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = shifted_2d(field_hpa, di, dj, np.inf)
            mask &= field_hpa < shifted
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


def suppress_close_candidates(candidates: list[Candidate], min_separation_km: float, max_count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.center_pressure_hpa):
        if len(selected) >= max_count:
            break
        if not selected:
            selected.append(candidate)
            continue
        distances = haversine_km(
            candidate.lon,
            candidate.lat,
            np.array([item.lon for item in selected]),
            np.array([item.lat for item in selected]),
        )
        if np.nanmin(distances) >= min_separation_km:
            selected.append(candidate)
    return selected


def detect_candidates_for_time(
    field_pa: np.ndarray,
    date: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    max_center_pressure_hpa: float,
    min_depth_hpa: float,
    depth_radius_grid: int,
    min_separation_km: float,
    max_centers_per_day: int,
) -> list[Candidate]:
    field_hpa = np.asarray(field_pa, dtype=float) / 100.0
    lat_mask = lats >= lat_min
    minima = local_minima_mask(field_hpa)
    depth = surrounding_mean(field_hpa, depth_radius_grid) - field_hpa
    keep = minima & lat_mask[:, None] & (field_hpa <= max_center_pressure_hpa) & (depth >= min_depth_hpa)
    ii, jj = np.where(keep)
    candidates = [
        Candidate(
            date=date.normalize(),
            lat=float(lats[i]),
            lon=float(normalize_lon180(lons[j])),
            center_pressure_hpa=float(field_hpa[i, j]),
            depth_hpa=float(depth[i, j]),
            grid_i=int(i),
            grid_j=int(j),
        )
        for i, j in zip(ii, jj)
    ]
    return suppress_close_candidates(candidates, min_separation_km, max_centers_per_day)


def discover_input_files(path_glob: str, start_year: int, end_year: int) -> list[Path]:
    import glob
    import re

    files = []
    for raw in sorted(glob.glob(path_glob)):
        path = Path(raw)
        years = [int(match) for match in re.findall(r"(?:19|20)\d{2}", path.name)]
        if years and start_year <= years[-1] <= end_year:
            files.append(path)
    return files


def lon_mask_for_region(lons: np.ndarray, region: Region) -> np.ndarray:
    if abs(region.lon_max - region.lon_min) >= 360.0:
        return np.ones_like(lons, dtype=bool)
    lon360 = np.mod(lons, 360.0)
    lo = region.lon_min % 360.0
    hi = region.lon_max % 360.0
    if lo <= hi:
        return (lon360 >= lo) & (lon360 <= hi)
    return (lon360 >= lo) | (lon360 <= hi)


def subset_detection_domain(ds: xr.Dataset, region: Region) -> xr.Dataset:
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lat_mask = (lats >= region.lat_min) & (lats <= region.lat_max)
    lon_mask = lon_mask_for_region(lons, region)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) < 3 or len(lon_idx) < 3:
        raise ValueError(f"Detection region is too small or outside the ERA5 grid: {region}")
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def detect_all_candidates(files: list[Path], args: argparse.Namespace, months: list[int]) -> pd.DataFrame:
    rows = []
    detection_region = parse_region(args.detection_region)
    for path in files:
        print(f"detect candidates {path}", flush=True)
        with xr.open_dataset(path) as ds:
            time_name = find_time_name(ds)
            if args.variable not in ds:
                raise KeyError(f"{path} has no variable {args.variable!r}")
            ds = ds.sel({time_name: ds[time_name].dt.month.isin(months)})
            ds = subset_detection_domain(ds, detection_region)
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            for t_index, date in enumerate(pd.to_datetime(ds[time_name].values)):
                field = ds[args.variable].isel({time_name: t_index}).values
                candidates = detect_candidates_for_time(
                    field,
                    pd.Timestamp(date),
                    lats,
                    lons,
                    args.lat_min,
                    args.max_center_pressure_hpa,
                    args.min_depth_hpa,
                    args.depth_radius_grid,
                    args.min_separation_km,
                    args.max_centers_per_day,
                )
                for cand in candidates:
                    rows.append(cand.__dict__)
    return pd.DataFrame(rows)


def link_tracks(candidates: pd.DataFrame, max_link_distance_km: float, min_track_length: int) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values(["date", "center_pressure_hpa"]).reset_index(drop=True)
    active: dict[int, dict[str, object]] = {}
    next_track_id = 1
    assigned_rows = []
    for date, group in candidates.groupby("date", sort=True):
        date = pd.Timestamp(date)
        used_tracks: set[int] = set()
        for _, row in group.iterrows():
            best_track = None
            best_distance = np.inf
            for track_id, endpoint in active.items():
                if track_id in used_tracks:
                    continue
                if pd.Timestamp(endpoint["date"]) != date - pd.Timedelta(days=1):
                    continue
                distance = float(haversine_km(row["lon"], row["lat"], endpoint["lon"], endpoint["lat"]))
                if distance < best_distance and distance <= max_link_distance_km:
                    best_distance = distance
                    best_track = track_id
            if best_track is None:
                best_track = next_track_id
                next_track_id += 1
                link_distance = np.nan
            else:
                used_tracks.add(best_track)
                link_distance = best_distance
            record = row.to_dict()
            record["track_id_raw"] = best_track
            record["link_distance_km"] = link_distance
            assigned_rows.append(record)
            active[best_track] = {"date": date, "lat": row["lat"], "lon": row["lon"]}
        stale = [track_id for track_id, endpoint in active.items() if pd.Timestamp(endpoint["date"]) < date]
        for track_id in stale:
            active.pop(track_id, None)

    out = pd.DataFrame(assigned_rows)
    lengths = out.groupby("track_id_raw")["date"].nunique()
    keep_ids = set(lengths[lengths >= min_track_length].index)
    out = out[out["track_id_raw"].isin(keep_ids)].copy()
    if out.empty:
        return out
    id_map = {old: new for new, old in enumerate(sorted(out["track_id_raw"].unique()), start=1)}
    out["track_id"] = out["track_id_raw"].map(id_map).astype(int)
    out = out.sort_values(["track_id", "date"]).reset_index(drop=True)
    out["track_day_index"] = out.groupby("track_id").cumcount() + 1
    out["track_length_days"] = out.groupby("track_id")["date"].transform("nunique")
    cols = [
        "track_id",
        "date",
        "track_day_index",
        "track_length_days",
        "lat",
        "lon",
        "center_pressure_hpa",
        "depth_hpa",
        "link_distance_km",
        "grid_i",
        "grid_j",
    ]
    return out[cols]


def summarize_tracks(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    grouped = points.groupby("track_id")
    summary = grouped.agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        track_length_days=("date", "nunique"),
        min_center_pressure_hpa=("center_pressure_hpa", "min"),
        max_depth_hpa=("depth_hpa", "max"),
        mean_lat=("lat", "mean"),
        mean_lon=("lon", "mean"),
    ).reset_index()
    return summary


def daily_metrics(points: pd.DataFrame, start_year: int, end_year: int, months: list[int], region: Region, radius_km: float) -> pd.DataFrame:
    region_lon, region_lat = region_center(region)
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    dates = dates[dates.month.isin(months)]
    rows = []
    by_date = {pd.Timestamp(date): group for date, group in points.groupby("date")} if not points.empty else {}
    for date in dates:
        group = by_date.get(pd.Timestamp(date.normalize()))
        row = {
            "date": date,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "metric_region": region.name,
            "metric_region_center_lon": region_lon,
            "metric_region_center_lat": region_lat,
            "cyclone_count": 0,
            f"cyclone_count_within_{int(radius_km)}km": 0,
            f"cyclone_within_{int(radius_km)}km": 0,
            "nearest_cyclone_distance_km": np.nan,
            "nearest_cyclone_pressure_hpa": np.nan,
            "nearest_cyclone_depth_hpa": np.nan,
            "nearest_cyclone_lat": np.nan,
            "nearest_cyclone_lon": np.nan,
            "nearest_cyclone_track_id": np.nan,
        }
        if group is not None and not group.empty:
            distances = haversine_km(region_lon, region_lat, group["lon"].to_numpy(float), group["lat"].to_numpy(float))
            nearest_idx = int(np.nanargmin(distances))
            nearest = group.iloc[nearest_idx]
            row["cyclone_count"] = int(len(group))
            row[f"cyclone_count_within_{int(radius_km)}km"] = int(np.sum(distances <= radius_km))
            row[f"cyclone_within_{int(radius_km)}km"] = int(np.any(distances <= radius_km))
            row["nearest_cyclone_distance_km"] = float(distances[nearest_idx])
            row["nearest_cyclone_pressure_hpa"] = float(nearest["center_pressure_hpa"])
            row["nearest_cyclone_depth_hpa"] = float(nearest["depth_hpa"])
            row["nearest_cyclone_lat"] = float(nearest["lat"])
            row["nearest_cyclone_lon"] = float(nearest["lon"])
            row["nearest_cyclone_track_id"] = int(nearest["track_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(candidates: pd.DataFrame, points: pd.DataFrame, args: argparse.Namespace, months: list[int], files: list[Path]) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_region = parse_region(args.metric_region)
    tracks_summary = summarize_tracks(points)
    metrics = daily_metrics(points, args.start_year, args.end_year, months, metric_region, args.metrics_radius_km)

    candidates_path = out_dir / "cyclone_candidates.csv"
    points_path = out_dir / "cyclone_track_points.csv"
    summary_path = out_dir / "cyclone_tracks_summary.csv"
    metrics_path = out_dir / f"cyclone_daily_metrics_{args.metric_region}.csv"
    metadata_path = out_dir / "run_metadata.json"

    for path in [candidates_path, points_path, summary_path, metrics_path, metadata_path]:
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; use --overwrite to replace: {path}")

    candidates.to_csv(candidates_path, index=False)
    points.to_csv(points_path, index=False)
    tracks_summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metadata = {
        "method": "ERA5 MSL local-minimum detection plus adjacent-day nearest-neighbor linking",
        "input_glob": args.input_glob,
        "input_files": [str(path) for path in files],
        "detection_region": args.detection_region,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "months": months,
        "variable": args.variable,
        "lat_min": args.lat_min,
        "max_center_pressure_hpa": args.max_center_pressure_hpa,
        "min_depth_hpa": args.min_depth_hpa,
        "depth_radius_grid": args.depth_radius_grid,
        "min_separation_km": args.min_separation_km,
        "max_centers_per_day": args.max_centers_per_day,
        "max_link_distance_km": args.max_link_distance_km,
        "min_track_length": args.min_track_length,
        "metric_region": args.metric_region,
        "metric_region_note": "Auxiliary daily metrics reference region only; downstream event diagnostics use cyclone_candidates.csv.",
        "metrics_radius_km": args.metrics_radius_km,
        "outputs": {
            "candidates": str(candidates_path),
            "track_points": str(points_path),
            "track_summary": str(summary_path),
            "daily_metrics": str(metrics_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"WROTE {points_path}")
    print(f"WROTE {summary_path}")
    print(f"WROTE {metrics_path}")


def main() -> int:
    args = build_parser().parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    months = parse_months(args.months)
    files = discover_input_files(args.input_glob, args.start_year, args.end_year)
    if not files:
        raise SystemExit(f"No ERA5 files found for {args.start_year}-{args.end_year}: {args.input_glob}")
    print(f"ERA5 input files: {len(files)}", flush=True)
    if args.dry_run:
        out_dir = Path(args.out_dir)
        print(f"would read years={args.start_year}-{args.end_year} months={months}")
        print(f"would detect cyclone centers from variable={args.variable}")
        print(f"detection_region={args.detection_region}")
        print(f"would write {out_dir / 'cyclone_candidates.csv'}")
        print(f"would write {out_dir / 'cyclone_track_points.csv'}")
        print(f"would write {out_dir / 'cyclone_tracks_summary.csv'}")
        print(f"would write {out_dir / f'cyclone_daily_metrics_{args.metric_region}.csv'}")
        print(f"would write {out_dir / 'run_metadata.json'}")
        return 0
    candidates = detect_all_candidates(files, args, months)
    print(f"candidate centers: {len(candidates)}", flush=True)
    points = link_tracks(candidates, args.max_link_distance_km, args.min_track_length)
    print(f"track points after filtering: {len(points)}", flush=True)
    write_outputs(candidates, points, args, months, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
