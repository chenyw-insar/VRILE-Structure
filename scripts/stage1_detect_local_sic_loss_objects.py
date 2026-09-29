#!/usr/bin/env python
"""Detect local rapid SIC-loss objects independently of pan-Arctic VRILE dates.

The script scans daily NSIDC SIC fields, computes end-minus-start SIC change
for one or more windows, labels connected regions where SIC decreases beyond a
physical threshold, and summarizes each object.  It is intentionally separate
from the pan-Arctic SIE-based VRILE event list: pan-Arctic and regional overlaps
are added only as diagnostic flags after the local objects are detected.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vrile.io import _add_projected_lat_lon, projected_grid_geometry
from vrile.local_objects import detect_connected_components
from vrile.months import months_label, parse_months
from vrile.regions import parse_region, region_display_label
from common_utils_parallel import run_with_fallback


DEFAULT_REGIONS = [
    "central_arctic",
    "beaufort_sea",
    "chukchi_sea",
    "east_siberian_sea",
    "laptev_sea",
    "kara_sea",
    "barents_sea",
    "east_greenland_sea",
    "baffin_and_labrador_seas",
    "gulf_of_st_lawrence",
    "hudson_bay",
    "canadian_archipelago",
    "bering_sea",
    "sea_of_okhotsk",
    "sea_of_japan",
    "bohai_and_yellow_seas",
    "baltic_sea",
    "gulf_of_alaska",
    "pan_arctic",
]

REGION_LABELS = {
    "central_arctic": region_display_label("central_arctic"),
    "beaufort_sea": "Beaufort",
    "chukchi_sea": "Chukchi",
    "east_siberian_sea": "East Siberian",
    "laptev_sea": "Laptev",
    "kara_sea": "Kara",
    "barents_sea": "Barents",
    "east_greenland_sea": "East Greenland",
    "baffin_and_labrador_seas": "Baffin/Labrador",
    "gulf_of_st_lawrence": "Gulf of St. Lawrence",
    "hudson_bay": "Hudson Bay",
    "canadian_archipelago": "Canadian Arch.",
    "bering_sea": "Bering",
    "sea_of_okhotsk": "Okhotsk",
    "sea_of_japan": "Japan",
    "bohai_and_yellow_seas": "Bohai/Yellow",
    "baltic_sea": "Baltic",
    "gulf_of_alaska": "Gulf of Alaska",
    "pan_arctic": "Pan-Arctic",
    "barents_extended": "Barents ext.",
    "greenland_nordic_seas": "Greenland/Nordic",
    "central_basin": region_display_label("central_basin"),
}

PRIMARY_REGION_ORDER = [
    "central_arctic",
    "beaufort_sea",
    "chukchi_sea",
    "east_siberian_sea",
    "laptev_sea",
    "kara_sea",
    "barents_sea",
    "east_greenland_sea",
    "baffin_and_labrador_seas",
    "gulf_of_st_lawrence",
    "hudson_bay",
    "canadian_archipelago",
    "bering_sea",
    "sea_of_okhotsk",
    "sea_of_japan",
    "bohai_and_yellow_seas",
    "baltic_sea",
    "gulf_of_alaska",
    "pan_arctic",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sic_dir', default="data/raw/nsidc_sic", dest='sic_root')
    parser.add_argument("--panarctic-events", default="outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv")
    parser.add_argument(
        "--regional-events",
        default="outputs/regional_vrile_enhanced/regional_vrile_events_unique.csv",
        help="Optional regional-response diagnostic used only to label overlap; it never limits scanned dates or detected objects.",
    )
    parser.add_argument('--out_dir', default="outputs/local_vrile")
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", default="jja", help="'jja', 'all', or comma-separated month numbers.")
    parser.add_argument("--window-days", type=int, nargs="+", default=[5])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-0.10])
    parser.add_argument(
        "--min-cells",
        type=int,
        default=4,
        help="Minimum connected SIC-loss grid cells required to keep an object; 4 cells on the 25 km grid is about 2500 km2.",
    )
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--quick", action="store_true", help="Run only 2019-2021 and first threshold/window.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    parser.add_argument("--disable-parallel", action="store_true")
    return parser


def canonical_regions(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(parse_region(item).name)
    return out


def find_sic_file(root: Path, date: pd.Timestamp) -> Path:
    matches = sorted(glob.glob(str(root / f"{date.year}" / f"sic_psn25_{date:%Y%m%d}_*.nc")))
    if not matches:
        raise FileNotFoundError(f"No SIC file found for {date:%Y-%m-%d}")
    return Path(matches[0])


def guess_sic_var(ds: xr.Dataset) -> str:
    for name in ("cdr_seaice_conc", "seaice_conc_cdr", "sic", "ice_conc"):
        if name in ds:
            return name
    for name in ds.data_vars:
        if "time" in ds[name].dims:
            return name
    raise ValueError(f"Cannot infer SIC variable from {list(ds.data_vars)}")


@lru_cache(maxsize=64)
def read_sic(path_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    with xr.open_dataset(path_text) as raw:
        ds = _add_projected_lat_lon(raw)
        native_x2, native_y2, inverse_transformer, _ = projected_grid_geometry(raw)
        var = guess_sic_var(ds)
        sic = np.asarray(ds[var].isel(time=0).values, dtype=float).squeeze()
        sic = np.where((sic >= 0.0) & (sic <= 1.0), sic, np.nan)
        return (
            sic,
            np.asarray(ds["lon"].values),
            np.asarray(ds["lat"].values),
            native_x2,
            native_y2,
            inverse_transformer,
        )


def load_event_dates(path: Path) -> set[pd.Timestamp]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    date_col = "event_date" if "event_date" in df.columns else "date"
    return set(pd.to_datetime(df[date_col]).dt.normalize())


def load_regional_event_lookup(path: Path) -> set[tuple[str, pd.Timestamp]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, parse_dates=["event_date"])
    out: set[tuple[str, pd.Timestamp]] = set()
    if {"region", "event_date"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            out.add((str(row.region), pd.Timestamp(row.event_date).normalize()))
    return out


def assign_regions(lon: float, lat: float, regions: list[str]) -> tuple[str, str]:
    memberships = [name for name in regions if parse_region(name).contains(np.asarray([lon]), np.asarray([lat]))[0]]
    primary = ""
    for name in PRIMARY_REGION_ORDER:
        if name in memberships:
            primary = name
            break
    if not primary and memberships:
        primary = memberships[0]
    return primary, ";".join(memberships)


def _ordered_regions(regions: list[str], *, include_pan_arctic: bool = False) -> list[str]:
    region_set = set(regions)
    ordered = [name for name in PRIMARY_REGION_ORDER if name in region_set]
    ordered.extend(name for name in regions if name not in ordered)
    if not include_pan_arctic:
        ordered = [name for name in ordered if name != "pan_arctic"]
    return ordered


def assign_object_region(
    obj: np.ndarray,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
    regions: list[str],
) -> dict[str, float | str]:
    """Assign a SIC-loss object by its own grid cells instead of its centroid."""
    candidates = _ordered_regions(regions, include_pan_arctic=False)
    total_cells = int(np.count_nonzero(obj))
    total_weight = float(np.nansum(np.where(obj, weights, 0.0)))
    memberships = []
    stats = []
    for name in candidates:
        region_mask = masks.get(name)
        if region_mask is None:
            continue
        overlap = obj & region_mask
        cells = int(np.count_nonzero(overlap))
        if cells <= 0:
            continue
        loss_weight = float(np.nansum(np.where(overlap, weights, 0.0)))
        memberships.append(name)
        stats.append((name, loss_weight, cells))

    if not stats:
        return {
            "region": "",
            "region_memberships": "",
            "region_assignment_method": "unassigned_no_region_overlap",
            "region_weight_fraction": np.nan,
            "region_cell_fraction": np.nan,
        }

    # Prefer the region that contains most of the object's cumulative SIC loss;
    # fall back naturally to cell count when the weighted loss ties or is zero.
    order_index = {name: idx for idx, name in enumerate(candidates)}
    primary, primary_weight, primary_cells = max(
        stats,
        key=lambda item: (item[1], item[2], -order_index.get(item[0], len(order_index))),
    )
    return {
        "region": primary,
        "region_memberships": ";".join(memberships),
        "region_assignment_method": "object_loss_weighted_mask",
        "region_weight_fraction": primary_weight / total_weight if total_weight > 0 else np.nan,
        "region_cell_fraction": primary_cells / total_cells if total_cells > 0 else np.nan,
    }


def max_loss_location(obj: np.ndarray, diff: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> tuple[float, float]:
    values = np.where(obj, diff, np.nan)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    iy, ix = np.unravel_index(int(np.nanargmin(values)), values.shape)
    return float(lon2[iy, ix]), float(lat2[iy, ix])


def date_range(start_year: int, end_year: int, months: list[int]) -> list[pd.Timestamp]:
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    return [pd.Timestamp(d).normalize() for d in dates if int(d.month) in months]


_LOCAL_GRID_CACHE: dict[tuple[str, ...], tuple] = {}


def _detect_date_task(task: dict) -> list[dict]:
    event_date = pd.Timestamp(task["event_date"])
    regions = list(task["regions"])
    sic_root = Path(task["sic_root"])
    rows: list[dict] = []
    for window in task["window_days"]:
        start_date = event_date - pd.Timedelta(days=int(window))
        try:
            end_arr, lon2, lat2, native_x2, native_y2, inverse_transformer = read_sic(
                str(find_sic_file(sic_root, event_date))
            )
            start_arr, *_ = read_sic(str(find_sic_file(sic_root, start_date)))
        except FileNotFoundError:
            continue
        key = tuple(regions)
        cached = _LOCAL_GRID_CACHE.get(key)
        if cached is None:
            masks = {name: parse_region(name).contains(lon2, lat2) for name in regions}
            pan_mask = masks.get("pan_arctic")
            if pan_mask is None:
                pan_mask = parse_region("pan_arctic").contains(lon2, lat2)
            cached = (masks, pan_mask, lon2, lat2, native_x2, native_y2, inverse_transformer)
            _LOCAL_GRID_CACHE[key] = cached
        masks, pan_mask, lon2, lat2, native_x2, native_y2, inverse_transformer = cached
        diff = end_arr - start_arr
        valid = np.isfinite(diff) & pan_mask
        for threshold in task["thresholds"]:
            components = detect_connected_components(
                diff,
                valid,
                float(threshold),
                int(task["min_cells"]),
                lon2=lon2,
                lat2=lat2,
                native_x2=native_x2,
                native_y2=native_y2,
                inverse_transform=inverse_transformer.transform,
                canonical_local_geometry=True,
                binary_closing=False,
                strict_min_cells=False,
            )
            for component in components:
                obj = component.mask
                weights = np.where(obj, -np.minimum(diff, 0.0), 0.0)
                centroid_lon = component.weighted_centroid_lon
                centroid_lat = component.weighted_centroid_lat
                centroid_region, centroid_memberships = assign_regions(centroid_lon, centroid_lat, regions)
                region_info = assign_object_region(obj, weights, masks, regions)
                rows.append(
                    {
                        "date": event_date.strftime("%Y-%m-%d"),
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "year": event_date.year,
                        "month": event_date.month,
                        "window_days": int(window),
                        "threshold": float(threshold),
                        "centroid_lon": centroid_lon,
                        "centroid_lat": centroid_lat,
                        "longitude_geometry_class": component.longitude_geometry_class,
                        "raw_signed_longitude_span_deg": component.raw_signed_longitude_span_deg,
                        "minimum_circular_covering_arc_deg": component.minimum_circular_covering_arc_deg,
                        "largest_longitude_gap_deg": component.largest_longitude_gap_deg,
                        "unwrap_arc_start_deg": component.unwrap_arc_start_deg,
                        "largest_gap_tie_count": component.largest_gap_tie_count,
                        "region": str(region_info["region"]),
                        "region_memberships": region_info["region_memberships"],
                        "centroid_region": centroid_region,
                        "centroid_region_memberships": centroid_memberships,
                        "max_loss_lon": component.max_loss_lon,
                        "max_loss_lat": component.max_loss_lat,
                        "region_assignment_method": region_info["region_assignment_method"],
                        "region_weight_fraction": region_info["region_weight_fraction"],
                        "region_cell_fraction": region_info["region_cell_fraction"],
                        "object_area_cells": int(component.cell_count),
                        "object_area_km2": float(component.cell_count) * 25.0 * 25.0,
                        "mean_sic_change": component.mean_sic_change,
                        "min_sic_change": component.min_sic_change,
                        "cumulative_sic_loss": component.cumulative_sic_loss,
                    }
                )
    return rows


def detect_objects(args: argparse.Namespace, regions: list[str]) -> pd.DataFrame:
    sic_root = Path(args.sic_root)
    months = list(parse_months(args.months))
    years = (max(args.start_year, 2019), min(args.end_year, 2021)) if args.quick else (args.start_year, args.end_year)
    windows = args.window_days[:1] if args.quick else args.window_days
    thresholds = args.thresholds[:1] if args.quick else args.thresholds
    dates = date_range(years[0], years[1], months)
    pan_dates = load_event_dates(Path(args.panarctic_events))
    regional_lookup = load_regional_event_lookup(Path(args.regional_events))
    tasks = [
        {
            "event_date": event_date,
            "sic_root": str(sic_root),
            "regions": tuple(regions),
            "window_days": tuple(windows),
            "thresholds": tuple(thresholds),
            "min_cells": args.min_cells,
        }
        for event_date in dates
    ]
    use_parallel = args.parallel and not args.disable_parallel
    print(f"parallel={use_parallel} backend={args.backend} workers={args.workers} date_tasks={len(tasks)}")
    nested = run_with_fallback(
        _detect_date_task,
        tasks,
        workers=args.workers,
        parallel=use_parallel,
        backend=args.backend,
    )
    rows = [row for group in nested for row in group]
    for object_seq, row in enumerate(rows, start=1):
        event_date = pd.Timestamp(row["date"])
        primary_region = str(row["region"])
        row["object_id"] = f"LOC{object_seq:07d}"
        row["overlap_with_panarctic_vrile"] = event_date in pan_dates
        row["overlap_with_regional_vrile"] = (primary_region, event_date) in regional_lookup
        row["overlap_with_regional_response_diagnostic"] = (primary_region, event_date) in regional_lookup
        row["event_catalog_role"] = "primary_local_spatial_event_candidate"
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[["object_id"] + [column for column in out.columns if column != "object_id"]]
    return out


def summarize_objects(objects: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    rows = []
    for region in regions:
        if objects.empty:
            df = objects
        elif region == "pan_arctic":
            # Pan-Arctic is the detection domain, so report it as an aggregate
            # total rather than as a mutually exclusive primary region.
            df = objects
        else:
            df = objects[objects["region"] == region]
        years = int(df["year"].nunique()) if not df.empty else 0
        rows.append(
            {
                "region": region,
                "label": REGION_LABELS.get(region, region),
                "object_count": int(len(df)),
                "years_with_objects": years,
                "mean_area_km2": float(df["object_area_km2"].mean()) if not df.empty else np.nan,
                "mean_sic_change": float(df["mean_sic_change"].mean()) if not df.empty else np.nan,
                "mean_cumulative_sic_loss": float(df["cumulative_sic_loss"].mean()) if not df.empty else np.nan,
                "panarctic_overlap_fraction": float(df["overlap_with_panarctic_vrile"].mean()) if not df.empty else np.nan,
                "regional_overlap_fraction": float(df["overlap_with_regional_vrile"].mean()) if not df.empty else np.nan,
                "regional_response_diagnostic_overlap_fraction": float(
                    df["overlap_with_regional_response_diagnostic"].mean()
                )
                if not df.empty
                else np.nan,
                "cross_region_count_comparable": False,
            }
        )
    return pd.DataFrame(rows)


def write_unique(objects: pd.DataFrame) -> pd.DataFrame:
    if objects.empty:
        return objects.copy()
    ordered = objects.sort_values(["region", "date", "window_days", "threshold", "cumulative_sic_loss"], ascending=[True, True, True, True, False])
    return ordered.groupby(["region", "date", "window_days", "threshold"], as_index=False, dropna=False).head(1).reset_index(drop=True)


def plot_outputs(objects: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, label: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary[~summary["region"].isin(["central_arctic", "pan_arctic"])].sort_values("object_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.barh(plot_df["label"], plot_df["object_count"], color="#386CB0")
    ax.set_xlabel("Local rapid SIC-loss objects")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(fig_dir / "local_object_region_counts.png", dpi=220)
    plt.close(fig)

    if not objects.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        sc = ax.scatter(
            objects["centroid_lon"],
            objects["centroid_lat"],
            c=objects["cumulative_sic_loss"],
            s=np.clip(objects["object_area_cells"], 8, 120),
            cmap="viridis",
            alpha=0.55,
            linewidths=0,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Local rapid SIC-loss object centroids")
        ax.grid(color="#E6E8EB", linewidth=0.8)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.86)
        cbar.set_label("Cumulative SIC loss (cell fraction)")
        fig.savefig(fig_dir / "local_object_density_map.png", dpi=220)
        plt.close(fig)

        month_counts = objects.groupby(["region", "month"]).size().reset_index(name="count")
        month_counts["label"] = month_counts["region"].map(REGION_LABELS).fillna(month_counts["region"])
        pivot = month_counts.pivot_table(index="label", columns="month", values="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_xlabel("Month")
        ax.set_title(f"{label.upper()} timing of local SIC-loss objects")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Object count")
        fig.savefig(fig_dir / f"local_object_timing_{label}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be >= --start-year")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = canonical_regions(args.regions)
    months = parse_months(args.months)
    label = months_label(months)
    primary = out_dir / f"local_objects_all_{label}.csv"
    if primary.exists() and not args.overwrite:
        print(f"reuse existing {primary}; pass --overwrite to recompute")
        objects = pd.read_csv(primary, parse_dates=["date", "start_date"])
    else:
        objects = detect_objects(args, regions)
        objects.to_csv(primary, index=False)
    unique = write_unique(objects)
    summary = summarize_objects(unique, regions)
    unique.to_csv(out_dir / "local_objects_unique.csv", index=False)
    summary.to_csv(out_dir / "local_object_region_summary.csv", index=False)
    plot_outputs(unique, summary, out_dir, label)
    metadata = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "months": list(months),
        "months_label": label,
        "window_days": args.window_days,
        "thresholds": args.thresholds,
        "min_cells": args.min_cells,
        "quick": args.quick,
        "parallel": args.parallel and not args.disable_parallel,
        "backend": args.backend,
        "workers": args.workers,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"WROTE {out_dir}")


if __name__ == "__main__":
    main()
