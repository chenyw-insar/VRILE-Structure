#!/usr/bin/env python
"""Filter daily local SIC-loss patches and merge them into unique local events."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vrile.local_objects import haversine_km, merge_local_patches
from vrile.months import months_label, parse_months
from vrile.regions import parse_region, region_display_label


REGION_LABELS = {
    "barents_sea": "Barents",
    "barents_extended": "Barents ext.",
    "kara_sea": "Kara",
    "laptev_sea": "Laptev",
    "east_siberian_sea": "East Siberian",
    "chukchi_sea": "Chukchi",
    "beaufort_sea": "Beaufort",
    "greenland_nordic_seas": "Greenland/Nordic",
    "central_basin": region_display_label("central_basin"),
    "central_arctic": region_display_label("central_arctic"),
    "pan_arctic": "Pan-Arctic reference",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-objects", default="outputs/local_vrile/local_objects_all_jja.csv")
    parser.add_argument('--out_dir', default="outputs/local_vrile_enhanced")
    parser.add_argument("--months", default="jja", help="'jja', 'all', or comma-separated month numbers.")
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--window-days", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--sic-change-thresholds", nargs="+", type=float, default=[-0.05, -0.10, -0.15])
    parser.add_argument("--min-cells", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--primary-window-days", type=int, default=5)
    parser.add_argument("--primary-sic-change-threshold", type=float, default=-0.10)
    parser.add_argument("--primary-min-cells", type=int, default=50)
    parser.add_argument("--merge-days", type=int, default=2)
    parser.add_argument("--merge-distance-km", type=float, default=300.0)
    parser.add_argument("--regions", default="all")
    parser.add_argument("--quick", action="store_true")
    return parser


def canonical_regions(value: str, objects: pd.DataFrame) -> list[str]:
    if value.strip().lower() == "all":
        return sorted([r for r in objects["region"].dropna().unique() if r != "central_arctic"])
    return [parse_region(x.strip()).name for x in value.split(",") if x.strip()]


def load_objects(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.local_objects, parse_dates=["date", "start_date"])
    df = df[(df["year"] >= args.start_year) & (df["year"] <= args.end_year)].copy()
    months = parse_months(args.months)
    if "month" in df:
        df = df[df["month"].isin(months)].copy()
    if args.quick:
        df = df[(df["year"] >= 2019) & (df["year"] <= 2021)].copy()
    df = df[df["window_days"].isin(args.window_days)].copy()
    return df


def filter_objects(df: pd.DataFrame, window_days: int, threshold: float, min_cells: int) -> pd.DataFrame:
    out = df[
        (df["window_days"].astype(int) == int(window_days))
        & np.isclose(df["threshold"].astype(float), float(threshold))
        & (df["object_area_cells"].astype(int) >= int(min_cells))
    ].copy()
    return out.sort_values(["region", "date", "cumulative_sic_loss"], ascending=[True, True, False])


def merge_unique_events(filtered: pd.DataFrame, merge_days: int, merge_distance_km: float) -> pd.DataFrame:
    return merge_local_patches(filtered, merge_days, merge_distance_km)


def sensitivity(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for window in args.window_days:
        for threshold in args.sic_change_thresholds:
            threshold_available = np.isclose(df["threshold"].astype(float), float(threshold)).any()
            for min_cells in args.min_cells:
                if not threshold_available:
                    rows.append(
                        {
                            "window_days": window,
                            "sic_change_threshold": threshold,
                            "min_cells": min_cells,
                            "daily_patch_count": np.nan,
                            "unique_event_count": np.nan,
                            "status": "not_available_in_existing_daily_object_file",
                        }
                    )
                    continue
                sub = filter_objects(df, window, threshold, min_cells)
                unique = merge_unique_events(sub, args.merge_days, args.merge_distance_km)
                rows.append(
                    {
                        "window_days": window,
                        "sic_change_threshold": threshold,
                        "min_cells": min_cells,
                        "daily_patch_count": int(len(sub)),
                        "unique_event_count": int(len(unique)),
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def region_summary(filtered: pd.DataFrame, unique: pd.DataFrame) -> pd.DataFrame:
    regions = sorted(set(filtered["region"].dropna()) | set(unique["dominant_region"].dropna()))
    rows = []
    for region in regions:
        f = filtered[filtered["region"] == region]
        u = unique[unique["dominant_region"] == region]
        rows.append(
            {
                "region": region,
                "label": REGION_LABELS.get(region, region),
                "daily_patch_count": int(len(f)),
                "unique_event_count": int(len(u)),
                "patch_to_event_ratio": float(len(f) / len(u)) if len(u) else np.nan,
                "mean_duration_days": float(u["duration_days"].mean()) if len(u) else np.nan,
                "mean_max_area_km2": float(u["max_area_km2"].mean()) if len(u) else np.nan,
                "mean_cumulative_loss": float(u["cumulative_loss"].mean()) if len(u) else np.nan,
                "panarctic_overlap_fraction": float(u["overlap_with_panarctic_vrile"].mean()) if len(u) else np.nan,
                "regional_response_diagnostic_overlap_fraction": float(
                    u["overlap_with_regional_response_diagnostic"].mean()
                )
                if len(u) and "overlap_with_regional_response_diagnostic" in u
                else np.nan,
                "event_catalog_role": "primary_local_spatial_event",
                "cross_region_count_comparable": False,
                "comparison_note": (
                    "Raw counts are descriptive; formal cross-region comparisons require area or ice-exposure normalization."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("unique_event_count", ascending=False)


def plot_outputs(filtered: pd.DataFrame, unique: pd.DataFrame, summary: pd.DataFrame, sens: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot = summary[summary["region"] != "central_arctic"].sort_values("unique_event_count", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    ax.barh(y - 0.18, plot["daily_patch_count"], height=0.34, label="daily patches", color="#9AA0A6")
    ax.barh(y + 0.18, plot["unique_event_count"], height=0.34, label="unique local events", color="#386CB0")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Count")
    ax.set_title("Daily local patches vs unique local events")
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "daily_objects_vs_unique_events_by_region.png", dpi=220)
    plt.close(fig)

    if not unique.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        sc = ax.scatter(
            unique["track_centroid_lon"],
            unique["track_centroid_lat"],
            c=unique["cumulative_loss"],
            s=np.clip(unique["max_area_cells"] / 6, 8, 150),
            cmap="viridis",
            alpha=0.58,
            linewidths=0,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Filtered unique local rapid-loss events")
        ax.grid(color="#E6E8EB", linewidth=0.8)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Cumulative SIC loss")
        fig.savefig(fig_dir / "local_event_density_filtered.png", dpi=220)
        plt.close(fig)

    ok = sens[sens["status"] == "ok"]
    if not ok.empty:
        pivot = ok.pivot_table(index="min_cells", columns="sic_change_threshold", values="unique_event_count", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(x)) for x in pivot.index])
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
        ax.set_ylabel("min_cells")
        ax.set_xlabel("SIC-change threshold")
        ax.set_title("Object filter sensitivity")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, str(int(pivot.values[i, j])), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.82, label="Unique events")
        fig.savefig(fig_dir / "object_filter_sensitivity_heatmap.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = load_objects(args)
    primary = filter_objects(objects, args.primary_window_days, args.primary_sic_change_threshold, args.primary_min_cells)
    unique = merge_unique_events(primary, args.merge_days, args.merge_distance_km)
    summary = region_summary(primary, unique)
    sens = sensitivity(objects, args)
    primary.to_csv(out_dir / "local_objects_filtered.csv", index=False)
    unique.to_csv(out_dir / "unique_local_events.csv", index=False)
    summary.to_csv(out_dir / "unique_local_event_region_summary.csv", index=False)
    sens.to_csv(out_dir / "local_object_filter_sensitivity.csv", index=False)
    plot_outputs(primary, unique, summary, sens, out_dir)
    metadata = vars(args).copy()
    metadata["months"] = list(parse_months(args.months))
    metadata["months_label"] = months_label(args.months)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"WROTE {out_dir}")


if __name__ == "__main__":
    main()
