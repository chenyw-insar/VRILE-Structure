#!/usr/bin/env python3
"""Filter broad unique local SIC-loss events into severe local events."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_utils_parallel import configure_blas_threads, safe_write_csv, safe_write_text, set_reproducible_seed


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vrile.months import months_label, parse_months

LOGGER = logging.getLogger("filter_severe_local_events")


def df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", default="jja", help="'jja', 'all', or comma-separated month numbers.")
    p.add_argument("--start-year", type=int, default=1989)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--input", default="outputs/local_vrile_enhanced/unique_local_events.csv")
    p.add_argument('--out_dir', default="outputs/local_vrile_severe")
    p.add_argument("--severity-percentiles", nargs="+", type=float, default=[10, 20, 30])
    p.add_argument("--min-cells", nargs="+", type=int, default=[50, 100, 200])
    p.add_argument("--min-duration", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--sic-thresholds", nargs="+", type=float, default=[-0.10, -0.15])
    p.add_argument("--quick", action="store_true")
    # Reproducibility seed only; this is not an analysis date.
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_region(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str)


def load_broad(path: Path, start_year: int, end_year: int, months: tuple[int, ...], quick: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing broad local event table: {path}")
    df = pd.read_csv(path, parse_dates=["event_start", "event_end"])
    if "event_start" in df:
        df = df[(df["event_start"].dt.year >= start_year) & (df["event_start"].dt.year <= end_year)].copy()
        df = df[df["event_start"].dt.month.isin(months)].copy()
    df["dominant_region"] = normalize_region(df["dominant_region"])
    if quick:
        keep = ["beaufort_sea", "laptev_sea", "kara_sea", "barents_sea", "central_arctic"]
        df = df[df["dominant_region"].isin(keep)].head(1000).copy()
    return df.reset_index(drop=True)


def mark_severe(
    df: pd.DataFrame,
    severity_percentile: float,
    min_cells: int,
    min_duration: int,
    sic_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    loss_q = out["cumulative_loss"].quantile(1 - severity_percentile / 100)
    area_col = "max_area_cells" if "max_area_cells" in out.columns else "min_cells"
    area_q = out[area_col].quantile(1 - severity_percentile / 100)
    intensity_q = out["max_intensity"].quantile(1 - severity_percentile / 100)

    region_loss_q = out.groupby("dominant_region")["cumulative_loss"].transform(lambda s: s.quantile(1 - severity_percentile / 100))
    region_area_q = out.groupby("dominant_region")[area_col].transform(lambda s: s.quantile(1 - severity_percentile / 100))

    thresh_ok = out.get("sic_change_threshold", pd.Series([-0.10] * len(out))).astype(float) <= sic_threshold + 1e-12
    if sic_threshold < out.get("sic_change_threshold", pd.Series([-0.10] * len(out))).min() - 1e-12:
        thresh_ok = pd.Series(False, index=out.index)

    size_ok = out[area_col] >= min_cells
    duration_or_intensity_ok = (out["duration_days"] >= min_duration) | (out["max_intensity"] >= intensity_q)
    global_ok = ((out["cumulative_loss"] >= loss_q) | (out[area_col] >= area_q)) & size_ok & duration_or_intensity_ok & thresh_ok
    region_ok = ((out["cumulative_loss"] >= region_loss_q) | (out[area_col] >= region_area_q)) & size_ok & duration_or_intensity_ok & thresh_ok

    out["severity_percentile"] = severity_percentile
    out["severity_loss_threshold"] = loss_q
    out["severity_area_threshold_cells"] = area_q
    out["min_cells_threshold"] = min_cells
    out["min_duration_threshold"] = min_duration
    out["sic_threshold_filter"] = sic_threshold
    out["is_severe_global"] = global_ok
    out["is_severe_region_normalized"] = region_ok
    out["is_severe_primary"] = global_ok | region_ok
    return out


def sensitivity_table(df: pd.DataFrame, percentiles: list[float], min_cells_values: list[int], min_duration_values: list[int], sic_thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for pct in percentiles:
        for cells in min_cells_values:
            for dur in min_duration_values:
                for sic_thr in sic_thresholds:
                    marked = mark_severe(df, pct, cells, dur, sic_thr)
                    for scope, col in [("global", "is_severe_global"), ("region_normalized", "is_severe_region_normalized"), ("primary_union", "is_severe_primary")]:
                        sub = marked[marked[col]]
                        rows.append(
                            {
                                "severity_percentile": pct,
                                "min_cells": cells,
                                "min_duration": dur,
                                "sic_threshold": sic_thr,
                                "scope": scope,
                                "severe_event_count": len(sub),
                                "broad_event_count": len(df),
                                "severe_fraction_of_broad": len(sub) / len(df) if len(df) else np.nan,
                                "n_regions_with_events": sub["dominant_region"].nunique() if len(sub) else 0,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_regions(broad: pd.DataFrame, severe: pd.DataFrame) -> pd.DataFrame:
    regional_overlap_col = (
        "overlap_with_regional_response_diagnostic"
        if "overlap_with_regional_response_diagnostic" in severe.columns
        else "overlap_with_regional_vrile_v1"
    )
    b = broad.groupby("dominant_region").agg(
        broad_event_count=("unique_local_event_id", "count"),
        broad_median_cumulative_loss=("cumulative_loss", "median"),
        broad_median_max_area_cells=("max_area_cells", "median"),
    )
    s = severe.groupby("dominant_region").agg(
        severe_event_count=("unique_local_event_id", "count"),
        severe_median_cumulative_loss=("cumulative_loss", "median"),
        severe_median_max_area_cells=("max_area_cells", "median"),
        severe_overlap_panarctic=("overlap_with_panarctic_vrile", "sum"),
        severe_overlap_regional_response_diagnostic=(regional_overlap_col, "sum"),
    )
    out = b.join(s, how="left").fillna(0).reset_index().rename(columns={"dominant_region": "region"})
    out["severe_fraction_of_broad"] = out["severe_event_count"] / out["broad_event_count"].replace(0, np.nan)
    # Compatibility alias; this is diagnostic overlap, not an event-definition gate.
    out["severe_overlap_regional"] = out["severe_overlap_regional_response_diagnostic"]
    out["event_catalog_role"] = "primary_local_spatial_event"
    out["cross_region_count_comparable"] = False
    out["interpretation_flag"] = np.where(
        out["severe_event_count"] == 0,
        "weak_or_not_supported_by_severe_definition",
        np.where(out["severe_fraction_of_broad"] < 0.05, "broad-dominated_or_threshold-sensitive", "supported_by_severe_definition"),
    )
    return out.sort_values(["severe_event_count", "broad_event_count"], ascending=False)


def make_figures(broad: pd.DataFrame, severe: pd.DataFrame, summary: pd.DataFrame, sensitivity: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_df = summary.sort_values("broad_event_count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["broad_event_count"], color="#9aa6b2", alpha=0.55, label="Broad unique local events")
    ax.barh(y, plot_df["severe_event_count"], color="#b33f3f", alpha=0.85, label="Severe unique local events")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["region"])
    ax.set_xlabel("Event count")
    ax.set_title("Broad vs severe local rapid SIC-loss events by region")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="0.85", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "broad_vs_severe_local_events_by_region.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not severe.empty:
        sc = ax.scatter(severe["track_centroid_lon"], severe["track_centroid_lat"], c=severe["cumulative_loss"], s=np.clip(severe["max_area_cells"] / 8, 10, 120), cmap="magma_r", alpha=0.75, edgecolor="white", linewidth=0.2)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Cumulative SIC loss")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Severe local rapid SIC-loss event centroids")
    ax.grid(color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "severe_local_event_density_map.png", dpi=220)
    plt.close(fig)

    main_sens = sensitivity[sensitivity["scope"] == "primary_union"].copy()
    pivot = main_sens.pivot_table(index="min_cells", columns="severity_percentile", values="severe_event_count", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Severity percentile (top %)")
    ax.set_ylabel("Minimum cells")
    ax.set_title("Severe local event sensitivity")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="white" if pivot.values[i, j] < np.nanmax(pivot.values) * 0.65 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Mean severe event count")
    fig.tight_layout()
    fig.savefig(fig_dir / "severe_event_sensitivity_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    configure_blas_threads(1)
    set_reproducible_seed(args.seed)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    broad = load_broad(ROOT / args.input, args.start_year, args.end_year, months, args.quick)
    LOGGER.info("loaded broad events: %s", len(broad))

    safe_write_csv(broad, out_dir / "broad_unique_local_events.csv")
    severe_all = mark_severe(broad, 20, 100, 2, -0.10)
    severe = severe_all[severe_all["is_severe_primary"]].copy()
    severe["severe_definition"] = "primary_union_top20_min100_duration2_or_high_intensity_sic-0.10"
    safe_write_csv(severe, out_dir / "severe_unique_local_events.csv")

    summary = summarize_regions(broad, severe)
    safe_write_csv(summary, out_dir / "severe_local_event_region_summary.csv")
    sensitivity = sensitivity_table(broad, args.severity_percentiles, args.min_cells, args.min_duration, args.sic_thresholds)
    safe_write_csv(sensitivity, out_dir / "severe_event_sensitivity.csv")
    make_figures(broad, severe, summary, sensitivity, out_dir)

    answer = [
        "# Severe local event filtering summary",
        "",
        f"- Broad unique local events: {len(broad)}",
        f"- Primary severe unique local events: {len(severe)}",
        f"- Severe fraction: {len(severe) / len(broad):.3f}" if len(broad) else "- Severe fraction: NA",
        "- Broad events should be used only as the complete local SIC-loss activity background.",
        "- Severe events should be used as the main local rapid-loss event definition in the submission draft.",
        "",
        "## Region summary",
        "",
        df_to_md(summary),
    ]
    safe_write_text(out_dir / "severe_local_event_summary.md", "\n".join(answer))
    safe_write_text(
        out_dir / "run_metadata.json",
        json.dumps({**vars(args), "months": list(months), "months_label": months_label(months)}, indent=2),
    )
    LOGGER.info("wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
