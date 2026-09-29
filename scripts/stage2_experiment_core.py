#!/usr/bin/env python
"""VRILE Stage 2 credibility-enhancement pipeline.

This stage uses raw-regenerated Stage-1 outputs in this workspace. It does not copy old
outputs as formal results. Old scripts are reused as tooling only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Geod, Transformer
from common_utils_parallel import run_with_fallback

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from vrile.regions import NSIDC_0780_REGION_IDS, region_display_label

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


PHYSICAL_REGIONS = [
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
]
FOCUS_REGIONS = ["beaufort_sea", "laptev_sea", "kara_sea", "barents_sea", "central_arctic"]
AGGREGATE_REGIONS = ["pan_arctic", "pan_arctic_reference"]
NSIDC_0780_OCEAN_CODES: tuple[int, ...] = tuple(range(0, 19))
NSIDC_0780_NON_OCEAN_CODES: frozenset[int] = frozenset({30, 32, 33, 34, 35, 40})
BACKGROUND_LAGS: tuple[int, ...] = (-10, -7, -5)
PRE_EVENT_LAGS: tuple[int, ...] = (-3, -1)
EVENT_TIME_LAGS: tuple[int, ...] = (0,)
POST_EVENT_LAGS: tuple[int, ...] = (1, 3, 5)
LAG_GROUPS: dict[str, tuple[int, ...]] = {
    "pre-event": PRE_EVENT_LAGS,
    "event-time": EVENT_TIME_LAGS,
    "post-event": POST_EVENT_LAGS,
}
ALL_DIAGNOSTIC_LAGS: tuple[int, ...] = BACKGROUND_LAGS + PRE_EVENT_LAGS + EVENT_TIME_LAGS + POST_EVENT_LAGS
CYCLONE_EXPECTED_DIRECTION: dict[str, str] = {
    "nearest_cyclone_distance_km": "negative",
    "nearest_cyclone_pressure_hpa": "negative",
    "cyclone_count_500km": "positive",
    "cyclone_count_1000km": "positive",
}
QA_ORDER = {"ok": 0, "degraded": 1, "reject": 2}
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
    "pan_arctic": "Pan-Arctic reference",
}


def _assert_lag_groups_disjoint() -> None:
    groups = {
        "background": BACKGROUND_LAGS,
        "pre-event": PRE_EVENT_LAGS,
        "event-time": EVENT_TIME_LAGS,
        "post-event": POST_EVENT_LAGS,
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(groups[a]) & set(groups[b])
            assert not overlap, f"overlapping lag groups: {a} and {b}: {sorted(overlap)}"


_assert_lag_groups_disjoint()
ADJACENT = {
    "central_arctic": {"beaufort_sea", "chukchi_sea", "east_siberian_sea", "laptev_sea", "kara_sea", "barents_sea", "east_greenland_sea", "canadian_archipelago"},
    "beaufort_sea": {"central_arctic", "chukchi_sea", "canadian_archipelago"},
    "chukchi_sea": {"central_arctic", "beaufort_sea", "east_siberian_sea", "bering_sea"},
    "east_siberian_sea": {"central_arctic", "chukchi_sea", "laptev_sea"},
    "laptev_sea": {"central_arctic", "east_siberian_sea", "kara_sea"},
    "kara_sea": {"central_arctic", "laptev_sea", "barents_sea"},
    "barents_sea": {"central_arctic", "kara_sea", "east_greenland_sea"},
    "east_greenland_sea": {"central_arctic", "barents_sea", "baffin_and_labrador_seas"},
    "baffin_and_labrador_seas": {"east_greenland_sea", "canadian_archipelago", "hudson_bay", "gulf_of_st_lawrence"},
    "gulf_of_st_lawrence": {"baffin_and_labrador_seas"},
    "hudson_bay": {"baffin_and_labrador_seas", "canadian_archipelago"},
    "canadian_archipelago": {"central_arctic", "beaufort_sea", "baffin_and_labrador_seas", "hudson_bay"},
    "bering_sea": {"chukchi_sea", "sea_of_okhotsk", "gulf_of_alaska"},
    "sea_of_okhotsk": {"bering_sea", "sea_of_japan"},
    "sea_of_japan": {"sea_of_okhotsk"},
    "bohai_and_yellow_seas": set(),
    "baltic_sea": set(),
    "gulf_of_alaska": {"bering_sea"},
}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def outdir(name: str) -> Path:
    p = root() / "outputs" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def figdir(out: Path) -> Path:
    p = out / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_major_severe_region_counts(summary: pd.DataFrame, output_path: Path) -> None:
    p = summary.sort_values("major_event_count")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    ax.barh(y - 0.2, p["broad_event_count"], height=0.18, label="broad", color="#BDBDBD")
    ax.barh(y, p["severe_event_count"], height=0.18, label="severe", color="#4C78A8")
    ax.barh(y + 0.2, p["major_event_count"], height=0.18, label="major_severe", color="#D95F02")
    ax.set_yticks(y); ax.set_yticklabels([REGION_LABELS.get(x, x) for x in p["region"]])
    ax.set_xlabel("Event count"); ax.set_title("Broad / severe / major_severe events by physical region")
    ax.grid(axis="x", color="#E6E6E6"); ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220); plt.close(fig)


def plot_cluster_region_counts(summary: pd.DataFrame, output_path: Path) -> None:
    p = summary.sort_values("severe_cluster_count")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(y - 0.17, p["severe_cluster_count"], height=0.32, label="severe clusters", color="#4C78A8")
    ax.barh(y + 0.17, p["major_cluster_count"], height=0.32, label="major clusters", color="#D95F02")
    ax.set_yticks(y); ax.set_yticklabels([REGION_LABELS.get(x, x) for x in p["region"]]); ax.set_xlabel("Cluster count"); ax.set_title("Event-level vs cluster-level region counts")
    ax.grid(axis="x", color="#E6E6E6"); ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220); plt.close(fig)


def plot_cyclone_proximity_by_region(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    regs = [r for r in FOCUS_REGIONS if r in df["region"].unique()]
    ax.boxplot([df.loc[df["region"] == r, "nearest_cyclone_distance_km"].dropna() for r in regs], labels=[REGION_LABELS.get(r, r) for r in regs], showfliers=False)
    ax.set_ylabel("Nearest cyclone distance (km)"); ax.set_title("Cyclone proximity for major_severe clusters"); ax.tick_params(axis="x", rotation=35)
    fig.savefig(output_path, dpi=220); plt.close(fig)


def plot_ice_edge_wind_by_region(samples: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for region in [r for r in FOCUS_REGIONS if r in samples["region"].unique()]:
        sub = samples[samples["region"] == region].groupby("lag", as_index=False).mean(numeric_only=True)
        ax.plot(sub["lag"], sub["outward_edge_normal_wind_ms"], marker="o", label=REGION_LABELS.get(region, region))
    ax.axhline(0, color="#BBBBBB", ls="--", lw=.8)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=.8)
    ax.set_xlabel("Lag day")
    ax.set_ylabel("Outward edge-normal wind (m s-1)")
    ax.set_title("Ice-edge-relative wind around major_severe clusters")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_evidence_strength_heatmap(trace: pd.DataFrame, output_path: Path) -> None:
    heat = trace.pivot_table(index="region", columns="evidence_dimension", values="score", aggfunc="max")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True); im = ax.imshow(heat.values, aspect="auto", vmin=0, vmax=4, cmap="YlGnBu")
    ax.set_yticks(np.arange(len(heat.index))); ax.set_yticklabels([REGION_LABELS.get(x, x) for x in heat.index]); ax.set_xticks(np.arange(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=40, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Evidence score"); ax.set_title("Evidence strength matrix"); fig.savefig(output_path, dpi=220); plt.close(fig)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--backend", default="process", choices=["process", "thread", "serial"])
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--severe-events", default="outputs/local_vrile_severe/severe_unique_local_events.csv")
    parser.add_argument("--major-events", default="outputs/local_vrile_major_severe/major_severe_events_union.csv")
    parser.add_argument("--panarctic-events", default="outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv")
    parser.add_argument("--panarctic-locations", default="outputs/reproduce_sie/vrile_locations.csv")
    parser.add_argument(
        "--regional-events",
        default="outputs/regional_vrile_enhanced/regional_vrile_events_unique.csv",
        help="Regional-response diagnostic used only in event-definition sensitivity matching.",
    )
    parser.add_argument("--severe-clusters", default="outputs/event_clusters/synoptic_event_clusters_severe.csv")
    parser.add_argument("--major-clusters", default="outputs/event_clusters/synoptic_event_clusters_major.csv")
    parser.add_argument('--sic_dir', default="data/raw/nsidc_sic", dest='sic_root')
    parser.add_argument("--era5-single-glob", default="data/raw/era5/single_levels/*.nc")
    parser.add_argument("--cyclone-tracks", default="data/processed/cyclone_tracks/era5_derived/cyclone_track_points.csv")
    parser.add_argument("--location-buffer-values", default="outputs/location_buffer_mechanism/location_buffer_lag_values.csv")
    parser.add_argument(
        "--wrong-region-location-buffer-values",
        default="outputs/location_buffer_mechanism/wrong_region_location_buffer_lag_values.csv",
    )
    parser.add_argument(
        "--location-buffer-evidence",
        default="outputs/location_buffer_mechanism/location_buffer_evidence_table.csv",
    )
    parser.add_argument("--broad-summary", default="outputs/local_vrile_enhanced/unique_local_event_region_summary.csv")
    parser.add_argument("--severe-summary", default="outputs/local_vrile_severe/severe_local_event_region_summary.csv")
    parser.add_argument("--ice-edge-sic-threshold", type=float, default=0.15)
    parser.add_argument("--ice-edge-search-km", type=float, default=750.0)
    parser.add_argument("--ice-edge-coastal-cells", type=int, default=2)
    parser.add_argument("--ice-edge-min-cells", type=int, default=8)
    parser.add_argument("--ice-edge-component-fraction-min", type=float, default=0.45)
    parser.add_argument("--normal-orientation-sic-sampling-distance-m", type=float, default=50000.0)
    parser.add_argument("--normal-orientation-min-sic-difference", type=float, default=0.05)
    parser.add_argument("--normal-orientation-ok-fraction", type=float, default=0.80)
    parser.add_argument("--normal-orientation-degraded-fraction", type=float, default=0.50)
    parser.add_argument('--out_dir', default=None)
    return parser


def analysis_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root() / p


def parallel_enabled(args) -> bool:
    return bool(getattr(args, "parallel", False) and not getattr(args, "disable_parallel", False) and int(getattr(args, "workers", 1)) > 1)


def stable_seed(base_seed: int, *parts) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def haversine_km(lon1, lat1, lon2, lat2):
    if any(pd.isna(x) for x in [lon1, lat1, lon2, lat2]):
        return np.nan
    r = 6371.0
    p1, p2 = np.deg2rad(float(lat1)), np.deg2rad(float(lat2))
    dphi = np.deg2rad(float(lat2) - float(lat1))
    dlambda = np.deg2rad(float(lon2) - float(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def region_match(a, b, adjacent=True):
    if pd.isna(a) or pd.isna(b):
        return False
    if a == b:
        return True
    return adjacent and b in ADJACENT.get(a, set())


def cohen_d(a, b):
    a, b = pd.Series(a).dropna().to_numpy(float), pd.Series(b).dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled else np.nan


def welch_p(a, b):
    a, b = pd.Series(a).dropna().to_numpy(float), pd.Series(b).dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2 or stats is None:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def ci_bootstrap_diff(a, b, seed=20260625, n=1000):
    a, b = pd.Series(a).dropna().to_numpy(float), pd.Series(b).dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(np.mean(rng.choice(a, len(a), replace=True)) - np.mean(rng.choice(b, len(b), replace=True)))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def ci_bootstrap_mean(diffs, seed=20260625, n=1000):
    diffs = pd.Series(diffs).dropna().to_numpy(float)
    if len(diffs) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def one_sample_difference_stats(diffs, seed=20260625, nboot=1000):
    diffs = pd.Series(diffs).dropna().to_numpy(float)
    n_units = int(len(diffs))
    mean_difference = float(np.mean(diffs)) if n_units else np.nan
    sd_difference = float(np.std(diffs, ddof=1)) if n_units >= 2 else np.nan
    standardized = mean_difference / sd_difference if np.isfinite(sd_difference) and sd_difference != 0 else np.nan
    p_raw = (
        float(stats.ttest_1samp(diffs, 0.0, nan_policy="omit").pvalue)
        if stats is not None and n_units >= 2 and np.isfinite(sd_difference) and sd_difference != 0
        else np.nan
    )
    lo, hi = ci_bootstrap_mean(diffs, seed, nboot)
    return {
        "n_units": n_units,
        "mean_difference": mean_difference,
        "difference": mean_difference,
        "standardized_mean_difference_zero": standardized,
        "p_raw": p_raw,
        "ci_low": lo,
        "ci_high": hi,
    }


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(vals)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.minimum(adj, 1.0)
    out[np.where(ok)[0]] = tmp
    return out


def load_severe(path: str | Path = "outputs/local_vrile_severe/severe_unique_local_events.csv"):
    df = pd.read_csv(analysis_path(path))
    df["event_start"] = pd.to_datetime(df["event_start"])
    df["event_end"] = pd.to_datetime(df["event_end"])
    df["event_date"] = df["event_start"]
    return df[df["dominant_region"].isin(PHYSICAL_REGIONS)].copy()


def load_panarctic(
    events_path: str | Path = "outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv",
    locations_path: str | Path = "outputs/reproduce_sie/vrile_locations.csv",
):
    pan = pd.read_csv(analysis_path(events_path))
    loc = pd.read_csv(analysis_path(locations_path))
    pan["date"] = pd.to_datetime(pan["date"])
    loc["date"] = pd.to_datetime(loc["date"])
    df = pan.merge(loc[["date", "region", "center_lon", "center_lat"]], on="date", how="left")
    matched = df["region"].notna() & df["center_lon"].notna() & df["center_lat"].notna()
    print(f"panarctic_location_matched_n={int(matched.sum())} panarctic_location_unmatched_n={int((~matched).sum())}")
    if (~matched).any():
        print("WARNING: pan-Arctic events without matched locations remain unassigned; no central_arctic fill was applied.")
    return pd.DataFrame(
        {
            "event_id": [f"PAN{i:04d}" for i in range(1, len(df) + 1)],
            "definition": "panarctic",
            "region": df["region"],
            "location_match_status": np.where(matched, "matched", "unmatched_no_location_record"),
            "event_date": df["date"],
            "start_date": df["date"],
            "end_date": df["date"],
            "centroid_lon": df["center_lon"],
            "centroid_lat": df["center_lat"],
        }
    )


def primary_regional(path: str | Path = "outputs/regional_vrile_enhanced/regional_vrile_events_unique.csv"):
    """Load the regional-coherence diagnostic, not a formal event catalogue."""
    df = pd.read_csv(analysis_path(path))
    mask = (
        (df["window_days"].astype(int) == 5)
        & np.isclose(df["percentile"].astype(float), 5.0)
        & np.isclose(df["absolute_threshold"].astype(float), -0.03)
        & np.isclose(df["area_fraction_threshold"].astype(float), 0.02)
        & df["region"].isin(PHYSICAL_REGIONS)
    )
    df = df[mask].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    return pd.DataFrame(
        {
            "event_id": df["unique_event_id"].astype(str).radd("REG"),
            "definition": "regional_response_diagnostic",
            "region": df["region"],
            "event_date": df["event_date"],
            "start_date": pd.to_datetime(df["start_date"]),
            "end_date": pd.to_datetime(df["end_date"]),
            "centroid_lon": np.nan,
            "centroid_lat": np.nan,
        }
    )


def event_frame(kind, args=None):
    if kind == "panarctic":
        return load_panarctic(args.panarctic_events, args.panarctic_locations) if args else load_panarctic()
    if kind == "regional":
        return primary_regional(args.regional_events) if args else primary_regional()
    if kind == "severe":
        df = load_severe(args.severe_events) if args else load_severe()
    elif kind == "major":
        df = pd.read_csv(analysis_path(args.major_events) if args else root() / "outputs/local_vrile_major_severe/major_severe_events_union.csv")
        df["event_start"] = pd.to_datetime(df["event_start"])
        df["event_end"] = pd.to_datetime(df["event_end"])
        df["event_date"] = df["event_start"]
    elif kind in {"severe_cluster", "major_cluster"}:
        if args:
            path = analysis_path(args.severe_clusters if kind == "severe_cluster" else args.major_clusters)
        else:
            path = root() / "outputs/event_clusters" / f"synoptic_event_clusters_{'severe' if kind == 'severe_cluster' else 'major'}.csv"
        df = pd.read_csv(path)
        required = {
            "cluster_id",
            "dominant_region",
            "start_date",
            "end_date",
            "representative_date",
            "representative_date_method",
            "representative_event_id",
            "representative_region",
            "representative_lon",
            "representative_lat",
            "peak_loss_date",
            "peak_intensity_date",
            "centroid_lon",
            "centroid_lat",
            "cluster_centroid_lon",
            "cluster_centroid_lat",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Cluster table missing representative-date fields: "
                f"{sorted(missing)}. Rerun clustering before atmospheric diagnostics."
            )
        return pd.DataFrame(
            {
                "event_id": df["cluster_id"].astype(str),
                "definition": kind,
                "region": df["representative_region"],
                "event_date": pd.to_datetime(df["representative_date"]),
                "representative_date": pd.to_datetime(df["representative_date"]),
                "representative_event_id": df["representative_event_id"],
                "representative_date_method": df["representative_date_method"],
                "cluster_start_date": pd.to_datetime(df["start_date"]),
                "start_date": pd.to_datetime(df["start_date"]),
                "end_date": pd.to_datetime(df["end_date"]),
                "peak_loss_date": pd.to_datetime(df["peak_loss_date"]),
                "peak_intensity_date": pd.to_datetime(df["peak_intensity_date"]),
                "centroid_lon": df["representative_lon"],
                "centroid_lat": df["representative_lat"],
                "cluster_centroid_lon": df["cluster_centroid_lon"],
                "cluster_centroid_lat": df["cluster_centroid_lat"],
            }
        )
    else:
        raise ValueError(kind)
    return pd.DataFrame(
        {
            "event_id": df["unique_local_event_id"].astype(str),
            "definition": kind,
            "region": df["dominant_region"],
            "event_date": pd.to_datetime(df["event_date"]),
            "start_date": pd.to_datetime(df["event_start"]),
            "end_date": pd.to_datetime(df["event_end"]),
            "centroid_lon": df["track_centroid_lon"],
            "centroid_lat": df["track_centroid_lat"],
        }
    )


def stage2_taxonomy(args):
    out = outdir("region_taxonomy")
    rows = []
    for r in PHYSICAL_REGIONS:
        rows.append(
            {
                "region_name": r,
                "region_type": "physical_region",
                "used_in_ranking": True,
                "used_in_matching": True,
                "used_in_evidence_matrix": True,
                "used_in_mechanism_classification": r in FOCUS_REGIONS,
                "notes": "NSIDC-0780 official Arctic sea-ice region",
            }
        )
    for r in AGGREGATE_REGIONS:
        rows.append(
            {
                "region_name": r,
                "region_type": "aggregate_reference",
                "used_in_ranking": False,
                "used_in_matching": r == "pan_arctic",
                "used_in_evidence_matrix": False,
                "used_in_mechanism_classification": False,
                "notes": "aggregate/reference row; excluded from rankings and mechanism classes",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "region_taxonomy_check.csv", index=False)
    md = [
        "# Region taxonomy check",
        "",
        "- Physical rows are NSIDC-0780 official Arctic sea-ice regions.",
        "- `pan_arctic` and `pan_arctic_reference` are aggregate/reference rows.",
        "- Custom/legacy regions such as `barents_extended` and `central_basin` are excluded from the formal taxonomy.",
    ]
    (out / "region_taxonomy_check.md").write_text("\n".join(md), encoding="utf-8")
    print(f"WROTE {out}")


def select_major_severe_events(severe: pd.DataFrame) -> dict[str, object]:
    """Apply the canonical major_severe hierarchy without file-system side effects."""
    df = severe.copy()
    for c in ["cumulative_loss", "max_area_km2", "max_intensity", "duration_days"]:
        df[f"global_rank_{c}"] = df[c].rank(pct=True)
        df[f"region_rank_{c}"] = df.groupby("dominant_region")[c].transform(lambda s: s.rank(pct=True))
    df["severity_index_global"] = df[[f"global_rank_{c}" for c in ["cumulative_loss", "max_area_km2", "max_intensity", "duration_days"]]].mean(axis=1)
    df["severity_index_region"] = df[[f"region_rank_{c}" for c in ["cumulative_loss", "max_area_km2", "max_intensity", "duration_days"]]].mean(axis=1)
    outputs, sens = [], []
    global_frames, region_frames, index_frames = [], [], []
    for top in [0.10, 0.15]:
        loss_cut = df["cumulative_loss"].quantile(1 - top)
        area_cut = df["max_area_km2"].quantile(0.80)
        tmp = df[(df["cumulative_loss"] >= loss_cut) & (df["max_area_km2"] >= area_cut) & (df["duration_days"] >= 2)].copy()
        tmp["major_definition"] = f"global_loss_top{int(top*100)}_area_top20_duration2"
        global_frames.append(tmp)
        sens.append({"family": "global", "threshold": top, "event_count": len(tmp)})
        mask = (df["region_rank_cumulative_loss"] >= 1 - top) & (df["region_rank_max_area_km2"] >= 0.80) & (df["duration_days"] >= 2)
        tmp = df[mask].copy()
        tmp["major_definition"] = f"region_loss_top{int(top*100)}_area_top20_duration2"
        region_frames.append(tmp)
        sens.append({"family": "region_normalized", "threshold": top, "event_count": len(tmp)})
    for top in [0.10, 0.15, 0.20]:
        tmp = df[df["severity_index_global"] >= df["severity_index_global"].quantile(1 - top)].copy()
        tmp["major_definition"] = f"severity_index_top{int(top*100)}"
        tmp["global_severity_rank"] = tmp["severity_index_global"].rank(ascending=False, method="first")
        tmp["region_normalized_severity_rank"] = tmp.groupby("dominant_region")["severity_index_region"].rank(ascending=False, method="first")
        index_frames.append(tmp)
        sens.append({"family": "severity_index", "threshold": top, "event_count": len(tmp)})
    global_df = pd.concat(global_frames).drop_duplicates(["unique_local_event_id", "major_definition"])
    region_df = pd.concat(region_frames).drop_duplicates(["unique_local_event_id", "major_definition"])
    index_df = pd.concat(index_frames).drop_duplicates(["unique_local_event_id", "major_definition"])
    union = pd.concat([global_df, region_df, index_df]).drop_duplicates("unique_local_event_id")
    return {
        "ranked_severe": df,
        "global": global_df,
        "region_normalized": region_df,
        "severity_index": index_df,
        "union": union,
        "sensitivity": pd.DataFrame(sens),
    }


def stage3_major(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("local_vrile_major_severe")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    severe = load_severe(args.severe_events)
    if args.quick:
        severe = severe.sample(min(300, len(severe)), random_state=args.seed)
    selected = select_major_severe_events(severe)
    df = selected["ranked_severe"]
    global_df = selected["global"]
    region_df = selected["region_normalized"]
    index_df = selected["severity_index"]
    union = selected["union"]
    sensitivity = selected["sensitivity"]
    global_df.to_csv(out / "major_severe_events_global.csv", index=False)
    region_df.to_csv(out / "major_severe_events_region_normalized.csv", index=False)
    index_df.to_csv(out / "major_severe_events_severity_index.csv", index=False)
    union.to_csv(out / "major_severe_events_union.csv", index=False)
    broad = pd.read_csv(analysis_path(args.broad_summary))
    severe_summary = pd.read_csv(analysis_path(args.severe_summary))
    maj = union.groupby("dominant_region").agg(major_event_count=("unique_local_event_id", "nunique"), median_cumulative_loss=("cumulative_loss", "median"), median_max_area_km2=("max_area_km2", "median"), median_duration_days=("duration_days", "median")).reset_index().rename(columns={"dominant_region": "region"})
    summary = severe_summary[severe_summary["region"].isin(PHYSICAL_REGIONS)].merge(maj, on="region", how="left").fillna(0)
    summary["major_fraction_of_severe"] = summary["major_event_count"] / summary["severe_event_count"].replace(0, np.nan)
    summary.to_csv(out / "major_severe_event_region_summary.csv", index=False)
    sensitivity.to_csv(out / "major_severe_sensitivity.csv", index=False)
    plot_major_severe_region_counts(summary, fdir / "broad_vs_severe_vs_major_severe_by_region.png")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(df["severity_index_global"], bins=30, color="#4C78A8", alpha=0.8)
    ax.set_xlabel("Global severity index"); ax.set_ylabel("Severe event count"); ax.set_title("major_severe event severity distribution")
    fig.savefig(fdir / "major_severe_event_severity_distribution.png", dpi=220); plt.close(fig)
    print(f"WROTE {out}")


def cluster_events(events, gap_days=5, distance_km=500):
    if events.empty:
        return pd.DataFrame()
    rows, active, cid = [], [], 1
    for r in events.sort_values(["event_date", "dominant_region"]).to_dict("records"):
        date = pd.Timestamp(r["event_date"])
        still = []
        for c in active:
            if (date - c["last_date"]).days <= gap_days:
                still.append(c)
            else:
                rows.append(c)
        active = still
        best, bestd = None, np.inf
        for i, c in enumerate(active):
            if not region_match(c["dominant_region"], r["dominant_region"]):
                continue
            d = haversine_km(c["centroid_lon"], c["centroid_lat"], r["track_centroid_lon"], r["track_centroid_lat"])
            if np.isfinite(d) and d <= distance_km and d < bestd:
                best, bestd = i, d
        if best is None:
            active.append({"cluster_id": f"CL{cid:05d}", "dominant_region": r["dominant_region"], "rows": [r], "last_date": date, "centroid_lon": r["track_centroid_lon"], "centroid_lat": r["track_centroid_lat"]})
            cid += 1
        else:
            c = active[best]; c["rows"].append(r); c["last_date"] = date
            g = pd.DataFrame(c["rows"]); w = g["cumulative_loss"].clip(lower=1)
            c["centroid_lon"] = float(np.average(g["track_centroid_lon"], weights=w))
            c["centroid_lat"] = float(np.average(g["track_centroid_lat"], weights=w))
    rows.extend(active)
    out = []
    for c in rows:
        g = pd.DataFrame(c["rows"])
        start, end = pd.to_datetime(g["event_start"]).min(), pd.to_datetime(g["event_end"]).max()
        loss = pd.to_numeric(g["cumulative_loss"], errors="coerce")
        intensity = pd.to_numeric(g["max_intensity"], errors="coerce")
        if not loss.notna().any():
            raise ValueError(f"{c['cluster_id']}: no valid cumulative_loss")
        peak_loss_idx = loss.idxmax()
        peak_intensity_idx = intensity.idxmax() if intensity.notna().any() else peak_loss_idx
        peak_member = g.loc[peak_loss_idx]
        peak_loss_date = pd.to_datetime(g.loc[peak_loss_idx, "event_date"])
        peak_intensity_date = pd.to_datetime(g.loc[peak_intensity_idx, "event_date"])
        representative_date = peak_loss_date
        representative_date_method = "peak_cumulative_loss_member_event_date"
        cluster_centroid_lon = float(c["centroid_lon"])
        cluster_centroid_lat = float(c["centroid_lat"])
        representative_region = str(peak_member["dominant_region"])
        region_assignment = cluster_loss_dominant_region(g, representative_region)
        out.append({"cluster_id": c["cluster_id"], "dominant_region": region_assignment["loss_dominant_region"], "legacy_first_member_region": str(c["dominant_region"]), "loss_dominant_region_fraction": region_assignment["loss_dominant_region_fraction"], "cluster_region_assignment_method": "member_cumulative_loss_summed_by_region", "cluster_region_tie_count": region_assignment["tie_count"], "cluster_region_tie_break": region_assignment["tie_break"], "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"), "duration_days": int((end-start).days+1), "member_event_count": len(g), "total_cumulative_loss": float(g["cumulative_loss"].sum()), "max_area": float(g["max_area_km2"].max()), "max_intensity": float(g["max_intensity"].max()), "peak_loss_date": peak_loss_date.strftime("%Y-%m-%d"), "peak_intensity_date": peak_intensity_date.strftime("%Y-%m-%d"), "representative_date": representative_date.strftime("%Y-%m-%d"), "representative_date_method": representative_date_method, "representative_event_id": str(peak_member["unique_local_event_id"]), "representative_region": representative_region, "representative_lon": float(peak_member["track_centroid_lon"]), "representative_lat": float(peak_member["track_centroid_lat"]), "centroid_lon": cluster_centroid_lon, "centroid_lat": cluster_centroid_lat, "cluster_centroid_lon": cluster_centroid_lon, "cluster_centroid_lat": cluster_centroid_lat, "member_event_ids": ";".join(g["unique_local_event_id"].astype(str)), "matches_regional": bool(g["overlap_with_regional_vrile_v1"].any()), "matches_panarctic": bool(g["overlap_with_panarctic_vrile"].any())})
    return pd.DataFrame(out)


def cluster_loss_dominant_region(members, representative_region):
    loss = pd.to_numeric(members["cumulative_loss"], errors="coerce").to_numpy(float)
    if not np.isfinite(loss).all():
        raise ValueError("HOLD_NONFINITE_LOSS: cluster cumulative_loss contains NaN/Inf")
    if np.all(loss >= 0):
        weights = loss
    elif np.all(loss <= 0):
        weights = -loss
    else:
        raise ValueError("HOLD_MIXED_LOSS_SIGN: cluster cumulative_loss contains both signs")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("HOLD_ZERO_CLUSTER_LOSS: cluster total loss weight is not positive")
    weighted = members[["dominant_region"]].copy()
    weighted["loss_weight"] = weights
    region_weights = weighted.groupby("dominant_region", sort=False)["loss_weight"].sum()
    maximum = float(region_weights.max())
    tied = [
        str(region)
        for region, value in region_weights.items()
        if np.isclose(value, maximum, rtol=1e-12, atol=1e-12 * max(1.0, abs(maximum)))
    ]
    if len(tied) == 1:
        selected, tie_break = tied[0], "unique_maximum"
    elif representative_region in tied:
        selected, tie_break = representative_region, "representative_region"
    else:
        order = {region: index for index, region in enumerate(PHYSICAL_REGIONS)}
        unknown = [region for region in tied if region not in order]
        if unknown:
            raise ValueError(f"Cluster regions absent from PHYSICAL_REGIONS: {unknown}")
        selected, tie_break = min(tied, key=order.__getitem__), "physical_regions_order"
    return {
        "loss_dominant_region": selected,
        "loss_dominant_region_fraction": maximum / total,
        "tie_count": len(tied),
        "tie_break": tie_break,
        "region_weights": region_weights,
        "total_loss_weight": total,
    }


def cluster_region_audit_tables(event_level, clusters, events):
    lookup = events.set_index("unique_local_event_id", drop=False)
    audit_rows, breakdown_rows = [], []
    for cluster in clusters.to_dict("records"):
        member_ids = [item for item in str(cluster["member_event_ids"]).split(";") if item]
        if len(member_ids) != len(set(member_ids)) or len(member_ids) != int(cluster["member_event_count"]):
            raise ValueError(f"HOLD_MEMBER_RECONSTRUCTION: {event_level} {cluster['cluster_id']}")
        missing = [member for member in member_ids if member not in lookup.index]
        if missing:
            raise ValueError(f"HOLD_MEMBER_RECONSTRUCTION: missing members for {event_level} {cluster['cluster_id']}")
        members = lookup.loc[member_ids]
        if isinstance(members, pd.Series):
            members = members.to_frame().T
        assignment = cluster_loss_dominant_region(members, str(cluster["representative_region"]))
        if assignment["loss_dominant_region"] != cluster["dominant_region"]:
            raise ValueError(f"Cluster region reconstruction mismatch: {event_level} {cluster['cluster_id']}")
        audit_rows.append({
            "event_level": event_level,
            "cluster_id": cluster["cluster_id"],
            "member_event_count": len(member_ids),
            "legacy_first_member_region": cluster["legacy_first_member_region"],
            "representative_region": cluster["representative_region"],
            "loss_dominant_region": cluster["dominant_region"],
            "loss_dominant_region_fraction": cluster["loss_dominant_region_fraction"],
            "n_member_regions": int(len(assignment["region_weights"])),
            "tie_count": assignment["tie_count"],
            "tie_break": assignment["tie_break"],
            "legacy_differs_from_loss_dominant": cluster["legacy_first_member_region"] != cluster["dominant_region"],
            "representative_differs_from_loss_dominant": cluster["representative_region"] != cluster["dominant_region"],
        })
        for member_region, region_loss_weight in assignment["region_weights"].items():
            breakdown_rows.append({
                "event_level": event_level,
                "cluster_id": cluster["cluster_id"],
                "member_region": member_region,
                "member_count": int((members["dominant_region"] == member_region).sum()),
                "region_loss_weight": float(region_loss_weight),
                "region_loss_fraction": float(region_loss_weight / assignment["total_loss_weight"]),
                "is_loss_dominant_region": member_region == cluster["dominant_region"],
            })
    return pd.DataFrame(audit_rows), pd.DataFrame(breakdown_rows)


def stage4_clusters(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("event_clusters")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    severe = load_severe(args.severe_events); major = pd.read_csv(analysis_path(args.major_events))
    for d in [severe, major]:
        d["event_start"] = pd.to_datetime(d["event_start"]); d["event_end"] = pd.to_datetime(d["event_end"]); d["event_date"] = d["event_start"]
    sc, mc = cluster_events(severe), cluster_events(major)
    sc.to_csv(out / "synoptic_event_clusters_severe.csv", index=False)
    mc.to_csv(out / "synoptic_event_clusters_major.csv", index=False)
    region_audit_severe, region_breakdown_severe = cluster_region_audit_tables("severe", sc, severe)
    region_audit_major, region_breakdown_major = cluster_region_audit_tables("major_severe", mc, major)
    pd.concat([region_audit_severe, region_audit_major], ignore_index=True).to_csv(out / "cluster_region_assignment_audit.csv", index=False)
    pd.concat([region_breakdown_severe, region_breakdown_major], ignore_index=True).to_csv(out / "cluster_region_loss_breakdown.csv", index=False)
    audit = mc[["cluster_id", "dominant_region", "legacy_first_member_region", "loss_dominant_region_fraction", "cluster_region_assignment_method", "cluster_region_tie_count", "cluster_region_tie_break", "member_event_count", "start_date", "peak_loss_date", "peak_intensity_date", "representative_date", "representative_date_method", "representative_event_id", "representative_region", "representative_lon", "representative_lat", "cluster_centroid_lon", "cluster_centroid_lat"]].copy()
    if not audit.empty:
        audit["representative_minus_start_days"] = (pd.to_datetime(audit["representative_date"]) - pd.to_datetime(audit["start_date"])).dt.days
        audit["peak_intensity_minus_peak_loss_days"] = (pd.to_datetime(audit["peak_intensity_date"]) - pd.to_datetime(audit["peak_loss_date"])).dt.days
        audit["representative_to_cluster_centroid_km"] = audit.apply(lambda r: haversine_km(r["representative_lon"], r["representative_lat"], r["cluster_centroid_lon"], r["cluster_centroid_lat"]), axis=1)
    audit.to_csv(out / "cluster_representative_date_audit.csv", index=False)
    summary = pd.concat([sc.groupby("dominant_region").size().rename("severe_cluster_count"), mc.groupby("dominant_region").size().rename("major_cluster_count")], axis=1).fillna(0).reset_index().rename(columns={"dominant_region": "region"})
    summary.to_csv(out / "cluster_region_summary.csv", index=False)
    sens = []
    for gap in [3,5,7]:
        for dist in [300,500,700]:
            c = cluster_events(severe, gap, dist)
            sens.append({"event_level":"severe","gap_days":gap,"distance_km":dist,"cluster_count":len(c),"multi_event_cluster_fraction":float((c["member_event_count"]>1).mean()) if len(c) else np.nan})
    pd.DataFrame(sens).to_csv(out / "cluster_sensitivity.csv", index=False)
    plot_cluster_region_counts(summary, fdir / "event_vs_cluster_counts_by_region.png")
    print(f"WROTE {out}")


def match_one(src, tgt, tol, require_region=False, dist_km=None):
    matched, dists = 0, []
    for ev in src.to_dict("records"):
        t = tgt[(pd.to_datetime(tgt["event_date"]) - pd.Timestamp(ev["event_date"])).abs().dt.days <= tol].copy()
        if require_region:
            t = t[t["region"].apply(lambda r: region_match(ev["region"], r))]
        if dist_km is not None:
            if pd.isna(ev["centroid_lon"]):
                t = t.iloc[0:0]
            else:
                dd = t.apply(lambda r: haversine_km(ev["centroid_lon"], ev["centroid_lat"], r["centroid_lon"], r["centroid_lat"]), axis=1)
                t = t[dd <= dist_km]
                if len(t):
                    dists.append(float(dd.loc[t.index].min()))
        if len(t):
            matched += 1
    return matched, dists


def stage5_matching(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("spatiotemporal_matching")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    display_definition = {
        "major": "major_severe",
        "major_cluster": "major_severe_cluster",
        "regional": "regional_response_diagnostic",
    }
    defs = {k: event_frame(k, args) for k in ["panarctic","regional","severe","major","severe_cluster","major_cluster"]}
    pairs = [("panarctic","regional"),("regional","panarctic"),("panarctic","severe"),("severe","panarctic"),("panarctic","major"),("major","panarctic"),("panarctic","severe_cluster"),("severe_cluster","panarctic"),("regional","severe"),("severe","regional"),("regional","major"),("major","regional"),("regional","severe_cluster"),("severe_cluster","regional")]
    rows = []
    for s,t in pairs:
        for tol in [0,1,3]:
            for rule, rr, dk in [("time-only",False,None),("time+region",True,None),("time+centroid_300km",False,300),("time+centroid_500km",False,500)]:
                m,dists = match_one(defs[s], defs[t], tol, rr, dk)
                rows.append({"source_definition":display_definition.get(s,s),"target_definition":display_definition.get(t,t),"matching_rule":f"{rule}_pm{tol}d","numerator":m,"denominator":len(defs[s]),"ratio":m/len(defs[s]) if len(defs[s]) else np.nan,"median_distance_km":float(np.nanmedian(dists)) if dists else np.nan,"p90_distance_km":float(np.nanpercentile(dists,90)) if dists else np.nan,"region":"all","notes":"" if not dk or (defs[s]["centroid_lon"].notna().any() and defs[t]["centroid_lon"].notna().any()) else "location unavailable for regional response diagnostic"})
        rows.append({"source_definition":display_definition.get(s,s),"target_definition":display_definition.get(t,t),"matching_rule":"object_overlap_IoU","numerator":0,"denominator":len(defs[s]),"ratio":np.nan,"median_distance_km":np.nan,"p90_distance_km":np.nan,"region":"all","notes":"infeasible: object masks/polygons unavailable"})
    df = pd.DataFrame(rows)
    df["analysis_role"] = np.where(
        (df["source_definition"] == "regional_response_diagnostic")
        | (df["target_definition"] == "regional_response_diagnostic"),
        "regional_response_sensitivity",
        "event_definition_comparison",
    )
    df["scored_in_stage2_evidence"] = False
    df.to_csv(out/"panarctic_regional_severe_major_cluster_spatiotemporal_match.csv",index=False)
    df.to_csv(out/"matching_rule_sensitivity.csv",index=False)
    major = defs["major"]; pan = defs["panarctic"]
    missed = []
    for ev in major.to_dict("records"):
        t = pan[(pd.to_datetime(pan["event_date"]) - pd.Timestamp(ev["event_date"])).abs().dt.days <= 3]
        if t.empty:
            missed.append(ev)
    pd.DataFrame(missed).to_csv(out/"missed_major_severe_events.csv",index=False)
    dist = df[df["median_distance_km"].notna()].copy()
    dist.to_csv(out/"match_distance_distribution.csv",index=False)
    p = df[(df["region"]=="all") & df["matching_rule"].str.contains("pm3d") & df["source_definition"].isin(["panarctic","major_severe","regional_response_diagnostic"])]
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    ax.bar(np.arange(len(p)),p["ratio"],color="#4C78A8")
    ax.set_xticks(np.arange(len(p))); ax.set_xticklabels((p["source_definition"]+"->"+p["target_definition"]+"\n"+p["matching_rule"]),rotation=75,ha="right",fontsize=7)
    ax.set_ylabel("Match ratio"); ax.set_title("Bidirectional spatiotemporal matching")
    ax.grid(axis="y",color="#E6E6E6"); fig.savefig(fdir/"spatiotemporal_match_matrix.png",dpi=220); plt.close(fig)
    if not dist.empty:
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
        ax.scatter(dist["median_distance_km"],dist["p90_distance_km"],alpha=0.6); ax.axhline(0,color="#BBBBBB",ls="--",lw=.8); ax.axvline(0,color="#BBBBBB",ls="--",lw=.8)
        ax.set_xlabel("Median distance (km)"); ax.set_ylabel("P90 distance (km)"); ax.set_title("Match distance distribution")
        fig.savefig(fdir/"match_distance_distribution.png",dpi=220); plt.close(fig)
    if missed:
        mm = pd.DataFrame(missed)
        fig, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
        ax.scatter(mm["centroid_lon"],mm["centroid_lat"],s=18,alpha=.6,color="#D95F02"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("major_severe events missed by pan-Arctic VRILE")
        ax.grid(color="#E6E6E6"); fig.savefig(fdir/"missed_major_severe_events_map.png",dpi=220); plt.close(fig)
    print(f"WROTE {out}")


@lru_cache(maxsize=1)
def sic_grid():
    sample = next((root()/"data/raw/nsidc_sic").glob("1989/*.nc"))
    ds = xr.open_dataset(sample)
    crs = CRS.from_proj4(ds.crs.attrs["proj4text"])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return ds["x"].values, ds["y"].values, transformer


def sic_file(date, sic_root: str | Path = "data/raw/nsidc_sic"):
    d = pd.Timestamp(date)
    files = list((analysis_path(sic_root)/str(d.year)).glob(f"*{d.strftime('%Y%m%d')}*.nc"))
    return files[0] if files else None


def sic_sample_file(sic_root: str | Path = "data/raw/nsidc_sic") -> Path:
    files = sorted(analysis_path(sic_root).glob("*/*.nc"))
    if not files:
        raise FileNotFoundError(f"missing SIC NetCDF sample under {sic_root}")
    return files[0]


def surface_mask_file() -> Path:
    return root() / "data/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def crs_raw_attrs(ds: xr.Dataset) -> str:
    attrs = {k: json_safe(v) for k, v in ds["crs"].attrs.items()} if "crs" in ds else {}
    return json.dumps(attrs, sort_keys=True, ensure_ascii=True)


def crs_candidates_from_dataset(ds: xr.Dataset) -> list[dict[str, str]]:
    if "crs" not in ds:
        return []
    attrs = ds["crs"].attrs
    candidates = []
    for key in ("spatial_ref", "crs_wkt", "proj4text"):
        value = attrs.get(key)
        if not value:
            continue
        try:
            crs = CRS.from_proj4(value) if key == "proj4text" else CRS.from_user_input(value)
            epsg = crs.to_epsg()
            candidates.append({"source": key, "identifier": f"EPSG:{epsg}" if epsg else crs.name})
        except Exception as exc:
            candidates.append({"source": key, "identifier": f"unparsed:{type(exc).__name__}"})
    try:
        crs = CRS.from_cf(attrs)
        epsg = crs.to_epsg()
        candidates.append({"source": "cf_attrs", "identifier": f"EPSG:{epsg}" if epsg else crs.name})
    except Exception as exc:
        candidates.append({"source": "cf_attrs", "identifier": f"unparsed:{type(exc).__name__}"})
    return candidates


def crs_from_dataset(ds: xr.Dataset) -> CRS | None:
    if "crs" not in ds:
        return None
    attrs = ds["crs"].attrs
    for key in ("spatial_ref", "crs_wkt", "proj4text"):
        value = attrs.get(key)
        if value:
            try:
                if key == "proj4text":
                    return CRS.from_proj4(value)
                return CRS.from_user_input(value)
            except Exception:
                continue
    try:
        return CRS.from_cf(attrs)
    except Exception:
        return None


def crs_identifier(ds: xr.Dataset) -> str:
    crs = crs_from_dataset(ds)
    if crs is not None:
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.name
    if "crs" in ds:
        attrs = ds["crs"].attrs
        return str(attrs.get("srid") or attrs.get("long_name") or "unknown_crs")
    return "missing_crs"


def crs_exact_equal(a: CRS | None, b: CRS | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return bool(a.equals(b, ignore_axis_order=True))
    except TypeError:
        return bool(a == b)


def sampled_grid_geolocation_delta_m(x: np.ndarray, y: np.ndarray, sic_crs: CRS, mask_crs: CRS) -> float:
    xi = sorted(set([0, len(x) - 1, len(x) // 2] + np.linspace(0, len(x) - 1, 5, dtype=int).tolist()))
    yi = sorted(set([0, len(y) - 1, len(y) // 2] + np.linspace(0, len(y) - 1, 5, dtype=int).tolist()))
    xx, yy = np.meshgrid(np.asarray(x, dtype=float)[xi], np.asarray(y, dtype=float)[yi])
    sic_to_ll = Transformer.from_crs(sic_crs, "EPSG:4326", always_xy=True)
    mask_to_ll = Transformer.from_crs(mask_crs, "EPSG:4326", always_xy=True)
    lon_s, lat_s = sic_to_ll.transform(xx.ravel(), yy.ravel())
    lon_m, lat_m = mask_to_ll.transform(xx.ravel(), yy.ravel())
    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(lon_s, lat_s, lon_m, lat_m)
    return float(np.nanmax(dist))


@lru_cache(maxsize=128)
def open_sic(path):
    return xr.open_dataset(path)


@lru_cache(maxsize=16)
def open_era(year, era5_single_glob="data/raw/era5/single_levels/*.nc"):
    files = sorted(analysis_path(".").glob(str(era5_single_glob)) if not Path(era5_single_glob).is_absolute() else Path("/").glob(str(Path(era5_single_glob).relative_to("/"))))
    year_files = [p for p in files if str(year) in p.name]
    if not year_files:
        raise FileNotFoundError(f"missing ERA5 single-level file for {year}: {era5_single_glob}")
    return xr.open_dataset(year_files[0])


@lru_cache(maxsize=1)
def sic_lonlat_grid():
    sample = next((root() / "data/raw/nsidc_sic").glob("1989/*.nc"))
    ds = xr.open_dataset(sample)
    crs = CRS.from_proj4(ds.crs.attrs["proj4text"])
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)
    return x, y, lon, lat, transformer


@lru_cache(maxsize=1)
def nsidc_surface_mask():
    ds = xr.open_dataset(surface_mask_file())
    return ds["sea_ice_region_surface_mask"].values.astype(float)


def grid_monotonicity(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    xdiff = np.diff(x.astype(float))
    ydiff = np.diff(y.astype(float))
    return {
        "x_first": float(x[0]),
        "x_last": float(x[-1]),
        "x_step": float(x[1] - x[0]) if len(x) > 1 else np.nan,
        "x_monotonic_increasing": bool(np.all(xdiff > 0)),
        "x_monotonic_decreasing": bool(np.all(xdiff < 0)),
        "y_first": float(y[0]),
        "y_last": float(y[-1]),
        "y_step": float(y[1] - y[0]) if len(y) > 1 else np.nan,
        "y_monotonic_increasing": bool(np.all(ydiff > 0)),
        "y_monotonic_decreasing": bool(np.all(ydiff < 0)),
    }


def assert_strict_xy_monotonic(x: np.ndarray, y: np.ndarray, label: str = "grid") -> None:
    m = grid_monotonicity(np.asarray(x), np.asarray(y))
    x_ok = m["x_monotonic_increasing"] or m["x_monotonic_decreasing"]
    y_ok = m["y_monotonic_increasing"] or m["y_monotonic_decreasing"]
    if not x_ok or not y_ok:
        raise ValueError(f"{label} x/y coordinates must be strictly monotonic")


@lru_cache(maxsize=8)
def nsidc_surface_mask_grid_audit(sic_root: str | Path = "data/raw/nsidc_sic") -> pd.DataFrame:
    sic_ds = xr.open_dataset(sic_sample_file(sic_root))
    mask_ds = xr.open_dataset(surface_mask_file())
    sic = sic_ds["cdr_seaice_conc"].isel(time=0)
    mask = mask_ds["sea_ice_region_surface_mask"]
    sic_x = sic_ds["x"].values.astype(float)
    sic_y = sic_ds["y"].values.astype(float)
    mask_x = mask_ds["x"].values.astype(float)
    mask_y = mask_ds["y"].values.astype(float)
    shape_equal = tuple(sic.shape) == tuple(mask.shape)
    x_size_equal = len(sic_x) == len(mask_x)
    y_size_equal = len(sic_y) == len(mask_y)
    x_allclose = bool(x_size_equal and np.allclose(sic_x, mask_x, equal_nan=True))
    y_allclose = bool(y_size_equal and np.allclose(sic_y, mask_y, equal_nan=True))
    max_abs_x_diff = float(np.nanmax(np.abs(sic_x - mask_x))) if x_size_equal else np.nan
    max_abs_y_diff = float(np.nanmax(np.abs(sic_y - mask_y))) if y_size_equal else np.nan
    sic_crs = crs_from_dataset(sic_ds)
    mask_crs = crs_from_dataset(mask_ds)
    grid_index_equal = bool(shape_equal and x_allclose and y_allclose)
    exact_equal = crs_exact_equal(sic_crs, mask_crs)
    step = min(abs(float(sic_x[1] - sic_x[0])), abs(float(sic_y[1] - sic_y[0]))) if len(sic_x) > 1 and len(sic_y) > 1 else np.nan
    threshold = float(min(1000.0, 0.05 * step)) if np.isfinite(step) else np.nan
    max_geo_delta = sampled_grid_geolocation_delta_m(sic_x, sic_y, sic_crs, mask_crs) if sic_crs is not None and mask_crs is not None and grid_index_equal else np.nan
    crs_compatible = bool(sic_crs is not None and mask_crs is not None and (exact_equal or (np.isfinite(max_geo_delta) and max_geo_delta <= threshold)))
    if exact_equal:
        alignment_status = "exact_crs_match"
    elif grid_index_equal and crs_compatible:
        alignment_status = "compatible_crs_metadata_difference"
    else:
        alignment_status = "incompatible_grid_or_crs"
    row = {
        "sic_y_size": int(sic.shape[0]),
        "sic_x_size": int(sic.shape[1]),
        "mask_y_size": int(mask.shape[0]),
        "mask_x_size": int(mask.shape[1]),
        "shape_equal": bool(shape_equal),
        "sic_x_size_coord": int(len(sic_x)),
        "sic_y_size_coord": int(len(sic_y)),
        "mask_x_size_coord": int(len(mask_x)),
        "mask_y_size_coord": int(len(mask_y)),
        "x_allclose": x_allclose,
        "y_allclose": y_allclose,
        "max_abs_x_diff": max_abs_x_diff,
        "max_abs_y_diff": max_abs_y_diff,
        "sic_crs_identifier": crs_identifier(sic_ds),
        "mask_crs_identifier": crs_identifier(mask_ds),
        "crs_exact_equal": exact_equal,
        "sic_crs_candidates": json.dumps(crs_candidates_from_dataset(sic_ds), ensure_ascii=True),
        "mask_crs_candidates": json.dumps(crs_candidates_from_dataset(mask_ds), ensure_ascii=True),
        "sic_crs_raw_attrs": crs_raw_attrs(sic_ds),
        "mask_crs_raw_attrs": crs_raw_attrs(mask_ds),
        "grid_index_equal": grid_index_equal,
        "max_geolocation_delta_m": max_geo_delta,
        "compatibility_threshold_m": threshold,
        "crs_compatible": crs_compatible,
        "alignment_status": alignment_status,
    }
    row["grid_alignment_pass"] = bool(grid_index_equal and crs_compatible)
    return pd.DataFrame([row])


def write_surface_mask_grid_audit(out: Path, sic_root: str | Path = "data/raw/nsidc_sic") -> pd.DataFrame:
    df = nsidc_surface_mask_grid_audit(str(sic_root))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nsidc_surface_mask_grid_audit.csv", index=False)
    if not bool(df.loc[0, "grid_alignment_pass"]):
        raise ValueError("SIC grid and NSIDC-0780 surface mask grid/CRS are not aligned; see nsidc_surface_mask_grid_audit.csv")
    if df.loc[0, "alignment_status"] == "compatible_crs_metadata_difference":
        print(
            "WARNING: SIC and NSIDC-0780 CRS metadata are not exactly equal, "
            f"but sampled geolocation delta ({df.loc[0, 'max_geolocation_delta_m']:.3f} m) "
            f"is within threshold ({df.loc[0, 'compatibility_threshold_m']:.3f} m); direct mask slicing allowed."
        )
    return df


def build_wrong_region_reference_points(out: Path, sic_root: str | Path) -> pd.DataFrame:
    """Build one fixed, mask-backed projected medoid for each focus region."""
    grid_audit = write_surface_mask_grid_audit(out, sic_root)
    alignment_status = str(grid_audit.loc[0, "alignment_status"])
    with xr.open_dataset(surface_mask_file(), mask_and_scale=False) as ds:
        if "sea_ice_region_surface_mask" not in ds:
            raise KeyError("sea_ice_region_surface_mask missing from NSIDC-0780 mask")
        surface = np.asarray(ds["sea_ice_region_surface_mask"].values, dtype=float)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        mask_crs = crs_from_dataset(ds)
    if mask_crs is None:
        raise ValueError("NSIDC-0780 mask CRS is not parseable")
    if surface.shape != (len(y), len(x)):
        raise ValueError(
            f"NSIDC-0780 mask shape {surface.shape} does not match y/x coordinates {(len(y), len(x))}"
        )
    to_lonlat = Transformer.from_crs(mask_crs, "EPSG:4326", always_xy=True)
    to_equal_area = Transformer.from_crs(mask_crs, "EPSG:6931", always_xy=True)
    rows = []
    for region in FOCUS_REGIONS:
        region_code = int(NSIDC_0780_REGION_IDS[region])
        iy, ix = np.where(surface == region_code)
        if len(iy) == 0:
            raise ValueError(f"NSIDC-0780 region {region} (code {region_code}) has no cells")
        source_x = x[ix]
        source_y = y[iy]
        region_x, region_y = to_equal_area.transform(source_x, source_y)
        region_x = np.asarray(region_x, dtype=float)
        region_y = np.asarray(region_y, dtype=float)
        centroid_x = float(region_x.mean())
        centroid_y = float(region_y.mean())
        medoid_pos = int(np.argmin((region_x - centroid_x) ** 2 + (region_y - centroid_y) ** 2))
        row_idx = int(iy[medoid_pos])
        col_idx = int(ix[medoid_pos])
        reference_grid_x = float(x[col_idx])
        reference_grid_y = float(y[row_idx])
        reference_x = float(region_x[medoid_pos])
        reference_y = float(region_y[medoid_pos])
        reference_lon, reference_lat = to_lonlat.transform(reference_grid_x, reference_grid_y)
        inside = bool(int(surface[row_idx, col_idx]) == region_code)
        if not inside:
            raise ValueError(f"reference medoid for {region} is outside its NSIDC-0780 mask cells")
        rows.append(
            {
                "region": region,
                "region_code": region_code,
                "reference_x_m": reference_x,
                "reference_y_m": reference_y,
                "reference_lon": float(reference_lon),
                "reference_lat": float(reference_lat),
                "method": "nsidc0780_equal_area_cell_medoid_epsg6931",
                "region_cell_count": int(len(iy)),
                "grid_alignment_status": alignment_status,
                "source_grid_x_m": reference_grid_x,
                "source_grid_y_m": reference_grid_y,
                "reference_row": row_idx,
                "reference_col": col_idx,
                "reference_inside_region_cell": inside,
            }
        )
    refs = pd.DataFrame(rows)
    if len(refs) != len(FOCUS_REGIONS) or set(refs["region"]) != set(FOCUS_REGIONS):
        raise ValueError("wrong-region reference-point coverage is not exactly 5/5 focus regions")
    refs.to_csv(out / "wrong_region_reference_points.csv", index=False)
    return refs


def write_ice_edge_grid_coordinate_audit(out: Path, sic_root: str | Path = "data/raw/nsidc_sic") -> pd.DataFrame:
    ds = xr.open_dataset(sic_sample_file(sic_root))
    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)
    row = grid_monotonicity(x, y)
    df = pd.DataFrame([row])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ice_edge_grid_coordinate_audit.csv", index=False)
    assert_strict_xy_monotonic(x, y, "SIC sample grid")
    return df


def nsidc_surface_ocean_mask(surf: np.ndarray) -> np.ndarray:
    finite = surf[np.isfinite(surf)]
    observed_codes = set(np.unique(finite).astype(int).tolist())
    known_codes = set(NSIDC_0780_OCEAN_CODES) | set(NSIDC_0780_NON_OCEAN_CODES)
    unknown_codes = observed_codes - known_codes
    if unknown_codes:
        raise ValueError(f"Unknown NSIDC-0780 surface-mask codes: {sorted(unknown_codes)}")
    return np.isin(surf, NSIDC_0780_OCEAN_CODES)


def write_surface_mask_code_audit(out: Path) -> pd.DataFrame:
    surf = nsidc_surface_mask()
    vals, counts = np.unique(surf[np.isfinite(surf)].astype(int), return_counts=True)
    rows = []
    known = set(NSIDC_0780_OCEAN_CODES) | set(NSIDC_0780_NON_OCEAN_CODES)
    for code, count in zip(vals, counts):
        if int(code) in NSIDC_0780_OCEAN_CODES:
            cls = "ocean"
        elif int(code) in NSIDC_0780_NON_OCEAN_CODES:
            cls = "non_ocean"
        else:
            cls = "unknown"
        rows.append({"surface_code": int(code), "cell_count": int(count), "classification": cls})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nsidc_surface_mask_code_audit.csv", index=False)
    unknown = set(df.loc[df["classification"] == "unknown", "surface_code"].astype(int)) - known
    if unknown:
        raise ValueError(f"Unknown NSIDC-0780 surface-mask codes: {sorted(unknown)}")
    return df


def neighbor_count(mask):
    m = mask.astype(int)
    out = np.zeros_like(m, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            out += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0
    return out


def expand_mask(mask, steps):
    out = mask.copy()
    for _ in range(max(int(steps), 0)):
        out |= neighbor_count(out) > 0
    return out


def connected_components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        comp_y, comp_x = [], []
        while stack:
            y0, x0 = stack.pop()
            comp_y.append(y0)
            comp_x.append(x0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        comps.append((np.asarray(comp_y), np.asarray(comp_x)))
    return comps


def era_at_points(date, lons, lats, variables=("u10", "v10"), era5_single_glob="data/raw/era5/single_levels/*.nc"):
    if len(lons) == 0:
        return {v: np.asarray([], dtype=float) for v in variables}
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon360 = ((lons + 180) % 360) - 180
    year = pd.Timestamp(date).year
    for attempt in range(3):
        ds = open_era(year, era5_single_glob)
        try:
            tname = "valid_time"
            target = np.datetime64(pd.Timestamp(date).date())
            tidx = int(np.argmin(np.abs(ds[tname].values.astype("datetime64[D]") - target)))
            lat_vals = ds.latitude.values
            lon_vals = ds.longitude.values
            lat_idx = np.abs(lat_vals[:, None] - lats[None, :]).argmin(axis=0)
            lon_idx = np.abs(((lon_vals[:, None] - lon360[None, :] + 180) % 360) - 180).argmin(axis=0)
            out = {}
            for var in variables:
                arr = ds[var].isel({tname: tidx}).values
                out[var] = arr[lat_idx, lon_idx].astype(float)
            return out
        except RuntimeError as exc:
            if "HDF error" not in str(exc) or attempt == 2:
                raise
            print(
                f"WARNING: transient ERA5 NetCDF/HDF read failure for {year}; "
                f"closing cached handles and retrying ({attempt + 1}/2)",
                file=sys.stderr,
            )
            ds.close()
            open_era.cache_clear()
            time.sleep(1.0)
    raise AssertionError("unreachable ERA5 retry state")


def combine_qa_status(*statuses: str) -> str:
    if not statuses:
        return "reject"
    return max(statuses, key=lambda s: QA_ORDER.get(s, QA_ORDER["reject"]))


def sample_local_grid_nearest(values: np.ndarray, sx: np.ndarray, sy: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    step_x = float(sx[1] - sx[0]) if len(sx) > 1 else np.nan
    step_y = float(sy[1] - sy[0]) if len(sy) > 1 else np.nan
    ix = np.rint((qx - float(sx[0])) / step_x).astype(int) if np.isfinite(step_x) and step_x else np.full(qx.shape, -1)
    iy = np.rint((qy - float(sy[0])) / step_y).astype(int) if np.isfinite(step_y) and step_y else np.full(qy.shape, -1)
    valid = (ix >= 0) & (ix < len(sx)) & (iy >= 0) & (iy < len(sy))
    out = np.full(qx.shape, np.nan, dtype=float)
    out[valid] = values[iy[valid], ix[valid]]
    return out, valid


def difference_matches_direction(value: float, expected: str) -> bool:
    if not np.isfinite(value):
        return False
    if expected == "negative":
        return value < 0
    if expected == "positive":
        return value > 0
    raise ValueError(f"Unsupported expected direction: {expected}")


def status_score(status: str) -> int:
    return {"supported": 3, "not_supported": 1, "insufficient_samples": 1, "weak_or_insufficient": 1, "unavailable": 0}.get(str(status), 0)


def read_status_file_for_regions(path: Path, regions: list[str], evidence_name: str, supported_col: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    warnings = []
    required = {"region", "evidence_status"}
    cols = ["region", "evidence_status", supported_col]
    if not path.exists():
        warnings.append({"evidence_name": evidence_name, "warning": f"missing status file: {path}"})
        return pd.DataFrame([{c: (region if c == "region" else "unavailable" if c == "evidence_status" else "") for c in cols} for region in regions]), warnings
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        warnings.append({"evidence_name": evidence_name, "warning": f"schema mismatch in {path}: missing {sorted(missing)}"})
        return pd.DataFrame([{c: (region if c == "region" else "unavailable" if c == "evidence_status" else "") for c in cols} for region in regions]), warnings
    if supported_col not in df.columns:
        warnings.append({"evidence_name": evidence_name, "warning": f"optional column missing in {path}: {supported_col}"})
        df[supported_col] = ""
    rows = []
    for region in regions:
        sub = df[df["region"] == region]
        if sub.empty:
            warnings.append({"evidence_name": evidence_name, "warning": f"region missing in {path}: {region}"})
            rows.append({"region": region, "evidence_status": "unavailable", supported_col: ""})
        else:
            row = sub.iloc[0]
            rows.append({
                "region": region,
                "evidence_status": row.get("evidence_status", "unavailable"),
                supported_col: row.get(supported_col, ""),
            })
    return pd.DataFrame(rows), warnings


def cyclone_evidence_status(cyclone_df: pd.DataFrame, region: str, min_n: int = 3) -> dict[str, object]:
    required = {"region", "variable", "mean_difference", "p_FDR", "ci_excludes_zero", "n_units"}
    if cyclone_df is None or cyclone_df.empty:
        return {"region": region, "evidence_status": "unavailable", "supported_variable_count": 0, "supported_variables": "", "evaluated_test_count": 0, "max_n_units": 0, "reason": "empty cyclone tests"}
    if missing := required - set(cyclone_df.columns):
        return {"region": region, "evidence_status": "unavailable", "supported_variable_count": 0, "supported_variables": "", "evaluated_test_count": 0, "max_n_units": 0, "reason": f"missing columns: {sorted(missing)}"}
    sub = cyclone_df[cyclone_df["region"] == region].copy()
    if sub.empty:
        return {"region": region, "evidence_status": "unavailable", "supported_variable_count": 0, "supported_variables": "", "evaluated_test_count": 0, "max_n_units": 0, "reason": "region absent"}
    evaluable = sub[
        (pd.to_numeric(sub["n_units"], errors="coerce") >= min_n)
        & pd.to_numeric(sub["p_FDR"], errors="coerce").notna()
        & pd.to_numeric(sub["mean_difference"], errors="coerce").notna()
        & sub["ci_excludes_zero"].notna()
    ].copy()
    max_n = int(pd.to_numeric(sub["n_units"], errors="coerce").max()) if pd.to_numeric(sub["n_units"], errors="coerce").notna().any() else 0
    if evaluable.empty:
        return {"region": region, "evidence_status": "insufficient_samples", "supported_variable_count": 0, "supported_variables": "", "evaluated_test_count": 0, "max_n_units": max_n, "reason": "no tests with n_units >= 3 and computable statistics"}
    supported = []
    for row in evaluable.to_dict("records"):
        var = row.get("variable")
        expected = CYCLONE_EXPECTED_DIRECTION.get(var)
        if expected is None:
            continue
        ok = (
            pd.notna(row.get("p_FDR"))
            and float(row.get("p_FDR")) < 0.05
            and bool(row.get("ci_excludes_zero"))
            and int(row.get("n_units", 0)) >= min_n
            and difference_matches_direction(float(row.get("mean_difference", np.nan)), expected)
        )
        if ok:
            supported.append(str(var))
    supported = sorted(set(supported))
    return {
        "region": region,
        "evidence_status": "supported" if supported else "not_supported",
        "supported_variable_count": len(supported),
        "supported_variables": ";".join(supported),
        "evaluated_test_count": int(len(evaluable)),
        "max_n_units": max_n,
        "reason": "directional statistically supported variables" if supported else "no statistically supported test with expected physical direction",
    }


def edge_wind_evidence_status(edge_tests: pd.DataFrame, region: str) -> dict[str, object]:
    required = {"region", "variable", "lag_group", "analysis_tier", "mean_difference", "mechanism_evidence_status", "n_units"}
    if edge_tests is None or edge_tests.empty:
        return {"region": region, "evidence_status": "unavailable", "supported_lag_groups": "", "supported_test_count": 0, "evaluated_test_count": 0, "max_n_units": 0, "reason": "empty edge-wind tests"}
    if missing := required - set(edge_tests.columns):
        return {"region": region, "evidence_status": "unavailable", "supported_lag_groups": "", "supported_test_count": 0, "evaluated_test_count": 0, "max_n_units": 0, "reason": f"missing columns: {sorted(missing)}"}
    sub = edge_tests[
        (edge_tests["region"] == region)
        & (edge_tests["variable"] == "outward_edge_normal_wind_ms")
        & (edge_tests["analysis_tier"] == "primary")
        & (edge_tests["lag_group"].isin(["pre-event", "event-time"]))
    ].copy()
    if sub.empty:
        return {"region": region, "evidence_status": "unavailable", "supported_lag_groups": "", "supported_test_count": 0, "evaluated_test_count": 0, "max_n_units": 0, "reason": "no primary outward normal tests"}
    evaluable = sub[
        (pd.to_numeric(sub["n_units"], errors="coerce") >= 3)
        & pd.to_numeric(sub["p_FDR"], errors="coerce").notna()
        & pd.to_numeric(sub["mean_difference"], errors="coerce").notna()
        & sub["ci_excludes_zero"].notna()
    ].copy()
    max_n = int(pd.to_numeric(sub["n_units"], errors="coerce").max()) if pd.to_numeric(sub["n_units"], errors="coerce").notna().any() else 0
    if evaluable.empty:
        return {"region": region, "evidence_status": "insufficient_samples", "supported_lag_groups": "", "supported_test_count": 0, "evaluated_test_count": 0, "max_n_units": max_n, "reason": "no primary outward tests with n_units >= 3 and computable statistics"}
    support = evaluable[(evaluable["mean_difference"] > 0) & (evaluable["mechanism_evidence_status"] == "statistically_supported_dynamic_indicator")]
    groups = sorted(support["lag_group"].dropna().astype(str).unique().tolist())
    return {
        "region": region,
        "evidence_status": "supported" if groups else "not_supported",
        "supported_lag_groups": ";".join(groups),
        "supported_test_count": int(len(support)),
        "evaluated_test_count": int(len(evaluable)),
        "max_n_units": max_n,
        "reason": "primary outward edge-normal wind support" if groups else "no primary significant outward-positive pre/event-time support",
    }


def apply_quick_event_subset(events: pd.DataFrame, quick: bool, n: int = 5) -> pd.DataFrame:
    if not quick:
        return events.copy()
    return events.sort_values(["event_date", "event_id"]).head(n).copy()


def apply_quick_cluster_subset(events: pd.DataFrame, quick: bool, per_region: int = 3) -> pd.DataFrame:
    if not quick:
        return events.copy()
    base = events[events["region"].isin(FOCUS_REGIONS)].copy()
    selected = (
        base.sort_values(["region", "event_date", "event_id"])
        .groupby("region", group_keys=False)
        .head(per_region)
    )
    if "cluster_start_date" in base.columns:
        shifted = base[pd.to_datetime(base["event_date"]) != pd.to_datetime(base["cluster_start_date"])]
        if not shifted.empty:
            extra = shifted.sort_values(["event_date", "event_id"]).head(1)
            selected = pd.concat([selected, extra], ignore_index=True)
    return selected.drop_duplicates("event_id").sort_values(["event_date", "event_id"]).copy()


def apply_quick_major_event_subset(events: pd.DataFrame, quick: bool, per_region: int = 5) -> pd.DataFrame:
    if not quick:
        return events.copy()
    return (
        events[events["dominant_region"].isin(FOCUS_REGIONS)]
        .sort_values(["dominant_region", "event_date", "unique_local_event_id"])
        .groupby("dominant_region", group_keys=False)
        .head(per_region)
        .sort_values(["event_date", "unique_local_event_id"])
        .copy()
    )


def write_quick_event_metadata(out: Path, events: pd.DataFrame, quick: bool, module: str, region_col: str = "region", id_col: str = "event_id", date_col: str = "event_date") -> None:
    if not quick:
        return
    rows = []
    for region in FOCUS_REGIONS:
        sub = events[events[region_col] == region] if region_col in events else events.iloc[0:0]
        dates = pd.to_datetime(sub[date_col]) if date_col in sub else pd.Series(dtype="datetime64[ns]")
        rows.append({
            "module": module,
            "region": region,
            "event_count": len(sub),
            "event_ids": ";".join(sub[id_col].astype(str).tolist()) if id_col in sub else "",
            "date_min": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
            "date_max": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
        })
    pd.DataFrame(rows).to_csv(out / "quick_event_subset.csv", index=False)


def extract_local_ice_edge(date, lon, lat, args):
    p = sic_file(date, args.sic_root)
    if p is None or pd.isna(lon) or pd.isna(lat):
        return pd.DataFrame(), {"qa_status": "reject", "qa_reason": "missing_sic_or_location"}
    if not bool(nsidc_surface_mask_grid_audit(str(args.sic_root)).loc[0, "grid_alignment_pass"]):
        raise ValueError("SIC grid and NSIDC-0780 surface mask grid/CRS are not aligned")
    ds = open_sic(str(p))
    sic_full = ds["cdr_seaice_conc"].isel(time=0).values.astype(float)
    sic_full = np.where(sic_full > 1.2, np.nan, sic_full)
    x, y, glon_full, glat_full, inv_transformer = sic_lonlat_grid()
    _, _, to_proj = sic_grid()
    surf_full = nsidc_surface_mask()
    assert_strict_xy_monotonic(x, y, "SIC grid")
    x_step = float(x[1] - x[0])
    y_step = float(y[1] - y[0])
    pad = int(math.ceil(args.ice_edge_search_km * 1000.0 / min(abs(x_step), abs(y_step)))) + args.ice_edge_coastal_cells + 3
    cx, cy = to_proj.transform(float(lon), float(lat))
    ix = int(np.argmin(np.abs(x - cx)))
    iy = int(np.argmin(np.abs(y - cy)))
    y0 = max(0, iy - pad)
    y1 = min(len(y), iy + pad + 1)
    x0 = max(0, ix - pad)
    x1 = min(len(x), ix + pad + 1)
    sic = sic_full[y0:y1, x0:x1]
    surf = surf_full[y0:y1, x0:x1]
    glon = glon_full[y0:y1, x0:x1]
    glat = glat_full[y0:y1, x0:x1]
    sx = x[x0:x1]
    sy = y[y0:y1]
    assert_strict_xy_monotonic(sx, sy, "local SIC grid")
    ocean = nsidc_surface_ocean_mask(surf)
    land_like = ~ocean
    coastal = expand_mask(land_like, args.ice_edge_coastal_cells)
    xx, yy = np.meshgrid(sx, sy)
    local = (xx - cx) ** 2 + (yy - cy) ** 2 <= (args.ice_edge_search_km * 1000.0) ** 2
    ice = (sic >= args.ice_edge_sic_threshold) & ocean
    open_water = (sic < args.ice_edge_sic_threshold) & ocean
    edge = ice & (neighbor_count(open_water) > 0) & local & ~coastal
    comps = connected_components(edge)
    if not comps:
        return pd.DataFrame(), {"qa_status": "reject", "qa_reason": "no_stable_edge_after_ocean_coastal_mask"}
    sizes = np.asarray([len(c[0]) for c in comps])
    order = np.argsort(sizes)[::-1]
    comp_y, comp_x = comps[int(order[0])]
    total_edge_cells = int(sizes.sum())
    component_fraction = float(sizes[order[0]] / total_edge_cells) if total_edge_cells else np.nan
    if len(comp_y) < args.ice_edge_min_cells:
        return pd.DataFrame(), {"qa_status": "reject", "qa_reason": "dominant_edge_component_too_small", "edge_cells": int(len(comp_y)), "component_fraction": component_fraction}
    topology_qa_status = "ok"
    topology_qa_reason = "dominant_main_edge"
    if component_fraction < args.ice_edge_component_fraction_min:
        topology_qa_status = "degraded"
        topology_qa_reason = "fragmented_edge_topology_dominant_component_used"
    gy, gx = np.gradient(sic, sy, sx)
    nx_ps = -gx[comp_y, comp_x]
    ny_ps = -gy[comp_y, comp_x]
    norm = np.hypot(nx_ps, ny_ps)
    ok = np.isfinite(norm) & (norm > 0)
    comp_y, comp_x, nx_ps, ny_ps, norm = comp_y[ok], comp_x[ok], nx_ps[ok], ny_ps[ok], norm[ok]
    candidate_normal_cells = int(len(comp_y))
    if len(comp_y) < args.ice_edge_min_cells:
        return pd.DataFrame(), {"qa_status": "reject", "qa_reason": "normal_orientation_failed", "edge_cells": int(len(comp_y)), "component_fraction": component_fraction}
    nx_ps = nx_ps / norm
    ny_ps = ny_ps / norm
    qdist = float(args.normal_orientation_sic_sampling_distance_m)
    px_center = sx[comp_x]
    py_center = sy[comp_y]
    sic_plus, valid_plus = sample_local_grid_nearest(sic, sx, sy, px_center + nx_ps * qdist, py_center + ny_ps * qdist)
    sic_minus, valid_minus = sample_local_grid_nearest(sic, sx, sy, px_center - nx_ps * qdist, py_center - ny_ps * qdist)
    ocean_plus, valid_ocean_plus = sample_local_grid_nearest(ocean.astype(float), sx, sy, px_center + nx_ps * qdist, py_center + ny_ps * qdist)
    ocean_minus, valid_ocean_minus = sample_local_grid_nearest(ocean.astype(float), sx, sy, px_center - nx_ps * qdist, py_center - ny_ps * qdist)
    valid_orientation = (
        valid_plus
        & valid_minus
        & valid_ocean_plus
        & valid_ocean_minus
        & (ocean_plus > 0.5)
        & (ocean_minus > 0.5)
        & np.isfinite(sic_plus)
        & np.isfinite(sic_minus)
    )
    orientation_total_cells = candidate_normal_cells
    orientation_valid_cells = int(valid_orientation.sum())
    if orientation_total_cells == 0 or orientation_valid_cells == 0:
        qa = {
            "qa_status": "reject",
            "qa_reason": "normal_orientation_failed:no_valid_bilateral_sic_samples",
            "edge_cells": int(len(comp_y)),
            "total_edge_cells": total_edge_cells,
            "component_fraction": component_fraction,
            "orientation_total_cells": orientation_total_cells,
            "orientation_valid_cells": orientation_valid_cells,
            "orientation_strong_cells": 0,
            "orientation_flip_cells": 0,
            "valid_orientation_fraction": 0.0,
            "strong_orientation_fraction": np.nan,
            "flip_fraction": np.nan,
            "orientation_mean_sic_ice_side": np.nan,
            "orientation_mean_sic_low_side": np.nan,
            "orientation_mean_sic_difference": np.nan,
            "orientation_qa_status": "reject",
        }
        return pd.DataFrame(), qa
    orientation_diff = sic_minus - sic_plus
    flip = valid_orientation & (orientation_diff < 0)
    nx_ps[flip] *= -1.0
    ny_ps[flip] *= -1.0
    contrast = np.full_like(orientation_diff, np.nan, dtype=float)
    contrast[valid_orientation] = np.abs(orientation_diff[valid_orientation])
    strong_orientation = valid_orientation & (contrast >= float(args.normal_orientation_min_sic_difference))
    orientation_strong_cells = int(strong_orientation.sum())
    orientation_flip_cells = int(flip.sum())
    valid_orientation_fraction = float(orientation_valid_cells / orientation_total_cells)
    strong_orientation_fraction = float(orientation_strong_cells / orientation_valid_cells)
    flip_fraction = float(orientation_flip_cells / orientation_valid_cells)
    ice_side = np.where(orientation_diff >= 0, sic_minus, sic_plus)
    low_side = np.where(orientation_diff >= 0, sic_plus, sic_minus)
    if valid_orientation_fraction < args.normal_orientation_degraded_fraction or strong_orientation_fraction < args.normal_orientation_degraded_fraction:
        orientation_qa_status = "reject"
        orientation_reason = "weak_orientation_contrast"
    elif strong_orientation_fraction >= args.normal_orientation_ok_fraction:
        orientation_qa_status = "ok"
        orientation_reason = "orientation_ok"
    else:
        orientation_qa_status = "degraded"
        orientation_reason = "orientation_degraded"
    combined_qa_status = combine_qa_status(topology_qa_status, orientation_qa_status)
    qa_reason = f"{topology_qa_reason};{orientation_reason}"
    if combined_qa_status == "reject":
        qa = {
            "qa_status": "reject",
            "qa_reason": qa_reason,
            "edge_cells": int(len(comp_y)),
            "total_edge_cells": total_edge_cells,
            "component_fraction": component_fraction,
            "orientation_total_cells": orientation_total_cells,
            "orientation_valid_cells": orientation_valid_cells,
            "orientation_strong_cells": orientation_strong_cells,
            "orientation_flip_cells": orientation_flip_cells,
            "valid_orientation_fraction": valid_orientation_fraction,
            "strong_orientation_fraction": strong_orientation_fraction,
            "flip_fraction": flip_fraction,
            "orientation_mean_sic_ice_side": float(np.nanmean(np.where(valid_orientation, ice_side, np.nan))),
            "orientation_mean_sic_low_side": float(np.nanmean(np.where(valid_orientation, low_side, np.nan))),
            "orientation_mean_sic_difference": float(np.nanmean(np.where(valid_orientation, contrast, np.nan))),
            "orientation_qa_status": orientation_qa_status,
        }
        return pd.DataFrame(), qa
    keep = valid_orientation
    comp_y, comp_x, nx_ps, ny_ps = comp_y[keep], comp_x[keep], nx_ps[keep], ny_ps[keep]
    px0 = sx[comp_x]
    py0 = sy[comp_y]
    lon2, lat2 = inv_transformer.transform(px0 + nx_ps * 25000.0, py0 + ny_ps * 25000.0)
    east = np.deg2rad(lon2 - glon[comp_y, comp_x]) * 6371000.0 * np.cos(np.deg2rad(glat[comp_y, comp_x]))
    north = np.deg2rad(lat2 - glat[comp_y, comp_x]) * 6371000.0
    en_norm = np.hypot(east, north)
    good = np.isfinite(en_norm) & (en_norm > 0)
    rows = pd.DataFrame({
        "edge_lon": glon[comp_y[good], comp_x[good]],
        "edge_lat": glat[comp_y[good], comp_x[good]],
        "sic_edge": sic[comp_y[good], comp_x[good]],
        "normal_east": east[good] / en_norm[good],
        "normal_north": north[good] / en_norm[good],
    })
    if len(rows) < args.ice_edge_min_cells:
        return pd.DataFrame(), {"qa_status": "reject", "qa_reason": "normal_orientation_failed", "edge_cells": int(len(rows)), "component_fraction": component_fraction}
    qa = {
        "qa_status": combined_qa_status,
        "qa_reason": qa_reason,
        "edge_cells": int(len(rows)),
        "total_edge_cells": total_edge_cells,
        "component_fraction": component_fraction,
        "orientation_total_cells": orientation_total_cells,
        "orientation_valid_cells": orientation_valid_cells,
        "orientation_strong_cells": orientation_strong_cells,
        "orientation_flip_cells": orientation_flip_cells,
        "valid_orientation_fraction": valid_orientation_fraction,
        "strong_orientation_fraction": strong_orientation_fraction,
        "flip_fraction": flip_fraction,
        "orientation_mean_sic_ice_side": float(np.nanmean(np.where(valid_orientation, ice_side, np.nan))),
        "orientation_mean_sic_low_side": float(np.nanmean(np.where(valid_orientation, low_side, np.nan))),
        "orientation_mean_sic_difference": float(np.nanmean(np.where(valid_orientation, contrast, np.nan))),
        "orientation_qa_status": orientation_qa_status,
    }
    return rows, qa


def representative_location(ev):
    candidates = [
        ("max_loss_point", "max_loss_lon", "max_loss_lat"),
        ("loss_weighted_point", "loss_weighted_lon", "loss_weighted_lat"),
        ("track_centroid_fallback", "track_centroid_lon", "track_centroid_lat"),
        ("centroid_fallback", "centroid_lon", "centroid_lat"),
    ]
    for method, lon_col, lat_col in candidates:
        lon = ev.get(lon_col, np.nan)
        lat = ev.get(lat_col, np.nan)
        if pd.notna(lon) and pd.notna(lat):
            return float(lon), float(lat), method
    return np.nan, np.nan, "missing"


def buffer_sic(date, lon, lat, radius_km, sic_root="data/raw/nsidc_sic"):
    p = sic_file(date, sic_root)
    if p is None or pd.isna(lon) or pd.isna(lat):
        return np.nan, 0
    x, y, transformer = sic_grid()
    cx, cy = transformer.transform(float(lon), float(lat))
    mx = np.abs(x - cx) <= radius_km * 1000
    my = np.abs(y - cy) <= radius_km * 1000
    if not mx.any() or not my.any():
        return np.nan, 0
    xx, yy = np.meshgrid(x[mx], y[my])
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius_km * 1000) ** 2
    arr = open_sic(str(p))["cdr_seaice_conc"].isel(time=0, x=np.where(mx)[0], y=np.where(my)[0]).values.astype(float)
    arr = np.where(arr > 1.2, np.nan, arr)
    vals = arr[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan, 0
    return float(finite.mean()), int(finite.size)


def buffer_era(date, lon, lat, radius_km, era5_single_glob="data/raw/era5/single_levels/*.nc"):
    if pd.isna(lon) or pd.isna(lat):
        return {}
    ds = open_era(pd.Timestamp(date).year, era5_single_glob)
    tname = "valid_time"
    target = np.datetime64(pd.Timestamp(date).date())
    idx = int(np.argmin(np.abs(ds[tname].values.astype("datetime64[D]") - target)))
    lats = ds.latitude.values
    lons = ds.longitude.values
    lon360 = float(lon) % 360
    lat_mask = np.abs(lats - float(lat)) <= radius_km / 111.0 + 1
    lon_mask = np.abs(((lons - lon360 + 180) % 360) - 180) <= radius_km / (111.0 * max(np.cos(np.deg2rad(lat)), .15)) + 1
    sub_lat = lats[lat_mask]; sub_lon = lons[lon_mask]
    if len(sub_lat)==0 or len(sub_lon)==0:
        return {}
    lon2, lat2 = np.meshgrid(sub_lon, sub_lat)
    dist = np.vectorize(haversine_km)(lon, lat, lon2, lat2)
    mask = dist <= radius_km
    out = {}
    for v in ["msl","u10","v10"]:
        arr = ds[v].isel({tname: idx, "latitude": np.where(lat_mask)[0], "longitude": np.where(lon_mask)[0]}).values.astype(float)
        val = float(np.nanmean(np.where(mask, arr, np.nan)))
        out[v] = val/100 if v=="msl" else val
    if "u10" in out and "v10" in out:
        out["wspd10"] = float(np.hypot(out["u10"], out["v10"]))
    return out


def stage6_event_buffer_rows_task(payload):
    ev, lags, radii, sic_root, era5_single_glob = payload
    lon, lat, center_method = representative_location(ev)
    rows = []
    for lag in lags:
        date = pd.Timestamp(ev["event_date"]) + pd.Timedelta(days=lag)
        prev = date - pd.Timedelta(days=5)
        for r in radii:
            sic, n = buffer_sic(date, lon, lat, r, sic_root)
            sic0, _ = buffer_sic(prev, lon, lat, r, sic_root)
            era = buffer_era(date, lon, lat, r, era5_single_glob)
            base = {
                "event_id": ev["unique_local_event_id"],
                "region": ev["dominant_region"],
                "event_date": ev["event_date"],
                "lag": lag,
                "buffer_km": r,
                "buffer_center_lon": lon,
                "buffer_center_lat": lat,
                "buffer_center_method": center_method,
                "sic_mean_buffer": sic,
                "sic_change_5d_buffer": sic - sic0 if np.isfinite(sic) and np.isfinite(sic0) else np.nan,
                "sic_valid_cells": n,
            }
            base.update({f"era5_{k}_buffer": v for k, v in era.items()})
            rows.append(base)
    return rows


def stage6_wrong_region_rows_task(payload):
    ev, references, lags, radii, sic_root, era5_single_glob = payload
    source_region = str(ev["dominant_region"])
    rows = []
    for ref in references:
        sample_region = str(ref["region"])
        if sample_region == source_region:
            continue
        lon = float(ref["reference_lon"])
        lat = float(ref["reference_lat"])
        for lag in lags:
            date = pd.Timestamp(ev["event_date"]) + pd.Timedelta(days=lag)
            prev = date - pd.Timedelta(days=5)
            for radius_km in radii:
                sic, n = buffer_sic(date, lon, lat, radius_km, sic_root)
                sic0, _ = buffer_sic(prev, lon, lat, radius_km, sic_root)
                era = buffer_era(date, lon, lat, radius_km, era5_single_glob)
                row = {
                    "source_event_id": ev["unique_local_event_id"],
                    "source_region": source_region,
                    "source_event_date": ev["event_date"],
                    "sample_date": date.strftime("%Y-%m-%d"),
                    "sample_region": sample_region,
                    "sampling_role": "wrong_region_displacement",
                    "lag": lag,
                    "buffer_km": radius_km,
                    "sample_lon": lon,
                    "sample_lat": lat,
                    "sample_method": ref["method"],
                    "sic_mean_buffer": sic,
                    "sic_change_5d_buffer": sic - sic0 if np.isfinite(sic) and np.isfinite(sic0) else np.nan,
                    "sic_valid_cells": n,
                }
                row.update({f"era5_{key}_buffer": value for key, value in era.items()})
                rows.append(row)
    return rows


def validate_wrong_region_sampling_rows(
    primary: pd.DataFrame,
    displaced: pd.DataFrame,
    references: pd.DataFrame,
    lags: list[int],
    radii: list[int],
) -> pd.DataFrame:
    required = {
        "source_event_id", "source_region", "source_event_date", "sample_date", "sample_region",
        "sampling_role", "lag", "buffer_km", "sample_lon", "sample_lat", "sample_method",
    }
    missing = required - set(displaced.columns)
    if missing:
        raise ValueError(f"wrong-region displaced samples missing columns: {sorted(missing)}")
    logical_key_cols = ["source_event_id", "source_region", "sample_region", "lag", "buffer_km"]
    duplicate_count = int(displaced.duplicated(logical_key_cols).sum())
    if duplicate_count:
        raise ValueError(f"wrong-region displaced sample keys are duplicated: {duplicate_count}")
    persisted_key_cols = [
        "source_event_id", "source_region", "sample_region", "sample_date", "lag", "buffer_km",
    ]
    persisted_duplicate_count = int(displaced.duplicated(persisted_key_cols).sum())
    if persisted_duplicate_count:
        raise ValueError(
            f"wrong-region displaced persisted sample keys are duplicated: {persisted_duplicate_count}"
        )
    same_region_count = int((displaced["source_region"] == displaced["sample_region"]).sum())
    if same_region_count:
        raise ValueError(f"wrong-region samples include source_region == sample_region: {same_region_count}")
    if not displaced["sampling_role"].eq("wrong_region_displacement").all():
        raise ValueError("wrong-region samples contain an invalid sampling_role")

    source_events = primary[["event_id", "region", "event_date"]].drop_duplicates().copy()
    source_events["event_id"] = source_events["event_id"].astype(str)
    source_events["region"] = source_events["region"].astype(str)
    expected = {
        (event_id, region, sample_region, int(lag), float(radius))
        for event_id, region, _ in source_events.itertuples(index=False, name=None)
        for sample_region in FOCUS_REGIONS
        if sample_region != region
        for lag in lags
        for radius in radii
    }
    actual = set(
        zip(
            displaced["source_event_id"].astype(str),
            displaced["source_region"].astype(str),
            displaced["sample_region"].astype(str),
            displaced["lag"].astype(int),
            displaced["buffer_km"].astype(float),
        )
    )
    if actual != expected:
        raise ValueError(
            "wrong-region displaced sample key coverage mismatch: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    event_dates = {
        str(event_id): pd.Timestamp(event_date).normalize()
        for event_id, _, event_date in source_events.itertuples(index=False, name=None)
    }
    displaced_dates = pd.to_datetime(displaced["source_event_date"]).dt.normalize()
    date_mismatch = sum(
        event_dates.get(str(event_id)) != event_date
        for event_id, event_date in zip(displaced["source_event_id"], displaced_dates)
    )
    if date_mismatch:
        raise ValueError(f"wrong-region samples do not preserve source event dates: {date_mismatch}")
    parsed_sample_dates = pd.to_datetime(displaced["sample_date"], errors="coerce").dt.normalize()
    sample_date_parse_failures = int(parsed_sample_dates.isna().sum())
    expected_sample_dates = displaced_dates + pd.to_timedelta(displaced["lag"], unit="D")
    valid_sample_dates = parsed_sample_dates.notna() & displaced_dates.notna()
    sample_date_mismatches = int(
        (valid_sample_dates & parsed_sample_dates.ne(expected_sample_dates)).sum()
    )
    if sample_date_parse_failures or sample_date_mismatches:
        raise ValueError(
            "wrong-region sample-date contract failed: "
            f"parse_failures={sample_date_parse_failures} mismatches={sample_date_mismatches}"
        )
    wrong_regions_per_event = displaced.groupby("source_event_id")["sample_region"].nunique()
    four_region_events = int((wrong_regions_per_event == 4).sum())
    if four_region_events != len(source_events):
        raise ValueError("not every source event has exactly four displaced wrong regions")
    reference_inside = int(references["reference_inside_region_cell"].astype(bool).sum())
    if len(references) != 5 or reference_inside != 5:
        raise ValueError("NSIDC wrong-region reference-point gate is not 5/5")
    audit = pd.DataFrame(
        [
            {"check": "reference_points", "value": len(references), "expected": 5, "status": "PASS"},
            {"check": "reference_points_inside_region", "value": reference_inside, "expected": 5, "status": "PASS"},
            {"check": "displaced_rows", "value": len(displaced), "expected": len(expected), "status": "PASS"},
            {"check": "duplicate_displaced_keys", "value": duplicate_count, "expected": 0, "status": "PASS"},
            {"check": "duplicate_persisted_sample_keys", "value": persisted_duplicate_count, "expected": 0, "status": "PASS"},
            {"check": "same_source_sample_region_rows", "value": same_region_count, "expected": 0, "status": "PASS"},
            {"check": "source_date_mismatches", "value": date_mismatch, "expected": 0, "status": "PASS"},
            {"check": "sample_date_parse_failures", "value": sample_date_parse_failures, "expected": 0, "status": "PASS"},
            {"check": "sample_date_mismatches", "value": sample_date_mismatches, "expected": 0, "status": "PASS"},
            {"check": "events_with_four_wrong_regions", "value": four_region_events, "expected": len(source_events), "status": "PASS"},
        ]
    )
    return audit


def stage6_centroid_buffer(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("location_buffer_mechanism")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    events = pd.read_csv(analysis_path(args.major_events))
    events["event_date"] = pd.to_datetime(events["event_start"])
    events = events[events["dominant_region"].isin(FOCUS_REGIONS)].copy()
    events = apply_quick_major_event_subset(events, args.quick)
    write_quick_event_metadata(out, events, args.quick, "location_buffer", region_col="dominant_region", id_col="unique_local_event_id")
    lags = [-10,-7,-5,-3,-1,0,1,3,5]
    radii = [100,300,500]
    references = build_wrong_region_reference_points(out, args.sic_root)
    tasks = [(ev, lags, radii, args.sic_root, args.era5_single_glob) for ev in events.to_dict("records")]
    nested_rows = run_with_fallback(
        stage6_event_buffer_rows_task,
        tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend=args.backend,
    )
    rows = [row for part in nested_rows for row in part]
    df = pd.DataFrame(rows)
    df.to_csv(out/"location_buffer_lag_values.csv",index=False)
    tests = []
    for region in FOCUS_REGIONS:
        for r in radii:
            for var in [c for c in df.columns if c.endswith("_buffer")]:
                sub = df[(df["region"] == region) & (df["buffer_km"] == r)]
                unit_rows = []
                event_observation_count = 0
                background_observation_count = 0
                for event_id, evsub in sub.groupby("event_id"):
                    event_vals = evsub[evsub["lag"].isin([-1, 0])][var].dropna()
                    background_vals = evsub[evsub["lag"].isin(BACKGROUND_LAGS)][var].dropna()
                    event_observation_count += int(event_vals.notna().sum())
                    background_observation_count += int(background_vals.notna().sum())
                    if event_vals.empty or background_vals.empty:
                        continue
                    event_mean = float(event_vals.mean())
                    background_mean = float(background_vals.mean())
                    unit_rows.append({
                        "event_id": event_id,
                        "event_mean": event_mean,
                        "background_mean": background_mean,
                        "difference": event_mean - background_mean,
                    })
                unit_df = pd.DataFrame(unit_rows)
                diffs = unit_df["difference"] if not unit_df.empty else pd.Series(dtype=float)
                tests.append({
                    "region": region,
                    "buffer_km": r,
                    "variable": var,
                    "event_observation_count": event_observation_count,
                    "background_observation_count": background_observation_count,
                    "mean_event_value": unit_df["event_mean"].mean() if not unit_df.empty else np.nan,
                    "mean_background_value": unit_df["background_mean"].mean() if not unit_df.empty else np.nan,
                    **one_sample_difference_stats(diffs, args.seed, 100 if args.quick else 500),
                    "note": "event-level location-buffer test; each event_id contributes at most one event-minus-background difference",
                })
    td = pd.DataFrame(tests)
    td["p_FDR"] = fdr_bh(td["p_raw"])
    td["ci_excludes_zero"] = td["ci_low"] * td["ci_high"] > 0
    td.to_csv(out/"location_buffer_evidence_table.csv",index=False)
    df.groupby(["region","buffer_km","lag"],as_index=False).mean(numeric_only=True).to_csv(out/"location_buffer_lag_composites.csv",index=False)
    td.groupby(["region","variable"],as_index=False).agg(best_p_FDR=("p_FDR","min"), max_abs_effect=("standardized_mean_difference_zero",lambda s: np.nanmax(np.abs(s)))).to_csv(out/"buffer_size_sensitivity.csv",index=False)
    wrong_tasks = [
        (ev, references.to_dict("records"), lags, radii, args.sic_root, args.era5_single_glob)
        for ev in events.to_dict("records")
    ]
    wrong_nested = run_with_fallback(
        stage6_wrong_region_rows_task,
        wrong_tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend=args.backend,
    )
    wrong_df = pd.DataFrame([row for part in wrong_nested for row in part])
    sampling_audit = validate_wrong_region_sampling_rows(df, wrong_df, references, lags, radii)
    wrong_df.to_csv(out / "wrong_region_location_buffer_lag_values.csv", index=False)
    sampling_audit.to_csv(out / "wrong_region_sampling_audit.csv", index=False)
    for region, fname in [("beaufort_sea","beaufort_location_buffer_panel.png"),("laptev_sea","laptev_location_buffer_dynamic_panel.png")]:
        sub = df[(df["region"]==region)&(df["buffer_km"]==300)].groupby("lag",as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
        for v in [c for c in ["sic_change_5d_buffer","era5_wspd10_buffer","era5_v10_buffer"] if c in sub]:
            ax.plot(sub["lag"],sub[v],marker="o",label=v)
        ax.axhline(0,color="#BBBBBB",ls="--",lw=.8); ax.axvline(0,color="#BBBBBB",ls="--",lw=.8)
        ax.set_xlabel("Lag day"); ax.set_title(REGION_LABELS.get(region,region)+" location-buffer diagnostics"); ax.legend(frameon=False,fontsize=7)
        fig.savefig(fdir/fname,dpi=220); plt.close(fig)
    print(f"WROTE {out}")


def stage7_lead_lag(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("lead_lag")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    df = pd.read_csv(analysis_path(args.location_buffer_values))
    bg_rows = []
    for region in FOCUS_REGIONS:
        for var in [c for c in df.columns if c.endswith("_buffer")]:
            sub = df[(df["region"]==region)&(df["buffer_km"]==300)]
            for lag_group, lags in LAG_GROUPS.items():
                event_rows = []
                for event_id, evsub in sub.groupby("event_id"):
                    event_vals = evsub[evsub["lag"].isin(lags)][var].dropna()
                    background_vals = evsub[evsub["lag"].isin(BACKGROUND_LAGS)][var].dropna()
                    if event_vals.empty or background_vals.empty:
                        continue
                    event_mean = float(event_vals.mean())
                    background_mean = float(background_vals.mean())
                    event_rows.append(
                        {
                            "event_id": event_id,
                            "event_mean": event_mean,
                            "background_mean": background_mean,
                            "difference": event_mean - background_mean,
                        }
                    )
                unit_df = pd.DataFrame(event_rows)
                diffs = unit_df["difference"].astype(float) if not unit_df.empty else pd.Series(dtype=float)
                n_units = int(diffs.notna().sum())
                mean_difference = float(diffs.mean()) if n_units else np.nan
                sd_difference = float(diffs.std(ddof=1)) if n_units >= 2 else np.nan
                standardized = mean_difference / sd_difference if np.isfinite(sd_difference) and sd_difference != 0 else np.nan
                p_raw = float(stats.ttest_1samp(diffs.dropna().to_numpy(float), 0.0, nan_policy="omit").pvalue) if stats is not None and n_units >= 2 and np.isfinite(sd_difference) and sd_difference != 0 else np.nan
                lo, hi = ci_bootstrap_mean(diffs, args.seed, 200 if args.quick else 1000)
                bg_rows.append({
                    "region": region,
                    "variable": var,
                    "source": "SIC_ERA5_location_buffer",
                    "lag_group": lag_group,
                    "test_lags": ",".join(str(x) for x in lags),
                    "background_lags": ",".join(str(x) for x in BACKGROUND_LAGS),
                    "n_units": n_units,
                    "event_mean": unit_df["event_mean"].mean() if not unit_df.empty else np.nan,
                    "background_mean": unit_df["background_mean"].mean() if not unit_df.empty else np.nan,
                    "mean_difference": mean_difference,
                    "difference": mean_difference,
                    "standardized_mean_difference_zero": standardized,
                    "p_raw": p_raw,
                    "ci_low": lo,
                    "ci_high": hi,
                })
    bgdf = pd.DataFrame(bg_rows)
    if not bgdf.empty:
        bgdf["p_FDR"] = fdr_bh(bgdf["p_raw"])
        bgdf["ci_excludes_zero"] = (bgdf["ci_low"] * bgdf["ci_high"] > 0)
        bgdf["significant_background_test"] = bgdf["ci_excludes_zero"] & (bgdf["p_FDR"] < 0.05) & (bgdf["n_units"] >= 3)
        bgdf["classification"] = np.where(
            bgdf["significant_background_test"],
            bgdf["lag_group"].map(
                {
                    "pre-event": "precondition / precursor",
                    "event-time": "contemporaneous forcing",
                    "post-event": "response / adjustment",
                }
            ),
            "weak / unreliable",
        )
        for (region, var), sub in bgdf.groupby(["region", "variable"]):
            if len(sub) == len(LAG_GROUPS) and sub["significant_background_test"].all():
                bgdf.loc[(bgdf["region"] == region) & (bgdf["variable"] == var), "classification"] = "persistent background state"
    class_rows = []
    if not bgdf.empty:
        for (region, var), sub in bgdf.groupby(["region", "variable"]):
            sig = sub[sub["significant_background_test"]].copy()
            if not sig.empty:
                pick = sig.loc[sig["mean_difference"].abs().idxmax()]
                lag_classification = pick["classification"]
                pre_event_mean = sub.loc[sub["lag_group"] == "pre-event", "mean_difference"].mean()
                event_time_mean = sub.loc[sub["lag_group"] == "event-time", "mean_difference"].mean()
                post_event_mean = sub.loc[sub["lag_group"] == "post-event", "mean_difference"].mean()
            else:
                lag_classification = "weak / unreliable"
                pre_event_mean = sub.loc[sub["lag_group"] == "pre-event", "mean_difference"].mean()
                event_time_mean = sub.loc[sub["lag_group"] == "event-time", "mean_difference"].mean()
                post_event_mean = sub.loc[sub["lag_group"] == "post-event", "mean_difference"].mean()
            class_rows.append(
                {
                    "region": region,
                    "variable": var,
                    "pre_event_mean": pre_event_mean,
                    "event_time_mean": event_time_mean,
                    "post_event_mean": post_event_mean,
                    "lag_classification": lag_classification,
                    "n_pre_units": int(sub.loc[sub["lag_group"] == "pre-event", "n_units"].sum()),
                    "n_event_units": int(sub.loc[sub["lag_group"] == "event-time", "n_units"].sum()),
                    "n_post_units": int(sub.loc[sub["lag_group"] == "post-event", "n_units"].sum()),
                }
            )
    outdf = pd.DataFrame(class_rows)
    outdf.to_csv(out/"lead_lag_directionality_summary.csv",index=False)
    outdf.to_csv(out/"region_variable_lag_classification.csv",index=False)
    outdf[outdf["region"]=="laptev_sea"].to_csv(out/"laptev_dynamic_trigger_assessment.csv",index=False)
    outdf[outdf["region"]=="beaufort_sea"].to_csv(out/"beaufort_precondition_assessment.csv",index=False)
    bgdf.to_csv(out/"lead_lag_background_tests.csv",index=False)
    for region, fname in [("laptev_sea","laptev_lead_lag_dynamic_panel.png"),("beaufort_sea","beaufort_precondition_lead_lag_panel.png")]:
        sub = df[(df["region"]==region)&(df["buffer_km"]==300)].groupby("lag",as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7,4.5),constrained_layout=True)
        for v in [c for c in ["sic_change_5d_buffer","era5_wspd10_buffer","era5_v10_buffer"] if c in sub]:
            ax.plot(sub["lag"],sub[v],marker="o",label=v)
        ax.axhline(0,color="#BBBBBB",ls="--",lw=.8); ax.axvline(0,color="#BBBBBB",ls="--",lw=.8); ax.set_xlabel("Lag day"); ax.set_title(REGION_LABELS.get(region,region)+" lead-lag")
        ax.legend(frameon=False,fontsize=7); fig.savefig(fdir/fname,dpi=220); plt.close(fig)
    print(f"WROTE {out}")


def stage8_control_group_task(payload):
    key, sub_records, seed, nboot = payload
    region, buffer_km, var = key
    sub_all = pd.DataFrame(sub_records)
    sub = sub_all.dropna(subset=["difference"]) if not sub_all.empty else pd.DataFrame()
    diffs = sub["difference"].astype(float).to_numpy() if not sub.empty else np.asarray([], dtype=float)
    n_units = int(len(diffs))
    obs = float(np.mean(diffs)) if n_units else np.nan
    rng = np.random.default_rng(stable_seed(seed, "stage8", region, buffer_km, var))
    if n_units < 3:
        sign_row = {"region": region, "buffer_km": buffer_km, "variable": var, "n_units": n_units, "observed_effect": obs, "p_empirical": np.nan, "effect_percentile": np.nan, "ci_low": np.nan, "ci_high": np.nan, "test_role": "event_unit_sign_flip_robustness", "scored_in_evidence": False, "sign_flip_status": "insufficient_event_units"}
    else:
        null = []
        for _ in range(nboot):
            signs = rng.choice([-1.0, 1.0], size=n_units)
            null.append(float(np.mean(diffs * signs)))
        null = np.asarray(null, dtype=float)
        sign_row = {
            "region": region,
            "buffer_km": buffer_km,
            "variable": var,
            "n_units": n_units,
            "observed_effect": obs,
            "p_empirical": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)),
            "effect_percentile": float((null < obs).mean()),
            "ci_low": float(np.percentile(null, 2.5)),
            "ci_high": float(np.percentile(null, 97.5)),
            "test_role": "event_unit_sign_flip_robustness",
            "scored_in_evidence": False,
            "sign_flip_status": "ok_event_unit_sign_flip",
        }

    years = sorted(sub_all.loc[sub_all["difference"].notna(), "year"].unique()) if not sub_all.empty else []
    diffs_series = sub_all["difference"].astype(float).dropna() if not sub_all.empty else pd.Series(dtype=float)
    if len(years) < 2 or len(diffs_series) < 3:
        boot_row = {"region": region, "buffer_km": buffer_km, "variable": var, "event_year_count": len(years), "mean_difference": diffs_series.mean() if len(diffs_series) else np.nan, "cohens_d": np.nan, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_n": 0, "block_bootstrap_status": "unavailable"}
    else:
        boot = []
        for _ in range(nboot):
            ys = rng.choice(years, len(years), replace=True)
            sample = pd.concat([sub_all[sub_all["year"] == y] for y in ys], ignore_index=True)
            boot.append(sample["difference"].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        std = diffs_series.std(ddof=1)
        boot_row = {"region": region, "buffer_km": buffer_km, "variable": var, "event_year_count": len(years), "mean_difference": diffs_series.mean(), "cohens_d": float(diffs_series.mean()/std) if std else np.nan, "ci_low": lo, "ci_high": hi, "bootstrap_n": nboot, "block_bootstrap_status": "ok_event_level_year_resampling"}
    return sign_row, boot_row


def primary_location_buffer_supported_rows(evidence: pd.DataFrame) -> pd.DataFrame:
    required = {
        "region", "buffer_km", "variable", "n_units", "p_FDR",
        "ci_excludes_zero", "standardized_mean_difference_zero",
    }
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"location-buffer evidence missing columns: {sorted(missing)}")
    ci_ok = (
        evidence["ci_excludes_zero"].astype(str).str.lower().eq("true")
        if evidence["ci_excludes_zero"].dtype == object
        else evidence["ci_excludes_zero"].astype(bool)
    )
    return evidence[
        (pd.to_numeric(evidence["n_units"], errors="coerce") >= 3)
        & (pd.to_numeric(evidence["p_FDR"], errors="coerce") < 0.05)
        & ci_ok
        & (pd.to_numeric(evidence["standardized_mean_difference_zero"], errors="coerce").abs() >= 0.2)
    ].copy()


def validate_wrong_region_control_input(primary_lag: pd.DataFrame, wrong_lag: pd.DataFrame) -> pd.DataFrame:
    primary_required = {"event_id", "region", "event_date", "lag", "buffer_km"}
    wrong_required = {
        "source_event_id", "source_region", "source_event_date", "sample_date", "sample_region",
        "sampling_role", "lag", "buffer_km",
    }
    if missing := primary_required - set(primary_lag.columns):
        raise ValueError(f"primary location-buffer input missing columns: {sorted(missing)}")
    if missing := wrong_required - set(wrong_lag.columns):
        raise ValueError(f"wrong-region location-buffer input missing columns: {sorted(missing)}")
    primary_events = primary_lag[["event_id", "region", "event_date"]].drop_duplicates().copy()
    if primary_events["event_id"].duplicated().any():
        raise ValueError("primary event_id maps to multiple region/date records")
    wrong_key = ["source_event_id", "source_region", "sample_region", "lag", "buffer_km"]
    duplicate_count = int(wrong_lag.duplicated(wrong_key).sum())
    if duplicate_count:
        raise ValueError(f"wrong-region lag rows have duplicate keys: {duplicate_count}")
    persisted_wrong_key = [
        "source_event_id", "source_region", "sample_region", "sample_date", "lag", "buffer_km",
    ]
    persisted_duplicate_count = int(wrong_lag.duplicated(persisted_wrong_key).sum())
    if persisted_duplicate_count:
        raise ValueError(
            f"wrong-region lag rows have duplicate persisted sample keys: {persisted_duplicate_count}"
        )
    if not wrong_lag["sampling_role"].eq("wrong_region_displacement").all():
        raise ValueError("wrong-region input contains a non-displacement sampling role")
    same_region = int((wrong_lag["source_region"] == wrong_lag["sample_region"]).sum())
    if same_region:
        raise ValueError(f"wrong-region input includes source_region == sample_region: {same_region}")
    lags = sorted(primary_lag["lag"].dropna().astype(int).unique())
    radii = sorted(primary_lag["buffer_km"].dropna().astype(float).unique())
    expected = {
        (str(event_id), str(region), sample_region, lag, radius)
        for event_id, region, _ in primary_events.itertuples(index=False, name=None)
        for sample_region in FOCUS_REGIONS
        if sample_region != region
        for lag in lags
        for radius in radii
    }
    actual = set(
        zip(
            wrong_lag["source_event_id"].astype(str),
            wrong_lag["source_region"].astype(str),
            wrong_lag["sample_region"].astype(str),
            wrong_lag["lag"].astype(int),
            wrong_lag["buffer_km"].astype(float),
        )
    )
    if actual != expected:
        raise ValueError(
            "unpaired/mismatched source event IDs or displaced row keys: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    event_map = {
        str(event_id): (str(region), pd.Timestamp(event_date).normalize())
        for event_id, region, event_date in primary_events.itertuples(index=False, name=None)
    }
    mismatch = 0
    for event_id, region, date in wrong_lag[["source_event_id", "source_region", "source_event_date"]].itertuples(index=False, name=None):
        mismatch += event_map.get(str(event_id)) != (str(region), pd.Timestamp(date).normalize())
    if mismatch:
        raise ValueError(f"wrong-region input does not preserve source event region/date: {mismatch}")
    source_dates = pd.to_datetime(wrong_lag["source_event_date"], errors="coerce").dt.normalize()
    sample_dates = pd.to_datetime(wrong_lag["sample_date"], errors="coerce").dt.normalize()
    sample_date_parse_failures = int(sample_dates.isna().sum())
    expected_sample_dates = source_dates + pd.to_timedelta(wrong_lag["lag"], unit="D")
    valid_sample_dates = source_dates.notna() & sample_dates.notna()
    sample_date_mismatches = int(
        (valid_sample_dates & sample_dates.ne(expected_sample_dates)).sum()
    )
    dated = wrong_lag.assign(_sample_date_normalized=sample_dates)
    dates_per_event_region_lag = dated.groupby(
        ["source_event_id", "sample_region", "lag"], dropna=False
    )["_sample_date_normalized"].nunique(dropna=False)
    inconsistent_buffer_dates = int((dates_per_event_region_lag != 1).sum())
    if sample_date_parse_failures or sample_date_mismatches or inconsistent_buffer_dates:
        raise ValueError(
            "wrong-region Stage 8 sample-date contract failed: "
            f"parse_failures={sample_date_parse_failures} "
            f"mismatches={sample_date_mismatches} "
            f"inconsistent_buffer_dates={inconsistent_buffer_dates}"
        )
    return pd.DataFrame(
        [
            {"check": "displaced_input_key_coverage", "value": len(actual), "expected": len(expected), "status": "PASS"},
            {"check": "duplicate_displaced_keys", "value": duplicate_count, "expected": 0, "status": "PASS"},
            {"check": "duplicate_persisted_sample_keys", "value": persisted_duplicate_count, "expected": 0, "status": "PASS"},
            {"check": "same_source_sample_region_rows", "value": same_region, "expected": 0, "status": "PASS"},
            {"check": "source_event_region_date_mismatches", "value": mismatch, "expected": 0, "status": "PASS"},
            {"check": "sample_date_parse_failures", "value": sample_date_parse_failures, "expected": 0, "status": "PASS"},
            {"check": "sample_date_mismatches", "value": sample_date_mismatches, "expected": 0, "status": "PASS"},
            {"check": "inconsistent_sample_dates_across_buffers", "value": inconsistent_buffer_dates, "expected": 0, "status": "PASS"},
        ]
    )


def event_level_wrong_region_rows(wrong_lag: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["source_event_id", "source_region", "source_event_date", "sample_region", "buffer_km"]
    for key, sub in wrong_lag.groupby(group_cols, sort=True):
        source_event_id, source_region, source_event_date, sample_region, buffer_km = key
        for variable in variables:
            event_values = sub[sub["lag"].isin([-1, 0])][variable].dropna()
            background_values = sub[sub["lag"].isin(BACKGROUND_LAGS)][variable].dropna()
            event_value = float(event_values.mean()) if not event_values.empty else np.nan
            background_value = float(background_values.mean()) if not background_values.empty else np.nan
            rows.append(
                {
                    "source_event_id": source_event_id,
                    "source_region": source_region,
                    "source_event_date": source_event_date,
                    "sample_region": sample_region,
                    "buffer_km": buffer_km,
                    "variable": variable,
                    "wrong_event_value": event_value,
                    "wrong_background_value": background_value,
                    "wrong_difference": event_value - background_value
                    if np.isfinite(event_value) and np.isfinite(background_value)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def stage8_paired_wrong_region_task(payload):
    key, pair_records, source_primary_supported, seed, nboot = payload
    source_region, sample_region, buffer_km, variable = key
    pairs = pd.DataFrame(pair_records).dropna(subset=["source_difference", "wrong_difference", "specificity"])
    n_paired = int(len(pairs))
    source_effect = float(pairs["source_difference"].mean()) if n_paired else np.nan
    wrong_effect = float(pairs["wrong_difference"].mean()) if n_paired else np.nan
    mean_specificity = float(pairs["specificity"].mean()) if n_paired else np.nan
    if n_paired < 3:
        ci_low = ci_high = p_raw = np.nan
        status = "insufficient_paired_event_units"
    else:
        values = pairs["specificity"].to_numpy(dtype=float)
        rng = np.random.default_rng(
            stable_seed(seed, "paired_wrong_region", source_region, sample_region, buffer_km, variable)
        )
        boot = np.asarray(
            [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(nboot)],
            dtype=float,
        )
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        null = np.asarray(
            [float(np.mean(values * rng.choice([-1.0, 1.0], size=len(values)))) for _ in range(nboot)],
            dtype=float,
        )
        p_raw = float((np.sum(np.abs(null) >= abs(mean_specificity)) + 1) / (len(null) + 1))
        status = "ok_same_event_paired_bootstrap_sign_flip"
    return {
        "source_region": source_region,
        "wrong_region": sample_region,
        "buffer_km": buffer_km,
        "variable": variable,
        "source_n_units": n_paired,
        "wrong_n_units": n_paired,
        "paired_n": n_paired,
        "source_n": n_paired,
        "wrong_n": n_paired,
        "source_effect": source_effect,
        "wrong_region_effect": wrong_effect,
        "mean_specificity": mean_specificity,
        "difference": mean_specificity,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": p_raw,
        "p_empirical": p_raw,
        "p_FDR": np.nan,
        "source_primary_supported": bool(source_primary_supported),
        "control_passed": False,
        "control_status": status,
        "pairing_unit": "source_event_id",
        "test_method": "paired_bootstrap_and_paired_sign_flip",
    }


def finalize_paired_wrong_region_statistics(wrong_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply source-region FDR, pair gates, and the required four-region gate."""
    required = {
        "source_region", "wrong_region", "buffer_km", "variable", "paired_n",
        "p_raw", "mean_specificity", "ci_low", "source_primary_supported",
    }
    if missing := required - set(wrong_df.columns):
        raise ValueError(f"paired wrong-region statistics missing columns: {sorted(missing)}")
    wrong_df = wrong_df.copy()
    wrong_df["p_FDR"] = np.nan
    for _, idx in wrong_df.groupby("source_region").groups.items():
        wrong_df.loc[idx, "p_FDR"] = fdr_bh(wrong_df.loc[idx, "p_raw"])
    wrong_df["control_passed"] = (
        (wrong_df["paired_n"] >= 3)
        & wrong_df["source_primary_supported"].astype(bool)
        & (wrong_df["mean_specificity"] > 0)
        & (wrong_df["ci_low"] > 0)
        & (wrong_df["p_FDR"] < 0.05)
    )
    wrong_df.loc[wrong_df["paired_n"] < 3, "control_status"] = "insufficient_paired_event_units"
    wrong_df.loc[
        (wrong_df["paired_n"] >= 3) & ~wrong_df["source_primary_supported"].astype(bool),
        "control_status",
    ] = "source_primary_not_supported"
    wrong_df.loc[wrong_df["control_passed"], "control_status"] = "passed_same_event_displaced_control"
    failed_evaluable = (
        (wrong_df["paired_n"] >= 3)
        & wrong_df["source_primary_supported"].astype(bool)
        & ~wrong_df["control_passed"]
    )
    wrong_df.loc[failed_evaluable, "control_status"] = "specificity_not_supported"

    summary_rows = []
    for key, sub in wrong_df.groupby(["source_region", "buffer_km", "variable"], sort=True):
        source_region, buffer_km, variable = key
        evaluable = (sub["paired_n"] >= 3) & sub["p_FDR"].notna()
        n_evaluable = int(evaluable.sum())
        n_passed = int(sub["control_passed"].astype(bool).sum())
        all_passed = bool(n_evaluable == 4 and n_passed == 4 and len(sub) == 4)
        if n_evaluable < 4 or len(sub) < 4:
            pair_status = "insufficient_wrong_region_coverage"
        elif all_passed:
            pair_status = "passed_all_wrong_regions"
        else:
            pair_status = "failed_one_or_more_wrong_regions"
        summary_rows.append(
            {
                "source_region": source_region,
                "buffer_km": buffer_km,
                "variable": variable,
                "source_primary_supported": bool(sub["source_primary_supported"].all()),
                "n_expected_wrong_regions": 4,
                "n_evaluable_wrong_regions": n_evaluable,
                "n_passed_wrong_regions": n_passed,
                "all_wrong_regions_passed": all_passed,
                "pair_control_status": pair_status,
            }
        )
    return wrong_df, pd.DataFrame(summary_rows)


def stage8_controls(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("controls")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    df = pd.read_csv(analysis_path(args.location_buffer_values))
    wrong_lag = pd.read_csv(analysis_path(args.wrong_region_location_buffer_values))
    location_evidence = pd.read_csv(analysis_path(args.location_buffer_evidence))
    input_audit = validate_wrong_region_control_input(df, wrong_lag)
    nboot = 200 if args.quick else 1000
    variables = [column for column in df.columns if column.endswith("_buffer")]
    missing_wrong_variables = set(variables) - set(wrong_lag.columns)
    if missing_wrong_variables:
        raise ValueError(f"wrong-region input missing variables: {sorted(missing_wrong_variables)}")
    tests, event_rows = [], []
    for region in FOCUS_REGIONS:
        for buffer_km in sorted(df.loc[df["region"] == region, "buffer_km"].dropna().unique()):
            for var in variables:
                base_sub = df[(df["region"] == region) & (df["buffer_km"] == buffer_km)]
                for event_id, sub in base_sub.groupby("event_id"):
                    ev_vals = sub[sub["lag"].isin([-1,0])][var].dropna()
                    bg_vals = sub[sub["lag"].isin(BACKGROUND_LAGS)][var].dropna()
                    evv = ev_vals.mean() if not ev_vals.empty else np.nan
                    bgg = bg_vals.mean() if not bg_vals.empty else np.nan
                    event_rows.append({
                        "event_id": event_id,
                        "event_or_cluster_id": event_id,
                        "year": int(pd.to_datetime(sub["event_date"].iloc[0]).year),
                        "region": region,
                        "buffer_km": buffer_km,
                        "event_definition": "major_severe local event",
                        "variable": var,
                        "lag_group": "event_minus1_to_0_vs_pre_minus10_minus7_minus5",
                        "event_value": evv,
                        "background_value": bgg,
                        "difference": evv - bgg if pd.notna(evv) and pd.notna(bgg) else np.nan,
                    })
    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(out/"event_level_yearly_samples.csv",index=False)

    group_tasks = []
    if not event_df.empty:
        for key, sub in event_df.groupby(["region", "buffer_km", "variable"]):
            group_tasks.append((key, sub.to_dict("records"), args.seed, nboot))
    group_results = run_with_fallback(
        stage8_control_group_task,
        group_tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend=args.backend,
    )
    boot_rows = []
    for sign_row, boot_row in group_results:
        tests.append(sign_row)
        boot_rows.append(boot_row)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out/"sign_flip_robustness_by_region_variable.csv",index=False)
    if not tests_df.empty:
        tests_df.query("region in ['beaufort_sea','laptev_sea']").to_csv(out/"sign_flip_robustness_summary_beaufort_laptev.csv",index=False)
    else:
        tests_df.to_csv(out/"sign_flip_robustness_summary_beaufort_laptev.csv",index=False)
    supported = primary_location_buffer_supported_rows(location_evidence)
    supported_pairs = {
        (str(row.region), float(row.buffer_km), str(row.variable))
        for row in supported[["region", "buffer_km", "variable"]].itertuples(index=False)
    }
    wrong_event_df = event_level_wrong_region_rows(wrong_lag, variables)
    primary_pair_df = event_df.rename(
        columns={
            "event_id": "source_event_id",
            "region": "source_region",
            "event_value": "source_event_value",
            "background_value": "source_background_value",
            "difference": "source_difference",
        }
    )[
        [
            "source_event_id", "source_region", "buffer_km", "variable",
            "source_event_value", "source_background_value", "source_difference",
        ]
    ]
    paired = wrong_event_df.merge(
        primary_pair_df,
        on=["source_event_id", "source_region", "buffer_km", "variable"],
        how="inner",
        validate="many_to_one",
    )
    if len(paired) != len(wrong_event_df):
        raise ValueError(
            f"same-event pairing lost wrong-region event rows: paired={len(paired)} wrong={len(wrong_event_df)}"
        )
    paired["specificity"] = paired["source_difference"].abs() - paired["wrong_difference"].abs()
    paired["source_primary_supported"] = [
        (str(region), float(buffer_km), str(variable)) in supported_pairs
        for region, buffer_km, variable in paired[["source_region", "buffer_km", "variable"]].itertuples(index=False, name=None)
    ]
    complete_pairs = paired.dropna(subset=["source_difference", "wrong_difference", "specificity"]).copy()
    pair_key = ["source_event_id", "source_region", "sample_region", "buffer_km", "variable"]
    pair_duplicates = int(complete_pairs.duplicated(pair_key).sum())
    if pair_duplicates:
        raise ValueError(f"same-event wrong-region pair keys are duplicated: {pair_duplicates}")
    complete_pairs.to_csv(out / "wrong_region_control_event_pairs.csv", index=False)

    paired_tasks = []
    for key, sub in paired.groupby(["source_region", "sample_region", "buffer_km", "variable"], sort=True):
        source_region, _, buffer_km, variable = key
        source_primary = (str(source_region), float(buffer_km), str(variable)) in supported_pairs
        paired_tasks.append((key, sub.to_dict("records"), source_primary, args.seed, nboot))
    wrong_rows = run_with_fallback(
        stage8_paired_wrong_region_task,
        paired_tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend=args.backend,
    )
    wrong_df, pair_summary = finalize_paired_wrong_region_statistics(pd.DataFrame(wrong_rows))
    wrong_df.to_csv(out/"wrong_region_control_statistical.csv",index=False)
    wrong_df.to_csv(out/"wrong_region_control_tests.csv",index=False)
    pair_summary.to_csv(out / "wrong_region_control_pair_summary.csv", index=False)

    passed = wrong_df["control_passed"].astype(bool)
    passed_contract_violations = int(
        (
            passed
            & (
                ~wrong_df["source_primary_supported"].astype(bool)
                | (wrong_df["mean_specificity"] <= 0)
                | (wrong_df["ci_low"] <= 0)
                | (wrong_df["p_FDR"] >= 0.05)
            )
        ).sum()
    )
    n_mismatch = int(
        (
            (wrong_df["source_n_units"] != wrong_df["wrong_n_units"])
            | (wrong_df["source_n_units"] != wrong_df["paired_n"])
        ).sum()
    )
    invalid_summary_pass = int(
        (
            pair_summary["all_wrong_regions_passed"].astype(bool)
            & (
                (pair_summary["n_evaluable_wrong_regions"] != 4)
                | (pair_summary["n_passed_wrong_regions"] != 4)
            )
        ).sum()
    )
    hard_audit = pd.concat(
        [
            input_audit,
            pd.DataFrame(
                [
                    {"check": "event_level_pair_duplicates", "value": pair_duplicates, "expected": 0, "status": "PASS" if pair_duplicates == 0 else "FAIL"},
                    {"check": "source_wrong_paired_n_mismatches", "value": n_mismatch, "expected": 0, "status": "PASS" if n_mismatch == 0 else "FAIL"},
                    {"check": "unpaired_test_methods", "value": int((wrong_df["test_method"] != "paired_bootstrap_and_paired_sign_flip").sum()), "expected": 0, "status": "PASS" if (wrong_df["test_method"] == "paired_bootstrap_and_paired_sign_flip").all() else "FAIL"},
                    {"check": "control_passed_contract_violations", "value": passed_contract_violations, "expected": 0, "status": "PASS" if passed_contract_violations == 0 else "FAIL"},
                    {"check": "invalid_4_of_4_summary_passes", "value": invalid_summary_pass, "expected": 0, "status": "PASS" if invalid_summary_pass == 0 else "FAIL"},
                ]
            ),
        ],
        ignore_index=True,
    )
    if not hard_audit["status"].eq("PASS").all():
        raise ValueError("wrong-region control hard QA failed")
    hard_audit.to_csv(out / "wrong_region_control_audit.csv", index=False)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out/"event_year_block_bootstrap.csv",index=False)
    boot_df.to_csv(out/"year_block_bootstrap_summary.csv",index=False)
    if tests:
        p = tests_df
        fig, ax = plt.subplots(figsize=(7,4),constrained_layout=True)
        ax.scatter(p["observed_effect"],p["p_empirical"],alpha=.7); ax.axvline(0,color="#BBBBBB",ls="--",lw=.8); ax.set_xlabel("Observed effect"); ax.set_ylabel("Empirical p"); ax.set_title("Event-unit sign-flip robustness")
        fig.savefig(fdir/"sign_flip_robustness_summary.png",dpi=220); plt.close(fig)
    print(f"WROTE {out}")


def stage9_cyclone_event_rows_task(payload):
    ev, lags, cyclones_by_date = payload
    rows = []
    for lag in lags:
        d = pd.Timestamp(ev["event_date"]) + pd.Timedelta(days=lag)
        records = cyclones_by_date.get(d.strftime("%Y-%m-%d"), [])
        if not records or pd.isna(ev["centroid_lon"]):
            rows.append({"event_id": ev["event_id"], "region": ev["region"], "lag": lag, "nearest_cyclone_distance_km": np.nan, "nearest_cyclone_pressure_hpa": np.nan, "cyclone_count_500km": 0, "cyclone_count_1000km": 0})
            continue
        c = pd.DataFrame(records)
        dist = np.array([haversine_km(ev["centroid_lon"], ev["centroid_lat"], lon, lat) for lon, lat in zip(c["lon"], c["lat"])], dtype=float)
        if not np.isfinite(dist).any():
            rows.append({"event_id": ev["event_id"], "region": ev["region"], "lag": lag, "nearest_cyclone_distance_km": np.nan, "nearest_cyclone_pressure_hpa": np.nan, "cyclone_count_500km": 0, "cyclone_count_1000km": 0})
            continue
        idx = int(np.nanargmin(dist))
        rows.append({
            "event_id": ev["event_id"],
            "region": ev["region"],
            "lag": lag,
            "nearest_cyclone_distance_km": float(dist[idx]),
            "nearest_cyclone_pressure_hpa": float(c.iloc[idx]["center_pressure_hpa"]),
            "cyclone_count_500km": int((dist <= 500).sum()),
            "cyclone_count_1000km": int((dist <= 1000).sum()),
        })
    return rows


def stage9_cyclone(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("atmospheric_forcing/cyclone_proximity")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    cyc_path = analysis_path(args.cyclone_tracks)
    if not cyc_path.exists():
        (out/"cyclone_proximity_feasibility.md").write_text("WARNING: cyclone tracks unavailable.\n",encoding="utf-8")
        pd.DataFrame([cyclone_evidence_status(pd.DataFrame(), region) for region in FOCUS_REGIONS]).to_csv(out/"cyclone_proximity_evidence_status.csv", index=False)
        print(f"WROTE {out}")
        return
    cyclones = pd.read_csv(cyc_path, parse_dates=["date"])
    events = apply_quick_cluster_subset(event_frame("major_cluster", args), args.quick)
    write_quick_event_metadata(out, events, args.quick, "cyclone_proximity")
    lags = list(ALL_DIAGNOSTIC_LAGS)
    cyclones = cyclones.copy()
    cyclones["date_key"] = cyclones["date"].dt.strftime("%Y-%m-%d")
    cyclones_by_date = {k: v[["lon", "lat", "center_pressure_hpa"]].to_dict("records") for k, v in cyclones.groupby("date_key")}
    tasks = [(ev, lags, cyclones_by_date) for ev in events.to_dict("records")]
    nested_rows = run_with_fallback(
        stage9_cyclone_event_rows_task,
        tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend="thread",
    )
    rows = [row for part in nested_rows for row in part]
    df = pd.DataFrame(rows); df.to_csv(out/"cyclone_proximity_by_event.csv",index=False)
    summary = df.groupby("region",as_index=False).agg(event_n=("event_id","nunique"),mean_nearest_distance_km=("nearest_cyclone_distance_km","mean"),median_nearest_distance_km=("nearest_cyclone_distance_km","median"),mean_count_500km=("cyclone_count_500km","mean"),mean_count_1000km=("cyclone_count_1000km","mean"))
    summary.to_csv(out/"cyclone_proximity_by_region.csv",index=False)
    tests = []
    for region in FOCUS_REGIONS:
        sub = df[df["region"] == region]
        for var in ["nearest_cyclone_distance_km", "nearest_cyclone_pressure_hpa", "cyclone_count_500km", "cyclone_count_1000km"]:
            unit_rows = []
            event_observation_count = 0
            background_observation_count = 0
            for event_id, evsub in sub.groupby("event_id"):
                event_vals = evsub[evsub["lag"].isin([-1, 0, 1])][var].dropna()
                bg_vals = evsub[evsub["lag"].isin(BACKGROUND_LAGS)][var].dropna()
                event_observation_count += int(event_vals.notna().sum())
                background_observation_count += int(bg_vals.notna().sum())
                if event_vals.empty or bg_vals.empty:
                    continue
                event_mean = float(event_vals.mean())
                background_mean = float(bg_vals.mean())
                unit_rows.append(
                    {
                        "event_id": event_id,
                        "event_mean": event_mean,
                        "background_mean": background_mean,
                        "difference": event_mean - background_mean,
                    }
                )
            unit_df = pd.DataFrame(unit_rows)
            diffs = unit_df["difference"] if not unit_df.empty else pd.Series(dtype=float)
            stats_row = one_sample_difference_stats(diffs, args.seed, 200 if args.quick else 1000)
            tests.append({
                "region": region,
                "variable": var,
                "source": "ERA5_derived_cyclone_tracks",
                "event_lags": "-1,0,1",
                "background_lags": "-10,-7,-5",
                "event_observation_count": event_observation_count,
                "background_observation_count": background_observation_count,
                "mean_event_value": unit_df["event_mean"].mean() if not unit_df.empty else np.nan,
                "mean_background_value": unit_df["background_mean"].mean() if not unit_df.empty else np.nan,
                **stats_row,
                "interpretation_note": "weather-scale cyclone proximity context only; timing/correlation is not causal attribution",
            })
    tdf = pd.DataFrame(tests)
    if not tdf.empty:
        tdf["p_FDR"] = fdr_bh(tdf["p_raw"])
        tdf["ci_excludes_zero"] = tdf["ci_low"] * tdf["ci_high"] > 0
    tdf.to_csv(out/"cyclone_proximity_tests.csv",index=False)
    pd.DataFrame([cyclone_evidence_status(tdf, region) for region in FOCUS_REGIONS]).to_csv(out/"cyclone_proximity_evidence_status.csv", index=False)
    plot_cyclone_proximity_by_region(df, fdir / "cyclone_distance_event_vs_background.png")
    print(f"WROTE {out}")


def stage10_ice_edge_event_task(payload):
    ev, lags, args_dict = payload
    task_args = argparse.Namespace(**args_dict)
    sample_rows, qa_rows = [], []
    for lag in lags:
        date = pd.Timestamp(ev["event_date"]) + pd.Timedelta(days=lag)
        edge, qa = extract_local_ice_edge(date, ev["centroid_lon"], ev["centroid_lat"], task_args)
        qa_base = {
            "event_id": ev["event_id"],
            "region": ev["region"],
            "event_date": ev["event_date"],
            "representative_date": ev.get("representative_date", ev["event_date"]),
            "representative_date_method": ev.get("representative_date_method", ""),
            "sample_date": date.strftime("%Y-%m-%d"),
            "lag": lag,
            "center_lon": ev["centroid_lon"],
            "center_lat": ev["centroid_lat"],
        }
        qa_rows.append({**qa_base, **qa})
        if qa.get("qa_status") == "reject" or edge.empty:
            continue
        era = era_at_points(date, edge["edge_lon"].values, edge["edge_lat"].values, ("u10", "v10"), task_args.era5_single_glob)
        u = era["u10"]
        v = era["v10"]
        normal = u * edge["normal_east"].values + v * edge["normal_north"].values
        tangent = -u * edge["normal_north"].values + v * edge["normal_east"].values
        sample_rows.append({
            **qa_base,
            "qa_status": qa.get("qa_status"),
            "qa_reason": qa.get("qa_reason"),
            "edge_cells": qa.get("edge_cells"),
            "component_fraction": qa.get("component_fraction"),
            "outward_edge_normal_wind_ms": float(np.nanmean(normal)),
            "abs_edge_normal_wind_ms": float(np.nanmean(np.abs(normal))),
            "edge_tangential_wind_ms": float(np.nanmean(tangent)),
            "wind_speed_ms": float(np.nanmean(np.hypot(u, v))),
            "edge_normal_positive_definition": "positive normal points from high-SIC ice side toward low-SIC/open-water side",
        })
    return sample_rows, qa_rows


def stage10_ice_edge(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("atmospheric_forcing/ice_edge_relative_wind")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    if args.normal_orientation_ok_fraction <= args.normal_orientation_degraded_fraction:
        raise ValueError("--normal-orientation-ok-fraction must be greater than --normal-orientation-degraded-fraction")
    write_surface_mask_code_audit(out)
    write_ice_edge_grid_coordinate_audit(out, args.sic_root)
    write_surface_mask_grid_audit(out, args.sic_root)
    events = apply_quick_cluster_subset(event_frame("major_cluster", args), args.quick)
    write_quick_event_metadata(out, events, args.quick, "ice_edge_relative_wind")
    lags = list(ALL_DIAGNOSTIC_LAGS)
    args_dict = vars(args).copy()
    tasks = [(ev, lags, args_dict) for ev in events.to_dict("records")]
    event_results = run_with_fallback(
        stage10_ice_edge_event_task,
        tasks,
        workers=args.workers,
        parallel=parallel_enabled(args),
        backend=args.backend,
    )
    sample_rows, qa_rows = [], []
    for sample_part, qa_part in event_results:
        sample_rows.extend(sample_part)
        qa_rows.extend(qa_part)
    samples = pd.DataFrame(sample_rows)
    qa_df = pd.DataFrame(qa_rows)
    samples.to_csv(out/"ice_edge_relative_wind_samples.csv",index=False)
    qa_df.to_csv(out/"ice_edge_relative_wind_qa.csv",index=False)
    tests = []
    variables = ["outward_edge_normal_wind_ms", "abs_edge_normal_wind_ms", "edge_tangential_wind_ms", "wind_speed_ms"]
    if not samples.empty:
        samples.groupby(["region", "lag"], as_index=False).mean(numeric_only=True).to_csv(out/"ice_edge_relative_wind_lag_composites.csv", index=False)
        analysis_tiers = {
            "primary": samples[samples["qa_status"] == "ok"].copy(),
            "sensitivity": samples[samples["qa_status"].isin(["ok", "degraded"])].copy(),
        }
        for tier, tier_df in analysis_tiers.items():
            for region in FOCUS_REGIONS:
                sub = tier_df[tier_df["region"] == region]
                for var in variables:
                    for group, glags in LAG_GROUPS.items():
                        unit_rows = []
                        event_observation_count = 0
                        background_observation_count = 0
                        group_rows = sub[sub["lag"].isin(glags)]
                        for event_id, evsub in sub.groupby("event_id"):
                            vals = evsub[evsub["lag"].isin(glags)][var].dropna() if var in evsub else pd.Series(dtype=float)
                            bg = evsub[evsub["lag"].isin(BACKGROUND_LAGS)][var].dropna() if var in evsub else pd.Series(dtype=float)
                            event_observation_count += int(vals.notna().sum())
                            background_observation_count += int(bg.notna().sum())
                            if vals.empty or bg.empty:
                                continue
                            event_mean = float(vals.mean())
                            background_mean = float(bg.mean())
                            unit_rows.append(
                                {
                                    "event_id": event_id,
                                    "event_mean": event_mean,
                                    "background_mean": background_mean,
                                    "difference": event_mean - background_mean,
                                }
                            )
                        unit_df = pd.DataFrame(unit_rows)
                        diffs = unit_df["difference"] if not unit_df.empty else pd.Series(dtype=float)
                        stats_row = one_sample_difference_stats(diffs, args.seed, 200 if args.quick else 1000)
                        tests.append({
                            "analysis_tier": tier,
                            "region": region,
                            "variable": var,
                            "source": "SIC_ice_edge_ERA5_relative_wind",
                            "lag_group": group,
                            "test_lags": ",".join(str(x) for x in glags),
                            "background_lags": ",".join(str(x) for x in BACKGROUND_LAGS),
                            "event_observation_count": event_observation_count,
                            "background_observation_count": background_observation_count,
                            "mean_event_value": unit_df["event_mean"].mean() if not unit_df.empty else np.nan,
                            "mean_background_value": unit_df["background_mean"].mean() if not unit_df.empty else np.nan,
                            **stats_row,
                            "qa_ok_samples": int((group_rows["qa_status"] == "ok").sum()) if "qa_status" in group_rows else 0,
                            "qa_degraded_samples": int((group_rows["qa_status"] == "degraded").sum()) if "qa_status" in group_rows else 0,
                            "interpretation_note": "edge-relative wind tests dynamic plausibility; it must be interpreted jointly with ice motion/divergence and not as standalone causal proof",
                        })
        plot_ice_edge_wind_by_region(samples, fdir / "outward_edge_normal_wind_lead_lag.png")
    else:
        pd.DataFrame(columns=["region", "lag"]).to_csv(out/"ice_edge_relative_wind_lag_composites.csv", index=False)
    tdf = pd.DataFrame(tests)
    if not tdf.empty:
        tdf["p_FDR"] = np.nan
        for _, idx in tdf.groupby("analysis_tier").groups.items():
            tdf.loc[idx, "p_FDR"] = fdr_bh(tdf.loc[idx, "p_raw"])
        tdf["ci_excludes_zero"] = tdf["ci_low"] * tdf["ci_high"] > 0
        primary_sig = (
            (tdf["analysis_tier"] == "primary")
            & (tdf["p_FDR"] < 0.05)
            & tdf["ci_excludes_zero"]
            & (tdf["n_units"] >= 3)
            & (tdf["qa_ok_samples"] > 0)
        )
        sensitivity_sig = (
            (tdf["analysis_tier"] == "sensitivity")
            & (tdf["p_FDR"] < 0.05)
            & tdf["ci_excludes_zero"]
            & (tdf["n_units"] >= 3)
        )
        tdf["mechanism_evidence_status"] = np.select(
            [
                primary_sig,
                (tdf["analysis_tier"] == "primary") & (tdf["qa_ok_samples"] <= 0),
                sensitivity_sig,
            ],
            [
                "statistically_supported_dynamic_indicator",
                "unsupported_zero_ok_samples",
                "sensitivity_dependent_suggestive",
            ],
            default="weak_or_insufficient",
        )
    tdf.to_csv(out/"ice_edge_relative_wind_tests.csv",index=False)
    pd.DataFrame([edge_wind_evidence_status(tdf, region) for region in FOCUS_REGIONS]).to_csv(out/"ice_edge_relative_wind_evidence_status.csv", index=False)
    qa_summary = qa_df.groupby(["region", "qa_status", "qa_reason"], as_index=False).size() if not qa_df.empty else pd.DataFrame()
    qa_summary.to_csv(out/"ice_edge_relative_wind_qa_summary.csv", index=False)
    note = [
        "# Ice-edge-relative wind diagnostics",
        "",
        "- The diagnostic extracts a local dominant SIC ice edge on the NSIDC 25 km grid.",
        "- Land, ice-on-land, floating-shelf, disconnected-ocean, and coastal-adjacent cells are excluded with the NSIDC-0780 surface mask.",
        "- Fragmented topology is downgraded to `degraded` when the dominant component is below the configured fraction; failed topology or normal-orientation QA is rejected.",
        "- Positive edge-normal wind is defined from the high-SIC ice side toward the low-SIC/open-water side.",
        "- Statistical signals describe dynamic plausibility and must be interpreted with ice motion/divergence, not as direct causal attribution.",
    ]
    (out/"ice_edge_relative_wind_method.md").write_text("\n".join(note),encoding="utf-8")
    print(f"WROTE {out}")


def stage11_evidence(args):
    out = analysis_path(args.out_dir) if args.out_dir else outdir("evidence")
    out.mkdir(parents=True, exist_ok=True)
    fdir = figdir(out)
    major = pd.read_csv(root()/"outputs/local_vrile_major_severe/major_severe_event_region_summary.csv")
    clusters = pd.read_csv(root()/"outputs/event_clusters/cluster_region_summary.csv")
    buf = pd.read_csv(root()/"outputs/location_buffer_mechanism/location_buffer_evidence_table.csv")
    lead = pd.read_csv(root()/"outputs/lead_lag/region_variable_lag_classification.csv")
    lead_bg_path = root()/"outputs/lead_lag/lead_lag_background_tests.csv"
    lead_bg = pd.read_csv(lead_bg_path) if lead_bg_path.exists() else pd.DataFrame()
    wrong_path = root()/"outputs/controls/wrong_region_control_pair_summary.csv"
    wrong = pd.read_csv(wrong_path) if wrong_path.exists() else pd.DataFrame()
    boot = pd.read_csv(root()/"outputs/controls/year_block_bootstrap_summary.csv")
    cyclone_status, cyclone_warnings = read_status_file_for_regions(
        root()/"outputs/atmospheric_forcing/cyclone_proximity/cyclone_proximity_evidence_status.csv",
        FOCUS_REGIONS,
        "cyclone_proximity",
        "supported_variables",
    )
    edge_status, edge_warnings = read_status_file_for_regions(
        root()/"outputs/atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_evidence_status.csv",
        FOCUS_REGIONS,
        "edge_relative_wind",
        "supported_lag_groups",
    )
    status_warnings = cyclone_warnings + edge_warnings
    if status_warnings:
        pd.DataFrame(status_warnings).to_csv(out/"evidence_status_input_warnings.csv", index=False)
        for row in status_warnings:
            print(f"WARNING: {row['warning']}")
    rows, trace, buffer_summary_rows, wrong_pair_audit_rows = [], [], [], []
    for region in FOCUS_REGIONS:
        cstat_row = cyclone_status[cyclone_status["region"] == region] if "region" in cyclone_status else pd.DataFrame()
        estat_row = edge_status[edge_status["region"] == region] if "region" in edge_status else pd.DataFrame()
        cyclone_status_value = cstat_row["evidence_status"].iloc[0] if not cstat_row.empty else "unavailable"
        edge_status_value = estat_row["evidence_status"].iloc[0] if not estat_row.empty else "unavailable"
        cyclone_supported_variables = cstat_row["supported_variables"].iloc[0] if not cstat_row.empty else ""
        edge_supported_lag_groups = estat_row["supported_lag_groups"].iloc[0] if not estat_row.empty else ""
        mr = major.loc[major["region"]==region]
        mf = float(mr["major_fraction_of_severe"].iloc[0]) if not mr.empty else 0
        major_count = int(mr["major_event_count"].iloc[0]) if (not mr.empty and "major_event_count" in mr) else 0
        cr = clusters.loc[clusters["region"]==region]
        major_cluster_count = int(cr["major_cluster_count"].iloc[0]) if (not cr.empty and "major_cluster_count" in cr) else 0
        major_score = 4 if (mf>=0.2 and major_count>=30 and major_cluster_count>=25) else 3 if (mf>=0.1 and major_count>=20 and major_cluster_count>=15) else 1
        b = buf[buf["region"]==region]
        sig = primary_location_buffer_supported_rows(b) if not b.empty else b.iloc[0:0]
        supported_variable_count = int(sig["variable"].dropna().nunique()) if "variable" in sig else 0
        buffer_score = 4 if supported_variable_count>=3 else 3 if supported_variable_count>=1 else 1
        supported_pairs = set()
        if {"variable", "buffer_km"}.issubset(sig.columns):
            for var, buffer_km in sig[["variable", "buffer_km"]].dropna().drop_duplicates().itertuples(index=False):
                supported_pairs.add((str(var), float(buffer_km)))
            for (var, sub) in sig.groupby("variable"):
                radii = sorted({int(x) if float(x).is_integer() else float(x) for x in sub["buffer_km"].dropna().astype(float)})
                buffer_summary_rows.append({
                    "region": region,
                    "variable": var,
                    "supported_radii_km": ";".join(str(x) for x in radii),
                    "n_supported_radii": len(radii),
                })
        w = wrong[wrong["source_region"]==region].copy() if not wrong.empty else pd.DataFrame()
        controlled_pairs = set()
        if not w.empty:
            required_wrong = {"variable", "buffer_km", "all_wrong_regions_passed"}
            if missing := required_wrong - set(w.columns):
                raise ValueError(f"wrong-region pair summary missing columns: {sorted(missing)}")
            for row in w.itertuples(index=False):
                pair = (str(row.variable), float(row.buffer_km))
                location_supported = pair in supported_pairs
                all_wrong_passed = str(row.all_wrong_regions_passed).lower() == "true"
                used = bool(location_supported and all_wrong_passed)
                if used:
                    controlled_pairs.add(pair)
                wrong_pair_audit_rows.append(
                    {
                        "region": region,
                        "variable": pair[0],
                        "buffer_km": pair[1],
                        "location_buffer_supported_pair": location_supported,
                        "all_wrong_regions_passed": all_wrong_passed,
                        "used_for_control_score": used,
                    }
                )
        control_score = 3 if controlled_pairs else 1
        bo = boot[boot["region"]==region]
        boot_pass = False
        if not bo.empty and supported_pairs and {"variable", "buffer_km", "ci_low", "ci_high"}.issubset(bo.columns):
            bo = bo.copy()
            bo["_pair"] = list(zip(bo["variable"].astype(str), bo["buffer_km"].astype(float)))
            boot_pass = (bo["_pair"].isin(supported_pairs) & ((bo["ci_low"] * bo["ci_high"]) > 0)).any()
        boot_score = 3 if (not bo.empty and boot_pass) else 1
        lg = lead[lead["region"]==region]
        lbg = lead_bg[lead_bg["region"]==region] if not lead_bg.empty else pd.DataFrame()
        dynamic_vars = {"era5_msl_buffer", "era5_u10_buffer", "era5_v10_buffer", "era5_wspd10_buffer"}
        sig_background = (
            lbg["significant_background_test"].astype(str).str.lower().eq("true")
            if not lbg.empty and "significant_background_test" in lbg.columns
            else pd.Series(False, index=lbg.index)
        )
        dyn_ok = (
            not lbg.empty
            and {"significant_background_test", "lag_group", "variable"}.issubset(lbg.columns)
            and (
                sig_background
                & lbg["lag_group"].isin(["pre-event", "event-time"])
                & lbg["variable"].isin(dynamic_vars)
            ).any()
        )
        lead_score = 3 if dyn_ok else 1
        if region=="beaufort_sea":
            beaufort_strong = major_score >= 4 and buffer_score >= 4 and control_score >= 3 and boot_score >= 3
            if beaufort_strong:
                regime = "Strong preconditioned/compound regime"
            elif major_score >= 3 and buffer_score >= 3 and (control_score >= 3 or boot_score >= 3):
                regime = "Moderate preconditioned/compound regime"
            else:
                regime = "Preconditioned/compound suggestive regime"
        elif region=="laptev_sea":
            regime = "Dynamic-loss regime with contemporaneous forcing" if lead_score>=3 and control_score>=2 else "Dynamic-loss suggestive regime"
        elif region=="kara_sea":
            cyclone_ok = cyclone_status_value == "supported"
            regime = "Moderate atmospheric-dynamic regime" if cyclone_ok and lead_score>=3 and buffer_score>=3 else "Atmospheric-dynamic suggestive regime"
        elif region=="barents_sea":
            regime = "Barents Sea atmospheric/compound suggestive regime; ocean heat not tested"
        elif region=="central_arctic":
            regime = f"{region_display_label('central_arctic')} weak/threshold-sensitive regime"
        else:
            regime = "Relative-threshold-sensitive / weak absolute-loss regime"
        scores = {"major_severe_cluster_support":major_score,"location_buffer_support":buffer_score,"lead_lag_directionality":lead_score,"wrong_region_control":control_score,"year_block_bootstrap":boot_score}
        mean = np.mean(list(scores.values()))
        conf = "strong" if mean>=3.5 else "moderate" if mean>=2.7 else "suggestive" if mean>=1.8 else "weak"
        if region == "central_arctic":
            conf = "weak"
            scores["major_severe_cluster_support"] = min(scores["major_severe_cluster_support"], 1)
            mean = np.mean(list(scores.values()))
        if region == "barents_sea":
            conf = "weak"
        if region == "kara_sea" and regime.startswith("Atmospheric-dynamic suggestive"):
            conf = "suggestive"
        rows.append({"region":region,"assigned_regime":regime,"confidence_level":conf,"mean_score":mean,**scores,"cyclone_proximity_status":cyclone_status_value,"edge_relative_wind_status":edge_status_value,"cyclone_supported_variables":cyclone_supported_variables,"edge_supported_lag_groups":edge_supported_lag_groups,"notes":"Evidence scoring uses the original five dimensions; cyclone proximity and edge-relative wind are supplementary dynamic evidence and do not enter mean_score."})
        for k,v in scores.items():
            trace.append({"region":region,"evidence_dimension":k,"evidence_status":"","score":v,"score_label":{4:"strong",3:"moderate",2:"suggestive",1:"weak",0:"unreliable"}.get(v,""),"is_supplementary":False,"notes":"included in mean_score"})
        trace.append({"region":region,"evidence_dimension":"cyclone_proximity_support","evidence_status":cyclone_status_value,"score":status_score(cyclone_status_value),"score_label":cyclone_status_value,"is_supplementary":True,"notes":"supplementary dynamic evidence; not included in mean_score"})
        trace.append({"region":region,"evidence_dimension":"edge_relative_wind_support","evidence_status":edge_status_value,"score":status_score(edge_status_value),"score_label":edge_status_value,"is_supplementary":True,"notes":"supplementary dynamic evidence; not included in mean_score"})
    mat = pd.DataFrame(rows); tr = pd.DataFrame(trace)
    mat.to_csv(out/"evidence_strength_matrix.csv",index=False); tr.to_csv(out/"evidence_strength_matrix_trace.csv",index=False)
    pd.DataFrame(buffer_summary_rows, columns=["region", "variable", "supported_radii_km", "n_supported_radii"]).to_csv(out/"buffer_multiscale_support_summary.csv", index=False)
    wrong_pair_audit = pd.DataFrame(wrong_pair_audit_rows)
    wrong_pair_audit.to_csv(out / "wrong_region_evidence_pair_audit.csv", index=False)
    if not wrong_pair_audit.empty:
        invalid_use = wrong_pair_audit[
            wrong_pair_audit["used_for_control_score"].astype(bool)
            & ~(
                wrong_pair_audit["location_buffer_supported_pair"].astype(bool)
                & wrong_pair_audit["all_wrong_regions_passed"].astype(bool)
            )
        ]
        if not invalid_use.empty:
            raise ValueError("evidence used a wrong-region control that did not match the same supported variable-buffer pair")
    mat[["region","assigned_regime","confidence_level","notes"]].to_csv(out/"region_mechanism_classification.csv",index=False)
    (out/"region_mechanism_classification.md").write_text("\n".join(["# Region mechanism classification",""]+[f"- **{REGION_LABELS.get(r.region,r.region)}**: {r.assigned_regime} ({r.confidence_level})." for r in mat.itertuples()]),encoding="utf-8")
    plot_evidence_strength_heatmap(tr, fdir / "evidence_strength_heatmap.png")
    print(f"WROTE {out}")


def stage12_package(args):
    out = outdir("review_package")
    manifest = []
    patterns = [
        "outputs/region_taxonomy/*",
        "outputs/local_vrile_major_severe/*.csv",
        "outputs/local_vrile_major_severe/figures/*.png",
        "outputs/event_clusters/*.csv",
        "outputs/event_clusters/figures/*.png",
        "outputs/spatiotemporal_matching/*.csv",
        "outputs/spatiotemporal_matching/figures/*.png",
        "outputs/location_buffer_mechanism/*.csv",
        "outputs/location_buffer_mechanism/figures/*.png",
        "outputs/lead_lag/*.csv",
        "outputs/controls/*.csv",
        "outputs/atmospheric_forcing/cyclone_proximity/*.csv",
        "outputs/atmospheric_forcing/ice_edge_relative_wind/*.csv",
        "outputs/atmospheric_forcing/ice_edge_relative_wind/*.md",
        "outputs/evidence/*",
        "outputs/experiment_summary/*.md",
        "outputs/report/*.md",
    ]
    for pat in patterns:
        for p in root().glob(pat):
            if p.is_file():
                rel = p.relative_to(root())
                dest = out/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest)
                manifest.append(str(rel))
    pd.DataFrame({"included_file":manifest}).to_csv(out/"review_package_manifest.csv",index=False)
    zip_path = out/"VRILE_review_package.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for rel in manifest:
            z.write(root()/rel, rel)
        z.write(out/"review_package_manifest.csv","review_package_manifest.csv")
    print(f"WROTE {out}")


def stage13_report(args):
    out = outdir("report")
    major = pd.read_csv(root()/"outputs/local_vrile_major_severe/major_severe_event_region_summary.csv")
    clusters = pd.read_csv(root()/"outputs/event_clusters/cluster_region_summary.csv")
    match = pd.read_csv(root()/"outputs/spatiotemporal_matching/panarctic_regional_severe_major_cluster_spatiotemporal_match.csv")
    evidence = pd.read_csv(root()/"outputs/evidence/evidence_strength_matrix.csv")
    lines = ["# VRILE Stage 2 实验结果解释报告", "", "## 1. 摘要式结论", ""]
    lines.append("当前 workflow 在独立目录中从 raw 数据重跑基础事件体系，并在此基础上新增 region taxonomy、major_severe events、synoptic clustering、双向时空匹配、location-buffer 诊断、lead-lag、event-unit sign-flip robustness、same-date displaced-location wrong-region control、year-block bootstrap、cyclone proximity、ice-edge-relative wind 和 evidence matrix。")
    lines.append("")
    lines.append("## 2. 事件定义增强结果")
    lines.append(f"- severe local events 经 major_severe 筛选后，物理区域 major_severe 事件数显著减少。Kara={int(major.loc[major.region=='kara_sea','major_event_count'].sum())}, Laptev={int(major.loc[major.region=='laptev_sea','major_event_count'].sum())}, Beaufort={int(major.loc[major.region=='beaufort_sea','major_event_count'].sum())}。对应表：`outputs/local_vrile_major_severe/major_severe_event_region_summary.csv`。")
    lines.append("- aggregate/reference rows 已通过 `outputs/region_taxonomy/region_taxonomy_check.csv` 排除出区域排名和机制分型。")
    lines.append("")
    lines.append("## 3. 时空匹配结果")
    m = match[(match.source_definition=="panarctic")&(match.target_definition=="major_severe")&(match.matching_rule=="time-only_pm3d")]
    mr = match[(match.source_definition=="panarctic")&(match.target_definition=="major_severe")&(match.matching_rule=="time+region_pm3d")]
    mc = match[(match.source_definition=="panarctic")&(match.target_definition=="major_severe")&(match.matching_rule=="time+centroid_500km_pm3d")]
    if not m.empty:
        lines.append(f"- pan-Arctic -> major_severe 的 +/-3 day time-only ratio={m.ratio.iloc[0]:.3f}；加入 region 后为 {mr.ratio.iloc[0]:.3f}；加入 500 km centroid 后为 {mc.ratio.iloc[0]:.3f}。这说明 time-only 会高估空间对应。对应表：`outputs/spatiotemporal_matching/panarctic_regional_severe_major_cluster_spatiotemporal_match.csv`。")
    lines.append("")
    lines.append("## 4. 机制诊断结果")
    lines.append("- location-buffer 诊断按事件代表位置 100/300/500 km buffer 从 raw SIC 和 ERA5 提取；不是 object-core，因为当前没有 object mask/polygon。对应表：`outputs/location_buffer_mechanism/location_buffer_evidence_table.csv`。")
    lines.append("- PIOMAS/ice motion 中不能做到 object-core 的部分不得称为 object-core 证据。")
    lines.append("")
    lines.append("## 5. 控制实验结果")
    lines.append("- event-unit sign-flip robustness、same-date displaced-location wrong-region control 和 year-block bootstrap 输出在 `outputs/controls/`。wrong-region control 将同一 source event/date 的真实位置与四个 NSIDC-0780 固定错位位置配对比较；未通过这些控制的信号不得写成 strong mechanism。")
    lines.append("- sign-flip robustness is descriptive/robustness evidence and is not scored in `mean_score`; control score is based only on a FDR-gated same-date displaced-location wrong-region control for the same supported variable-buffer pair.")
    lines.append("")
    lines.append("## 6. Evidence matrix 与区域分型")
    lines.append("以下分型仅来自当前 `outputs/evidence/evidence_strength_matrix.csv`，report 不重新实现显著性、方向或样本量判据。")
    for r in evidence.itertuples():
        lines.append(f"- {REGION_LABELS.get(r.region,r.region)}: {r.assigned_regime}，confidence={r.confidence_level}，cyclone proximity={getattr(r, 'cyclone_proximity_status', 'unavailable')}，edge-relative wind={getattr(r, 'edge_relative_wind_status', 'unavailable')}。")
    lines.append("")
    lines.append("## 7. 大气动力机制诊断")
    status_notes = {
        "supported": "当前统计、独立样本和物理方向支持该证据线",
        "not_supported": "样本足以评价，但当前结果不支持",
        "insufficient_samples": "独立事件/cluster 数不足，不能评价；这不是“没有机制”",
        "unavailable": "数据或诊断不可用",
    }
    for r in evidence.itertuples():
        cyclone_status = getattr(r, "cyclone_proximity_status", "unavailable")
        edge_status = getattr(r, "edge_relative_wind_status", "unavailable")
        cyclone_vars = getattr(r, "cyclone_supported_variables", "")
        edge_lags = getattr(r, "edge_supported_lag_groups", "")
        lines.append(f"- {REGION_LABELS.get(r.region,r.region)}: cyclone proximity={cyclone_status}（{status_notes.get(str(cyclone_status), '状态来自 evidence status 输出')}；supported variables={cyclone_vars or 'none'}）；edge-relative wind={edge_status}（{status_notes.get(str(edge_status), '状态来自 evidence status 输出')}；supported lag groups={edge_lags or 'none'}）。")
    lines.append("")
    lines.append("## 8. 可保留的当前结论")
    lines.append("- pan-Arctic VRILE 不能完整代表 local rapid-loss 的结论被增强，因为当前 workflow 提供了双向、空间约束匹配。")
    lines.append("- cyclone proximity 与 edge-relative wind 是 supplementary evidence，不进入 `mean_score`，也不改变原 confidence thresholds。")
    lines.append("")
    lines.append("## 9. 不作出的结论")
    lines.append("- 不把相关、时序一致性或动力合理性写成确定因果归因。")
    lines.append("- 不把 `insufficient_samples` 写成“没有机制”。")
    lines.append("- 不写不属于当前 formal taxonomy/evidence 输出的 legacy 区域结论。")
    lines.append("")
    lines.append("## 10. 剩余风险")
    lines.append("- 当前没有 object mask/polygon，因此 IoU 和 object-core composite 不可行。")
    lines.append("- ice-edge-relative wind 已升级为 QA-gated 诊断；closed-contour/polynya topology、cluster-level bootstrap 和集中式 config 仍未在本轮解决。")
    lines.append("")
    lines.append("## 11. 下一轮 report 修改建议")
    lines.append("- 在 full Stage 2 后重新生成 evidence/report，避免用 quick validation 输出替代正式结果。")
    lines.append("- 明确区分 time-only matching 和 spatial matching。")
    lines.append("- 将 evidence matrix 的 strong/moderate/suggestive/weak 作为结论强度边界。")
    (out/"experiment_results_interpretation.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"WROTE {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["taxonomy","major","clusters","matching","location_buffer","centroid_buffer","lead_lag","controls","cyclone","ice_edge","evidence","package","report"])
    add_args(p)
    args = p.parse_args()
    parallel_stages = {"location_buffer", "centroid_buffer", "controls", "cyclone", "ice_edge"}
    if args.parallel and args.stage not in parallel_stages:
        p.error("--parallel is implemented only for: location_buffer, centroid_buffer, controls, cyclone, ice_edge")
    if args.stage == "cyclone" and args.parallel and args.backend != "thread":
        p.error("cyclone parallel execution requires --backend thread")
    {
        "taxonomy": stage2_taxonomy,
        "major": stage3_major,
        "clusters": stage4_clusters,
        "matching": stage5_matching,
        "location_buffer": stage6_centroid_buffer,
        "centroid_buffer": stage6_centroid_buffer,
        "lead_lag": stage7_lead_lag,
        "controls": stage8_controls,
        "cyclone": stage9_cyclone,
        "ice_edge": stage10_ice_edge,
        "evidence": stage11_evidence,
        "package": stage12_package,
        "report": stage13_report,
    }[args.stage](args)


if __name__ == "__main__":
    main()
