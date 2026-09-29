#!/usr/bin/env python3
"""Reusable Round-1 drawing primitives for the F1 and F2 scientific drafts.

The module deliberately has no command-line entry point and writes no files.
It prepares transparent descriptive payloads from locked inputs and returns
Matplotlib figures to the controlled-build orchestrator.  Composite and
standalone figures call the same panel functions, preventing visual drift.

Nothing here performs inference, changes a catalog or assigns a formal event
status.  In particular, the observed F1 window is a deterministic illustration
selected from columns already present in the locked daily table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import xarray as xr


F1_SELECTION_RULE = (
    "MEDIAN_ANNUAL_MEAN_EXTENT_NEAREST_YEAR_EARLIEST_TIE__"
    "MIN_DELTA_SIE_EARLIEST_TIE__PLUS_MINUS_15_DAYS"
)
F2_EXPECTED_COUNTS = {"broad": 9530, "severe": 2427, "major_severe": 554}
F2_EXCLUSIVE_COUNTS = {"broad_only": 7103, "severe_nonmajor": 1873, "major_severe": 554}
F2_HALF_MONTH_LABELS = (
    "Jun 1–15",
    "Jun 16–30",
    "Jul 1–15",
    "Jul 16–31",
    "Aug 1–15",
    "Aug 16–31",
)
F2_MASK_VARIABLE = "sea_ice_region_surface_mask"
ROUND1_OUTPUT_STEMS = {
    "F1": {
        "composite": "F1_revision_round1_composite",
        "panel_A": "F1_revision_round1_panel_A",
        "panel_B": "F1_revision_round1_panel_B",
        "panel_C": "F1_revision_round1_panel_C",
    },
    "F2": {
        "composite": "F2_revision_round1_composite",
        "panel_A": "F2_revision_round1_panel_A",
        "panel_B": "F2_revision_round1_panel_B",
        "panel_C": "F2_revision_round1_panel_C",
        "panel_D": "F2_revision_round1_panel_D",
    },
}
ROUND1_PAYLOAD_FILENAMES = {
    "F1": {
        "annual": "F1_R1_annual_mean_context.csv",
        "window": "F1_R1_observed_window.csv",
        "scale": "F1_R1_scale_payload.json",
    },
    "F2": {
        "events": "F2_R1_event_display.csv",
        "timing": "F2_R1_year_halfmonth_counts.csv",
        "mask": "F2_R1_mask_payload.json",
    },
}


def _read_frame(source: str | Path | pd.DataFrame, *, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    else:
        frame = pd.read_csv(Path(source))
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], role: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{role} missing required columns: {missing}")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{role} required columns contain missing values")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _colors(style: Mapping[str, Any]) -> Mapping[str, str]:
    return style.get("colors", style)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.035,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _horizontal_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")


def prepare_f1(daily_source: str | Path | pd.DataFrame) -> dict[str, Any]:
    """Prepare deterministic F1 annual and observed-window payloads.

    Selection is visualization-only.  The year is chosen before the rapid-loss
    row: nearest to the median of the 37 annual mean raw extents, with earliest
    year as the tie break.  The existing minimum ``delta_sie`` in that year is
    then selected, with earliest date as the tie break.  No threshold is made.
    """

    daily = _read_frame(daily_source, date_columns=("date",))
    required = ("date", "extent", "delta_sie", "year")
    _require_columns(daily, required, "F1 daily input")
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
        raise ValueError("F1 daily dates must be unique and ordered")
    if not np.array_equal(daily["year"].astype(int).to_numpy(), daily["date"].dt.year.to_numpy()):
        raise ValueError("F1 year column disagrees with date")

    annual = (
        daily.groupby("year", sort=True, as_index=False)
        .agg(n_days=("date", "size"), annual_mean_extent=("extent", "mean"))
        .astype({"year": int, "n_days": int})
    )
    if len(annual) != 37 or set(annual["n_days"].tolist()) != {365}:
        raise ValueError("F1 representative-year rule requires 37 equally sampled 365-day years")
    median_mean = float(annual["annual_mean_extent"].median())
    annual["absolute_distance_to_median"] = (annual["annual_mean_extent"] - median_mean).abs()
    annual = annual.sort_values(["absolute_distance_to_median", "year"], kind="mergesort")
    selected_year = int(annual.iloc[0]["year"])
    annual["selected_year"] = annual["year"].eq(selected_year)
    annual = annual.sort_values("year", kind="mergesort").reset_index(drop=True)

    selected_rows = daily.loc[daily["year"].astype(int).eq(selected_year)].copy()
    minimum_delta = float(selected_rows["delta_sie"].min())
    candidates = selected_rows.loc[selected_rows["delta_sie"].eq(minimum_delta)].sort_values("date")
    selected = candidates.iloc[0]
    selected_date = pd.Timestamp(selected["date"])
    delta_start = selected_date - pd.Timedelta(days=2)
    delta_end = selected_date + pd.Timedelta(days=1)
    window_start = selected_date - pd.Timedelta(days=15)
    window_end = selected_date + pd.Timedelta(days=15)
    window = daily.loc[daily["date"].between(window_start, window_end), list(required)].copy()
    if len(window) != 31:
        raise ValueError("F1 fixed illustrative window must contain 31 rows")
    date_set = set(window["date"])
    if delta_start not in date_set or delta_end not in date_set:
        raise ValueError("F1 delta endpoints fall outside the illustrative window")
    window["is_delta_start"] = window["date"].eq(delta_start)
    window["is_delta_center"] = window["date"].eq(selected_date)
    window["is_delta_end"] = window["date"].eq(delta_end)

    endpoint = window.set_index("date")["extent"]
    endpoint_delta = float(endpoint.loc[delta_end] - endpoint.loc[delta_start])
    if not np.isclose(endpoint_delta, minimum_delta, rtol=0.0, atol=1e-12):
        raise ValueError("F1 displayed endpoints do not reproduce existing delta_sie")

    scale_payload = {
        "relationship": "ANALYTICAL_SCALE_ONLY_NO_CAUSAL_DIRECTION",
        "nodes": [
            {"stage": "Stage 1", "title": "Local event population", "status": "corrected catalog"},
            {"stage": "Stage 2", "title": "Regional environmental organization", "status": "corrected evidence"},
            {"stage": "Stage 3", "title": "Pan-Arctic organization", "status": "question pending corrected rebuild"},
        ],
    }
    metadata = {
        "selection_rule": F1_SELECTION_RULE,
        "daily_rows": int(len(daily)),
        "annual_years": int(len(annual)),
        "days_per_year": sorted(annual["n_days"].unique().astype(int).tolist()),
        "median_annual_mean": median_mean,
        "selected_year": selected_year,
        "selected_annual_mean": float(annual.loc[annual["selected_year"], "annual_mean_extent"].iloc[0]),
        "selected_date": selected_date.strftime("%Y-%m-%d"),
        "selected_delta_sie": minimum_delta,
        "delta_start": delta_start.strftime("%Y-%m-%d"),
        "delta_end": delta_end.strftime("%Y-%m-%d"),
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "window_rows": int(len(window)),
        "tie_count": int(len(candidates)),
        "formal_event_classification": False,
        "fitted_trend": False,
        "causal_connectors": False,
        "horizontal_tick_rotation_max": 0,
    }
    return {"annual": annual, "window": window, "scale": scale_payload, "metadata": metadata}


def _draw_f1_a(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "a") -> None:
    colors = _colors(style)
    annual = payload["annual"]
    meta = payload["metadata"]
    selected = annual.loc[annual["selected_year"]].iloc[0]
    ax.plot(annual["year"], annual["annual_mean_extent"], color=colors["formal"], lw=1.05)
    ax.scatter([selected["year"]], [selected["annual_mean_extent"]], s=32, color=colors["interpretation"], zorder=4)
    ax.axhline(meta["median_annual_mean"], color=colors["synthesis"], lw=0.75, ls=(0, (3, 2)))
    ax.text(
        selected["year"] + 0.45,
        selected["annual_mean_extent"] + 0.03,
        f"{meta['selected_year']}: median annual-mean year",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=colors["interpretation"],
    )
    ax.set_xlim(1988.3, 2026.0)
    ax.set_xticks(np.arange(1990, 2026, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean sea-ice extent\n(10$^6$ km$^2$)")
    ax.grid(axis="y", color=colors["grid"], lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Pan-Arctic aggregate context", loc="left", fontweight="semibold")
    ax.text(0.995, 0.965, "Direct annual means · no fitted trend", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=colors["synthesis"])
    _horizontal_ticks(ax)
    _panel_label(ax, label)


def _draw_f1_b(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "b") -> None:
    colors = _colors(style)
    window = payload["window"]
    meta = payload["metadata"]
    start = pd.Timestamp(meta["delta_start"])
    end = pd.Timestamp(meta["delta_end"])
    selected = pd.Timestamp(meta["selected_date"])
    highlight = window.loc[window["date"].between(start, end)]
    ax.plot(window["date"], window["extent"], color=colors["formal"], lw=1.15)
    ax.plot(highlight["date"], highlight["extent"], color=colors["interpretation"], lw=2.25, zorder=3)
    endpoint_rows = window.loc[window["is_delta_start"] | window["is_delta_end"]]
    ax.scatter(endpoint_rows["date"], endpoint_rows["extent"], s=25, color=colors["interpretation"], edgecolor="white", lw=0.45, zorder=4)
    ax.axvspan(start, end, color=colors["context"], alpha=0.35, zorder=-2)
    y_top = float(window["extent"].max())
    ax.text(selected, y_top + 0.025, f"ΔSIE$_{{3d}}$ = {meta['selected_delta_sie']:.3f} × 10$^6$ km$^2$", ha="center", va="bottom", fontsize=7.0, color=colors["interpretation"], fontweight="semibold")
    tick_start = pd.Timestamp(meta["window_start"])
    ticks = [tick_start + pd.Timedelta(days=7 * index) for index in range(5)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel(f"Date in {meta['selected_year']}")
    ax.set_ylabel("Sea-ice extent (10$^6$ km$^2$)")
    ax.grid(axis="y", color=colors["grid"], lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Observed rapid-loss illustration", loc="left", fontweight="semibold")
    ax.text(0.995, 0.05, "Fixed selection rule · not formal event classification", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=colors["synthesis"])
    _horizontal_ticks(ax)
    _panel_label(ax, label)


def _draw_f1_c(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "c") -> None:
    colors = _colors(style)
    nodes = payload["scale"]["nodes"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = (0.17, 0.50, 0.83)
    box_width = 0.255
    display_titles = ("Local\nevent", "Regional\norganization", "Pan-Arctic\norganization")
    display_statuses = ("event unit", "study-region\nframework", "Stage 3 pending")
    for xpos, node, title, status in zip(centers, nodes, display_titles, display_statuses):
        box = FancyBboxPatch((xpos - box_width / 2, 0.18), box_width, 0.61, boxstyle="round,pad=0.014,rounding_size=0.025", facecolor="#F4F7F9", edgecolor=colors["synthesis"], lw=0.9)
        ax.add_patch(box)
        ax.text(xpos, 0.735, node["stage"], ha="center", va="center", fontsize=6.5, color=colors["synthesis"], fontweight="semibold")
        circle = Circle((xpos, 0.565), 0.085, facecolor="#E5F0F5", edgecolor=colors["formal"], lw=0.8)
        ax.add_patch(circle)
        if xpos == centers[0]:
            for dx, dy, size in ((-0.034, 0.018, 18), (0.030, 0.030, 11), (0.012, -0.034, 15)):
                ax.scatter([xpos + dx], [0.565 + dy], s=size, color=colors["major"], alpha=0.88, zorder=3)
        elif xpos == centers[1]:
            for angle, color in zip(np.linspace(0, 2 * np.pi, 6)[:-1], (colors["formal"], colors["interpretation"], colors["severe"], colors["major"], colors["synthesis"])):
                ax.scatter([xpos + 0.054 * np.cos(angle)], [0.565 + 0.054 * np.sin(angle)], s=15, color=color, zorder=3)
        else:
            for angle, size in zip(np.linspace(0, 2 * np.pi, 7)[:-1], (13, 9, 15, 10, 12, 8)):
                ax.scatter([xpos + 0.058 * np.cos(angle)], [0.565 + 0.058 * np.sin(angle)], s=size, color=colors["formal"], alpha=0.75, zorder=3)
            ax.scatter([xpos], [0.565], s=22, facecolor="white", edgecolor=colors["formal"], lw=0.8, zorder=3)
        ax.text(xpos, 0.405, title, ha="center", va="center", fontsize=6.35, fontweight="semibold", linespacing=1.05)
        ax.text(xpos, 0.275, status, ha="center", va="center", fontsize=5.65, color=colors["synthesis"], linespacing=1.05)
    for left, right in zip(centers[:-1], centers[1:]):
        arrow = FancyArrowPatch((left + 0.132, 0.505), (right - 0.132, 0.505), arrowstyle="-|>", mutation_scale=7, lw=0.8, linestyle=(0, (2, 2)), color=colors["synthesis"])
        ax.add_patch(arrow)
    ax.text(0.50, 0.91, "Scientific organization across analytical scales", ha="center", va="center", fontsize=7.0, fontweight="semibold", color=colors["synthesis"])
    ax.text(0.50, 0.095, "Open connectors denote scale linkage, not a causal process", ha="center", va="center", fontsize=6.2, color=colors["synthesis"])
    ax.set_title("Multi-scale analytical framework", loc="left", fontweight="semibold", pad=8)
    _panel_label(ax, label)


def draw_f1_panel(panel: str, ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], *, label: str | None = None) -> None:
    """Draw one F1 panel; used identically by composite and standalone views."""

    key = panel.upper()
    functions = {"A": _draw_f1_a, "B": _draw_f1_b, "C": _draw_f1_c}
    if key not in functions:
        raise ValueError(f"unknown F1 panel: {panel}")
    functions[key](ax, payload, style, label or key.lower())


def make_f1_figures(payload: Mapping[str, Any], style: Mapping[str, Any]) -> dict[str, plt.Figure]:
    """Return the Round-1 F1 composite and three standalone panel figures."""

    composite = plt.figure(figsize=(7.1, 5.7), constrained_layout=False)
    grid = composite.add_gridspec(2, 2, height_ratios=(0.86, 1.14), left=0.09, right=0.98, top=0.90, bottom=0.105, hspace=0.40, wspace=0.29)
    axes = {"A": composite.add_subplot(grid[0, :]), "B": composite.add_subplot(grid[1, 0]), "C": composite.add_subplot(grid[1, 1])}
    for panel, ax in axes.items():
        draw_f1_panel(panel, ax, payload, style)
    composite.suptitle("Why an event-based framework is needed", x=0.09, y=0.965, ha="left", fontsize=12.5, fontweight="bold")
    composite.text(0.50, 0.022, "Aggregate and event views are complementary; scale links are analytical, not causal.", ha="center", va="bottom", fontsize=6.8, color=_colors(style)["synthesis"])

    figures: dict[str, plt.Figure] = {"composite": composite}
    for panel in ("A", "B", "C"):
        figure, axis = plt.subplots(figsize=(5.3, 3.25))
        figure.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.18)
        draw_f1_panel(panel, axis, payload, style)
        figures[f"panel_{panel}"] = figure
    return figures


def _load_f2_mask(mask_source: str | Path) -> dict[str, Any]:
    path = Path(mask_source)
    with xr.open_dataset(path, decode_cf=False) as dataset:
        if F2_MASK_VARIABLE not in dataset or "x" not in dataset or "y" not in dataset or "crs" not in dataset:
            raise ValueError("F2 NSIDC-0780 mask contract is incomplete")
        mask = dataset[F2_MASK_VARIABLE].values.copy()
        x = dataset["x"].values.astype(float).copy()
        y = dataset["y"].values.astype(float).copy()
        crs_attrs = dict(dataset["crs"].attrs)
    spatial_ref = crs_attrs.get("spatial_ref")
    if not spatial_ref:
        raise ValueError("F2 NSIDC-0780 mask lacks native spatial_ref")
    crs = CRS.from_wkt(spatial_ref)
    if crs.to_epsg() != 3413:
        raise ValueError("F2 native mask WKT does not resolve to EPSG:3413")
    return {
        "path": path,
        "mask": mask,
        "x": x,
        "y": y,
        "crs": crs,
        "metadata": {
            "path": str(path),
            "sha256": _sha256(path),
            "variable": F2_MASK_VARIABLE,
            "shape": [int(mask.shape[0]), int(mask.shape[1])],
            "x_min_m": float(x.min()),
            "x_max_m": float(x.max()),
            "y_min_m": float(y.min()),
            "y_max_m": float(y.max()),
            "projection": "EPSG:3413_WITH_DOCUMENTED_VENDOR_3411_EXCEPTION",
            "crs_contract": "EPSG:3413_WITH_DOCUMENTED_VENDOR_3411_EXCEPTION",
            "native_wkt_epsg": int(crs.to_epsg()),
            "x_range_m": [float(x.min()), float(x.max())],
            "y_range_m": [float(y.min()), float(y.max())],
        },
    }


def prepare_f2(
    broad_source: str | Path | pd.DataFrame,
    severe_source: str | Path | pd.DataFrame,
    major_source: str | Path | pd.DataFrame,
    mask_source: str | Path,
    *,
    expected_counts: Mapping[str, int] = F2_EXPECTED_COUNTS,
) -> dict[str, Any]:
    """Prepare F2 exclusive display classes, native map and timing payloads."""

    broad = _read_frame(broad_source, date_columns=("event_start",))
    severe = _read_frame(severe_source)
    major = _read_frame(major_source)
    key = "unique_local_event_id"
    required = (key, "track_centroid_lon", "track_centroid_lat", "duration_days", "max_area_km2", "event_start")
    _require_columns(broad, required, "F2 broad events")
    _require_columns(severe, (key,), "F2 severe events")
    _require_columns(major, (key,), "F2 major-severe events")
    frames = {"broad": broad, "severe": severe, "major_severe": major}
    for name, frame in frames.items():
        expected = int(expected_counts[name])
        if len(frame) != expected or frame[key].nunique() != expected or frame[key].duplicated().any():
            raise ValueError(f"F2 {name} identity/count gate failed")
    broad_ids = set(broad[key])
    severe_ids = set(severe[key])
    major_ids = set(major[key])
    if not severe_ids < broad_ids or not major_ids < severe_ids:
        raise ValueError("F2 nested severity membership gate failed")

    display = broad.loc[:, list(required)].copy()
    display["exclusive_class"] = "broad_only"
    display.loc[display[key].isin(severe_ids), "exclusive_class"] = "severe_nonmajor"
    display.loc[display[key].isin(major_ids), "exclusive_class"] = "major_severe"
    counts = display["exclusive_class"].value_counts().to_dict()
    if counts != dict(F2_EXCLUSIVE_COUNTS):
        raise ValueError(f"F2 exclusive display count mismatch: {counts}")

    mask_payload = _load_f2_mask(mask_source)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), mask_payload["crs"], always_xy=True)
    projected_x, projected_y = transformer.transform(
        display["track_centroid_lon"].astype(float).to_numpy(),
        display["track_centroid_lat"].astype(float).to_numpy(),
    )
    display["projected_x_m"] = projected_x
    display["projected_y_m"] = projected_y
    if not np.isfinite(display[["projected_x_m", "projected_y_m"]].to_numpy()).all():
        raise ValueError("F2 event projection contains nonfinite coordinates")
    display = display.sort_values(key, kind="mergesort").reset_index(drop=True)

    months = set(display["event_start"].dt.month.unique().astype(int).tolist())
    if not months.issubset({6, 7, 8}):
        raise ValueError(f"F2 half-month display encountered non-JJA start months: {sorted(months)}")
    display["half_month_index"] = (display["event_start"].dt.month - 6) * 2 + (display["event_start"].dt.day > 15).astype(int)
    years = list(range(int(display["event_start"].dt.year.min()), int(display["event_start"].dt.year.max()) + 1))
    grid = pd.MultiIndex.from_product([years, range(6)], names=["year", "half_month_index"]).to_frame(index=False)
    grouped = display.assign(year=display["event_start"].dt.year).groupby(["year", "half_month_index", "exclusive_class"]).size().unstack(fill_value=0).reset_index()
    timing = grid.merge(grouped, on=["year", "half_month_index"], how="left").fillna(0)
    for category in ("broad_only", "severe_nonmajor", "major_severe"):
        if category not in timing:
            timing[category] = 0
        timing[category] = timing[category].astype(int)
    timing["half_month_label"] = timing["half_month_index"].map(dict(enumerate(F2_HALF_MONTH_LABELS)))
    timing["total_events"] = timing[["broad_only", "severe_nonmajor", "major_severe"]].sum(axis=1)
    timing = timing[["year", "half_month_index", "half_month_label", "broad_only", "severe_nonmajor", "major_severe", "total_events"]]
    if int(timing["total_events"].sum()) != int(expected_counts["broad"]) or timing.shape != (len(years) * 6, 7):
        raise ValueError("F2 year-by-half-month payload failed count/shape gate")

    metadata = {
        "expected_counts": dict(expected_counts),
        "exclusive_counts": counts,
        "nested_membership": True,
        "native_mask_used": True,
        "mask": mask_payload["metadata"],
        "statistical_unit": key,
        "duplicate_event_ids": 0,
        "year_halfmonth_total": int(timing["total_events"].sum()),
        "heatmap_shape": [len(years), 6],
        "aggregation": "DESCRIPTIVE_COUNT_ONLY_NO_TREND",
        "patches_as_independent_samples": False,
        "nested_levels_as_independent_samples": False,
        "causal_connectors": False,
        "composite_score": False,
        "regional_rank": False,
        "horizontal_tick_rotation_max": 0,
    }
    return {"events": display, "timing": timing, "mask": mask_payload, "metadata": metadata}


def _f2_class_style(style: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    colors = _colors(style)
    color_map = {"broad_only": colors["broad"], "severe_nonmajor": colors["severe"], "major_severe": colors["major"]}
    sizes = {"broad_only": 2.0, "severe_nonmajor": 3.5, "major_severe": 7.5}
    alphas = {"broad_only": 0.30, "severe_nonmajor": 0.52, "major_severe": 0.82}
    return color_map, sizes, alphas


def _draw_f2_a(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "a") -> None:
    colors = _colors(style)
    events = payload["events"]
    mask_payload = payload["mask"]
    mask = mask_payload["mask"]
    x = mask_payload["x"] / 1.0e6
    y = mask_payload["y"] / 1.0e6
    surface = np.zeros(mask.shape, dtype=np.uint8)
    surface[np.isin(mask, np.arange(0, 19))] = 1
    surface[mask >= 30] = 2
    cmap = ListedColormap(["#F7FAFC", "#E8F2F7", "#D7DADD"])
    norm = BoundaryNorm(np.arange(-0.5, 3.5, 1), cmap.N)
    ax.pcolormesh(x, y, surface, cmap=cmap, norm=norm, shading="nearest", rasterized=True, zorder=-3)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), mask_payload["crs"], always_xy=True)
    longitude = np.linspace(-180.0, 180.0, 721)
    for latitude in (50.0, 60.0, 70.0, 80.0):
        gx, gy = transformer.transform(longitude, np.full_like(longitude, latitude))
        ax.plot(np.asarray(gx) / 1.0e6, np.asarray(gy) / 1.0e6, color="#B9C6CE", lw=0.38, alpha=0.8, zorder=-1)
        label_x, label_y = transformer.transform(135.0, latitude)
        ax.text(label_x / 1.0e6, label_y / 1.0e6, f"{int(latitude)}°N", ha="center", va="center", fontsize=5.5, color=colors["synthesis"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5}, zorder=4)
    latitude = np.linspace(49.0, 89.0, 161)
    for longitude_value in (-135.0, -45.0, 45.0, 135.0):
        gx, gy = transformer.transform(np.full_like(latitude, longitude_value), latitude)
        ax.plot(np.asarray(gx) / 1.0e6, np.asarray(gy) / 1.0e6, color="#C9D3D9", lw=0.30, alpha=0.65, zorder=-1)
    color_map, sizes, alphas = _f2_class_style(style)
    for category in ("broad_only", "severe_nonmajor", "major_severe"):
        part = events.loc[events["exclusive_class"].eq(category)]
        ax.scatter(part["projected_x_m"] / 1.0e6, part["projected_y_m"] / 1.0e6, s=sizes[category], color=color_map[category], alpha=alphas[category], linewidths=0, rasterized=True, zorder=2)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pan-Arctic event-centroid occurrence", loc="left", fontweight="semibold")
    ax.text(0.01, 0.015, "Locked NSIDC-0780 native-grid basemap", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, color=colors["synthesis"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5})
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[key], markeredgecolor="none", markersize=4.5, label=label_text)
        for key, label_text in (("broad_only", "Broad only"), ("severe_nonmajor", "Severe, nonmajor"), ("major_severe", "Major"))
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.0, handletextpad=0.25)
    _panel_label(ax, label)


def _draw_f2_b(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "b") -> None:
    colors = _colors(style)
    expected = payload["metadata"]["expected_counts"]
    labels = ["Broad population", "Severe subset", "Major subset"]
    counts = np.array([expected["broad"], expected["severe"], expected["major_severe"]], dtype=float)
    shares = counts / counts[0] * 100.0
    y = np.arange(3)[::-1]
    bar_colors = [colors["broad"], colors["severe"], colors["major"]]
    ax.barh(y, shares, height=0.50, color=bar_colors, edgecolor="white", lw=0.8)
    for ypos, count, share in zip(y, counts.astype(int), shares):
        if share > 80:
            ax.text(98.0, ypos, f"n = {count:,}  ·  {share:.1f}%", ha="right", va="center", fontsize=6.8, fontweight="semibold", color="#26343D")
        else:
            ax.text(share + 2.0, ypos, f"n = {count:,}  ·  {share:.1f}%", ha="left", va="center", fontsize=6.8, fontweight="semibold")
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.70, 2.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of broad-event catalog (%)")
    ax.grid(axis="x", color=colors["grid"], lw=0.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.995, 0.04, "Containment hierarchy · not independent samples", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.25, color=colors["synthesis"])
    ax.set_title("Nested severity hierarchy", loc="left", fontweight="semibold")
    _horizontal_ticks(ax)
    _panel_label(ax, label)


def _draw_f2_c(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "c") -> None:
    colors = _colors(style)
    events = payload["events"]
    color_map, sizes, alphas = _f2_class_style(style)
    for category in ("broad_only", "severe_nonmajor", "major_severe"):
        part = events.loc[events["exclusive_class"].eq(category)]
        ax.scatter(part["duration_days"], part["max_area_km2"], s=sizes[category] * 1.1, color=color_map[category], alpha=alphas[category], linewidths=0, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlim(0, 65)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_ylim(2.5e4, 1.0e7)
    ax.set_yticks([5e4, 1e5, 5e5, 1e6, 5e6])
    ax.set_yticklabels(["5×10⁴", "10⁵", "5×10⁵", "10⁶", "5×10⁶"])
    ax.set_xlabel("Event duration (days)")
    ax.set_ylabel("Maximum area (km$^2$; log scale)")
    ax.grid(color=colors["grid"], lw=0.45, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Duration and area diversity", loc="left", fontweight="semibold")
    ax.text(0.98, 0.97, "one point per unique event · exclusive display classes", transform=ax.transAxes, ha="right", va="top", fontsize=6.1, color=colors["synthesis"])
    _horizontal_ticks(ax)
    _panel_label(ax, label)


def _draw_f2_d(ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], label: str = "d") -> None:
    colors = _colors(style)
    timing = payload["timing"]
    years = sorted(timing["year"].unique().astype(int).tolist())
    matrix = timing.pivot(index="year", columns="half_month_index", values="total_events").reindex(index=years, columns=range(6)).to_numpy()
    image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=ListedColormap(["#F4F7F9", "#D8E8F1", "#AFCBDE", "#79A5C5", "#2F5D8C"]), extent=(-0.5, 5.5, years[0] - 0.5, years[-1] + 0.5))
    ax.set_xticks(range(6))
    ax.set_xticklabels([label_text.replace(" ", "\n", 1) for label_text in F2_HALF_MONTH_LABELS])
    year_ticks = [year for year in years if year % 5 == 0]
    ax.set_yticks(year_ticks)
    ax.set_yticklabels([str(year) for year in year_ticks], rotation=0)
    ax.set_xlabel("Event-start half-month")
    ax.set_ylabel("Event-start year")
    ax.set_title("Seasonal timing through the record", loc="left", fontweight="semibold")
    ax.text(0.99, 0.01, "descriptive counts only · no fitted trend", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color=colors["synthesis"])
    colorbar = ax.figure.colorbar(image, ax=ax, orientation="horizontal", fraction=0.06, pad=0.18, shrink=0.80)
    colorbar.set_label("Unique-event count", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    _horizontal_ticks(ax)
    _panel_label(ax, label)


def draw_f2_panel(panel: str, ax: plt.Axes, payload: Mapping[str, Any], style: Mapping[str, Any], *, label: str | None = None) -> None:
    """Draw one F2 panel; used identically by composite and standalone views."""

    key = panel.upper()
    functions = {"A": _draw_f2_a, "B": _draw_f2_b, "C": _draw_f2_c, "D": _draw_f2_d}
    if key not in functions:
        raise ValueError(f"unknown F2 panel: {panel}")
    functions[key](ax, payload, style, label or key.lower())


def make_f2_figures(payload: Mapping[str, Any], style: Mapping[str, Any]) -> dict[str, plt.Figure]:
    """Return the Round-1 F2 composite and four standalone panel figures."""

    composite = plt.figure(figsize=(7.1, 7.7), constrained_layout=False)
    grid = composite.add_gridspec(2, 2, left=0.085, right=0.955, top=0.89, bottom=0.085, hspace=0.40, wspace=0.31)
    axes = {"A": composite.add_subplot(grid[0, 0]), "B": composite.add_subplot(grid[0, 1]), "C": composite.add_subplot(grid[1, 0]), "D": composite.add_subplot(grid[1, 1])}
    for panel, ax in axes.items():
        draw_f2_panel(panel, ax, payload, style)
    composite.suptitle("Heterogeneous population of localized VRILEs", x=0.075, y=0.96, ha="left", fontsize=12.5, fontweight="bold")
    composite.text(0.50, 0.018, "Broad, severe, and major-severe form nested catalog levels; display classes are mutually exclusive.", ha="center", va="bottom", fontsize=6.7, color=_colors(style)["synthesis"])

    figures: dict[str, plt.Figure] = {"composite": composite}
    for panel in ("A", "B", "C", "D"):
        canvas = {"A": (4.0, 5.3), "D": (7.1, 4.3)}.get(panel, (5.3, 4.0))
        figure, axis = plt.subplots(figsize=canvas)
        if panel == "A":
            figure.subplots_adjust(left=0.08, right=0.96, top=0.89, bottom=0.08)
        elif panel == "B":
            figure.subplots_adjust(left=0.25, right=0.94, top=0.86, bottom=0.20)
        elif panel == "C":
            figure.subplots_adjust(left=0.20, right=0.95, top=0.86, bottom=0.20)
        else:
            figure.subplots_adjust(left=0.10, right=0.96, top=0.84, bottom=0.24)
        draw_f2_panel(panel, axis, payload, style)
        figures[f"panel_{panel}"] = figure
    return figures


__all__ = [
    "F1_SELECTION_RULE",
    "F2_EXPECTED_COUNTS",
    "F2_EXCLUSIVE_COUNTS",
    "F2_HALF_MONTH_LABELS",
    "ROUND1_OUTPUT_STEMS",
    "ROUND1_PAYLOAD_FILENAMES",
    "prepare_f1",
    "draw_f1_panel",
    "make_f1_figures",
    "prepare_f2",
    "draw_f2_panel",
    "make_f2_figures",
]
