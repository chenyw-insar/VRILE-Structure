#!/usr/bin/env python3
"""Round-1 F3 rendering API: typography and panel exports only.

This module deliberately has no command-line entry point.  A controlled-build
driver must verify the registered input hashes before calling it and must keep
the resulting artifacts in an isolated scientific-draft directory.  The
module preserves the accepted F3 data selection and scientific semantics while
providing one set of draw functions for the composite and standalone A-D
exports.

No inference, evidence classification, catalogue operation, or scientific
recalculation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd
import xarray as xr


ROUND1_STATUS = "SCIENTIFIC_DRAFT_REVISION_ROUND1_NOT_FINAL"
COMPOSITE_STEM = "F3_revised_draft"
PANEL_STEMS = {
    "A": "F3_panelA_regional_framework",
    "B": "F3_panelB_formal_evidence",
    "C": "F3_panelC_interpretation_layer",
    "D": "F3_panelD_scientific_synthesis",
}
PANEL_EXPORT_INCHES = {
    "A": (3.2, 2.6),
    "B": (4.8, 3.2),
    "C": (7.1, 4.1),
    "D": (7.1, 1.55),
}
FORMATS = ("pdf", "png", "svg")
EXPECTED_REGIONS = (
    "Beaufort Sea",
    "Laptev Sea",
    "Kara Sea",
    "Barents Sea",
    "Central Arctic",
)
EXPECTED_FAMILY_LABELS = (
    "SIC state",
    "SIC change",
    "Wind",
    "Wind",
    "Wind",
    "Pressure",
    "Cyclone proximity",
    "Ice-edge-relative wind",
)
FORMAL_FAMILIES = tuple(dict.fromkeys(EXPECTED_FAMILY_LABELS))

REQUIRED_INPUT_BASENAMES = {
    "regions.py",
    "NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc",
    "corrected_raw_evidence_770.csv",
    "corrected_region_summary.csv",
    "primary_floor015_paired_raw_770.csv",
    "STAGE2_TREND_PRESERVATION_TABLE.csv",
    "FIGURE_CASE_SELECTION_RECORD.md",
    "stage2_evidence_change_claim_map.csv",
}

PANEL_DATA_FILENAMES = {
    "formal_matrix": "F3_revision_round1_formal_evidence_matrix.csv",
    "interpretation_effects": "F3_revision_round1_interpretation_effect_rows.csv",
    "kara_changes": "F3_revision_round1_kara_registered_changes.csv",
}


@dataclass
class F3Payload:
    """Validated, display-ready payload derived from the accepted F3 inputs."""

    input_paths: dict[str, Path]
    raw: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame
    trend: pd.DataFrame
    claim_map: pd.DataFrame
    formal_matrix: pd.DataFrame
    laptev_effects: pd.DataFrame
    beaufort_effects: pd.DataFrame
    kara_claims: pd.DataFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input_paths(
    inputs: Sequence[Mapping[str, str]] | Mapping[str, Path | str],
    project_root: Path,
) -> dict[str, Path]:
    """Resolve the already-verified controlled-test inputs by basename."""

    resolved: dict[str, Path] = {}
    if isinstance(inputs, Mapping):
        iterator: Iterable[tuple[str, Path | str]] = inputs.items()
        for name, value in iterator:
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            resolved[name] = path
    else:
        for row in inputs:
            path = project_root / row["file_path"]
            name = path.name
            if name in resolved:
                raise ValueError(f"duplicate F3 input basename: {name}")
            resolved[name] = path

    missing = REQUIRED_INPUT_BASENAMES - set(resolved)
    if missing:
        raise ValueError(f"missing registered F3 inputs: {sorted(missing)}")
    for name in REQUIRED_INPUT_BASENAMES:
        if not resolved[name].is_file():
            raise FileNotFoundError(f"registered F3 input is unavailable: {resolved[name]}")
    return resolved


def _effect_rows(paired: pd.DataFrame, region: str, variable: str) -> pd.DataFrame:
    location = paired[
        (paired["diagnostic_type"] == "location_buffer")
        & (paired["region"] == region)
        & (paired["variable"] == variable)
    ].copy()
    location["display_label"] = location["buffer_km"].astype(int).astype(str) + " km"
    location["display_order"] = location["buffer_km"].map({100.0: 0, 300.0: 1, 500.0: 2})
    lag = paired[
        (paired["diagnostic_type"] == "lead_lag")
        & (paired["region"] == region)
        & (paired["variable"] == variable)
    ].copy()
    lag["display_label"] = lag["lag_group"].map(
        {"pre-event": "Pre-event", "event-time": "Event-time", "post-event": "Post-event"}
    )
    lag["display_order"] = lag["lag_group"].map(
        {"pre-event": 3, "event-time": 4, "post-event": 5}
    )
    rows = pd.concat([location, lag], ignore_index=True).sort_values("display_order")
    if len(rows) != 6 or rows["display_order"].isna().any():
        raise ValueError(f"F3 incomplete interpretation effect rows: {region} {variable}")
    return rows.reset_index(drop=True)


def prepare_f3_payload(
    config: Mapping[str, Any],
    inputs: Sequence[Mapping[str, str]] | Mapping[str, Path | str],
    project_root: Path,
) -> F3Payload:
    """Load and validate the same accepted Round-1 scientific payload."""

    if config.get("composite_stem") != COMPOSITE_STEM:
        raise ValueError("F3 composite-stem contract differs from the locked Round-1 config")
    if config.get("panel_stems") != PANEL_STEMS:
        raise ValueError("F3 standalone-panel stem contract differs from the locked Round-1 config")
    if tuple(config["regions"]) != EXPECTED_REGIONS:
        raise ValueError("F3 five-region display contract changed")
    if tuple(config["family_labels"]) != EXPECTED_FAMILY_LABELS:
        raise ValueError("F3 uniform six-family display contract changed")

    paths = resolve_input_paths(inputs, project_root)
    raw = pd.read_csv(paths["corrected_raw_evidence_770.csv"])
    summary = pd.read_csv(paths["corrected_region_summary.csv"])
    paired = pd.read_csv(paths["primary_floor015_paired_raw_770.csv"])
    trend = pd.read_csv(paths["STAGE2_TREND_PRESERVATION_TABLE.csv"])
    claim_map = pd.read_csv(paths["stage2_evidence_change_claim_map.csv"])

    expected_rows = int(config["expected_raw_evidence_rows"])
    if len(raw) != expected_rows or raw["evidence_id"].nunique() != expected_rows:
        raise ValueError("F3 raw-evidence identity gate failed")
    if len(paired) != expected_rows:
        raise ValueError("F3 paired primary/floor015 row gate failed")
    if len(summary) != int(config["expected_region_summary_rows"]):
        raise ValueError("F3 region-summary row gate failed")

    expected_pairs = {(region, mechanism) for region in config["region_keys"] for mechanism in config["mechanisms"]}
    actual_pairs = set(zip(summary["region"], summary["mechanism"]))
    if actual_pairs != expected_pairs or summary.duplicated(["region", "mechanism"]).any():
        raise ValueError("F3 complete 8-by-5 region/mechanism gate failed")
    if set(summary["mechanism"]) != set(config["mechanisms"]):
        raise ValueError("F3 diagnostic-family completeness gate failed")
    if not set(summary["manuscript_status"]).issubset(set(config["allowed_statuses"])):
        raise ValueError("F3 unexpected formal status")

    kara_claims = claim_map[
        claim_map["change_class"] == "KARA_FLOOR015_SENSITIVITY"
    ].sort_values("change_id")
    if (
        len(kara_claims) != int(config["expected_kara_registered_changes"])
        or set(kara_claims["author_review_row_id"]) != {"I003"}
    ):
        raise ValueError("F3 registered Kara sensitivity gate failed")

    region_order = {value: index for index, value in enumerate(config["region_keys"])}
    mechanism_order = {value: index for index, value in enumerate(config["mechanisms"])}
    formal = summary[
        ["region", "mechanism", "manuscript_status", "primary_supported", "manuscript_safe_statement"]
    ].copy()
    formal["region_order"] = formal["region"].map(region_order)
    formal["mechanism_order"] = formal["mechanism"].map(mechanism_order)
    formal = formal.sort_values(["mechanism_order", "region_order"]).reset_index(drop=True)

    laptev = _effect_rows(paired, "laptev_sea", "era5_v10_buffer")
    beaufort = _effect_rows(paired, "beaufort_sea", "era5_wspd10_buffer")
    kara_wrong = paired[
        (paired["diagnostic_type"] == "wrong_region")
        & (paired["source_region"] == "kara_sea")
        & (paired["variable"] == "era5_v10_buffer")
        & (paired["buffer_km"] == 100)
        & (paired["wrong_region"] == "beaufort_sea")
    ]
    if len(kara_wrong) != 1 or kara_wrong.iloc[0]["floor015_control_status"] != "source_primary_not_supported":
        raise ValueError("F3 Kara not-evaluable specificity gate failed")

    return F3Payload(
        input_paths=paths,
        raw=raw,
        summary=summary,
        paired=paired,
        trend=trend,
        claim_map=claim_map,
        formal_matrix=formal,
        laptev_effects=laptev,
        beaufort_effects=beaufort,
        kara_claims=kara_claims.reset_index(drop=True),
    )


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


def _draft_label(fig: plt.Figure, style: Mapping[str, Any]) -> None:
    fig.text(
        0.995,
        0.997,
        style["draft_label"],
        ha="right",
        va="top",
        fontsize=6.0,
        color="#7A3E00",
    )


def draw_f3_panel_a(
    ax: plt.Axes,
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
    *,
    show_panel_label: bool = True,
) -> list[plt.Axes]:
    """Draw Panel A from the locked NSIDC-0780 mask."""

    colors = style["colors"]
    mask_path = payload.input_paths["NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"]
    with xr.open_dataset(mask_path, decode_cf=False) as dataset:
        mask = dataset["sea_ice_region_surface_mask"].values
        x = dataset["x"].values / 1.0e6
        y = dataset["y"].values / 1.0e6

    display = np.zeros(mask.shape, dtype=np.int16)
    region_palette = ["#8FB3CF", "#D8A36A", "#8EBE9A", "#B49BC8", "#E0C36E"]
    for index, code in enumerate(config["region_codes"], start=1):
        display[mask == code] = index
    display[mask == 30] = 6
    cmap = ListedColormap(["#F7FAFC", *region_palette, "#D7D7D7"])
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N)
    ax.pcolormesh(x, y, display, cmap=cmap, norm=norm, shading="nearest", rasterized=True)

    for code, name in zip(config["region_codes"], config["regions"]):
        target = mask == code
        ax.contour(x, y, target.astype(int), levels=[0.5], colors=["#40464B"], linewidths=0.65)
        iy, ix = np.where(target)
        tx = float(np.mean(x[ix]))
        ty = float(np.mean(y[iy]))
        label = name.replace(" Sea", "\nSea") if name != "Central Arctic" else "Central\nArctic"
        ax.text(tx, ty, label, ha="center", va="center", fontsize=6.2, fontweight="semibold", color="#222222")

    ax.set_xlim(-2.65, 3.20)
    ax.set_ylim(-1.25, 2.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Five Arctic study regions", loc="left", fontweight="semibold", fontsize=9.0)
    ax.text(
        0.01,
        -0.060,
        "NSIDC-0780 native polar-stereographic mask · region colors only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.15,
        color=colors["synthesis"],
    )
    ax.text(
        0.01,
        0.985,
        "Defined before regional diagnostics",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.45,
        color=colors["synthesis"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )
    if show_panel_label:
        _panel_label(ax, "a")
    return [ax]


def draw_f3_panel_b(
    ax: plt.Axes,
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
    *,
    show_panel_label: bool = True,
) -> list[plt.Axes]:
    """Draw the complete formal 8-by-5 matrix with horizontal region labels."""

    colors = style["colors"]
    status_style = {
        "SUPPORTED_WITH_ROBUSTNESS": ("s", colors["formal"], colors["formal"], "R"),
        "SUPPORTED_PRIMARY_ONLY": ("o", colors["formal_light"], colors["formal"], "P"),
        "NOT_SUPPORTED": ("o", "white", colors["not_supported"], "N"),
    }
    family_background = {
        "SIC state": "#F4F8FB",
        "SIC change": "#EEF5F8",
        "Wind": "#F8F8F8",
        "Pressure": "#F4F8FB",
        "Cyclone proximity": "#F8F8F8",
        "Ice-edge-relative wind": "#F4F8FB",
    }
    for row_index, family in enumerate(config["family_labels"]):
        yloc = len(config["mechanisms"]) - 1 - row_index
        ax.axhspan(yloc - 0.48, yloc + 0.48, color=family_background[family], zorder=-3)

    lookup = payload.formal_matrix.set_index(["region", "mechanism"])["manuscript_status"].to_dict()
    for row_index, mechanism in enumerate(config["mechanisms"]):
        yloc = len(config["mechanisms"]) - 1 - row_index
        for column, region in enumerate(config["region_keys"]):
            marker, face, edge, letter = status_style[lookup[(region, mechanism)]]
            ax.scatter([column], [yloc], marker=marker, s=72, facecolor=face, edgecolor=edge, linewidth=1.0, zorder=2)
            ax.text(column, yloc, letter, ha="center", va="center", fontsize=5.7, fontweight="bold", color="#23313A")

    ax.set_xlim(-1.55, 4.55)
    ax.set_ylim(-0.65, 7.65)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Beaufort", "Laptev", "Kara", "Barents", "Central"], rotation=0, ha="center", fontsize=6.7)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=4)
    concise_labels = [
        "SIC state",
        "SIC change",
        "Wind speed",
        "Zonal wind u10",
        "Meridional wind v10",
        "Pressure (MSL)",
        "Cyclone proximity",
        "Edge-relative wind",
    ]
    ax.set_yticks(range(8))
    ax.set_yticklabels(list(reversed(concise_labels)), fontsize=6.45)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", color="white", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Formal evidence landscape (common workflow)", loc="left", fontweight="semibold", fontsize=8.5, pad=6)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["formal"], markeredgecolor=colors["formal"], markersize=6, label="R  supported with robustness"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["formal_light"], markeredgecolor=colors["formal"], markersize=6, label="P  supported in primary only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=colors["not_supported"], markersize=6, label="N  not supported"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.01, -0.155),
        frameon=False,
        ncol=2,
        fontsize=6.05,
        handletextpad=0.25,
        columnspacing=0.65,
    )
    ax.text(1.0, -0.315, "No score, tally, or regional rank", transform=ax.transAxes, ha="right", va="top", fontsize=6.15, color=colors["synthesis"])
    if show_panel_label:
        _panel_label(ax, "b")
    return [ax]


def _plot_effect_card(
    ax: plt.Axes,
    rows: pd.DataFrame,
    title: str,
    subtitle: str,
    colors: Mapping[str, str],
    *,
    show_legend: bool,
) -> None:
    y = np.arange(5, -1, -1, dtype=float)
    for offset, prefix, label, marker, color in [
        (0.10, "primary", "Primary", "o", colors["formal"]),
        (-0.10, "floor015", "floor015", "D", colors["interpretation"]),
    ]:
        values = rows[f"{prefix}_difference"].astype(float).to_numpy()
        low = rows[f"{prefix}_ci_low"].astype(float).to_numpy()
        high = rows[f"{prefix}_ci_high"].astype(float).to_numpy()
        ax.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - low, high - values]),
            fmt=marker,
            ms=3.2,
            color=color,
            ecolor=color,
            elinewidth=0.75,
            capsize=1.5,
            label=label,
            zorder=3,
        )
    ax.axvline(0, color="#777777", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.axhspan(2.45, 5.55, color="#F4F7F9", zorder=-2)
    ax.axhline(2.5, color=colors["grid"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["display_label"].tolist(), fontsize=6.6)
    ax.set_xlim(-0.72, 3.05)
    ax.set_xlabel("Event − background effect (m s$^{-1}$)", fontsize=7.6)
    ax.grid(axis="x", color=colors["grid"], lw=0.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=8.2, pad=20)
    ax.text(0.0, 1.008, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color=colors["synthesis"], linespacing=1.05)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.25, columnspacing=0.6, fontsize=6.2)


def _draw_kara_card(ax: plt.Axes, colors: Mapping[str, str]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#FFFCF5")
    ax.text(0.0, 1.035, "Kara threshold sensitivity", ha="left", va="bottom", fontsize=8.1, fontweight="semibold")
    ax.text(0.0, 0.988, "Three registered primary → floor015 changes", ha="left", va="top", fontsize=6.0, color=colors["synthesis"])
    rows = [
        (0.73, "100-km v10", "Primary: supported\n(p=.0421)", "floor015: not supported\n(p=.0577)"),
        (0.46, "post-event u10", "Primary: supported\n(p=.0479)", "floor015: not supported\n(p=.0754)"),
        (0.19, "v10 specificity\nvs Beaufort", "Primary: control pass", "floor015: not evaluable —\nsource primary unsupported"),
    ]
    ax.plot([0.0, 1.0], [0.88, 0.88], color=colors["grid"], lw=0.6)
    for ypos, label, primary_text, floor_text in rows:
        ax.axhspan(ypos - 0.105, ypos + 0.105, color="#FFFCF5" if ypos != 0.46 else "#F7F4EC", zorder=-2)
        ax.text(0.0, ypos, label, ha="left", va="center", fontsize=6.05, fontweight="semibold")
        ax.text(0.40, ypos + 0.035, primary_text, ha="left", va="center", fontsize=5.9, color=colors["formal"])
        ax.text(0.40, ypos - 0.045, floor_text, ha="left", va="center", fontsize=5.8, color=colors["interpretation"], linespacing=1.05)
        ax.plot([0.0, 1.0], [ypos - 0.125, ypos - 0.125], color=colors["grid"], lw=0.45)
    ax.text(0.0, 0.015, "Threshold sensitivity is not a\nrobustness category.", ha="left", va="bottom", fontsize=5.9, color=colors["synthesis"], linespacing=1.05)


def draw_f3_panel_c(
    fig: plt.Figure,
    subplot_spec: Any,
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
    *,
    show_panel_label: bool = True,
) -> list[plt.Axes]:
    """Draw Panel C as one logical panel with three registered case views."""

    del config  # The selected cases are already validated in the payload.
    colors = style["colors"]
    grid = subplot_spec.subgridspec(
        2,
        3,
        height_ratios=[0.38, 1.0],
        width_ratios=[1.0, 1.0, 1.08],
        hspace=0.060,
        wspace=0.35,
    )
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    label_prefix = "c   " if show_panel_label else ""
    header.text(0.0, 0.78, f"{label_prefix}INTERPRETATION LAYER", ha="left", va="center", fontsize=9.0, fontweight="bold", color=colors["interpretation"])
    header.text(0.0, 0.43, "Explanatory cases do not replace the complete formal comparison in panel b", ha="left", va="center", fontsize=6.4, color=colors["synthesis"])

    ax_laptev = fig.add_subplot(grid[1, 0])
    ax_beaufort = fig.add_subplot(grid[1, 1])
    ax_kara = fig.add_subplot(grid[1, 2])
    _plot_effect_card(ax_laptev, payload.laptev_effects, "Laptev v10", "PRIMARY SUPPORT\nspecificity not established", colors, show_legend=True)
    _plot_effect_card(ax_beaufort, payload.beaufort_effects, "Beaufort wind speed", "NOT SUPPORTED\ndirectional tendency only", colors, show_legend=False)
    for ax in (ax_laptev, ax_beaufort):
        ax.set_facecolor("#FFFCF5")
    _draw_kara_card(ax_kara, colors)
    return [header, ax_laptev, ax_beaufort, ax_kara]


def draw_f3_panel_d(
    ax: plt.Axes,
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
    *,
    show_panel_label: bool = True,
) -> list[plt.Axes]:
    """Draw the exact bounded synthesis with improved lower clearance."""

    del payload
    colors = style["colors"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.08),
            0.98,
            0.82,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#EEF2F4",
            edgecolor=colors["synthesis"],
            linewidth=0.9,
        )
    )
    ax.text(0.035, 0.79, "Scientific synthesis", ha="left", va="top", fontsize=8.0, fontweight="semibold", color=colors["synthesis"])
    ax.text(0.035, 0.50, config["synthesis_statement"], ha="left", va="center", fontsize=7.8, fontweight="semibold", wrap=True)
    ax.text(0.035, 0.27, "Regional atmospheric associations occurred within different background sea-ice states.", ha="left", va="center", fontsize=7.0)
    ax.text(0.965, 0.16, "Event-associated covariation · not causal forcing · no between-region ranking", ha="right", va="center", fontsize=6.3, color=colors["synthesis"])
    if show_panel_label:
        _panel_label(ax, "d")
    return [ax]


def render_f3_composite(
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    """Create the Round-1 composite using the shared panel draw functions."""

    fig = plt.figure(figsize=config["canvas_inches"], constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[3.0, 3.72, 1.22],
        width_ratios=[0.90, 1.24],
        hspace=0.51,
        wspace=0.39,
        left=0.08,
        right=0.98,
        top=0.875,
        bottom=0.055,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_d = fig.add_subplot(outer[2, :])
    axes = {
        "A": draw_f3_panel_a(ax_a, payload, config, style),
        "B": draw_f3_panel_b(ax_b, payload, config, style),
        "C": draw_f3_panel_c(fig, outer[1, :], payload, config, style),
        "D": draw_f3_panel_d(ax_d, payload, config, style),
    }
    fig.suptitle("Regional environmental organization of VRILEs", x=0.08, y=0.955, ha="left", fontsize=12.5, fontweight="bold")
    _draft_label(fig, style)
    return fig, axes


def render_f3_standalone_panel(
    panel_id: str,
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create one independently exportable A-D panel without crop heuristics."""

    panel = panel_id.upper()
    if panel not in PANEL_EXPORT_INCHES:
        raise ValueError(f"unknown F3 panel: {panel_id}")
    fig = plt.figure(figsize=PANEL_EXPORT_INCHES[panel], constrained_layout=False)
    if panel == "A":
        grid = fig.add_gridspec(1, 1, left=0.10, right=0.97, top=0.88, bottom=0.16)
        axes = draw_f3_panel_a(fig.add_subplot(grid[0, 0]), payload, config, style)
    elif panel == "B":
        grid = fig.add_gridspec(1, 1, left=0.32, right=0.98, top=0.82, bottom=0.27)
        axes = draw_f3_panel_b(fig.add_subplot(grid[0, 0]), payload, config, style)
    elif panel == "C":
        grid = fig.add_gridspec(1, 1, left=0.095, right=0.985, top=0.90, bottom=0.12)
        axes = draw_f3_panel_c(fig, grid[0, 0], payload, config, style)
    else:
        grid = fig.add_gridspec(1, 1, left=0.065, right=0.985, top=0.84, bottom=0.10)
        axes = draw_f3_panel_d(fig.add_subplot(grid[0, 0]), payload, config, style)
    _draft_label(fig, style)
    return fig, axes


def render_f3_round1_set(
    payload: F3Payload,
    config: Mapping[str, Any],
    style: Mapping[str, Any],
) -> tuple[dict[str, plt.Figure], dict[str, dict[str, list[plt.Axes]] | list[plt.Axes]]]:
    """Return the composite and standalone A-D figures without writing files."""

    composite, composite_axes = render_f3_composite(payload, config, style)
    figures: dict[str, plt.Figure] = {COMPOSITE_STEM: composite}
    axes: dict[str, dict[str, list[plt.Axes]] | list[plt.Axes]] = {COMPOSITE_STEM: composite_axes}
    for panel, stem in PANEL_STEMS.items():
        figure, panel_axes = render_f3_standalone_panel(panel, payload, config, style)
        figures[stem] = figure
        axes[stem] = panel_axes
    return figures, axes


def panel_data_frames(payload: F3Payload) -> dict[str, pd.DataFrame]:
    """Return the unchanged Round-1 panel-data tables for deterministic writing."""

    effects = pd.concat(
        [
            payload.laptev_effects.assign(case="Laptev v10"),
            payload.beaufort_effects.assign(case="Beaufort wspd10"),
        ],
        ignore_index=True,
    )
    keep = [
        "case",
        "diagnostic_type",
        "region",
        "variable",
        "buffer_km",
        "lag_group",
        "display_label",
        "primary_difference",
        "primary_ci_low",
        "primary_ci_high",
        "primary_p_FDR",
        "floor015_difference",
        "floor015_ci_low",
        "floor015_ci_high",
        "floor015_p_FDR",
    ]
    return {
        PANEL_DATA_FILENAMES["formal_matrix"]: payload.formal_matrix.copy(),
        PANEL_DATA_FILENAMES["interpretation_effects"]: effects[keep].copy(),
        PANEL_DATA_FILENAMES["kara_changes"]: payload.kara_claims.copy(),
    }


def write_panel_data(payload: F3Payload, out_dir: Path) -> list[Path]:
    """Write the three stable panel-data tables; no scientific fields change."""

    outputs: list[Path] = []
    for filename, frame in panel_data_frames(payload).items():
        path = out_dir / filename
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")
        outputs.append(path)
    return outputs


def _format_metadata(title: str, suffix: str, style: Mapping[str, Any]) -> dict[str, Any]:
    if suffix == "pdf":
        return {
            "Title": title,
            "Author": "VRILE Round-1 controlled-test workflow",
            "Subject": "Scientific draft figure; not for submission",
            "Creator": style["fixed_metadata"]["Creator"],
            "Producer": style["fixed_metadata"]["Producer"],
            "CreationDate": None,
            "ModDate": None,
        }
    if suffix == "svg":
        return {
            "Title": title,
            "Description": "Scientific draft figure; not for submission",
            "Creator": style["fixed_metadata"]["Creator"],
            "Date": None,
        }
    return {
        "Software": style["fixed_metadata"]["Creator"],
        "Title": title,
        "Description": "Scientific draft figure; not for submission",
    }


def save_f3_figure_set(
    figures: Mapping[str, plt.Figure],
    out_dir: Path,
    style: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Save five figures × three formats and return hashable output records."""

    records: list[dict[str, Any]] = []
    panel_by_stem = {value: key for key, value in PANEL_STEMS.items()}
    for stem, fig in figures.items():
        title = "Regional environmental organization of VRILEs"
        for suffix in FORMATS:
            path = out_dir / f"{stem}.{suffix}"
            metadata = _format_metadata(title, suffix, style)
            if suffix == "png":
                fig.savefig(path, dpi=style["dpi"], metadata=metadata)
            else:
                fig.savefig(path, metadata=metadata)
            records.append(
                {
                    "path": path.name,
                    "panel_id": "COMPOSITE" if stem == COMPOSITE_STEM else panel_by_stem[stem],
                    "format": suffix,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        plt.close(fig)
    if len(records) != 15:
        raise ValueError(f"unexpected F3 Round-1 rendered-file count: {len(records)}")
    return records


def text_bounds_issues(fig: plt.Figure, tolerance_pixels: float = 1.0) -> list[dict[str, Any]]:
    """Report text/legend extents outside the explicit standalone canvas."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    issues: list[dict[str, Any]] = []
    artists = fig.findobj(lambda artist: isinstance(artist, (Text, Legend)))
    for artist in artists:
        if not artist.get_visible():
            continue
        if isinstance(artist, Text) and not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < frame.x0 - tolerance_pixels
            or bounds.y0 < frame.y0 - tolerance_pixels
            or bounds.x1 > frame.x1 + tolerance_pixels
            or bounds.y1 > frame.y1 + tolerance_pixels
        ):
            issues.append(
                {
                    "artist_type": type(artist).__name__,
                    "text": artist.get_text() if isinstance(artist, Text) else "LEGEND",
                    "bounds_pixels": [float(value) for value in bounds.bounds],
                    "figure_bounds_pixels": [float(value) for value in frame.bounds],
                }
            )
    return issues


def f3_qa_metadata(payload: F3Payload) -> dict[str, Any]:
    """Return the unchanged scientific gates plus Round-1 presentation records."""

    counts = payload.formal_matrix["manuscript_status"].value_counts().sort_index().to_dict()
    return {
        "round": "ROUND1_TYPOGRAPHY_AND_PANEL_EXPORT_ONLY",
        "raw_evidence_rows": int(len(payload.raw)),
        "unique_evidence_ids": int(payload.raw["evidence_id"].nunique()),
        "region_summary_rows": int(len(payload.summary)),
        "formal_region_count": int(payload.formal_matrix["region"].nunique()),
        "formal_mechanism_count": int(payload.formal_matrix["mechanism"].nunique()),
        "formal_family_count": len(FORMAL_FAMILIES),
        "formal_families": list(FORMAL_FAMILIES),
        "formal_matrix_cells": int(len(payload.formal_matrix)),
        "formal_uniform_region_mechanism_grid": True,
        "formal_status_counts": {key: int(value) for key, value in counts.items()},
        "trend_table_rows_verified": int(len(payload.trend)),
        "kara_registered_changes": int(len(payload.kara_claims)),
        "interpretation_case_rows": {"Laptev v10": 6, "Beaufort wspd10": 6},
        "interpretation_examples": [
            "Laptev v10",
            "Beaufort wind-speed tendency",
            "Kara threshold sensitivity",
        ],
        "interpretation_label": "INTERPRETATION LAYER",
        "causal_arrows": False,
        "composite_scores": False,
        "regional_ranking": False,
        "scientific_content_changed": False,
        "region_tick_rotation_deg": 0,
        "nonhorizontal_tick_labels": [],
        "standalone_panels": ["A", "B", "C", "D"],
        "panel_output_formats": list(FORMATS),
        "expected_rendered_file_count": 15,
        "annotation_inventory": [
            "INTERPRETATION LAYER",
            "The corrected analysis did not identify a single stable and spatially specific atmospheric fingerprint shared across Arctic study regions.",
            "NOT EVALUABLE",
            "No score, tally, or regional rank",
        ],
    }


def metadata_record(
    *,
    payload: F3Payload,
    run_id: str,
    verified_inputs: Sequence[Mapping[str, Any]],
    workflow: Mapping[str, Any],
    configuration: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    panel_data: Sequence[Mapping[str, Any]],
    clipping_issues: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Create an F3 record suitable for the shared Round-1 metadata list."""

    qa = f3_qa_metadata(payload)
    if len(outputs) != 15:
        raise ValueError(f"F3 metadata requires 15 rendered files, got {len(outputs)}")
    if len(panel_data) != 3:
        raise ValueError(f"F3 metadata requires three panel-data files, got {len(panel_data)}")
    qa["actual_rendered_file_count"] = len(outputs)
    qa["composite_rendered_file_count"] = 3
    qa["standalone_panel_rendered_file_count"] = 12
    qa["panel_data_file_count"] = len(panel_data)
    qa["text_clipping_issue_counts"] = {
        name: len(items) for name, items in clipping_issues.items()
    }
    return {
        "figure_id": "F3",
        "run_id": run_id,
        "status": ROUND1_STATUS,
        "inputs": list(verified_inputs),
        "workflow": dict(workflow),
        "configuration": list(configuration),
        "outputs": list(outputs),
        "panel_data": list(panel_data),
        "qa": qa,
        "final_figure_generation_authorized": False,
        "scientific_recalculation_authorized": False,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic JSON for a calling controlled-build driver."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "COMPOSITE_STEM",
    "FORMATS",
    "F3Payload",
    "PANEL_DATA_FILENAMES",
    "PANEL_EXPORT_INCHES",
    "PANEL_STEMS",
    "ROUND1_STATUS",
    "draw_f3_panel_a",
    "draw_f3_panel_b",
    "draw_f3_panel_c",
    "draw_f3_panel_d",
    "f3_qa_metadata",
    "metadata_record",
    "panel_data_frames",
    "prepare_f3_payload",
    "render_f3_composite",
    "render_f3_round1_set",
    "render_f3_standalone_panel",
    "resolve_input_paths",
    "save_f3_figure_set",
    "sha256",
    "text_bounds_issues",
    "write_json",
    "write_panel_data",
]
