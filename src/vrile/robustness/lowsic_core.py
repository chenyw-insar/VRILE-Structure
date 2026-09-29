"""Stage-3 primitives for the prespecified low-SIC robustness audit.

This module deliberately writes no production artifacts.  It reproduces the
archived, frozen Stage-3 component procedure and applies only the two
prespecified start-SIC floors and three strict component-size filters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats

from vrile.regions import NSIDC_0780_REGION_IDS
from vrile.stage3.grid import OCEAN_CODES, read_sic


REGION_NAMES = {code: name for name, code in NSIDC_0780_REGION_IDS.items()}
KNOWN_NON_OCEAN_CODES = {30, 32, 33, 34, 35, 40}


def validate_surface_codes(codes: np.ndarray) -> None:
    """Reject region-mask codes outside the current NSIDC-0780 contract."""

    observed = set(int(value) for value in np.unique(codes[np.isfinite(codes)]))
    unknown = observed - set(OCEAN_CODES) - KNOWN_NON_OCEAN_CODES
    if unknown:
        raise ValueError(f"Unknown NSIDC-0780 surface codes: {sorted(unknown)}")


FLOORS: tuple[float | None, ...] = (None, 0.15, 0.30)
MIN_CELLS: tuple[int, ...] = (4, 20, 50)
LOSS_THRESHOLD = -0.10
EXPECTED_EVENTS = 99
EXPECTED_COMPONENTS = 3463


@dataclass(frozen=True)
class Component:
    """One 8-neighbour loss component retained by a strict size threshold."""

    component_id: int
    mask: np.ndarray
    cell_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_components(
    signed: np.ndarray,
    valid_mask: np.ndarray,
    strict_min_cells: int,
) -> list[Component]:
    """Reproduce production closing, valid-mask reapplication and strict size gate."""

    arr = np.asarray(signed, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    loss = np.isfinite(arr) & valid & (arr <= LOSS_THRESHOLD)
    loss = ndimage.binary_closing(loss, structure=np.ones((3, 3), dtype=bool), iterations=1)
    # This second intersection is essential: closing may not restore a floor-excluded cell.
    loss &= np.isfinite(arr) & valid
    labels, count = ndimage.label(loss, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return []
    sizes = ndimage.sum(np.ones_like(arr), labels, index=np.arange(1, count + 1))
    output: list[Component] = []
    for label_id, size in enumerate(sizes, start=1):
        if int(size) <= int(strict_min_cells):
            continue
        mask = labels == label_id
        output.append(Component(len(output) + 1, mask, int(size)))
    return output


def effective_number(losses: np.ndarray) -> float:
    """Compute the production Hill-number effective component count."""

    values = np.asarray(losses, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    total = float(values.sum())
    if total <= 0:
        return np.nan
    proportions = values / total
    return float(1.0 / np.sum(proportions**2))


def component_metrics(losses: list[float]) -> tuple[float, float, float]:
    """Return Neff, largest share and top-three share for retained losses."""

    values = np.asarray(losses, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    total = float(values.sum())
    if total <= 0:
        return np.nan, np.nan, np.nan
    shares = np.sort(values / total)[::-1]
    return effective_number(values), float(shares[0]), float(shares[:3].sum())


def load_cell_area_and_regions(baseline_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NSIDC-0771 cell areas and NSIDC-0780 surface-region codes."""

    area_path = baseline_root / "data/raw/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc"
    mask_path = baseline_root / "data/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
    with xr.open_dataset(area_path) as dataset:
        area = np.asarray(dataset["cell_area"].values, dtype=float) / 1_000_000.0
    with xr.open_dataset(mask_path, mask_and_scale=False) as dataset:
        regions = np.asarray(dataset["sea_ice_region_surface_mask"].values, dtype=np.int16)
    if area.shape != regions.shape:
        raise ValueError(f"grid mismatch: area={area.shape}, region={regions.shape}")
    validate_surface_codes(regions)
    return area, regions


def load_primary_tables(baseline_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three frozen Stage-3 tables needed by the audit."""

    root = baseline_root / "outputs/panarctic_local_contribution"
    components = pd.read_csv(root / "panarctic_loss_components.csv")
    events = pd.read_csv(root / "panarctic_loss_field_event_summary.csv")
    metrics = pd.read_csv(root / "panarctic_spatial_composition_metrics.csv")
    if len(components) != EXPECTED_COMPONENTS or len(events) != EXPECTED_EVENTS:
        raise ValueError(
            f"frozen baseline cardinality mismatch: components={len(components)}, events={len(events)}"
        )
    return components, events, metrics


def reproduce_min_cell_baseline(
    components: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute the five supplied min-cells-only sanity rows from the frozen table."""

    rows = []
    event_ids = events["pan_event_id"].astype(str).tolist()
    for minimum in (4, 10, 20, 50, 100):
        retained = components[components["cell_count"].astype(int) > minimum]
        per_event = []
        for event_id in event_ids:
            losses = retained.loc[
                retained["pan_event_id"].astype(str) == event_id,
                "integrated_sic_loss_km2eq",
            ].to_numpy(float)
            neff, largest, top3 = component_metrics(losses.tolist())
            per_event.append((len(losses), neff, largest, top3))
        frame = pd.DataFrame(per_event, columns=["n_components", "Neff", "largest", "top3"])
        rows.append(
            {
                "strict_min_cells": minimum,
                "component_count": int(len(retained)),
                "event_count": int(len(frame)),
                "median_n_components": float(frame["n_components"].median()),
                "median_Neff": float(frame["Neff"].median()),
                "median_largest_component_share": float(frame["largest"].median()),
                "median_top3_component_share": float(frame["top3"].median()),
            }
        )
    expected = {
        4: (3463, 36.0, 7.76, 0.254, 0.544),
        10: (2465, 26.0, 7.43, 0.260, 0.561),
        20: (1774, 19.0, 7.01, 0.272, 0.579),
        50: (1078, 11.0, 5.92, 0.296, 0.622),
        100: (677, 7.0, 4.94, 0.327, 0.699),
    }
    for row in rows:
        count, ncomp, neff, largest, top3 = expected[int(row["strict_min_cells"])]
        observed = (
            int(row["component_count"]),
            round(row["median_n_components"], 0),
            round(row["median_Neff"], 2),
            round(row["median_largest_component_share"], 3),
            round(row["median_top3_component_share"], 3),
        )
        if observed != (count, ncomp, neff, largest, top3):
            raise RuntimeError(f"STOP baseline reproduction mismatch: observed={observed}")
    return pd.DataFrame(rows)


def _bin_fractions(start: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Summarize cell and positive-loss fractions in the three prescribed SIC bins."""

    finite = np.isfinite(start)
    bins = (
        ("lt_0.15", finite & (start < 0.15)),
        ("0.15_0.30", finite & (start >= 0.15) & (start < 0.30)),
        ("ge_0.30", finite & (start >= 0.30)),
    )
    n = int(finite.sum())
    total_loss = float(np.nansum(np.where(finite, weights, 0.0), dtype=np.float64))
    row: dict[str, float] = {}
    for label, mask in bins:
        row[f"fraction_cells_start_sic_{label}"] = float(mask.sum() / n) if n else np.nan
        loss = float(np.nansum(np.where(mask, weights, 0.0), dtype=np.float64))
        row[f"fraction_loss_start_sic_{label}"] = loss / total_loss if total_loss > 0 else np.nan
    return row


def _event_row(
    event_id: str,
    event_date: str,
    floor: float | None,
    minimum: int,
    losses: list[float],
    eligible_loss: float,
    primary_total: float,
    primary_neff: float,
) -> dict[str, object]:
    """Build one all-event scenario record without dropping zero-component events."""

    neff, largest, top3 = component_metrics(losses)
    resolved = float(np.sum(losses, dtype=np.float64))
    return {
        "pan_event_id": event_id,
        "event_date": event_date,
        "sic_floor": "none" if floor is None else float(floor),
        "strict_min_cells": int(minimum),
        "n_components": int(len(losses)),
        "Neff": neff,
        "largest_component_share": largest,
        "top3_component_share": top3,
        "resolved_loss_km2eq": resolved,
        "eligible_loss_km2eq": float(eligible_loss),
        "primary_total_loss_km2eq": float(primary_total),
        "resolved_fraction_within_eligible_loss": resolved / eligible_loss if eligible_loss > 0 else np.nan,
        "resolved_fraction_vs_primary_total_loss": resolved / primary_total if primary_total > 0 else np.nan,
        "primary_Neff": float(primary_neff),
        "spearman_pair_eligible": bool(np.isfinite(neff) and np.isfinite(primary_neff)),
    }


def run_stage3(
    baseline_root: Path,
    *,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Stage-3 component, low-SIC and named-region diagnostics."""

    components, events, metrics = load_primary_tables(baseline_root)
    if quick:
        events = events.head(3).copy()
        event_set = set(events["pan_event_id"].astype(str))
        components = components[components["pan_event_id"].astype(str).isin(event_set)].copy()
        metrics = metrics[metrics["pan_event_id"].astype(str).isin(event_set)].copy()
    area, regions = load_cell_area_and_regions(baseline_root)
    ocean = np.isin(regions, list(OCEAN_CODES))
    metric_lookup = metrics.set_index("pan_event_id")
    event_lookup = events.set_index("pan_event_id")
    scenario_rows: list[dict[str, object]] = []
    component_diagnostics: list[dict[str, object]] = []
    named_rows: list[dict[str, object]] = []

    # floor=none may use the frozen component table exactly as prespecified.
    for event in events.itertuples(index=False):
        event_id = str(event.pan_event_id)
        event_date = str(event.event_date)
        frozen = components[components["pan_event_id"].astype(str) == event_id]
        total = float(event.total_positive_sic_loss_km2eq)
        primary_neff = float(metric_lookup.loc[event_id, "effective_component_number_sic"])
        for minimum in MIN_CELLS:
            retained = frozen[frozen["cell_count"].astype(int) > minimum]
            losses = retained["integrated_sic_loss_km2eq"].astype(float).tolist()
            scenario_rows.append(
                _event_row(event_id, event_date, None, minimum, losses, total, total, primary_neff)
            )

    for event in events.itertuples(index=False):
        event_id = str(event.pan_event_id)
        event_date = pd.Timestamp(event.event_date).normalize()
        start_date = event_date - pd.Timedelta(days=5)
        start, _, _ = read_sic(start_date)
        end, _, _ = read_sic(event_date)
        signed = end - start
        # Production stores loss as float32 before area integration.
        loss_weight = np.maximum(0.0, -signed).astype(np.float32).astype(float) * area
        total = float(event.total_positive_sic_loss_km2eq)
        primary_neff = float(metric_lookup.loc[event_id, "effective_component_number_sic"])

        primary_components = detect_components(signed, ocean & np.isfinite(signed), 4)
        frozen = components[components["pan_event_id"].astype(str) == event_id].sort_values("component_id")
        if len(primary_components) != len(frozen):
            raise RuntimeError(
                f"STOP primary segmentation mismatch {event_id}: detected={len(primary_components)} frozen={len(frozen)}"
            )
        for component, frozen_row in zip(primary_components, frozen.itertuples(index=False)):
            weight = np.where(component.mask, loss_weight, 0.0)
            component_loss = float(np.nansum(weight, dtype=np.float64))
            if component.cell_count != int(frozen_row.cell_count) or not np.isclose(
                component_loss, float(frozen_row.integrated_sic_loss_km2eq), rtol=2e-7, atol=1e-5
            ):
                raise RuntimeError(
                    f"STOP primary component mismatch {event_id}/{component.component_id}: "
                    f"cells={component.cell_count}/{frozen_row.cell_count}, loss={component_loss}/{frozen_row.integrated_sic_loss_km2eq}"
                )
            start_values = start[component.mask]
            weights = weight[component.mask]
            total_weight = float(np.nansum(weights, dtype=np.float64))
            row = {
                "event_id": event_id,
                "pan_event_id": event_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "component_id": int(component.component_id),
                "cell_count": int(component.cell_count),
                "integrated_sic_loss_km2eq": component_loss,
                "loss_weighted_mean_start_sic": (
                    float(np.nansum(start_values * weights, dtype=np.float64) / total_weight)
                    if total_weight > 0
                    else np.nan
                ),
                "cell_weighted_mean_start_sic": float(np.nanmean(start_values)),
            }
            row.update(_bin_fractions(start_values, weights))
            value = row["loss_weighted_mean_start_sic"]
            row["component_start_sic_class"] = (
                "lt_0.15" if value < 0.15 else "0.15_0.30" if value < 0.30 else "ge_0.30"
            )
            component_diagnostics.append(row)

        valid_physical = ocean & np.isfinite(signed) & np.isfinite(start)
        for code, region_name in REGION_NAMES.items():
            region_mask = valid_physical & (regions == code)
            weights = np.where(region_mask, loss_weight, 0.0)
            total_region = float(np.nansum(weights, dtype=np.float64))
            row = {
                "event_id": event_id,
                "pan_event_id": event_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "region_code": int(code),
                "region": region_name,
                "total_loss_km2eq": total_region,
            }
            for label, mask in (
                ("lt_0.15", start < 0.15),
                ("0.15_0.30", (start >= 0.15) & (start < 0.30)),
                ("ge_0.30", start >= 0.30),
            ):
                value = float(np.nansum(np.where(region_mask & mask, loss_weight, 0.0), dtype=np.float64))
                row[f"loss_start_sic_{label}_km2eq"] = value
                row[f"fraction_loss_start_sic_{label}"] = value / total_region if total_region > 0 else np.nan
            named_rows.append(row)

        for floor in (0.15, 0.30):
            valid = valid_physical & (start >= floor)
            eligible = float(np.nansum(np.where(valid, loss_weight, 0.0), dtype=np.float64))
            for minimum in MIN_CELLS:
                detected = detect_components(signed, valid, minimum)
                losses = [
                    float(np.nansum(np.where(item.mask, loss_weight, 0.0), dtype=np.float64))
                    for item in detected
                ]
                scenario_rows.append(
                    _event_row(
                        event_id,
                        event_date.strftime("%Y-%m-%d"),
                        floor,
                        minimum,
                        losses,
                        eligible,
                        total,
                        primary_neff,
                    )
                )

    return (
        pd.DataFrame(component_diagnostics),
        pd.DataFrame(scenario_rows),
        pd.DataFrame(named_rows),
        events.reset_index(drop=True),
    )


def summarize_stage3(event_level: pd.DataFrame, primary_median: float) -> pd.DataFrame:
    """Summarize all nine scenarios and apply the prespecified rules."""

    rows: list[dict[str, object]] = []
    for (floor, minimum), group in event_level.groupby(["sic_floor", "strict_min_cells"], sort=False):
        paired = group[group["spearman_pair_eligible"]]
        rho = (
            float(stats.spearmanr(paired["Neff"], paired["primary_Neff"]).statistic)
            if len(paired) >= 2
            else np.nan
        )
        row: dict[str, object] = {
            "sic_floor": floor,
            "strict_min_cells": int(minimum),
            "n_events_total": int(len(group)),
            "n_events_with_components": int((group["n_components"] > 0).sum()),
            "n_events_without_components": int((group["n_components"] == 0).sum()),
            "fraction_events_with_ge2_components": float((group["n_components"] >= 2).mean()),
            "fraction_events_with_ge3_components": float((group["n_components"] >= 3).mean()),
            "event_level_spearman_rho_Neff_vs_primary": rho,
        }
        for column in (
            "n_components",
            "Neff",
            "largest_component_share",
            "top3_component_share",
            "resolved_loss_km2eq",
            "eligible_loss_km2eq",
            "resolved_fraction_within_eligible_loss",
            "resolved_fraction_vs_primary_total_loss",
        ):
            values = group[column].dropna().astype(float)
            for label, quantile in (("min", 0), ("p10", 0.10), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("p90", 0.90), ("max", 1)):
                row[f"{column}_{label}"] = float(values.quantile(quantile)) if len(values) else np.nan
        median = float(row["Neff_median"])
        row["Neff_absolute_change_vs_primary"] = median - primary_median
        row["Neff_relative_change_vs_primary"] = (median - primary_median) / primary_median
        row["headline_exact_value_within_15pct"] = bool(abs(row["Neff_relative_change_vs_primary"]) <= 0.15)
        row["qualitative_multicenter_supported"] = bool(
            median >= 3.0
            and float(row["largest_component_share_median"]) < 0.5
            and np.isfinite(rho)
            and rho >= 0.7
        )
        row["scenario_class"] = "STRICT_STRESS" if str(floor) == "0.3" else "CORE"
        rows.append(row)
    output = pd.DataFrame(rows)
    primary = output[
        output["sic_floor"].astype(str).eq("none") & output["strict_min_cells"].eq(4)
    ]
    if len(primary) != 1:
        raise RuntimeError(f"primary scenario must be unique, found {len(primary)} rows")
    change_metrics = (
        "n_components",
        "Neff",
        "largest_component_share",
        "top3_component_share",
        "resolved_loss_km2eq",
        "eligible_loss_km2eq",
        "resolved_fraction_within_eligible_loss",
        "resolved_fraction_vs_primary_total_loss",
    )
    for metric in change_metrics:
        baseline = float(primary.iloc[0][f"{metric}_median"])
        absolute = output[f"{metric}_median"] - baseline
        output[f"{metric}_absolute_change_vs_primary"] = absolute
        output[f"{metric}_relative_change_vs_primary"] = absolute / baseline if baseline != 0 else np.nan
    return output


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic UTF-8 JSON for audit metadata."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
