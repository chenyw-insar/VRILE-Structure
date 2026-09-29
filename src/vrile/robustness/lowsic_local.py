"""Frozen-catalog diagnostics and independent local-event floor reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from vrile.io import projected_grid_geometry
from vrile.local_objects import detect_connected_components, merge_local_patches
from vrile.stage3.grid import find_sic_file, read_sic, read_sic_with_lonlat
from vrile.robustness.lowsic_core import REGION_NAMES, load_cell_area_and_regions


LOCAL_FLOORS: tuple[float | None, ...] = (None, 0.15, 0.30)
EXPECTED_LEVEL_COUNTS = {"broad": 9530, "severe": 2427, "major_severe": 554}


def _floor_label(value: float | None) -> str:
    """Return the stable CSV label for one SIC-floor scenario."""

    return "none" if value is None else f"{value:.2f}"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance with the production 6371-km Earth radius."""

    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    term = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(term))


def _patch_components(diff: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    """Apply the primary local 8-neighbour detector with no binary closing and >=50 cells."""

    mask = np.isfinite(diff) & valid & (diff <= -0.10)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return []
    sizes = ndimage.sum(np.ones_like(diff), labels, index=np.arange(1, count + 1))
    return [labels == label for label, size in enumerate(sizes, start=1) if int(size) >= 50]


def _primary_region(mask: np.ndarray, loss: np.ndarray, region_codes: np.ndarray) -> str:
    """Assign the named region containing most component SIC-loss weight."""

    candidates: list[tuple[float, int, int, str]] = []
    for code, name in REGION_NAMES.items():
        overlap = mask & (region_codes == code)
        cells = int(overlap.sum())
        if cells:
            weight = float(np.nansum(np.where(overlap, loss, 0.0), dtype=np.float64))
            candidates.append((weight, cells, -code, name))
    if not candidates:
        return ""
    return max(candidates)[3]


def _merge_local_patches(
    patches: pd.DataFrame,
    *,
    return_membership: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce consolidation and optionally expose the patch/event membership."""

    if patches.empty:
        empty_events = pd.DataFrame()
        empty_membership = pd.DataFrame(
            columns=[
                "unique_local_event_id",
                "object_id",
                "date",
                "start_date",
                "window_days",
                "threshold",
            ]
        )
        return (empty_events, empty_membership) if return_membership else empty_events
    events: list[dict[str, object]] = []
    next_id = 1
    for region, group in patches.groupby("region", sort=True):
        active: list[dict[str, object]] = []
        ordered = group.sort_values(["date", "cumulative_sic_loss"], ascending=[True, False])
        for row in ordered.to_dict("records"):
            date = pd.Timestamp(row["date"])
            still_active = []
            for event in active:
                if (date - event["last_date"]).days <= 2:
                    still_active.append(event)
                else:
                    events.append(event)
            active = still_active
            best_index = None
            best_distance = float("inf")
            for index, event in enumerate(active):
                days = (date - event["last_date"]).days
                if days < 0 or days > 2:
                    continue
                distance = _haversine_km(
                    float(row["centroid_lon"]),
                    float(row["centroid_lat"]),
                    float(event["last_lon"]),
                    float(event["last_lat"]),
                )
                if distance <= 300.0 and distance < best_distance:
                    best_index, best_distance = index, distance
            if best_index is None:
                active.append(
                    {
                        "unique_local_event_id": f"ULE{next_id:06d}",
                        "dominant_region": region,
                        "rows": [row],
                        "last_date": date,
                        "last_lon": float(row["centroid_lon"]),
                        "last_lat": float(row["centroid_lat"]),
                    }
                )
                next_id += 1
            else:
                event = active[best_index]
                event["rows"].append(row)
                event["last_date"] = date
                event["last_lon"] = float(row["centroid_lon"])
                event["last_lat"] = float(row["centroid_lat"])
        events.extend(active)

    rows = []
    member_rows = []
    for event in events:
        frame = pd.DataFrame(event["rows"])
        weights = frame["cumulative_sic_loss"].clip(lower=0).to_numpy(float)
        if weights.sum() <= 0:
            weights = np.ones(len(frame))
        dates = pd.to_datetime(frame["date"])
        rows.append(
            {
                "unique_local_event_id": event["unique_local_event_id"],
                "dominant_region": event["dominant_region"],
                "event_start": dates.min().strftime("%Y-%m-%d"),
                "event_end": dates.max().strftime("%Y-%m-%d"),
                "duration_days": int((dates.max() - dates.min()).days + 1),
                "n_daily_patches": int(len(frame)),
                "track_centroid_lon": float(np.average(frame["centroid_lon"], weights=weights)),
                "track_centroid_lat": float(np.average(frame["centroid_lat"], weights=weights)),
                "max_area_cells": int(frame["object_area_cells"].max()),
                "max_area_km2": float(frame["object_area_km2"].max()),
                "max_intensity": float((-frame["mean_sic_change"]).max()),
                "min_sic_change": float(frame["min_sic_change"].min()),
                "mean_sic_change": float(frame["mean_sic_change"].mean()),
                "cumulative_loss": float(frame["cumulative_sic_loss"].sum()),
                "window_days": 5,
                "sic_change_threshold": -0.10,
                "min_cells": int(frame["object_area_cells"].min()),
            }
        )
        for row in frame.to_dict("records"):
            member_rows.append(
                {
                    "unique_local_event_id": event["unique_local_event_id"],
                    "object_id": row["object_id"],
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "start_date": pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d"),
                    "window_days": int(row.get("window_days", 5)),
                    "threshold": float(row.get("threshold", -0.10)),
                }
            )
    event_frame = pd.DataFrame(rows).sort_values(["dominant_region", "event_start"]).reset_index(drop=True)
    membership = pd.DataFrame(member_rows)
    return (event_frame, membership) if return_membership else event_frame


def mark_severe(broad: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen primary top-20%, 100-cell, 2-day severe rule."""

    output = broad.copy()
    percentile = 0.80
    loss_threshold = output["cumulative_loss"].quantile(percentile)
    area_threshold = output["max_area_cells"].quantile(percentile)
    intensity_threshold = output["max_intensity"].quantile(percentile)
    regional_loss = output.groupby("dominant_region")["cumulative_loss"].transform(
        lambda values: values.quantile(percentile)
    )
    regional_area = output.groupby("dominant_region")["max_area_cells"].transform(
        lambda values: values.quantile(percentile)
    )
    size_ok = output["max_area_cells"] >= 100
    duration_or_intensity = (output["duration_days"] >= 2) | (output["max_intensity"] >= intensity_threshold)
    global_ok = (
        ((output["cumulative_loss"] >= loss_threshold) | (output["max_area_cells"] >= area_threshold))
        & size_ok
        & duration_or_intensity
    )
    regional_ok = (
        ((output["cumulative_loss"] >= regional_loss) | (output["max_area_cells"] >= regional_area))
        & size_ok
        & duration_or_intensity
    )
    output["is_severe_primary"] = global_ok | regional_ok
    return output[output["is_severe_primary"]].copy()


def mark_major(severe: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen union of global, regional and severity-index major rules."""

    frame = severe.copy()
    columns = ["cumulative_loss", "max_area_km2", "max_intensity", "duration_days"]
    for column in columns:
        frame[f"global_rank_{column}"] = frame[column].rank(pct=True)
        frame[f"region_rank_{column}"] = frame.groupby("dominant_region")[column].transform(
            lambda values: values.rank(pct=True)
        )
    frame["severity_index_global"] = frame[[f"global_rank_{column}" for column in columns]].mean(axis=1)
    selected: list[pd.DataFrame] = []
    for top in (0.10, 0.15):
        global_mask = (
            (frame["cumulative_loss"] >= frame["cumulative_loss"].quantile(1 - top))
            & (frame["max_area_km2"] >= frame["max_area_km2"].quantile(0.80))
            & (frame["duration_days"] >= 2)
        )
        regional_mask = (
            (frame["region_rank_cumulative_loss"] >= 1 - top)
            & (frame["region_rank_max_area_km2"] >= 0.80)
            & (frame["duration_days"] >= 2)
        )
        selected.extend([frame[global_mask], frame[regional_mask]])
    for top in (0.10, 0.15, 0.20):
        selected.append(frame[frame["severity_index_global"] >= frame["severity_index_global"].quantile(1 - top)])
    return pd.concat(selected, ignore_index=True).drop_duplicates("unique_local_event_id")


def reconstruct_local_catalogs(
    baseline_root: Path,
    *,
    quick: bool = False,
    floors: tuple[float | None, ...] = LOCAL_FLOORS,
    sparse_cell_dirs: dict[str, Path] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Resegment, consolidate and classify local events under selected floors.

    ``sparse_cell_dirs`` is used only by the downstream robustness branch.  If
    supplied, it persists the exact reconstructed patch cells in the existing
    Stage-3 sparse NPZ/index schema; primary processed data are never touched.
    """

    area, region_codes = load_cell_area_and_regions(baseline_root)
    pan_mask = np.isin(region_codes, list(REGION_NAMES))
    years = range(2019, 2022) if quick else range(1989, 2026)
    dates = [
        pd.Timestamp(date).normalize()
        for date in pd.date_range(f"{min(years)}-06-01", f"{max(years)}-08-31", freq="D")
        if date.year in years and date.month in (6, 7, 8)
    ]
    patch_rows: dict[str, list[dict[str, object]]] = {_floor_label(floor): [] for floor in floors}
    sparse_stores: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        _floor_label(floor): {} for floor in floors
    }
    sparse_index_rows: dict[str, list[dict[str, object]]] = {
        _floor_label(floor): [] for floor in floors
    }
    lon2 = lat2 = native_x2 = native_y2 = inverse_transformer = None
    object_sequence = defaultdict(int)
    for event_date in dates:
        start_date = event_date - pd.Timedelta(days=5)
        try:
            find_sic_file(event_date)
            find_sic_file(start_date)
        except FileNotFoundError:
            continue
        if lon2 is None:
            end, _, _, lon2, lat2 = read_sic_with_lonlat(event_date)
            with xr.open_dataset(
                find_sic_file(event_date), decode_cf=False, mask_and_scale=False
            ) as source_grid:
                native_x2, native_y2, inverse_transformer, _ = projected_grid_geometry(
                    source_grid
                )
            if native_x2.shape != end.shape or native_y2.shape != end.shape:
                raise RuntimeError("floor015 native-grid geometry shape mismatch")
        else:
            end, _, _ = read_sic(event_date)
        start, _, _ = read_sic(start_date)
        diff = end - start
        unweighted_loss = np.maximum(0.0, -diff)
        for floor in floors:
            label = _floor_label(floor)
            valid = pan_mask & np.isfinite(diff)
            if floor is not None:
                valid &= np.isfinite(start) & (start >= floor)
            components = detect_connected_components(
                diff,
                valid,
                -0.10,
                50,
                lon2=lon2,
                lat2=lat2,
                native_x2=native_x2,
                native_y2=native_y2,
                inverse_transform=inverse_transformer.transform,
                canonical_local_geometry=True,
                binary_closing=False,
                strict_min_cells=False,
            )
            for detected in components:
                object_sequence[label] += 1
                component = detected.mask
                cells = int(detected.cell_count)
                weights = np.where(component, unweighted_loss, 0.0)
                values = diff[component]
                object_id = f"ROB_{label}_{object_sequence[label]:08d}"
                patch_rows[label].append(
                    {
                        "object_id": object_id,
                        "date": event_date.strftime("%Y-%m-%d"),
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "window_days": 5,
                        "threshold": -0.10,
                        "region": _primary_region(component, weights, region_codes),
                        "centroid_lon": detected.weighted_centroid_lon,
                        "centroid_lat": detected.weighted_centroid_lat,
                        "longitude_geometry_class": detected.longitude_geometry_class,
                        "raw_signed_longitude_span_deg": detected.raw_signed_longitude_span_deg,
                        "minimum_circular_covering_arc_deg": detected.minimum_circular_covering_arc_deg,
                        "largest_longitude_gap_deg": detected.largest_longitude_gap_deg,
                        "unwrap_arc_start_deg": detected.unwrap_arc_start_deg,
                        "largest_gap_tie_count": detected.largest_gap_tie_count,
                        "max_loss_lon": detected.max_loss_lon,
                        "max_loss_lat": detected.max_loss_lat,
                        "object_area_cells": cells,
                        "object_area_km2": float(cells) * 625.0,
                        "mean_sic_change": float(np.nanmean(values)),
                        "min_sic_change": float(np.nanmin(values)),
                        "cumulative_sic_loss": float(np.nansum(-np.minimum(values, 0.0), dtype=np.float64)),
                        "diagnostic_area_weighted_loss_km2eq": float(
                            np.nansum(np.where(component, unweighted_loss * area, 0.0), dtype=np.float64)
                        ),
                    }
                )
                if sparse_cell_dirs is not None and label in sparse_cell_dirs:
                    y_idx, x_idx = np.where(component)
                    year_store = sparse_stores[label].setdefault(
                        int(event_date.year),
                        {"object_index": [], "y_idx": [], "x_idx": [], "delta_sic": []},
                    )
                    object_index = len(sparse_index_rows[label])
                    year_store["object_index"].append(
                        np.full(len(y_idx), object_index, dtype=np.int32)
                    )
                    year_store["y_idx"].append(y_idx.astype(np.int32))
                    year_store["x_idx"].append(x_idx.astype(np.int32))
                    year_store["delta_sic"].append(diff[y_idx, x_idx].astype(np.float32))
                    sparse_index_rows[label].append(
                        {
                            "object_index": object_index,
                            "object_id": object_id,
                            "date": event_date.strftime("%Y-%m-%d"),
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "window_days": 5,
                            "threshold": -0.10,
                            "year": int(event_date.year),
                            "npz_file": f"{int(event_date.year)}.npz",
                            "cell_count": int(len(y_idx)),
                            "cell_area_km2": float(np.nansum(area[y_idx, x_idx], dtype=np.float64)),
                        }
                    )

    if sparse_cell_dirs is not None:
        for label, directory in sparse_cell_dirs.items():
            if label not in sparse_stores:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            for year, arrays in sparse_stores[label].items():
                np.savez_compressed(
                    directory / f"{year}.npz",
                    object_index=np.concatenate(arrays["object_index"]),
                    y_idx=np.concatenate(arrays["y_idx"]),
                    x_idx=np.concatenate(arrays["x_idx"]),
                    delta_sic=np.concatenate(arrays["delta_sic"]),
                )
            pd.DataFrame(sparse_index_rows[label]).to_csv(
                directory / "patch_cell_index.csv", index=False
            )

    catalogs: dict[str, dict[str, pd.DataFrame]] = {}
    count_rows: list[dict[str, object]] = []
    for floor in floors:
        label = _floor_label(floor)
        patches = pd.DataFrame(patch_rows[label])
        merge_input = patches.copy()
        merge_input["overlap_with_panarctic_vrile"] = False
        merge_input["overlap_with_regional_vrile"] = False
        merge_input["overlap_with_regional_response_diagnostic"] = False
        broad, membership = merge_local_patches(
            merge_input,
            2,
            300.0,
            return_membership=True,
        )
        severe = mark_severe(broad)
        major = mark_major(severe)
        catalogs[label] = {
            "patches": patches,
            "membership": membership,
            "broad": broad,
            "severe": severe,
            "major_severe": major,
        }
        levels = {"broad": broad, "severe": severe, "major_severe": major}
        for level, frame in levels.items():
            for scope, subset in [("all", frame), *[(name, frame[frame["dominant_region"] == name]) for name in REGION_NAMES.values()]]:
                count_rows.append(
                    {
                        "sic_floor": label,
                        "severity_level": level,
                        "region": scope,
                        "event_count": int(len(subset)),
                        "unique_start_dates": int(pd.to_datetime(subset["event_start"]).nunique()) if len(subset) else 0,
                    }
                )
        count_rows.extend(
            [
                {
                    "sic_floor": label,
                    "severity_level": "nesting_gate",
                    "region": "severe_subset_broad",
                    "event_count": int(set(severe["unique_local_event_id"]).issubset(set(broad["unique_local_event_id"]))),
                    "unique_start_dates": np.nan,
                },
                {
                    "sic_floor": label,
                    "severity_level": "nesting_gate",
                    "region": "major_subset_severe",
                    "event_count": int(set(major["unique_local_event_id"]).issubset(set(severe["unique_local_event_id"]))),
                    "unique_start_dates": np.nan,
                },
            ]
        )
    counts = pd.DataFrame(count_rows)
    primary = counts[(counts["sic_floor"] == "none") & (counts["region"] == "all")].set_index("severity_level")
    if not quick and None in floors:
        observed = {level: int(primary.loc[level, "event_count"]) for level in EXPECTED_LEVEL_COUNTS}
        if observed != EXPECTED_LEVEL_COUNTS:
            raise RuntimeError(f"local primary reconstruction mismatch: observed={observed}, expected={EXPECTED_LEVEL_COUNTS}")
    if None in floors:
        for level in EXPECTED_LEVEL_COUNTS:
            reference = float(primary.loc[level, "event_count"])
            selector = counts["severity_level"].eq(level) & counts["region"].eq("all")
            counts.loc[selector, "absolute_change_vs_primary"] = counts.loc[selector, "event_count"] - reference
            counts.loc[selector, "relative_change_vs_primary"] = (
                counts.loc[selector, "event_count"] - reference
            ) / reference
    return catalogs, counts


def frozen_local_lowsic_diagnostic(
    baseline_root: Path,
    *,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align every frozen patch cell to that patch's own start-SIC field."""

    processed = baseline_root / "data/processed/local_event_cells"
    index = pd.read_csv(processed / "patch_cell_index.csv")
    membership = pd.read_csv(processed / "unique_event_patch_membership.csv")
    broad = pd.read_csv(baseline_root / "outputs/local_vrile_enhanced/unique_local_events.csv")
    severe_ids = set(
        pd.read_csv(baseline_root / "outputs/local_vrile_severe/severe_unique_local_events.csv")["unique_local_event_id"]
    )
    major_ids = set(
        pd.read_csv(baseline_root / "outputs/local_vrile_major_severe/major_severe_events_union.csv")["unique_local_event_id"]
    )
    if quick:
        index = index[index["year"].between(2019, 2021)].copy()
        membership = membership[membership["patch_key"].isin(index["patch_key"])].copy()
        broad = broad[broad["unique_local_event_id"].isin(membership["unique_local_event_id"])].copy()
    patch_to_event = membership.set_index("patch_key")["unique_local_event_id"].to_dict()
    event_region = broad.set_index("unique_local_event_id")["dominant_region"].to_dict()
    area, _ = load_cell_area_and_regions(baseline_root)
    event_cells: dict[str, list[np.ndarray]] = defaultdict(list)
    event_losses: dict[str, list[np.ndarray]] = defaultdict(list)
    for year, year_index in index.groupby("year"):
        archive = np.load(processed / f"{int(year)}.npz")
        y_idx = archive["y_idx"]
        x_idx = archive["x_idx"]
        delta = archive["delta_sic"].astype(float)
        for row in year_index.itertuples(index=False):
            offset = int(row.start_offset)
            stop = offset + int(row.cell_count)
            y = y_idx[offset:stop]
            x = x_idx[offset:stop]
            start, _, _ = read_sic(pd.Timestamp(row.start_date))
            start_values = start[y, x]
            loss = np.maximum(0.0, -delta[offset:stop]) * area[y, x]
            event_id = str(patch_to_event[row.patch_key])
            event_cells[event_id].append(start_values)
            event_losses[event_id].append(loss)

    rows: list[dict[str, object]] = []
    for event_id in sorted(event_cells):
        start = np.concatenate(event_cells[event_id])
        loss = np.concatenate(event_losses[event_id])
        finite = np.isfinite(start)
        total_loss = float(np.nansum(loss, dtype=np.float64))
        row: dict[str, object] = {
            "unique_local_event_id": event_id,
            "region": event_region.get(event_id, ""),
            "is_broad": True,
            "is_severe": event_id in severe_ids,
            "is_major_severe": event_id in major_ids,
            "patch_cell_observations": int(len(start)),
            "start_sic_p10": float(np.nanquantile(start, 0.10)),
            "start_sic_q25": float(np.nanquantile(start, 0.25)),
            "start_sic_median": float(np.nanmedian(start)),
            "start_sic_q75": float(np.nanquantile(start, 0.75)),
            "start_sic_p90": float(np.nanquantile(start, 0.90)),
            "loss_weighted_mean_start_sic": (
                float(np.nansum(start * loss, dtype=np.float64) / total_loss) if total_loss > 0 else np.nan
            ),
            "total_patch_cell_loss_km2eq": total_loss,
        }
        for label, mask in (
            ("lt_0.15", finite & (start < 0.15)),
            ("0.15_0.30", finite & (start >= 0.15) & (start < 0.30)),
            ("ge_0.30", finite & (start >= 0.30)),
        ):
            value = float(np.nansum(np.where(mask, loss, 0.0), dtype=np.float64))
            row[f"loss_start_sic_{label}_km2eq"] = value
            row[f"fraction_loss_start_sic_{label}"] = value / total_loss if total_loss > 0 else np.nan
        rows.append(row)
    event_level = pd.DataFrame(rows)

    by_level, by_region = summarize_frozen_local_event_level(event_level)
    return event_level, by_level, by_region


def summarize_frozen_local_event_level(
    event_level: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all three levels over a fixed, complete 18-region taxonomy."""

    level_groups = {
        "broad": event_level,
        "severe": event_level[event_level["is_severe"]],
        "major_severe": event_level[event_level["is_major_severe"]],
    }
    summaries: list[dict[str, object]] = []
    for level, group in level_groups.items():
        total = float(group["total_patch_cell_loss_km2eq"].sum())
        row = {"severity_level": level, "event_count": int(len(group)), "total_loss_km2eq": total}
        for label in ("lt_0.15", "0.15_0.30", "ge_0.30"):
            value = float(group[f"loss_start_sic_{label}_km2eq"].sum())
            row[f"loss_start_sic_{label}_km2eq"] = value
            row[f"fraction_loss_start_sic_{label}"] = value / total if total > 0 else np.nan
        summaries.append(row)
    region_rows = []
    for level, level_group in level_groups.items():
        for region in REGION_NAMES.values():
            group = level_group[level_group["region"] == region]
            total = float(group["total_patch_cell_loss_km2eq"].sum())
            row = {
                "region": region,
                "severity_level": level,
                "event_count": int(len(group)),
                "total_loss_km2eq": total,
            }
            for label in ("lt_0.15", "0.15_0.30", "ge_0.30"):
                value = float(group[f"loss_start_sic_{label}_km2eq"].sum())
                row[f"loss_start_sic_{label}_km2eq"] = value
                row[f"fraction_loss_start_sic_{label}"] = value / total if total > 0 else np.nan
            region_rows.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(region_rows)
