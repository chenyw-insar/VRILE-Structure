"""Stage 3 detector and CDR spatial field construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from vrile.local_objects import detect_connected_components
from vrile.stage3.grid import (
    EXTENT_THRESHOLD,
    find_g02135_geotiff,
    load_cell_area_km2,
    read_geotiff,
    read_sic,
    valid_ocean_mask,
    xy_points_to_lonlat,
)


def detector_extent_field(event: pd.Series, out_dir: Path | None = None) -> dict:
    area_km2, x, y = load_cell_area_km2()
    event_date = pd.Timestamp(event["date"]).normalize()
    start_date = event_date - pd.Timedelta(days=2)
    end_date = event_date + pd.Timedelta(days=1)
    start_arr, _ = read_geotiff(find_g02135_geotiff(start_date))
    end_arr, _ = read_geotiff(find_g02135_geotiff(end_date))
    physical = np.isin(start_arr, [0, 1]) & np.isin(end_arr, [0, 1])
    loss_mask = physical & (start_arr == 1) & (end_arr == 0)
    gain_mask = physical & (start_arr == 0) & (end_arr == 1)
    transition = np.where(gain_mask, 1, np.where(loss_mask, -1, 0)).astype(np.int8)

    istart = (start_arr == 1).astype(np.int8)
    iend = (end_arr == 1).astype(np.int8)
    status_change = (~physical) & (istart != iend)
    status_loss = status_change & (istart == 1) & (iend == 0)
    status_gain = status_change & (istart == 0) & (iend == 1)
    status_transition = np.where(status_gain, 1, np.where(status_loss, -1, 0)).astype(np.int8)

    gross_loss = float(np.nansum(np.where(loss_mask, area_km2, 0.0)))
    gross_gain = float(np.nansum(np.where(gain_mask, area_km2, 0.0)))
    status_loss_km2 = float(np.nansum(np.where(status_loss, area_km2, 0.0)))
    status_gain_km2 = float(np.nansum(np.where(status_gain, area_km2, 0.0)))
    status_net = status_gain_km2 - status_loss_km2
    status_activity = status_loss_km2 + status_gain_km2
    geotiff_delta = float(np.nansum(np.where(end_arr == 1, area_km2, 0.0)) - np.nansum(np.where(start_arr == 1, area_km2, 0.0)))
    closure_error = (gross_gain - gross_loss + status_net) - geotiff_delta
    reported_km2 = float(event["raw_delta_sie"]) * 1_000_000.0
    rel = abs(geotiff_delta - reported_km2) / abs(reported_km2) if reported_km2 != 0 else np.nan
    missing_touch = float(np.nansum(np.where((start_arr == 255) | (end_arr == 255), area_km2, 0.0)))
    other_touch = float(np.nansum(np.where(np.isin(start_arr, [253, 254]) | np.isin(end_arr, [253, 254]), area_km2, 0.0)))
    status = "PASS" if np.isclose(closure_error, 0.0, rtol=1e-12, atol=1e-6) else "FAIL"

    pan_event_id = str(event["pan_event_id"])
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {
                "detector_extent_loss_mask": (("y", "x"), loss_mask.astype(np.uint8)),
                "detector_extent_gain_mask": (("y", "x"), gain_mask.astype(np.uint8)),
                "detector_extent_transition": (("y", "x"), transition),
                "detector_source_status_transition": (("y", "x"), status_transition),
            },
            coords={"x": x, "y": y},
            attrs={"pan_event_id": pan_event_id, "event_date": event_date.strftime("%Y-%m-%d")},
        )
        ds.to_netcdf(out_dir / f"{pan_event_id}.nc")
    return {
        "pan_event_id": pan_event_id,
        "event_date": event_date.strftime("%Y-%m-%d"),
        "detector_gross_extent_loss_km2": gross_loss,
        "detector_gross_extent_gain_km2": gross_gain,
        "detector_source_status_ice_loss_km2": status_loss_km2,
        "detector_source_status_ice_gain_km2": status_gain_km2,
        "detector_source_status_transition_net_km2": status_net,
        "detector_source_status_transition_activity_km2": status_activity,
        "detector_source_status_transition_activity_fraction": status_activity / (gross_loss + gross_gain + status_activity) if (gross_loss + gross_gain + status_activity) > 0 else np.nan,
        "detector_geotiff_delta_sie_km2": geotiff_delta,
        "reported_raw_delta_sie_km2": reported_km2,
        "detector_budget_closure_error_km2": float(closure_error),
        "detector_source_relative_difference": float(rel),
        "missing_touch_area_km2": missing_touch,
        "other_class_touch_area_km2": other_touch,
        "status": status,
    }


def cdr_loss_field(event: pd.Series, out_dir: Path | None = None) -> tuple[dict, list[dict]]:
    area_km2, x, y = load_cell_area_km2()
    ocean = valid_ocean_mask()
    event_date = pd.Timestamp(event["date"]).normalize()
    start_date = event_date - pd.Timedelta(days=5)
    start, _, _ = read_sic(start_date)
    end, _, _ = read_sic(event_date)
    signed = end - start
    signed = np.where(ocean & np.isfinite(signed), signed, np.nan)
    loss = np.where(np.isfinite(signed), np.maximum(0.0, -signed), np.nan)
    gain = np.where(np.isfinite(signed), np.maximum(0.0, signed), np.nan)
    signed_store = signed.astype(np.float32)
    loss_store = loss.astype(np.float32)
    gain_store = gain.astype(np.float32)
    signed_for_sum = signed_store.astype(float)
    loss_for_sum = loss_store.astype(float)
    gain_for_sum = gain_store.astype(float)

    valid_loss = np.isfinite(signed) & ocean
    comps = detect_connected_components(
        signed,
        valid_loss,
        -0.1,
        4,
        binary_closing=True,
        strict_min_cells=True,
    )
    labels = np.zeros_like(signed, dtype=np.int32)
    comp_rows = []
    resolved_loss = 0.0
    for cid, comp in enumerate(comps, start=1):
        labels[comp.mask] = cid
        comp_loss = float(np.nansum(np.where(comp.mask, loss_for_sum * area_km2, 0.0), dtype=np.float64))
        comp_area = float(np.nansum(np.where(comp.mask, area_km2, 0.0)))
        weights = np.where(comp.mask, loss_for_sum * area_km2, 0.0)
        wsum = float(np.nansum(weights, dtype=np.float64))
        if wsum > 0:
            x2, y2 = np.meshgrid(x, y)
            centroid_x = float(np.nansum(x2 * weights, dtype=np.float64) / wsum)
            centroid_y = float(np.nansum(y2 * weights, dtype=np.float64) / wsum)
            lon_arr, lat_arr = xy_points_to_lonlat(centroid_x, centroid_y)
            centroid_lon = float(lon_arr)
            centroid_lat = float(lat_arr)
        else:
            centroid_x = centroid_y = centroid_lon = centroid_lat = np.nan
        resolved_loss += comp_loss
        comp_rows.append(
            {
                "pan_event_id": str(event["pan_event_id"]),
                "component_id": cid,
                "cell_count": comp.cell_count,
                "area_km2": comp_area,
                "integrated_sic_loss_km2eq": comp_loss,
                "loss_area_weighted_centroid_x_m": centroid_x,
                "loss_area_weighted_centroid_y_m": centroid_y,
                "loss_area_weighted_centroid_lon": centroid_lon,
                "loss_area_weighted_centroid_lat": centroid_lat,
            }
        )

    total_loss = float(np.nansum(loss_for_sum * area_km2, dtype=np.float64))
    total_gain = float(np.nansum(gain_for_sum * area_km2, dtype=np.float64))
    signed_integral = float(np.nansum(signed_for_sum * area_km2, dtype=np.float64))
    budget_error = total_gain - total_loss - signed_integral
    pan_event_id = str(event["pan_event_id"])
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {
                "cdr_signed_sic_change": (("y", "x"), signed_store),
                "cdr_sic_loss": (("y", "x"), loss_store),
                "cdr_sic_gain": (("y", "x"), gain_store),
                "loss_component_id": (("y", "x"), labels.astype(np.int32)),
            },
            coords={"x": x, "y": y},
            attrs={"pan_event_id": pan_event_id, "event_date": event_date.strftime("%Y-%m-%d")},
        )
        ds.to_netcdf(out_dir / f"{pan_event_id}.nc")
    summary = {
        "pan_event_id": pan_event_id,
        "event_date": event_date.strftime("%Y-%m-%d"),
        "cdr_window_start": start_date.strftime("%Y-%m-%d"),
        "cdr_window_end": event_date.strftime("%Y-%m-%d"),
        "total_positive_sic_loss_km2eq": total_loss,
        "total_positive_sic_gain_km2eq": total_gain,
        "cdr_signed_sic_change_km2eq": signed_integral,
        "cdr_gross_sic_loss_km2eq": total_loss,
        "cdr_gross_sic_gain_km2eq": total_gain,
        "cdr_budget_closure_error_km2eq": budget_error,
        "cdr_budget_closure_status": "PASS" if np.isclose(budget_error, 0.0, rtol=1e-12, atol=1e-6) else "FAIL",
        "resolved_sic_loss_km2eq": resolved_loss,
        "component_loss_closure_error_km2eq": np.nan,
        "component_loss_closure_status": "PENDING_INDEPENDENT_RECONCILIATION",
        "component_resolved_sic_loss_fraction": resolved_loss / total_loss if total_loss > 0 else np.nan,
        "n_loss_components": len(comps),
    }
    return summary, comp_rows
