"""Batch 3 spatial composition metrics for pan-Arctic/local attribution."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pyproj import Geod


def _effective_number(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    total = float(values.sum())
    if total <= 0:
        return np.nan
    p = values / total
    return float(1.0 / np.sum(p * p))


def _largest_share(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    total = float(values.sum())
    return float(values.max() / total) if total > 0 and values.size else np.nan


def _topn_share(values: np.ndarray, n: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    total = float(values.sum())
    if total <= 0 or values.size == 0:
        return np.nan
    return float(np.sort(values)[-n:].sum() / total)


def _weighted_spherical_centroid(lon: np.ndarray, lat: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan
    lon_rad = np.deg2rad(lon[valid])
    lat_rad = np.deg2rad(lat[valid])
    w = weights[valid] / np.sum(weights[valid])
    x = np.sum(w * np.cos(lat_rad) * np.cos(lon_rad))
    y = np.sum(w * np.cos(lat_rad) * np.sin(lon_rad))
    z = np.sum(w * np.sin(lat_rad))
    hyp = math.hypot(x, y)
    return float(np.rad2deg(np.arctan2(y, x))), float(np.rad2deg(np.arctan2(z, hyp)))


def component_metrics(components: pd.DataFrame, event_summary: pd.DataFrame) -> pd.DataFrame:
    geod = Geod(ellps="WGS84")
    summary = event_summary.set_index("pan_event_id")
    rows = []
    for pan_id, grp in components.groupby("pan_event_id", sort=True):
        loss = pd.to_numeric(grp["integrated_sic_loss_km2eq"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(grp["loss_area_weighted_centroid_lon"], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(grp["loss_area_weighted_centroid_lat"], errors="coerce").to_numpy(float)
        valid = np.isfinite(loss) & (loss > 0)
        n_components = int(valid.sum())
        resolved = float(summary.loc[pan_id, "resolved_sic_loss_km2eq"]) if pan_id in summary.index else float(np.nansum(loss))
        top2_dist = np.nan
        spread = np.nan
        if n_components >= 2:
            order = np.argsort(loss[valid])[::-1]
            lon_v = lon[valid]
            lat_v = lat[valid]
            loss_v = loss[valid]
            lon1, lat1 = lon_v[order[0]], lat_v[order[0]]
            lon2, lat2 = lon_v[order[1]], lat_v[order[1]]
            _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            top2_dist = float(dist_m / 1000.0)
            cen_lon, cen_lat = _weighted_spherical_centroid(lon_v, lat_v, loss_v)
            if np.isfinite(cen_lon) and np.isfinite(cen_lat):
                distances = []
                for lo, la in zip(lon_v, lat_v):
                    _, _, d_m = geod.inv(cen_lon, cen_lat, lo, la)
                    distances.append(d_m / 1000.0)
                p = loss_v / loss_v.sum()
                spread = float(np.sqrt(np.sum(p * np.square(distances))))
        rows.append(
            {
                "pan_event_id": pan_id,
                "event_date": str(summary.loc[pan_id, "event_date"]) if pan_id in summary.index and "event_date" in summary.columns else "",
                "effective_component_number_sic": _effective_number(loss),
                "largest_component_resolved_sic_loss_share": _largest_share(loss),
                "top3_component_resolved_sic_loss_share": _topn_share(loss, 3),
                "n_loss_components": n_components,
                "top2_component_centroid_distance_km": top2_dist,
                "weighted_component_spatial_spread_km": spread,
                "component_resolved_sic_loss_fraction": float(summary.loc[pan_id, "component_resolved_sic_loss_fraction"])
                if pan_id in summary.index
                else np.nan,
                "resolved_sic_loss_km2eq": resolved,
            }
        )
    return pd.DataFrame(rows)


def _region_metric_rows(
    detector_region: pd.DataFrame,
    cdr_region: pd.DataFrame,
    detector_budget: pd.DataFrame,
    cdr_budget: pd.DataFrame,
) -> pd.DataFrame:
    det_global = detector_budget.set_index("pan_event_id")["detector_gross_extent_loss_km2"].to_dict()
    cdr_global = cdr_budget.set_index("pan_event_id")["cdr_gross_sic_loss_km2eq"].to_dict()
    rows = []
    for pan_id in sorted(set(detector_region["pan_event_id"].astype(str)) | set(cdr_region["pan_event_id"].astype(str))):
        det = detector_region[detector_region["pan_event_id"].astype(str) == pan_id].copy()
        cdr = cdr_region[cdr_region["pan_event_id"].astype(str) == pan_id].copy()
        det_named = det[det["region_code"].between(1, 18)]
        cdr_named = cdr[cdr["region_code"].between(1, 18)]
        det_loss = pd.to_numeric(det_named["detector_gross_extent_loss_km2"], errors="coerce").to_numpy(float)
        cdr_loss = pd.to_numeric(cdr_named["cdr_gross_sic_loss_km2eq"], errors="coerce").to_numpy(float)
        det_total = float(det_global.get(pan_id, np.nan))
        cdr_total = float(cdr_global.get(pan_id, np.nan))
        det_non_region = float(pd.to_numeric(det.loc[det["region_code"] == 0, "detector_gross_extent_loss_km2"], errors="coerce").sum())
        cdr_non_region = float(pd.to_numeric(cdr.loc[cdr["region_code"] == 0, "cdr_gross_sic_loss_km2eq"], errors="coerce").sum())
        rows.append(
            {
                "pan_event_id": pan_id,
                "effective_named_region_number_extent_loss": _effective_number(det_loss),
                "largest_named_region_extent_loss_share": _largest_share(det_loss),
                "top3_named_region_extent_loss_share": _topn_share(det_loss, 3),
                "non_region_ocean_extent_loss_fraction": det_non_region / det_total if det_total > 0 else np.nan,
                "effective_named_region_number_sic_loss": _effective_number(cdr_loss),
                "largest_named_region_sic_loss_share": _largest_share(cdr_loss),
                "top3_named_region_sic_loss_share": _topn_share(cdr_loss, 3),
                "non_region_ocean_sic_loss_fraction": cdr_non_region / cdr_total if cdr_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spatial_composition_metrics(
    components: pd.DataFrame,
    event_summary: pd.DataFrame,
    detector_region: pd.DataFrame,
    cdr_region: pd.DataFrame,
    detector_budget: pd.DataFrame,
    cdr_budget: pd.DataFrame,
) -> pd.DataFrame:
    comp = component_metrics(components, event_summary)
    region = _region_metric_rows(detector_region, cdr_region, detector_budget, cdr_budget)
    return comp.merge(region, on="pan_event_id", how="outer").sort_values("pan_event_id").reset_index(drop=True)
