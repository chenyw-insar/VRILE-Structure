"""Shared local SIC-loss object primitives.

These functions hold the scientific primitives used by both Stage 1 local
object scripts and Stage 3 footprint backfill.  Entry scripts remain
responsible for date traversal, object-id allocation, CLI behavior, and output
files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage


LONGITUDE_GAP_TIE_TOLERANCE_DEG = 1e-12
LONGITUDE_GEOMETRY_ORDINARY = "ORDINARY"
LONGITUDE_GEOMETRY_WRAP_ONLY = "WRAP_ONLY"
LONGITUDE_GEOMETRY_DISTRIBUTED = "LONGITUDE_DISTRIBUTED"


class EventLongitudeDistributedError(RuntimeError):
    """Raised when event-member longitudes require an unapproved method."""


@dataclass(frozen=True)
class LongitudeGeometry:
    geometry_class: str
    raw_signed_span_deg: float
    minimum_circular_covering_arc_deg: float
    largest_gap_deg: float
    unwrap_arc_start_deg: float
    largest_gap_tie_count: int


@dataclass
class ConnectedComponent:
    label_id: int
    cell_count: int
    y_idx: np.ndarray
    x_idx: np.ndarray
    mean_sic_change: float
    min_sic_change: float
    cumulative_sic_loss: float
    weighted_centroid_lon: float
    weighted_centroid_lat: float
    max_loss_lon: float
    max_loss_lat: float
    longitude_geometry_class: str
    raw_signed_longitude_span_deg: float
    minimum_circular_covering_arc_deg: float
    largest_longitude_gap_deg: float
    unwrap_arc_start_deg: float
    largest_gap_tie_count: int
    mask: np.ndarray


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def max_loss_location(mask: np.ndarray, diff: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> tuple[float, float]:
    values = np.where(mask, diff, np.nan)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    iy, ix = np.unravel_index(int(np.nanargmin(values)), values.shape)
    return float(lon2[iy, ix]), float(lat2[iy, ix])


def classify_longitude_geometry(longitudes: np.ndarray) -> LongitudeGeometry:
    """Classify signed longitudes by their deterministic minimum circular arc."""
    values = np.asarray(longitudes, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("longitude geometry requires at least one finite longitude")
    raw_span = float(np.max(finite) - np.min(finite)) if finite.size > 1 else 0.0
    normalized = np.unique(finite % 360.0)
    if normalized.size == 1:
        largest_gap = 360.0
        covering_arc = 0.0
        arc_start = float(normalized[0])
        tie_count = 1
    else:
        gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
        largest_gap = float(np.max(gaps))
        tied = np.flatnonzero(np.abs(gaps - largest_gap) <= LONGITUDE_GAP_TIE_TOLERANCE_DEG)
        starts = normalized[(tied + 1) % normalized.size]
        arc_start = float(np.min(starts))
        tie_count = int(tied.size)
        covering_arc = float(360.0 - largest_gap)
    if covering_arc > 180.0:
        geometry_class = LONGITUDE_GEOMETRY_DISTRIBUTED
    elif raw_span > 180.0:
        geometry_class = LONGITUDE_GEOMETRY_WRAP_ONLY
    else:
        geometry_class = LONGITUDE_GEOMETRY_ORDINARY
    return LongitudeGeometry(
        geometry_class=geometry_class,
        raw_signed_span_deg=raw_span,
        minimum_circular_covering_arc_deg=covering_arc,
        largest_gap_deg=largest_gap,
        unwrap_arc_start_deg=arc_start,
        largest_gap_tie_count=tie_count,
    )


def _positive_weights(weights: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=float)
    valid = np.isfinite(out) & (out > 0)
    return np.where(valid, out, 0.0) if valid.any() else np.ones(out.shape, dtype=float)


def deterministic_unwrapped_weighted_longitude(
    longitudes: np.ndarray,
    weights: np.ndarray,
    geometry: LongitudeGeometry | None = None,
) -> float:
    """Apply the canonical largest-gap branch and return [-180, 180)."""
    values = np.asarray(longitudes, dtype=float)
    w = _positive_weights(weights)
    geometry = geometry or classify_longitude_geometry(values)
    normalized = values % 360.0
    unwrapped = np.where(
        normalized < geometry.unwrap_arc_start_deg - LONGITUDE_GAP_TIE_TOLERANCE_DEG,
        normalized + 360.0,
        normalized,
    )
    mean = float(np.average(unwrapped, weights=w))
    return float((mean + 180.0) % 360.0 - 180.0)


def native_projected_weighted_centroid(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    inverse_transform: Callable[[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Return the weighted native-grid x/y barycenter in lon/lat."""
    w = _positive_weights(weights)
    x_centroid = float(np.average(np.asarray(x, dtype=float), weights=w))
    y_centroid = float(np.average(np.asarray(y, dtype=float), weights=w))
    lon, lat = inverse_transform(x_centroid, y_centroid)
    return float(lon), float(lat)


def weighted_patch_centroid(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    weights: np.ndarray,
    *,
    native_x: np.ndarray | None = None,
    native_y: np.ndarray | None = None,
    inverse_transform: Callable[[float, float], tuple[float, float]] | None = None,
    frozen_arithmetic_reference: tuple[float, float] | None = None,
) -> tuple[float, float, LongitudeGeometry]:
    """Calculate the approved local-patch centroid and geometry class.

    ``frozen_arithmetic_reference`` is used only when rebuilding from the
    sparse post-detection cell intermediate, which does not retain the exact
    source-file lon/lat arrays.  It preserves the previously calculated
    arithmetic coordinate for ORDINARY patches and the latitude for WRAP_ONLY
    patches; LONGITUDE_DISTRIBUTED patches always use native projected x/y.
    """
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    w = _positive_weights(weights)
    geometry = classify_longitude_geometry(lon)
    if geometry.geometry_class == LONGITUDE_GEOMETRY_DISTRIBUTED:
        if native_x is None or native_y is None or inverse_transform is None:
            raise ValueError("LONGITUDE_DISTRIBUTED patch requires native x/y and inverse CF projection")
        centroid_lon, centroid_lat = native_projected_weighted_centroid(
            native_x, native_y, w, inverse_transform
        )
    else:
        centroid_lat = (
            float(frozen_arithmetic_reference[1])
            if frozen_arithmetic_reference is not None
            else float(np.average(lat, weights=w))
        )
        centroid_lon = (
            deterministic_unwrapped_weighted_longitude(lon, w, geometry)
            if geometry.geometry_class == LONGITUDE_GEOMETRY_WRAP_ONLY
            else (
                float(frozen_arithmetic_reference[0])
                if frozen_arithmetic_reference is not None
                else float(np.average(lon, weights=w))
            )
        )
    return centroid_lon, centroid_lat, geometry


def weighted_event_centroid(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    weights: np.ndarray,
    *,
    member_ids: Iterable[str] | None = None,
) -> tuple[float, float, LongitudeGeometry]:
    """Calculate the approved event centroid or fail for distributed events."""
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    w = _positive_weights(weights)
    geometry = classify_longitude_geometry(lon)
    if geometry.geometry_class == LONGITUDE_GEOMETRY_DISTRIBUTED:
        members = ";".join(str(x) for x in (member_ids or []))
        raise EventLongitudeDistributedError(
            "EVENT_LONGITUDE_DISTRIBUTED_REQUIRES_METHOD_DECISION"
            + (f": {members}" if members else "")
        )
    centroid_lon = (
        deterministic_unwrapped_weighted_longitude(lon, w, geometry)
        if geometry.geometry_class == LONGITUDE_GEOMETRY_WRAP_ONLY
        else float(np.average(lon, weights=w))
    )
    return centroid_lon, float(np.average(lat, weights=w)), geometry


def component_weighted_centroid(
    mask: np.ndarray,
    diff: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    *,
    native_x2: np.ndarray | None = None,
    native_y2: np.ndarray | None = None,
    inverse_transform: Callable[[float, float], tuple[float, float]] | None = None,
    canonical_local_geometry: bool = False,
) -> tuple[float, float, LongitudeGeometry]:
    weights = -np.minimum(np.asarray(diff, dtype=float)[mask], 0.0)
    lon = np.asarray(lon2, dtype=float)[mask]
    lat = np.asarray(lat2, dtype=float)[mask]
    if canonical_local_geometry:
        return weighted_patch_centroid(
            lon,
            lat,
            weights,
            native_x=None if native_x2 is None else np.asarray(native_x2, dtype=float)[mask],
            native_y=None if native_y2 is None else np.asarray(native_y2, dtype=float)[mask],
            inverse_transform=inverse_transform,
        )
    geometry = classify_longitude_geometry(lon)
    w = _positive_weights(weights)
    return float(np.average(lon, weights=w)), float(np.average(lat, weights=w)), geometry


def detect_connected_components(
    diff: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    min_cells: int,
    *,
    lon2: np.ndarray | None = None,
    lat2: np.ndarray | None = None,
    native_x2: np.ndarray | None = None,
    native_y2: np.ndarray | None = None,
    inverse_transform: Callable[[float, float], tuple[float, float]] | None = None,
    canonical_local_geometry: bool = False,
    binary_closing: bool = False,
    strict_min_cells: bool = False,
) -> list[ConnectedComponent]:
    """Detect connected SIC-loss components using the confirmed local logic."""
    arr = np.asarray(diff, dtype=float)
    loss = np.isfinite(arr) & np.asarray(valid_mask, dtype=bool) & (arr <= float(threshold))
    if binary_closing:
        loss = ndimage.binary_closing(loss, structure=np.ones((3, 3), dtype=bool), iterations=1)
        loss = loss & np.isfinite(arr) & np.asarray(valid_mask, dtype=bool)
    labels, nlab = ndimage.label(loss, structure=np.ones((3, 3), dtype=bool))
    if nlab == 0:
        return []
    sizes = ndimage.sum(np.ones_like(arr, dtype=float), labels, index=np.arange(1, nlab + 1))
    out: list[ConnectedComponent] = []
    for label_id, size in enumerate(sizes, start=1):
        keep = int(size) > int(min_cells) if strict_min_cells else int(size) >= int(min_cells)
        if not keep:
            continue
        mask = labels == label_id
        y_idx, x_idx = np.where(mask)
        if lon2 is not None and lat2 is not None:
            centroid_lon, centroid_lat, longitude_geometry = component_weighted_centroid(
                mask,
                arr,
                lon2,
                lat2,
                native_x2=native_x2,
                native_y2=native_y2,
                inverse_transform=inverse_transform,
                canonical_local_geometry=canonical_local_geometry,
            )
            max_loss_lon, max_loss_lat = max_loss_location(mask, arr, lon2, lat2)
        else:
            centroid_lon = centroid_lat = max_loss_lon = max_loss_lat = np.nan
            longitude_geometry = LongitudeGeometry("NOT_COMPUTED", np.nan, np.nan, np.nan, np.nan, 0)
        out.append(
            ConnectedComponent(
                label_id=int(label_id),
                cell_count=int(size),
                y_idx=y_idx.astype(np.int32),
                x_idx=x_idx.astype(np.int32),
                mean_sic_change=float(np.nanmean(arr[mask])),
                min_sic_change=float(np.nanmin(arr[mask])),
                cumulative_sic_loss=float(np.nansum(-np.minimum(arr[mask], 0.0))),
                weighted_centroid_lon=centroid_lon,
                weighted_centroid_lat=centroid_lat,
                max_loss_lon=max_loss_lon,
                max_loss_lat=max_loss_lat,
                longitude_geometry_class=longitude_geometry.geometry_class,
                raw_signed_longitude_span_deg=longitude_geometry.raw_signed_span_deg,
                minimum_circular_covering_arc_deg=longitude_geometry.minimum_circular_covering_arc_deg,
                largest_longitude_gap_deg=longitude_geometry.largest_gap_deg,
                unwrap_arc_start_deg=longitude_geometry.unwrap_arc_start_deg,
                largest_gap_tie_count=longitude_geometry.largest_gap_tie_count,
                mask=mask,
            )
        )
    return out


def merge_local_patches(
    filtered: pd.DataFrame,
    merge_days: int,
    merge_distance_km: float,
    *,
    return_membership: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Merge filtered daily local patches with the canonical Stage 1 semantics."""
    if filtered.empty:
        empty_events = pd.DataFrame()
        empty_members = pd.DataFrame(columns=["unique_local_event_id", "object_id"])
        return (empty_events, empty_members) if return_membership else empty_events

    events = []
    event_id = 1
    for region, grp in filtered.groupby("region", sort=True):
        active: list[dict] = []
        for row in grp.sort_values(["date", "cumulative_sic_loss"], ascending=[True, False]).to_dict("records"):
            date = pd.Timestamp(row["date"])
            still_active: list[dict] = []
            for ev in active:
                if (date - ev["last_date"]).days <= merge_days:
                    still_active.append(ev)
                else:
                    events.append(ev)
            active = still_active
            best_idx = None
            best_dist = float("inf")
            for idx, ev in enumerate(active):
                days = (date - ev["last_date"]).days
                if days < 0 or days > merge_days:
                    continue
                dist = haversine_km(row["centroid_lon"], row["centroid_lat"], ev["last_lon"], ev["last_lat"])
                if dist <= merge_distance_km and dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if best_idx is None:
                ev = {
                    "unique_local_event_id": f"ULE{event_id:06d}",
                    "dominant_region": region,
                    "rows": [row],
                    "last_date": date,
                    "last_lon": float(row["centroid_lon"]),
                    "last_lat": float(row["centroid_lat"]),
                }
                active.append(ev)
                event_id += 1
            else:
                ev = active[best_idx]
                ev["rows"].append(row)
                ev["last_date"] = date
                ev["last_lon"] = float(row["centroid_lon"])
                ev["last_lat"] = float(row["centroid_lat"])
        events.extend(active)

    rows = []
    member_rows = []
    for ev in events:
        df = pd.DataFrame(ev["rows"])
        weights = df["cumulative_sic_loss"].clip(lower=0).to_numpy(float)
        if np.isfinite(weights).sum() == 0 or weights.sum() <= 0:
            weights = np.ones(len(df))
        track_lon, track_lat, event_longitude_geometry = weighted_event_centroid(
            df["centroid_lon"].to_numpy(float),
            df["centroid_lat"].to_numpy(float),
            weights,
            member_ids=df["object_id"].astype(str),
        )
        rows.append(
            {
                "unique_local_event_id": ev["unique_local_event_id"],
                "dominant_region": ev["dominant_region"],
                "event_start": pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d"),
                "event_end": pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d"),
                "duration_days": int((pd.to_datetime(df["date"]).max() - pd.to_datetime(df["date"]).min()).days + 1),
                "n_daily_patches": int(len(df)),
                "track_centroid_lon": track_lon,
                "track_centroid_lat": track_lat,
                "track_longitude_geometry_class": event_longitude_geometry.geometry_class,
                "track_raw_signed_longitude_span_deg": event_longitude_geometry.raw_signed_span_deg,
                "track_minimum_circular_covering_arc_deg": event_longitude_geometry.minimum_circular_covering_arc_deg,
                "max_area_cells": int(df["object_area_cells"].max()),
                "max_area_km2": float(df["object_area_km2"].max()),
                "max_intensity": float((-df["mean_sic_change"]).max()),
                "min_sic_change": float(df["min_sic_change"].min()),
                "mean_sic_change": float(df["mean_sic_change"].mean()),
                "cumulative_loss": float(df["cumulative_sic_loss"].sum()),
                "window_days": int(df["window_days"].iloc[0]),
                "sic_change_threshold": float(df["threshold"].iloc[0]),
                "min_cells": int(df["object_area_cells"].min()),
                "overlap_with_panarctic_vrile": bool(df["overlap_with_panarctic_vrile"].any()),
                # Keep the old column as a schema-compatible alias.
                "overlap_with_regional_vrile_v1": bool(df["overlap_with_regional_vrile"].any()),
                "overlap_with_regional_response_diagnostic": bool(
                    df.get("overlap_with_regional_response_diagnostic", df["overlap_with_regional_vrile"]).any()
                ),
                "event_catalog_role": "primary_local_spatial_event",
            }
        )
        for row in df.to_dict("records"):
            member_rows.append(
                {
                    "unique_local_event_id": ev["unique_local_event_id"],
                    "object_id": row["object_id"],
                    "patch_key": patch_key(row),
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "start_date": pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d"),
                    "window_days": int(row["window_days"]),
                    "threshold": float(row["threshold"]),
                }
            )
    unique = pd.DataFrame(rows).sort_values(["dominant_region", "event_start"]).reset_index(drop=True)
    membership = pd.DataFrame(member_rows)
    return (unique, membership) if return_membership else unique


def patch_key(row: dict | pd.Series) -> str:
    return "|".join(
        [
            str(row["object_id"]),
            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d"),
            str(int(row["window_days"])),
            f"{float(row['threshold']):.6g}",
        ]
    )
