#!/usr/bin/env python
"""Stage 3 Batch 0-1 audits for pan-Arctic/local contribution attribution.

This script establishes method semantics, input/schema readiness, NSIDC-0780
surface-mask accounting, footprint strategy, and detector-aligned spatial
closure. It intentionally does not implement attribution, overlap rasters,
residuals, N_eff, reconstruction, or phenotypes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pyproj import CRS, Transformer

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.frozen import frozen_output_root

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = frozen_output_root(ROOT)
MASK_PATH = ROOT / "data/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
PLAN_PATH = ROOT / 'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md'
ANCILLARY_DIR = ROOT / "data/raw/nsidc_ancillary"
CELL_AREA_FILENAME = "NSIDC0771_CellArea_PS_N25km_v1.1.nc"
CELL_AREA_CANDIDATES = [
    ANCILLARY_DIR / CELL_AREA_FILENAME,
]
G02135_GEOTIFF_ROOT = ROOT / "data/raw/nsidc_g02135_geotiff/north/daily/geotiff"
G02135_MANIFEST = G02135_GEOTIFF_ROOT / "stage3_download_manifest.csv"
G02135_REQUIRED_DATES = G02135_GEOTIFF_ROOT / "stage3_required_detector_dates.csv"
SIE_CSV = ROOT / "data/raw/nsidc_sie/N_seaice_extent_daily_v4.0.csv"
SIC_ROOT = ROOT / "data/raw/nsidc_sic"
PAN_UNIQUE = ROOT / "outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv"
PAN_ALL = ROOT / "outputs/reproduce_sie/vrile_events_all_both_jja_5p.csv"

NSIDC_0780_REGION_VARIABLE = "sea_ice_region_surface_mask"
NSIDC_0780_REGION_IDS: dict[str, int] = {
    "central_arctic": 1,
    "beaufort_sea": 2,
    "chukchi_sea": 3,
    "east_siberian_sea": 4,
    "laptev_sea": 5,
    "kara_sea": 6,
    "barents_sea": 7,
    "east_greenland_sea": 8,
    "baffin_and_labrador_seas": 9,
    "gulf_of_st_lawrence": 10,
    "hudson_bay": 11,
    "canadian_archipelago": 12,
    "bering_sea": 13,
    "sea_of_okhotsk": 14,
    "sea_of_japan": 15,
    "bohai_and_yellow_seas": 16,
    "baltic_sea": 17,
    "gulf_of_alaska": 18,
}
OCEAN_CODES = set(range(0, 19))
NAMED_REGION_CODES = set(NSIDC_0780_REGION_IDS.values())
NON_OCEAN_CODES = {30, 32, 33, 34, 35, 40}
KNOWN_CODES = OCEAN_CODES | NON_OCEAN_CODES
EXTENT_THRESHOLD = 0.15

REGION_BY_CODE = {v: k for k, v in NSIDC_0780_REGION_IDS.items()}

DETECTION_CONFIG = {
    "start_year": 1989,
    "end_year": 2025,
    "climatology_start_year": 1990,
    "climatology_end_year": 2018,
    "delta_days": 3,
    "percentile": 5.0,
    "butterworth_cutoff_days": 18.0,
    "butterworth_order": 12,
    "unique_gap_days": 1,
    "months": [6, 7, 8],
    "months_label": "jja",
    "exclude_month_boundary": True,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_columns(path: Path) -> list[str]:
    if not path.exists() or path.is_dir():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def csv_rows(path: Path) -> int | float:
    if not path.exists() or path.is_dir():
        return math.nan
    try:
        with path.open("r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return math.nan


def status_for(path: Path, required_cols: list[str] | None = None) -> tuple[str, str]:
    if not path.exists():
        return "missing", "path missing"
    if path.is_dir():
        files = list(path.iterdir())
        return ("ok" if files else "empty", f"directory entries={len(files)}")
    cols = csv_columns(path)
    if required_cols:
        missing = [c for c in required_cols if c not in cols]
        if missing:
            return "schema_mismatch", f"missing columns: {missing}"
    rows = csv_rows(path)
    if rows == 0:
        return "empty", "zero data rows"
    return "ok", ""


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_crs_identifier(ds: xr.Dataset) -> str:
    if "crs" not in ds:
        return ""
    attrs = ds["crs"].attrs
    return str(
        attrs.get("srid")
        or attrs.get("spatial_ref")
        or attrs.get("crs_wkt")
        or attrs.get("proj4text")
        or attrs.get("grid_mapping_name")
        or ""
    )


def crs_attrs(ds: xr.Dataset) -> dict[str, Any]:
    if "crs" not in ds:
        return {}
    out = {}
    for key, value in ds["crs"].attrs.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = str(value)
    return out


def parse_epsg_from_wkt(text: str) -> str:
    if "EPSG" in text and "3411" in text:
        return "EPSG:3411"
    if "EPSG" in text and "3413" in text:
        return "EPSG:3413"
    return ""


def normalize_crs_identifier(text: str, attrs: dict[str, Any] | None = None) -> str:
    attrs = attrs or {}
    for item in [text, str(attrs.get("srid", "")), str(attrs.get("spatial_ref", "")), str(attrs.get("crs_wkt", ""))]:
        epsg = parse_epsg_from_wkt(item)
        if epsg:
            return epsg
        if "EPSG:3411" in item or "EPSG::3411" in item:
            return "EPSG:3411"
        if "EPSG:3413" in item or "EPSG::3413" in item:
            return "EPSG:3413"
    proj4 = str(attrs.get("proj4text", ""))
    if "+proj=stere" in proj4 and "+lon_0=-45" in proj4 and "+lat_ts=70" in proj4:
        return "NSIDC_PS_N25KM_STERE"
    return text[:160]


def grid_step_m(x: np.ndarray, y: np.ndarray) -> float:
    vals = []
    if x.size > 1:
        vals.append(abs(float(x[1] - x[0])))
    if y.size > 1:
        vals.append(abs(float(y[1] - y[0])))
    return min(vals) if vals else np.nan


def spherical_inverse_psn(x: np.ndarray, y: np.ndarray, *, radius_m: float, lon0_deg: float = -45.0, lat_ts_deg: float = 70.0) -> tuple[np.ndarray, np.ndarray]:
    """Approximate inverse for sampled CRS-compatibility QA when pyproj is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    lat_ts = np.deg2rad(lat_ts_deg)
    t = rho * np.sqrt((1.0 + np.sin(lat_ts)) / (1.0 - np.sin(lat_ts))) / (2.0 * radius_m)
    lat = np.rad2deg(np.pi / 2.0 - 2.0 * np.arctan(t))
    lon = lon0_deg + np.rad2deg(np.arctan2(x, -y))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def sampled_geolocation_delta_m(x: np.ndarray, y: np.ndarray, attrs_a: dict[str, Any], attrs_b: dict[str, Any]) -> float:
    delta, _ = sampled_geolocation_delta_with_method(x, y, attrs_a, attrs_b)
    return delta


def sampled_geolocation_delta_with_method(x: np.ndarray, y: np.ndarray, attrs_a: dict[str, Any], attrs_b: dict[str, Any]) -> tuple[float, str]:
    if x.size == 0 or y.size == 0:
        return np.nan, "not_evaluable"
    ix = sorted(set([0, len(x) - 1, len(x) // 2] + np.linspace(0, len(x) - 1, 5, dtype=int).tolist()))
    iy = sorted(set([0, len(y) - 1, len(y) // 2] + np.linspace(0, len(y) - 1, 5, dtype=int).tolist()))
    xx, yy = np.meshgrid(x[ix], y[iy])
    try:
        crs_a = CRS.from_wkt(str(attrs_a.get("spatial_ref") or attrs_a.get("crs_wkt")))
        crs_b = CRS.from_wkt(str(attrs_b.get("spatial_ref") or attrs_b.get("crs_wkt")))
        lon_a, lat_a = Transformer.from_crs(crs_a, CRS.from_epsg(4326), always_xy=True).transform(xx, yy)
        lon_b, lat_b = Transformer.from_crs(crs_b, CRS.from_epsg(4326), always_xy=True).transform(xx, yy)
        return float(np.nanmax(haversine_m(np.asarray(lon_a), np.asarray(lat_a), np.asarray(lon_b), np.asarray(lat_b)))), "pyproj"
    except Exception:
        pass
    radius_a = float(attrs_a.get("semi_major_axis", 6378273.0))
    radius_b = float(attrs_b.get("semi_major_axis", 6378273.0))
    lon0_a = float(attrs_a.get("straight_vertical_longitude_from_pole", -45.0))
    lon0_b = float(attrs_b.get("straight_vertical_longitude_from_pole", -45.0))
    lat_ts_a = float(attrs_a.get("standard_parallel", 70.0))
    lat_ts_b = float(attrs_b.get("standard_parallel", 70.0))
    lon_a, lat_a = spherical_inverse_psn(xx, yy, radius_m=radius_a, lon0_deg=lon0_a, lat_ts_deg=lat_ts_a)
    lon_b, lat_b = spherical_inverse_psn(xx, yy, radius_m=radius_b, lon0_deg=lon0_b, lat_ts_deg=lat_ts_b)
    return float(np.nanmax(haversine_m(lon_a, lat_a, lon_b, lat_b))), "approximate_spherical_fallback"


def geotiff_tags(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        tags = dict(im.tag_v2.items())
        size = im.size
        mode = im.mode
        n_frames = int(getattr(im, "n_frames", 1))
    scale = tuple(float(v) for v in tags.get(33550, (np.nan, np.nan, np.nan)))
    tie = tuple(float(v) for v in tags.get(33922, (0.0, 0.0, 0.0, np.nan, np.nan, np.nan)))
    geokeys = tuple(int(v) for v in tags.get(34735, ()))
    geodouble = tags.get(34736, ())
    geoascii = str(tags.get(34737, ""))
    epsg = "EPSG:3411" if 3411 in geokeys or "NSIDC Sea Ice Polar Stereographic North" in geoascii else ""
    return {
        "width": int(size[0]),
        "height": int(size[1]),
        "mode": mode,
        "n_frames": n_frames,
        "pixel_scale_x": scale[0],
        "pixel_scale_y": scale[1],
        "tiepoint_i": tie[0],
        "tiepoint_j": tie[1],
        "tiepoint_x": tie[3],
        "tiepoint_y": tie[4],
        "epsg": epsg,
        "geo_key_directory": geokeys,
        "geo_double_params": geodouble,
        "geo_ascii_params": geoascii,
    }


def read_geotiff_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    info = geotiff_tags(path)
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-band GeoTIFF, got shape {arr.shape} for {path}")
    if info["n_frames"] != 1:
        raise ValueError(f"Expected one GeoTIFF frame, got {info['n_frames']} for {path}")
    return arr.astype(np.uint16), info


def geotiff_xy(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(int(info["width"]), dtype=float)
    rows = np.arange(int(info["height"]), dtype=float)
    # GeoTIFF tiepoint is the upper-left pixel corner; NSIDC NetCDF x/y
    # coordinates are pixel centers.
    x = float(info["tiepoint_x"]) + (cols + 0.5 - float(info["tiepoint_i"])) * float(info["pixel_scale_x"])
    y = float(info["tiepoint_y"]) - (rows + 0.5 - float(info["tiepoint_j"])) * float(info["pixel_scale_y"])
    return x, y


def parse_geotiff_version(path: Path) -> tuple[int, ...]:
    m = re.search(r"_v([0-9.]+)\.tif$", path.name)
    if not m:
        return (0,)
    return tuple(int(part) for part in m.group(1).split(".") if part.isdigit())


def find_g02135_geotiff(date: pd.Timestamp) -> Path | None:
    pattern = G02135_GEOTIFF_ROOT / f"{date.year}" / f"{date:%m_%b}" / f"N_{date:%Y%m%d}_extent_v*.tif"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        return None
    return max((Path(p) for p in matches), key=parse_geotiff_version)


def g02135_extent_million_km2(path: Path, area_km2: np.ndarray) -> tuple[float, dict[str, Any]]:
    arr, info = read_geotiff_array(path)
    if arr.shape != area_km2.shape:
        raise ValueError(f"G02135 GeoTIFF shape {arr.shape} does not match cell-area grid shape {area_km2.shape}: {path}")
    extent = float(np.nansum(np.where(arr == 1, area_km2, 0.0)) / 1_000_000.0)
    return extent, info


def geotiff_grid_audit(sample_path: Path, area_path: Path, area_km2: np.ndarray) -> pd.DataFrame:
    arr, info = read_geotiff_array(sample_path)
    with xr.open_dataset(area_path) as ds_area:
        area_x = np.asarray(ds_area["x"].values)
        area_y = np.asarray(ds_area["y"].values)
        area_attrs = crs_attrs(ds_area)
        area_crs = normalize_crs_identifier(dataset_crs_identifier(ds_area), area_attrs)
    gt_x, gt_y = geotiff_xy(info)
    shape_equal = bool(arr.shape == area_km2.shape)
    x_allclose = bool(gt_x.size == area_x.size and np.allclose(gt_x, area_x))
    y_allclose = bool(gt_y.size == area_y.size and np.allclose(gt_y, area_y))
    grid_index_equal = bool(shape_equal and x_allclose and y_allclose)
    geotiff_crs = info.get("epsg", "")
    crs_exact_equal = bool(geotiff_crs and geotiff_crs == area_crs)
    max_delta = 0.0 if crs_exact_equal and grid_index_equal else np.nan
    step = grid_step_m(area_x, area_y)
    threshold = float(min(1000.0, 0.05 * step)) if np.isfinite(step) else np.nan
    crs_compatible = bool(crs_exact_equal or (np.isfinite(max_delta) and np.isfinite(threshold) and max_delta <= threshold))
    pass_gate = bool(grid_index_equal and crs_compatible and info["n_frames"] == 1 and arr.ndim == 2)
    return pd.DataFrame(
        [
            {
                "sample_geotiff": rel(sample_path),
                "cell_area_source": rel(area_path),
                "reader": "Pillow.Image",
                "single_band": bool(arr.ndim == 2 and info["n_frames"] == 1),
                "geotiff_shape_y": int(arr.shape[0]),
                "geotiff_shape_x": int(arr.shape[1]),
                "area_shape_y": int(area_km2.shape[0]),
                "area_shape_x": int(area_km2.shape[1]),
                "shape_equal": shape_equal,
                "x_allclose": x_allclose,
                "y_allclose": y_allclose,
                "grid_index_equal": grid_index_equal,
                "pixel_scale_x": float(info["pixel_scale_x"]),
                "pixel_scale_y": float(info["pixel_scale_y"]),
                "tiepoint_x": float(info["tiepoint_x"]),
                "tiepoint_y": float(info["tiepoint_y"]),
                "geotiff_crs_identifier": geotiff_crs,
                "area_crs_identifier": area_crs,
                "crs_exact_equal": crs_exact_equal,
                "max_geolocation_delta_m": max_delta,
                "compatibility_threshold_m": threshold,
                "crs_compatible": crs_compatible,
                "alignment_status": "PASS" if pass_gate else "FAIL",
                "grid_alignment_pass": pass_gate,
                "geo_key_directory": json.dumps(info.get("geo_key_directory", ())),
                "geo_ascii_params": info.get("geo_ascii_params", ""),
            }
        ]
    )


def plan_metadata() -> dict[str, Any]:
    if PLAN_PATH.exists():
        return {
            "experiment_plan_path": rel(PLAN_PATH),
            "experiment_plan_status": "found",
            "experiment_plan_sha256": sha256_file(PLAN_PATH),
        }
    return {
        "experiment_plan_path": rel(PLAN_PATH),
        "experiment_plan_status": "missing",
        "experiment_plan_sha256": "",
    }


def build_method_contract() -> dict[str, Any]:
    plan = plan_metadata()
    return {
        "stage3_scope": "Batch 0-1 only: method contract, schema audit, surface accounting, footprint strategy, detector-aligned closure gate.",
        "stage_numbering_note": "Top-level Stage 3 is pan-Arctic/local contribution attribution; stage2_experiment_core.py internal stage3_major is historical naming only.",
        **plan,
        "stage3_experiment_plan_md": {**plan, "contract_source": "active Stage 3 experiment plan plus actual source code inspection"},
        "detector_delta_sie_formula": "For delta_days=3, compute_delta_sie sets delta_sie(date n) = extent(n+1) - extent(n-2); boundary values only use nearest-neighbour at series ends.",
        "detector_metric_window_semantics": "3-day detector-aligned Sea Ice Index extent difference spanning window_start=date-2 to window_end=date+1; units are million km2 from the Sea Ice Index CSV.",
        "detector_anchor_date_rule": "Event row date is the detector date n to which delta_sie(n)=SIE(n+1)-SIE(n-2) is assigned.",
        "unique_event_date_rule": "unique_gap_days=1 deduplicate_adjacent keeps the final row/date in each adjacent block, per src/vrile/utils.py; it does not choose the strongest member.",
        "detector_date_exclusion_rule": "Detection first selects requested months on the event row date, then excludes the first day and last two days of each calendar month when exclude_month_boundary=True.",
        "percentile_semantics": "_5p in Stage 1 output names comes from DetectionConfig.percentile=5.0, i.e. lower 5th percentile threshold; it is not a 5-day window label.",
        "detector_methods": {
            "butterworth": "high-pass Butterworth-filtered delta_sie with cutoff 18 days and order 12; threshold is nanpercentile(filtered delta, 5) over the full post-year/month/date-exclusion base series.",
            "mean_removed": "daily-climatology mean-removed extent anomaly is differenced with the same delta_days=3 formula; threshold is nanpercentile over base_mask rows.",
            "both": "union of butterworth and mean_removed event rows followed by unique-gap deduplication when unique_gap_days>0.",
        },
        "spatial_sic_change_window_days": "stage1_reproduce_sic_locations.py default --window-days=5; stage1_detect_local_sic_loss_objects.py scans --window-days 3 5 7 10 in Stage 1 and consolidation primary-window-days=5.",
        "spatial_window_end_rule": "SIC spatial loss windows end on the event/object date and use start_date = date - window_days; this is distinct from detector_delta_sie_formula.",
        "closure_object": "Batch 0-1R separates Track S CDR cross-product concordance from Track D detector-source closure. Spatial 5-day SIC windows are recorded but not used for detector-source closure.",
        "exact_attribution_scope": "g02135_detector_aligned_extent_only",
        "cdr_spatial_composition_scope": "g02202_5day_sic",
        "primary_panarctic_anchor": "Stage 3 primary pan-Arctic anchor is outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv column date with the unique-event date semantics above.",
        "strongest_member_date_rule": "Strongest-member date is reserved for future sensitivity only and is not used as the primary anchor in Batch 0-1.",
        "local_loss_thresholds": "Stage 1 broad detector uses thresholds [-0.05, -0.10, -0.15]; stage1_reproduce_sic_locations.py localization default loss_threshold=-0.1; consolidation primary_sic_change_threshold=-0.10.",
        "local_connectivity": "stage1_detect_local_sic_loss_objects.py uses scipy.ndimage.label with structure=np.ones((3,3), dtype=bool), i.e. 8-neighbour connectivity; stage1_reproduce_sic_locations.py additionally applies binary_closing before labeling its single largest locator object.",
        "local_min_cells": "Stage 1 broad daily object detection min-cells=4; consolidation sensitivity min-cells=[20,50,100] with primary-min-cells=50; stage1_reproduce_sic_locations.py default min_object_cells=4.",
        "local_patch_id_rule": "Daily broad local objects get sequential object_id LOC0000001... in detect_objects traversal order by date, window, threshold, connected-component label; unique local events get ULE ids in consolidation merge order by region/date/loss.",
        "surface_ocean_codes": sorted(OCEAN_CODES),
        "named_region_codes": dict(sorted(REGION_BY_CODE.items())),
        "stage3_surface_mask_accounting_rule": "1-18 -> named physical regions; 0 -> non_region_ocean; 30/32/33/34/35/40 -> non_ocean; unknown code -> hard error.",
        "stage3_valid_ocean_accounting_domain": "named_regions + non_region_ocean = valid ocean accounting domain",
        "detector_vs_spatial_window_distinction_locked": True,
        "closure_uses_spatial_sic_window": False,
        "detection_config": DETECTION_CONFIG,
    }


def input_schema_audit() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    items = [
        (ROOT / "outputs/reproduce_sie", "pan-Arctic detector output directory", [], "contains unique/all detector outputs and processed daily SIE"),
        (PAN_UNIQUE, "pan-Arctic unique event table", ["date", "method", "raw_delta_sie", "vrile_value", "threshold"], "Primary anchor date comes from date; reported delta is raw_delta_sie."),
        (PAN_ALL, "pan-Arctic all/member event table", ["date", "method", "raw_delta_sie", "vrile_value", "threshold"], "Fallback source for detector metric/member records."),
        (ROOT / "outputs/reproduce_sie/daily_sie_processed.csv", "processed Sea Ice Index daily table", ["date", "extent", "delta_sie"], "Can recompute detector delta_sie if needed."),
        (ROOT / "outputs/reproduce_sie/vrile_locations.csv", "pan-Arctic SIC localization table", ["date", "start_date", "found", "center_lon", "center_lat", "window_days", "loss_threshold"], "Centroid/location table is not a footprint and is not used as footprint membership."),
        (ROOT / "outputs/local_vrile", "local broad object output directory", [], "Daily local SIC-loss object outputs."),
        (ROOT / "outputs/local_vrile/local_objects_all_jja.csv", "daily local SIC-loss object table", ["object_id", "date", "start_date", "window_days", "threshold", "object_area_cells", "cumulative_sic_loss"], "Patch IDs and metrics exist; cell membership/y_idx/x_idx are not present."),
        (ROOT / "outputs/local_vrile/local_objects_unique.csv", "unique-by-region/date/window/threshold local object table", ["object_id", "date", "window_days", "threshold", "object_area_cells"], "Unique object summary only, no cell membership."),
        (ROOT / "outputs/local_vrile_enhanced", "consolidated local event output directory", [], "Unique local event outputs."),
        (ROOT / "outputs/local_vrile_enhanced/local_objects_filtered.csv", "primary filtered daily object table", ["object_id", "date", "window_days", "threshold", "object_area_cells"], "Candidate member patch table for ULE construction; no ULE member mapping column."),
        (ROOT / "outputs/local_vrile_enhanced/unique_local_events.csv", "unique local event table", ["unique_local_event_id", "event_start", "event_end", "window_days", "sic_change_threshold", "min_cells"], "Unique event table lacks explicit member object_id list."),
        (ROOT / "outputs/local_vrile_severe", "severe local event output directory", [], "Severe event outputs."),
        (ROOT / "outputs/local_vrile_severe/severe_unique_local_events.csv", "severe unique local events", ["unique_local_event_id", "event_start", "event_end", "cumulative_loss"], "Derived from unique local events."),
        (ROOT / "outputs/local_vrile_major_severe", "major_severe output directory", [], "major_severe event outputs."),
        (ROOT / "outputs/local_vrile_major_severe/major_severe_events_union.csv", "major_severe event union", ["unique_local_event_id", "event_date", "event_start", "event_end", "cumulative_loss"], "Stage 2 major_severe event definitions; not used for closure."),
        (SIC_ROOT, "raw NSIDC SIC root", [], "Contains year subdirectories of daily CDR SIC NetCDF files."),
        (MASK_PATH, "NSIDC-0780 surface region mask", [], "Required for Stage 3 accounting domain."),
        (CELL_AREA_CANDIDATES[0], "NSIDC-0771 PS-N25km cell-area ancillary v1.1", [], "Required for all Stage 3 area-weighted integration."),
        (G02135_MANIFEST, "G02135 daily extent GeoTIFF download manifest", ["date", "local_path", "status", "sha256"], "Required D1a inventory; use existing files only."),
        (G02135_REQUIRED_DATES, "G02135 detector-window required dates", ["date", "roles", "pan_event_count"], "Required D1a date inventory."),
    ]
    for path, role, required, note in items:
        cols = csv_columns(path)
        status, status_note = status_for(path, required)
        footprint_reconstructable = "not_applicable"
        if path.name in {"local_objects_all_jja.csv", "local_objects_unique.csv", "local_objects_filtered.csv"}:
            footprint_reconstructable = "summary_only_no_cell_membership"
        elif path.name == "unique_local_events.csv":
            footprint_reconstructable = "event_summary_no_member_patch_mapping"
        elif path == SIC_ROOT:
            footprint_reconstructable = "source_grids_available_for_future_reconstruction"
        elif path == MASK_PATH:
            footprint_reconstructable = "mask_reference_available"
        elif path.name == CELL_AREA_FILENAME:
            footprint_reconstructable = "cell_area_required_for_area_weighted_integration"
        records.append(
            {
                "source": rel(path),
                "required_role": role,
                "required_columns": ";".join(required),
                "available_columns": ";".join(cols),
                "rows": csv_rows(path) if path.is_file() else math.nan,
                "footprint_reconstructable": footprint_reconstructable,
                "status": status,
                "notes": "; ".join(x for x in [note, status_note] if x),
            }
        )
    return pd.DataFrame(records)


def load_surface_mask() -> np.ndarray:
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Required NSIDC-0780 mask missing: {MASK_PATH}")
    with xr.open_dataset(MASK_PATH, mask_and_scale=False) as ds:
        if NSIDC_0780_REGION_VARIABLE not in ds:
            raise KeyError(f"{NSIDC_0780_REGION_VARIABLE} not found in {MASK_PATH}")
        arr = np.asarray(ds[NSIDC_0780_REGION_VARIABLE].values, dtype=np.int16)
    return arr


def surface_mask_accounting_audit() -> pd.DataFrame:
    arr = load_surface_mask()
    vals, counts = np.unique(arr, return_counts=True)
    unknown = sorted(int(v) for v in vals if int(v) not in KNOWN_CODES)
    if unknown:
        raise ValueError(f"Unknown NSIDC-0780 surface mask codes encountered: {unknown}")
    rows = []
    for v, c in zip(vals, counts):
        code = int(v)
        if code == 0:
            klass = "non_region_ocean"
            name = "non_region_ocean"
        elif code in NAMED_REGION_CODES:
            klass = "named_region"
            name = REGION_BY_CODE[code]
        elif code in NON_OCEAN_CODES:
            klass = "non_ocean"
            name = ""
        else:
            raise ValueError(f"Unknown code {code}")
        rows.append({"surface_code": code, "cell_count": int(c), "stage3_accounting_class": klass, "named_region": name})
    return pd.DataFrame(rows).sort_values("surface_code")


def find_cell_area_file() -> Path | None:
    for path in CELL_AREA_CANDIDATES:
        if path.exists():
            return path
    return None


def choose_cell_area_var(ds: xr.Dataset) -> str:
    candidates = []
    for name, da in ds.data_vars.items():
        if name == "crs":
            continue
        lname = name.lower()
        units = str(da.attrs.get("units", "")).lower()
        long_name = str(da.attrs.get("long_name", "")).lower()
        if "area" in lname or "area" in long_name or units in {"km2", "km^2", "m2", "m^2", "square meters", "square kilometers"}:
            candidates.append(name)
    if not candidates:
        raise ValueError(f"Could not infer cell-area variable from {list(ds.data_vars)}")
    return candidates[0]


def area_to_km2(arr: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    u = units.lower().replace(" ", "")
    if u in {"km2", "km^2", "squarekilometers", "squarekilometres"}:
        return arr.astype(float), "km2"
    if u in {"m2", "m^2", "meters^2", "metres^2", "squaremeters", "squaremetres"}:
        return arr.astype(float) / 1_000_000.0, "m2_to_km2"
    raise ValueError(f"Unsupported cell-area units: {units!r}")


def read_grid_signature(path: Path, *, mask_and_scale: bool = True) -> dict[str, Any]:
    with xr.open_dataset(path, mask_and_scale=mask_and_scale) as ds:
        x = np.asarray(ds["x"].values) if "x" in ds.coords else np.asarray([])
        y = np.asarray(ds["y"].values) if "y" in ds.coords else np.asarray([])
        return {
            "shape_y": int(ds.sizes.get("y", -1)),
            "shape_x": int(ds.sizes.get("x", -1)),
            "x_size": int(x.size),
            "y_size": int(y.size),
            "x_first": float(x[0]) if x.size else np.nan,
            "x_last": float(x[-1]) if x.size else np.nan,
            "y_first": float(y[0]) if y.size else np.nan,
            "y_last": float(y[-1]) if y.size else np.nan,
            "x_step": float(x[1] - x[0]) if x.size > 1 else np.nan,
            "y_step": float(y[1] - y[0]) if y.size > 1 else np.nan,
            "crs_identifier": dataset_crs_identifier(ds),
        }


def load_cell_area_grid() -> tuple[np.ndarray | None, pd.DataFrame]:
    area_path = find_cell_area_file()
    sample_sic = find_sic_file(pd.Timestamp("1989-01-01"))
    if sample_sic is None:
        return None, pd.DataFrame([{"grid_alignment_pass": False, "alignment_status": "missing_sic_sample", "notes": "No sample SIC file found for alignment audit."}])
    if area_path is None:
        return None, pd.DataFrame([{"grid_alignment_pass": False, "alignment_status": "missing_cell_area", "source_path": "", "notes": f"{CELL_AREA_FILENAME} not found at the fixed Stage 3 contract path."}])

    with xr.open_dataset(area_path) as ds_area, xr.open_dataset(sample_sic) as ds_sic, xr.open_dataset(MASK_PATH, mask_and_scale=False) as ds_mask:
        area_var = choose_cell_area_var(ds_area)
        area_raw = np.asarray(ds_area[area_var].values, dtype=float).squeeze()
        units = str(ds_area[area_var].attrs.get("units", ""))
        area_km2, unit_conversion = area_to_km2(area_raw, units)
        sic_x = np.asarray(ds_sic["x"].values)
        sic_y = np.asarray(ds_sic["y"].values)
        area_x = np.asarray(ds_area["x"].values) if "x" in ds_area.coords else np.asarray([])
        area_y = np.asarray(ds_area["y"].values) if "y" in ds_area.coords else np.asarray([])
        mask_x = np.asarray(ds_mask["x"].values)
        mask_y = np.asarray(ds_mask["y"].values)
        shape_equal = area_km2.shape == (ds_sic.sizes["y"], ds_sic.sizes["x"]) == (ds_mask.sizes["y"], ds_mask.sizes["x"])
        x_equal = bool(area_x.size == sic_x.size == mask_x.size and np.allclose(area_x, sic_x) and np.allclose(area_x, mask_x))
        y_equal = bool(area_y.size == sic_y.size == mask_y.size and np.allclose(area_y, sic_y) and np.allclose(area_y, mask_y))
        grid_index_equal = bool(shape_equal and x_equal and y_equal)
        area_attrs = crs_attrs(ds_area)
        sic_attrs = crs_attrs(ds_sic)
        mask_attrs = crs_attrs(ds_mask)
        area_crs = normalize_crs_identifier(dataset_crs_identifier(ds_area), area_attrs)
        sic_crs = normalize_crs_identifier(dataset_crs_identifier(ds_sic), sic_attrs)
        mask_crs = normalize_crs_identifier(dataset_crs_identifier(ds_mask), mask_attrs)
        crs_exact_equal = bool(area_crs == sic_crs == mask_crs)
        if grid_index_equal:
            delta_area_sic, method_sic = sampled_geolocation_delta_with_method(area_x, area_y, area_attrs, sic_attrs)
            delta_area_mask, method_mask = sampled_geolocation_delta_with_method(area_x, area_y, area_attrs, mask_attrs)
            crs_delta_method = method_sic if method_sic == method_mask else f"{method_sic};{method_mask}"
        else:
            delta_area_sic = delta_area_mask = np.nan
            crs_delta_method = "not_evaluable"
        max_delta = float(np.nanmax([delta_area_sic, delta_area_mask]))
        step = grid_step_m(area_x, area_y)
        threshold = float(min(1000.0, 0.05 * step)) if np.isfinite(step) else np.nan
        crs_compatible = bool(crs_exact_equal or (np.isfinite(max_delta) and np.isfinite(threshold) and max_delta <= threshold))
        pass_gate = bool(grid_index_equal and crs_compatible)
        audit = pd.DataFrame(
            [
                {
                    "source_path": rel(area_path),
                    "area_variable": area_var,
                    "area_units": units,
                    "unit_conversion": unit_conversion,
                    "shape_equal": shape_equal,
                    "x_allclose": x_equal,
                    "y_allclose": y_equal,
                    "grid_index_equal": grid_index_equal,
                    "crs_exact_equal": crs_exact_equal,
                    "max_geolocation_delta_m": max_delta,
                    "compatibility_threshold_m": threshold,
                    "crs_compatible": crs_compatible,
                    "crs_delta_method": crs_delta_method,
                    "alignment_status": "PASS" if pass_gate else "FAIL",
                    "grid_alignment_pass": pass_gate,
                    "area_shape": "x".join(map(str, area_km2.shape)),
                    "min_area_km2": float(np.nanmin(area_km2)),
                    "median_area_km2": float(np.nanmedian(area_km2)),
                    "max_area_km2": float(np.nanmax(area_km2)),
                    "area_crs_identifier": area_crs,
                    "sic_crs_identifier": sic_crs,
                    "mask_crs_identifier": mask_crs,
                    "area_crs_raw": json.dumps(area_attrs, sort_keys=True),
                    "sic_crs_raw": json.dumps(sic_attrs, sort_keys=True),
                    "mask_crs_raw": json.dumps(mask_attrs, sort_keys=True),
                    "notes": "Sampled geolocation delta uses pyproj when available and the internal spherical polar-stereographic inverse as fallback.",
                }
            ]
        )
    if not pass_gate:
        return None, audit
    return area_km2, audit


def footprint_strategy_audit() -> dict[str, Any]:
    return {
        "selected_plan": "Plan A",
        "planned_strategy": "Plan A",
        "implementation_status": "pending_batch2_reconciliation",
        "temporary_batch0_1_strategy": "superseded_by_batch2_shared_primitives",
        "batch2_required_action": "run Stage 3 Batch 2 footprint reconciliation before treating implementation as complete",
        "final_release_architecture": True,
        "detector_function": "vrile.local_objects.detect_connected_components",
        "merge_function": "vrile.local_objects.merge_local_patches",
        "shared_core_path": "src/vrile/local_objects.py",
        "connectivity": "8-neighbour scipy.ndimage.label with np.ones((3,3), dtype=bool)",
        "threshold_semantics": "loss = finite pan-Arctic SIC difference end_arr-start_arr <= threshold; Stage 1 broad thresholds are -0.05, -0.10, -0.15.",
        "min_cells_semantics": "Connected component retained when object_area_cells >= --min-cells in stage1_detect_local_sic_loss_objects.py; Stage 1 broad --min-cells=4.",
        "patch_key_fields": ["object_id", "date", "start_date", "window_days", "threshold"],
        "reconciliation_required": True,
        "reason": "Batch 2 extracted shared import-safe primitives for connected-component detection and unique-event merge/membership. Stage 1 scripts and Stage 3 now call the same primitives while date traversal, object_id allocation, plotting, and CSV writing remain in the entry scripts.",
        "future_reconciliation_contract": {
            "required_file": "footprint_detector_reconciliation.csv",
            "required_fields": [
                "patch_key",
                "reported_cell_count",
                "recomputed_cell_count",
                "reported_area",
                "recomputed_area",
                "reported_loss",
                "recomputed_loss",
                "reported_centroid_x",
                "reported_centroid_y",
                "recomputed_centroid_x",
                "recomputed_centroid_y",
                "status",
            ],
            "pass_rule": "patch count exact; patch identity/key exact; cell count exact; numeric metrics np.isclose at original CSV precision; no systematic bias; no cell-membership hash unless original data include cell membership.",
        },
    }


def find_sic_file(date: pd.Timestamp) -> Path | None:
    matches = sorted(glob.glob(str(SIC_ROOT / f"{date.year}" / f"sic_psn25_{date:%Y%m%d}_*.nc")))
    return Path(matches[0]) if matches else None


def guess_sic_var(ds: xr.Dataset) -> str:
    for name in ("cdr_seaice_conc", "seaice_conc_cdr", "sic", "ice_conc"):
        if name in ds:
            return name
    for name in ds.data_vars:
        if "time" in ds[name].dims:
            return name
    raise ValueError(f"Cannot infer SIC variable from {list(ds.data_vars)}")


def read_sic_array(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        var = guess_sic_var(ds)
        da = ds[var]
        if "time" in da.dims:
            arr = np.asarray(da.isel(time=0).values, dtype=float).squeeze()
        else:
            arr = np.asarray(da.values, dtype=float).squeeze()
    arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
    return arr


def cdr_extent_million_km2(date: pd.Timestamp, ocean_mask: np.ndarray, area_km2: np.ndarray, cache: dict[str, float]) -> float:
    key = date.strftime("%Y-%m-%d")
    if key in cache:
        return cache[key]
    path = find_sic_file(date)
    if path is None:
        cache[key] = np.nan
        return np.nan
    arr = read_sic_array(path)
    if arr.shape != ocean_mask.shape:
        raise ValueError(f"SIC grid shape {arr.shape} does not match NSIDC-0780 mask shape {ocean_mask.shape} for {path}")
    if area_km2.shape != ocean_mask.shape:
        raise ValueError(f"Cell-area grid shape {area_km2.shape} does not match SIC/mask shape {ocean_mask.shape}")
    extent = float(np.nansum(np.where(np.isfinite(arr) & ocean_mask & (arr >= EXTENT_THRESHOLD), area_km2, 0.0)) / 1_000_000.0)
    cache[key] = extent
    return extent


def read_extent_csv_local(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, skiprows=2, header=None)
        if raw.shape[1] < 4:
            raise ValueError
        df = raw.iloc[:, :4].copy()
        df.columns = ["year", "month", "day", "extent"]
        df["date"] = pd.to_datetime(
            dict(year=df.year.astype(int), month=df.month.astype(int), day=df.day.astype(int)),
            errors="coerce",
        )
        out = df[["date", "extent"]].copy()
    except Exception:
        df = pd.read_csv(path)
        lower = {c.lower().strip(): c for c in df.columns}
        if "date" in lower and "extent" in lower:
            out = df[[lower["date"], lower["extent"]]].copy()
            out.columns = ["date", "extent"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
        else:
            required = ["year", "month", "day", "extent"]
            missing = [c for c in required if c not in lower]
            if missing:
                raise ValueError(f"Could not identify columns {missing} in {path}")
            tmp = df[[lower["year"], lower["month"], lower["day"], lower["extent"]]].copy()
            tmp.columns = required
            tmp["date"] = pd.to_datetime(
                dict(year=tmp.year.astype(int), month=tmp.month.astype(int), day=tmp.day.astype(int)),
                errors="coerce",
            )
            out = tmp[["date", "extent"]]
    out["extent"] = pd.to_numeric(out["extent"], errors="coerce")
    return out.dropna(subset=["date", "extent"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def compute_delta_sie_local(df: pd.DataFrame, delta_days: int = 3) -> pd.DataFrame:
    if delta_days != 3:
        raise ValueError("Stage 3 fallback currently only implements the confirmed delta_days=3 detector formula.")
    out = df.copy()
    x = out["extent"].to_numpy(dtype=float)
    dx = np.empty_like(x, dtype=float)
    dx[2:-1] = x[3:] - x[:-3]
    dx[:2] = dx[2]
    dx[-1] = dx[-2]
    out["delta_sie"] = dx
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    return out


def recompute_sie_delta_from_index(events: pd.DataFrame) -> pd.Series:
    raw = read_extent_csv_local(SIE_CSV)
    daily = raw.set_index("date").sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    daily["extent"] = daily["extent"].interpolate(method="time", limit_direction="both")
    daily = compute_delta_sie_local(daily.reset_index(), DETECTION_CONFIG["delta_days"])
    daily = daily[
        (daily["year"] >= DETECTION_CONFIG["start_year"])
        & (daily["year"] <= DETECTION_CONFIG["end_year"])
        & ~((daily["month"] == 2) & (daily["day"] == 29))
    ].reset_index(drop=True)
    lookup = daily.set_index(pd.to_datetime(daily["date"]).dt.normalize())["delta_sie"]
    return pd.to_datetime(events["date"]).dt.normalize().map(lookup)


def cdr_concordance_audit(area_km2: np.ndarray | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PAN_UNIQUE.exists():
        raise FileNotFoundError(f"Pan-Arctic unique event table missing: {PAN_UNIQUE}")
    unique = pd.read_csv(PAN_UNIQUE, parse_dates=["date"])
    if "raw_delta_sie" in unique.columns:
        reported = pd.to_numeric(unique["raw_delta_sie"], errors="coerce")
        metric_source = "unique_event_table.raw_delta_sie"
    elif PAN_ALL.exists() and "raw_delta_sie" in csv_columns(PAN_ALL):
        all_events = pd.read_csv(PAN_ALL, parse_dates=["date"])
        merged = unique[["date", "method"]].merge(all_events[["date", "method", "raw_delta_sie"]], on=["date", "method"], how="left")
        reported = pd.to_numeric(merged["raw_delta_sie"], errors="coerce")
        metric_source = "all_events_table.raw_delta_sie_joined_by_date_method"
    else:
        reported = recompute_sie_delta_from_index(unique)
        metric_source = "recomputed_from_raw_sea_ice_index_exact_delta_formula"

    mask = load_surface_mask()
    ocean_mask = np.isin(mask, list(OCEAN_CODES))
    cache: dict[str, float] = {}
    rows = []
    for i, row in unique.reset_index(drop=True).iterrows():
        event_date = pd.Timestamp(row["date"]).normalize()
        window_start = event_date - pd.Timedelta(days=2)
        window_end = event_date + pd.Timedelta(days=1)
        if area_km2 is None:
            start_extent = np.nan
            end_extent = np.nan
            cdr_delta = np.nan
        else:
            start_extent = cdr_extent_million_km2(window_start, ocean_mask, area_km2, cache)
            end_extent = cdr_extent_million_km2(window_end, ocean_mask, area_km2, cache)
            cdr_delta = end_extent - start_extent if np.isfinite(start_extent) and np.isfinite(end_extent) else np.nan
        rep = float(reported.iloc[i]) if i < len(reported) and pd.notna(reported.iloc[i]) else np.nan
        abs_diff = abs(cdr_delta - rep) if np.isfinite(cdr_delta) and np.isfinite(rep) else np.nan
        rel_diff = abs_diff / abs(rep) if np.isfinite(abs_diff) and np.isfinite(rep) and rep != 0 else np.nan
        if np.isfinite(cdr_delta) and np.isfinite(rep):
            sign_agree = bool(np.sign(cdr_delta) == np.sign(rep)) if rep != 0 and cdr_delta != 0 else bool(rep == cdr_delta)
        else:
            sign_agree = False
        if not np.isfinite(rep):
            status = "missing_reported_delta"
        elif rep == 0:
            status = "reported_zero_relative_difference_nan"
        elif area_km2 is None:
            status = "cell_area_unavailable"
        elif not np.isfinite(cdr_delta):
            status = "missing_cdr_reconstruction"
        else:
            status = "evaluable"
        rows.append(
            {
                "pan_event_id": f"PAN{i+1:06d}",
                "event_date": event_date.strftime("%Y-%m-%d"),
                "detector_formula": "delta_sie(date)=SIE(date+1)-SIE(date-2)",
                "window_start": window_start.strftime("%Y-%m-%d"),
                "window_end": window_end.strftime("%Y-%m-%d"),
                "reported_delta_sie": rep,
                "reported_metric_source": metric_source,
                "cell_area_source": rel(find_cell_area_file()) if find_cell_area_file() else "",
                "cdr_reconstructed_delta_sie": cdr_delta,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "sign_agree": sign_agree,
                "closure_status": status,
            }
        )
    audit = pd.DataFrame(rows)
    evaluable = audit[audit["closure_status"].eq("evaluable")].copy()
    event_total = int(len(audit))
    event_evaluable = int(len(evaluable))
    coverage = event_evaluable / event_total if event_total else 0.0
    if event_evaluable:
        median_rel = float(evaluable["relative_difference"].median())
        p90_rel = float(evaluable["relative_difference"].quantile(0.90))
        sign_fraction = float(evaluable["sign_agree"].mean())
        rho = float(evaluable[["reported_delta_sie", "cdr_reconstructed_delta_sie"]].corr(method="spearman").iloc[0, 1]) if event_evaluable >= 2 else np.nan
        bias = float((evaluable["cdr_reconstructed_delta_sie"] - evaluable["reported_delta_sie"]).mean())
        evaluable["event_year"] = pd.to_datetime(evaluable["event_date"]).dt.year
        pre_2025 = evaluable[evaluable["event_year"] < 2025]
        year_2025 = evaluable[evaluable["event_year"] == 2025]
        pre_2025_rho = float(pre_2025[["reported_delta_sie", "cdr_reconstructed_delta_sie"]].corr(method="spearman").iloc[0, 1]) if len(pre_2025) >= 2 else np.nan
        year_2025_rho = float(year_2025[["reported_delta_sie", "cdr_reconstructed_delta_sie"]].corr(method="spearman").iloc[0, 1]) if len(year_2025) >= 2 else np.nan
    else:
        median_rel = p90_rel = sign_fraction = rho = bias = np.nan
        pre_2025 = year_2025 = pd.DataFrame()
        pre_2025_rho = year_2025_rho = np.nan

    concordance_status = "evaluable" if event_evaluable else "unavailable"
    reason = "CDR cross-product concordance uses per-cell NSIDC-0771 area when available; Track D exact gates are not applied to this Track S comparison."
    if area_km2 is None:
        reason = "CDR concordance unavailable because authoritative NSIDC-0771 PS-N25km cell-area grid is not locally available."
    summary = pd.DataFrame(
        [
            {
                "comparison_type": "cdr_cross_product_concordance",
                "event_total": event_total,
                "event_evaluable": event_evaluable,
                "event_coverage_fraction": coverage,
                "median_relative_difference": median_rel,
                "p90_relative_difference": p90_rel,
                "sign_agreement_fraction": sign_fraction,
                "spearman_rho": rho,
                "signed_bias": bias,
                "pre_2025_n": int(len(pre_2025)),
                "pre_2025_spearman_rho": pre_2025_rho,
                "year_2025_n": int(len(year_2025)),
                "year_2025_spearman_rho": year_2025_rho,
                "concordance_status": concordance_status,
                "gate_reason": reason,
                "comparison_note": "Track S cross-product comparison: Stage 1 detector uses NSIDC Sea Ice Index extent CSV; reconstruction uses G02202/CDR SIC grid with SIC>=0.15, NSIDC-0771 per-cell area, and Stage 3 valid ocean accounting domain codes 0-18. This is not exact detector-source closure.",
            }
        ]
    )
    return audit, summary


def g02135_required_inventory(unique_events: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(G02135_MANIFEST) if G02135_MANIFEST.exists() else pd.DataFrame()
    manifest_by_date: dict[str, pd.Series] = {}
    if not manifest.empty and "date" in manifest.columns:
        for _, row in manifest.iterrows():
            manifest_by_date[str(row["date"])] = row

    roles: dict[str, dict[str, Any]] = {}
    for i, row in unique_events.reset_index(drop=True).iterrows():
        event_date = pd.Timestamp(row["date"]).normalize()
        for role, d in [("window_start", event_date - pd.Timedelta(days=2)), ("window_end", event_date + pd.Timedelta(days=1))]:
            key = d.strftime("%Y-%m-%d")
            item = roles.setdefault(key, {"date": key, "roles": set(), "pan_event_ids": []})
            item["roles"].add(role)
            item["pan_event_ids"].append(f"PAN{i+1:06d}")

    rows = []
    for key in sorted(roles):
        date = pd.Timestamp(key)
        path = find_g02135_geotiff(date)
        manifest_row = manifest_by_date.get(key)
        manifest_status = str(manifest_row.get("status", "")) if manifest_row is not None else "missing_manifest_row"
        manifest_sha = str(manifest_row.get("sha256", "")) if manifest_row is not None else ""
        actual_sha = sha256_file(path) if path and path.exists() else ""
        if path is None:
            status = "missing_file"
        elif manifest_sha and manifest_sha != actual_sha:
            status = "sha256_mismatch"
        elif manifest_status and manifest_status not in {"downloaded", "skipped_valid", "ok"}:
            status = f"manifest_status_{manifest_status}"
        else:
            status = "ok"
        rows.append(
            {
                "date": key,
                "roles": ";".join(sorted(roles[key]["roles"])),
                "pan_event_count": len(set(roles[key]["pan_event_ids"])),
                "pan_event_ids": ";".join(sorted(set(roles[key]["pan_event_ids"]))),
                "selected_file": rel(path) if path else "",
                "selected_version": ".".join(map(str, parse_geotiff_version(path))) if path else "",
                "manifest_status": manifest_status,
                "manifest_sha256": manifest_sha,
                "actual_sha256": actual_sha,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def detector_source_gate(
    event_total: int,
    event_evaluable: int,
    median_rel: float,
    p90_rel: float,
    sign_fraction: float,
    rho: float,
    *,
    unavailable_reason: str = "",
) -> tuple[str, str]:
    if unavailable_reason:
        return "unavailable", unavailable_reason
    coverage = event_evaluable / event_total if event_total else 0.0
    if (
        coverage >= 0.95
        and np.isfinite(median_rel)
        and np.isfinite(p90_rel)
        and np.isfinite(sign_fraction)
        and np.isfinite(rho)
        and median_rel <= 0.15
        and p90_rel <= 0.30
        and sign_fraction >= 0.95
        and rho >= 0.90
    ):
        return "PASS", "Track D detector-source closure meets all engineering thresholds."
    stop_reasons = []
    if coverage < 0.95:
        stop_reasons.append(f"coverage {coverage:.3f} < 0.95")
    if np.isfinite(median_rel) and median_rel > 0.20:
        stop_reasons.append(f"median relative difference {median_rel:.3f} > 0.20")
    if np.isfinite(p90_rel) and p90_rel > 0.40:
        stop_reasons.append(f"P90 relative difference {p90_rel:.3f} > 0.40")
    if np.isfinite(sign_fraction) and sign_fraction < 0.90:
        stop_reasons.append(f"sign agreement {sign_fraction:.3f} < 0.90")
    if np.isfinite(rho) and rho < 0.80:
        stop_reasons.append(f"Spearman rho {rho:.3f} < 0.80")
    if stop_reasons:
        return "STOP", "; ".join(stop_reasons)
    return "WARN", "Track D detector-source closure is between PASS and STOP thresholds; diagnose before exact attribution."


def detector_source_closure(unique_events: pd.DataFrame, area_km2: np.ndarray | None, inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    event_total = int(len(unique_events))
    if area_km2 is None:
        gate, reason = detector_source_gate(event_total, 0, np.nan, np.nan, np.nan, np.nan, unavailable_reason="cell-area grid unavailable or failed alignment")
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "detector_source_route": "D1a",
                    "detector_source_closure": gate,
                    "event_total": event_total,
                    "event_evaluable": 0,
                    "event_coverage_fraction": 0.0,
                    "median_relative_difference": np.nan,
                    "p90_relative_difference": np.nan,
                    "sign_agreement_fraction": np.nan,
                    "spearman_rho": np.nan,
                    "signed_bias": np.nan,
                    "engineering_gate": gate,
                    "gate_reason": reason,
                    "unavailable_reason": reason,
                }
            ]
        )
    missing = inventory[inventory["status"].ne("ok")] if not inventory.empty else pd.DataFrame()
    if inventory.empty or not missing.empty:
        reason = "required_g02135_geotiff_missing_or_failed_inventory"
        gate, gate_reason = detector_source_gate(event_total, 0, np.nan, np.nan, np.nan, np.nan, unavailable_reason=reason)
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "detector_source_route": "D1a",
                    "detector_source_closure": gate,
                    "event_total": event_total,
                    "event_evaluable": 0,
                    "event_coverage_fraction": 0.0,
                    "median_relative_difference": np.nan,
                    "p90_relative_difference": np.nan,
                    "sign_agreement_fraction": np.nan,
                    "spearman_rho": np.nan,
                    "signed_bias": np.nan,
                    "engineering_gate": gate,
                    "gate_reason": gate_reason,
                    "unavailable_reason": reason,
                }
            ]
        )

    cache: dict[str, tuple[float, dict[str, Any], Path]] = {}

    def extent_for(date: pd.Timestamp) -> tuple[float, dict[str, Any], Path]:
        key = date.strftime("%Y-%m-%d")
        if key in cache:
            return cache[key]
        path = find_g02135_geotiff(date)
        if path is None:
            raise FileNotFoundError(f"Required G02135 GeoTIFF missing for {key}")
        extent, info = g02135_extent_million_km2(path, area_km2)
        cache[key] = (extent, info, path)
        return cache[key]

    for i, row in unique_events.reset_index(drop=True).iterrows():
        event_date = pd.Timestamp(row["date"]).normalize()
        window_start = event_date - pd.Timedelta(days=2)
        window_end = event_date + pd.Timedelta(days=1)
        start_extent, start_info, start_path = extent_for(window_start)
        end_extent, end_info, end_path = extent_for(window_end)
        detector_delta = end_extent - start_extent
        rep = float(row.get("raw_delta_sie", np.nan)) if pd.notna(row.get("raw_delta_sie", np.nan)) else np.nan
        abs_diff = abs(detector_delta - rep) if np.isfinite(detector_delta) and np.isfinite(rep) else np.nan
        rel_diff = abs_diff / abs(rep) if np.isfinite(abs_diff) and np.isfinite(rep) and rep != 0 else np.nan
        sign_agree = bool(np.sign(detector_delta) == np.sign(rep)) if np.isfinite(detector_delta) and np.isfinite(rep) and rep != 0 and detector_delta != 0 else bool(detector_delta == rep)
        if not np.isfinite(rep):
            status = "missing_reported_delta"
        elif rep == 0:
            status = "reported_zero_relative_difference_nan"
        elif not np.isfinite(detector_delta):
            status = "missing_detector_source_reconstruction"
        else:
            status = "evaluable"
        rows.append(
            {
                "pan_event_id": f"PAN{i+1:06d}",
                "event_date": event_date.strftime("%Y-%m-%d"),
                "detector_source_route": "D1a",
                "detector_source_status": "evaluable",
                "detector_formula": "delta_sie(date)=SIE(date+1)-SIE(date-2)",
                "window_start": window_start.strftime("%Y-%m-%d"),
                "window_end": window_end.strftime("%Y-%m-%d"),
                "start_geotiff": rel(start_path),
                "end_geotiff": rel(end_path),
                "start_version": ".".join(map(str, parse_geotiff_version(start_path))),
                "end_version": ".".join(map(str, parse_geotiff_version(end_path))),
                "start_extent_million_km2": start_extent,
                "end_extent_million_km2": end_extent,
                "reported_delta_sie": rep,
                "reported_metric_source": "unique_event_table.raw_delta_sie",
                "geotiff_reconstructed_delta_sie": detector_delta,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "sign_agree": sign_agree,
                "closure_status": status,
                "reader": "Pillow.Image",
                "start_epsg": start_info.get("epsg", ""),
                "end_epsg": end_info.get("epsg", ""),
            }
        )
    audit = pd.DataFrame(rows)
    evaluable = audit[audit["closure_status"].eq("evaluable")].copy()
    event_evaluable = int(len(evaluable))
    coverage = event_evaluable / event_total if event_total else 0.0
    if event_evaluable:
        median_rel = float(evaluable["relative_difference"].median())
        p90_rel = float(evaluable["relative_difference"].quantile(0.90))
        sign_fraction = float(evaluable["sign_agree"].mean())
        rho = float(evaluable[["reported_delta_sie", "geotiff_reconstructed_delta_sie"]].corr(method="spearman").iloc[0, 1]) if event_evaluable >= 2 else np.nan
        bias = float((evaluable["geotiff_reconstructed_delta_sie"] - evaluable["reported_delta_sie"]).mean())
    else:
        median_rel = p90_rel = sign_fraction = rho = bias = np.nan
    gate, reason = detector_source_gate(event_total, event_evaluable, median_rel, p90_rel, sign_fraction, rho)
    summary = pd.DataFrame(
        [
            {
                "detector_source_route": "D1a",
                "detector_source_closure": gate,
                "event_total": event_total,
                "event_evaluable": event_evaluable,
                "event_coverage_fraction": coverage,
                "median_relative_difference": median_rel,
                "p90_relative_difference": p90_rel,
                "sign_agreement_fraction": sign_fraction,
                "spearman_rho": rho,
                "signed_bias": bias,
                "engineering_gate": gate,
                "gate_reason": reason,
                "unavailable_reason": "",
                "source_note": "D1a uses existing G02135 daily extent GeoTIFFs, GeoTIFF value 1 as ice, and NSIDC-0771 v1.1 PS-N25km per-cell area. No NSIDC-0051/0803 D2 route is downloaded or implemented.",
            }
        ]
    )
    return audit, summary


def validate_no_forbidden_batch_outputs() -> None:
    forbidden_terms = ["raster_overlap", "contribution_decomposition", "residual_co_loss", "n_eff", "masked_reconstruction", "phenotype"]
    existing = []
    if OUT_DIR.exists():
        for p in OUT_DIR.rglob("*"):
            low = p.name.lower()
            if any(term in low for term in forbidden_terms):
                existing.append(rel(p))
    if existing:
        raise RuntimeError(f"Forbidden Batch 2-4 outputs found in Stage 3 output directory: {existing}")


def remove_stale_closure_outputs() -> None:
    for name in [
        "panarctic_sie_spatial_closure_audit.csv",
        "panarctic_sie_spatial_closure_summary.csv",
    ]:
        path = OUT_DIR / name
        if path.exists():
            path.unlink()


def route_decision(detector_summary: pd.DataFrame, cdr_summary: pd.DataFrame, cell_area_alignment: pd.DataFrame) -> dict[str, Any]:
    d_gate = str(detector_summary.loc[0, "engineering_gate"])
    d_status = str(detector_summary.loc[0, "detector_source_closure"])
    if "grid_alignment_pass" in cell_area_alignment.columns:
        area_pass = bool(cell_area_alignment["grid_alignment_pass"].fillna(False).astype(bool).any())
    else:
        area_pass = False
    cdr_evaluable = bool(float(cdr_summary.loc[0, "event_coverage_fraction"]) >= 0.95) if not cdr_summary.empty else False
    if d_gate == "PASS":
        route = "exact_detector_attribution"
        exit_code = 0
        wording = "Exact detector-source attribution route is allowed by Track D engineering gate."
    elif (
        d_status == "unavailable"
        and area_pass
        and cdr_evaluable
        and "required_g02135_geotiff_missing" in str(detector_summary.loc[0].get("unavailable_reason", ""))
    ):
        route = "cdr_based_spatial_decomposition"
        exit_code = 0
        wording = "CDR-based spatial loss decomposition associated with Sea Ice Index-detected pan-Arctic VRILEs"
    else:
        route = "detector_source_diagnose_first"
        exit_code = 2
        wording = "Exact detector-source closure must be diagnosed first; do not claim exact contribution to G02135 delta SIE."
    return {
        "route": route,
        "exit_code": exit_code,
        "scientific_wording": wording,
        "detector_source_closure": d_status,
        "detector_source_route": str(detector_summary.loc[0, "detector_source_route"]),
        "detector_source_gate_reason": str(detector_summary.loc[0, "gate_reason"]),
        "cell_area_alignment_pass": area_pass,
        "cdr_concordance_status": str(cdr_summary.loc[0, "concordance_status"]) if not cdr_summary.empty else "unavailable",
        "cdr_event_coverage_fraction": float(cdr_summary.loc[0, "event_coverage_fraction"]) if not cdr_summary.empty else 0.0,
    }


def run(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_stale_closure_outputs()
    contract = build_method_contract()
    write_json(OUT_DIR / "stage3_method_contract.json", contract)

    schema = input_schema_audit()
    schema.to_csv(OUT_DIR / "input_schema_audit.csv", index=False)

    accounting = surface_mask_accounting_audit()
    accounting.to_csv(OUT_DIR / "surface_mask_accounting_audit.csv", index=False)

    area_grid, area_audit = load_cell_area_grid()
    area_audit.to_csv(OUT_DIR / "cell_area_alignment_audit.csv", index=False)

    strategy = footprint_strategy_audit()
    write_json(OUT_DIR / "footprint_strategy_audit.json", strategy)

    cdr_audit, cdr_summary = cdr_concordance_audit(area_grid)
    cdr_audit.to_csv(OUT_DIR / "panarctic_cdr_detector_concordance_audit.csv", index=False)
    cdr_summary.to_csv(OUT_DIR / "panarctic_cdr_detector_concordance_summary.csv", index=False)

    unique = pd.read_csv(PAN_UNIQUE, parse_dates=["date"])
    inventory = g02135_required_inventory(unique)
    inventory.to_csv(OUT_DIR / "g02135_required_file_inventory.csv", index=False)

    area_path = find_cell_area_file()
    if area_grid is not None and area_path is not None and not inventory.empty and inventory["status"].eq("ok").any():
        first_path = Path(ROOT / inventory.loc[inventory["status"].eq("ok"), "selected_file"].iloc[0])
        geotiff_audit = geotiff_grid_audit(first_path, area_path, area_grid)
    else:
        geotiff_audit = pd.DataFrame(
            [
                {
                    "grid_alignment_pass": False,
                    "alignment_status": "not_evaluable",
                    "notes": "GeoTIFF grid audit requires a valid cell-area grid and at least one available G02135 GeoTIFF.",
                }
            ]
        )
    geotiff_audit.to_csv(OUT_DIR / "g02135_geotiff_grid_audit.csv", index=False)
    if not bool(geotiff_audit["grid_alignment_pass"].fillna(False).astype(bool).any()):
        raise ValueError("G02135 GeoTIFF grid failed alignment against NSIDC-0771 v1.1 cell-area grid.")

    detector_audit, detector_summary = detector_source_closure(unique, area_grid, inventory)
    detector_audit.to_csv(OUT_DIR / "panarctic_detector_source_closure_audit.csv", index=False)
    detector_summary.to_csv(OUT_DIR / "panarctic_detector_source_closure_summary.csv", index=False)

    decision = route_decision(detector_summary, cdr_summary, area_audit)
    write_json(OUT_DIR / "stage3_route_decision.json", decision)

    validate_no_forbidden_batch_outputs()

    gate = str(decision["route"])
    print(f"WROTE {rel(OUT_DIR)}")
    print(f"STAGE3_ROUTE_DECISION {gate}")
    print(cdr_summary.to_string(index=False))
    print(detector_summary.to_string(index=False))
    return int(decision["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir', default=str(OUT_DIR), help="Reserved for compatibility; Stage 3 Batch 0-1 writes the canonical output directory.")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
