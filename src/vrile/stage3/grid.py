"""Grid, raster, and metadata helpers for Stage 3."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pyproj import CRS, Geod, Transformer
from pyproj.enums import WktVersion
from vrile.io import _add_projected_lat_lon
from vrile.regions import NSIDC_0780_REGION_IDS
from vrile.regions import parse_region
from vrile.stage3.frozen import frozen_output_root

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = Path(os.environ.get("VRILE_RAW_DATA_ROOT", ROOT / "data/raw")).expanduser().resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("VRILE_PROCESSED_DATA_ROOT", ROOT / "data/processed")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("VRILE_OUTPUT_ROOT", ROOT / "outputs")).expanduser().resolve()
OUT_DIR = frozen_output_root(ROOT)
MASK_PATH = RAW_DATA_ROOT / "nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
CELL_AREA_PATH = RAW_DATA_ROOT / "nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc"
G02135_ROOT = RAW_DATA_ROOT / "nsidc_g02135_geotiff/north/daily/geotiff"
SIC_ROOT = RAW_DATA_ROOT / "nsidc_sic"
PAN_UNIQUE = OUTPUT_ROOT / "reproduce_sie/vrile_events_unique_both_jja_5p.csv"
LOCAL_FILTERED = OUTPUT_ROOT / "local_vrile_enhanced/local_objects_filtered.csv"
LOCAL_UNIQUE = OUTPUT_ROOT / "local_vrile_enhanced/unique_local_events.csv"
LOCAL_SEVERE = OUTPUT_ROOT / "local_vrile_severe/severe_unique_local_events.csv"
LOCAL_MAJOR = OUTPUT_ROOT / "local_vrile_major_severe/major_severe_events_union.csv"

OCEAN_CODES = set(range(0, 19))
EXTENT_THRESHOLD = 0.15
REGION_BY_CODE = {0: "non_region_ocean", **{v: k for k, v in NSIDC_0780_REGION_IDS.items()}}

DOCUMENTED_G02202_NORTH_CRS = {
    "projection_method_contains": "Polar Stereographic",
    "latitude_of_projection_origin": 90.0,
    "standard_parallel": 70.0,
    "central_meridian": -45.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "semi_major_axis_m": 6378273.0,
    "semi_minor_axis_m": 6356889.449,
    "shape_y": 448,
    "shape_x": 304,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pan_events() -> pd.DataFrame:
    df = pd.read_csv(PAN_UNIQUE, parse_dates=["date"])
    if "pan_event_id" not in df.columns:
        df = df.copy()
        df.insert(0, "pan_event_id", [f"PAN{i+1:06d}" for i in range(len(df))])
    return df


def load_cell_area_km2() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(CELL_AREA_PATH) as ds:
        arr = np.asarray(ds["cell_area"].values, dtype=float) / 1_000_000.0
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
    return arr, x, y


def load_surface_mask() -> np.ndarray:
    with xr.open_dataset(MASK_PATH, mask_and_scale=False) as ds:
        var = "sea_ice_region_surface_mask"
        return np.asarray(ds[var].values, dtype=np.int16)


def valid_ocean_mask() -> np.ndarray:
    return np.isin(load_surface_mask(), list(OCEAN_CODES))


def parse_geotiff_version(path: Path) -> tuple[int, ...]:
    m = re.search(r"_v([0-9.]+)\.tif$", path.name)
    return tuple(int(part) for part in m.group(1).split(".") if part.isdigit()) if m else (0,)


def find_g02135_geotiff(date: pd.Timestamp) -> Path:
    pattern = G02135_ROOT / f"{date.year}" / f"{date:%m_%b}" / f"N_{date:%Y%m%d}_extent_v*.tif"
    matches = [Path(p) for p in glob.glob(str(pattern))]
    if not matches:
        raise FileNotFoundError(f"Missing G02135 GeoTIFF for {date:%Y-%m-%d}")
    return max(matches, key=parse_geotiff_version)


def geotiff_tags(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        tags = dict(im.tag_v2.items())
        size = im.size
        mode = im.mode
        n_frames = int(getattr(im, "n_frames", 1))
    scale = tuple(float(v) for v in tags.get(33550, (np.nan, np.nan, np.nan)))
    tie = tuple(float(v) for v in tags.get(33922, (0.0, 0.0, 0.0, np.nan, np.nan, np.nan)))
    geokeys = tuple(int(v) for v in tags.get(34735, ()))
    geoascii = str(tags.get(34737, ""))
    epsg = "EPSG:3411" if 3411 in geokeys or "NSIDC Sea Ice Polar Stereographic North" in geoascii else ""
    return {
        "width": int(size[0]),
        "height": int(size[1]),
        "mode": mode,
        "n_frames": n_frames,
        "pixel_scale_x": scale[0],
        "pixel_scale_y": scale[1],
        "tiepoint_x": tie[3],
        "tiepoint_y": tie[4],
        "epsg": epsg,
        "geo_key_directory": geokeys,
        "geo_ascii_params": geoascii,
    }


@lru_cache(maxsize=256)
def read_geotiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    info = geotiff_tags(path)
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2 or info["n_frames"] != 1:
        raise ValueError(f"Expected single-band GeoTIFF: {path}")
    return arr.astype(np.uint16), info


def geotiff_xy(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(int(info["width"]), dtype=float)
    rows = np.arange(int(info["height"]), dtype=float)
    x = float(info["tiepoint_x"]) + (cols + 0.5) * float(info["pixel_scale_x"])
    y = float(info["tiepoint_y"]) - (rows + 0.5) * float(info["pixel_scale_y"])
    return x, y


def grid_signature(info: dict[str, Any]) -> str:
    payload = {
        "width": info["width"],
        "height": info["height"],
        "pixel_scale_x": info["pixel_scale_x"],
        "pixel_scale_y": info["pixel_scale_y"],
        "tiepoint_x": info["tiepoint_x"],
        "tiepoint_y": info["tiepoint_y"],
        "epsg": info["epsg"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def required_detector_dates(events: pd.DataFrame) -> list[pd.Timestamp]:
    dates = set()
    for date in pd.to_datetime(events["date"]).dt.normalize():
        dates.add(date - pd.Timedelta(days=2))
        dates.add(date + pd.Timedelta(days=1))
    return sorted(dates)


def write_g02135_signature_audit(events: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for date in required_detector_dates(events):
        try:
            path = find_g02135_geotiff(date)
            arr, info = read_geotiff(path)
            status = "ok" if arr.ndim == 2 and info["n_frames"] == 1 and info["epsg"] == "EPSG:3411" else "invalid"
            sig = grid_signature(info)
        except Exception as exc:
            path = Path("")
            info = {}
            status = f"error:{exc}"
            sig = ""
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "file": rel(path) if str(path) else "",
                "width": info.get("width", np.nan),
                "height": info.get("height", np.nan),
                "pixel_scale_x": info.get("pixel_scale_x", np.nan),
                "pixel_scale_y": info.get("pixel_scale_y", np.nan),
                "tiepoint_x": info.get("tiepoint_x", np.nan),
                "tiepoint_y": info.get("tiepoint_y", np.nan),
                "epsg": info.get("epsg", ""),
                "grid_signature": sig,
                "status": status,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    valid = int(df["status"].eq("ok").sum())
    unique = int(df.loc[df["status"].eq("ok"), "grid_signature"].nunique())
    if valid != len(df) or unique != 1:
        raise ValueError(f"G02135 signature gate failed: valid={valid}/{len(df)}, unique_grid_signature_count={unique}")
    return df


def find_sic_file(date: pd.Timestamp) -> Path:
    matches = sorted(glob.glob(str(SIC_ROOT / f"{date.year}" / f"sic_psn25_{date:%Y%m%d}_*.nc")))
    if not matches:
        raise FileNotFoundError(f"No SIC file for {date:%Y-%m-%d}")
    return Path(matches[0])


def guess_sic_var(ds: xr.Dataset) -> str:
    for name in ("cdr_seaice_conc", "seaice_conc_cdr", "sic", "ice_conc"):
        if name in ds:
            return name
    raise ValueError(f"Cannot infer SIC variable from {list(ds.data_vars)}")


def _crs_signature(crs: CRS) -> str:
    text = crs.to_wkt(version=WktVersion.WKT2_2019, pretty=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _grid_signature(shape: tuple[int, int], dims: tuple[str, ...], x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"shape": shape, "dims": dims}, sort_keys=True).encode("utf-8"))
    for arr in (x, y):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def _parse_current_sic_crs(ds: xr.Dataset, da: xr.DataArray) -> dict[str, object]:
    grid_mapping = str(da.attrs.get("grid_mapping", "crs"))
    if grid_mapping not in ds:
        raise ValueError(f"Missing grid mapping variable {grid_mapping!r}")
    attrs = dict(ds[grid_mapping].attrs)
    wkt_keys = {"crs_wkt", "spatial_ref"}
    cf_attrs = {k: v for k, v in attrs.items() if k not in wkt_keys}
    cf_status = "UNPARSEABLE"
    wkt_status = "UNPARSEABLE"
    cf_crs = None
    wkt_crs = None
    try:
        cf_crs = CRS.from_cf(cf_attrs)
        cf_status = "OK"
    except Exception as exc:
        cf_status = f"UNPARSEABLE:{type(exc).__name__}"
    wkt_text = attrs.get("spatial_ref") or attrs.get("crs_wkt")
    if wkt_text:
        try:
            wkt_crs = CRS.from_user_input(wkt_text)
            wkt_status = "OK"
        except Exception as exc:
            wkt_status = f"UNPARSEABLE:{type(exc).__name__}"
    if cf_crs is None and wkt_crs is None:
        raise ValueError("No parseable CRS in current SIC file")
    if cf_crs is not None:
        authoritative = cf_crs
        source = "cf_grid_mapping_attrs"
    else:
        authoritative = wkt_crs
        source = "current_file_wkt_fallback"
    if cf_crs is not None and wkt_crs is not None:
        equals = bool(cf_crs.equals(wkt_crs))
        equals_axis = bool(cf_crs.equals(wkt_crs, ignore_axis_order=True))
        consistency = "CONSISTENT" if equals_axis else "REPRESENTATION_DIFFERENCE"
    elif cf_crs is not None:
        equals = False
        equals_axis = False
        consistency = "CF_ONLY"
    else:
        equals = False
        equals_axis = False
        consistency = "WKT_ONLY"
    return {
        "grid_mapping_var": grid_mapping,
        "cf_attrs": cf_attrs,
        "cf_crs": cf_crs,
        "wkt_crs": wkt_crs,
        "authoritative_crs": authoritative,
        "crs_parse_source": source,
        "cf_crs_parse_status": cf_status,
        "wkt_crs_parse_status": wkt_status,
        "cf_wkt_consistency_status": consistency,
        "crs_equals": equals,
        "crs_equals_ignore_axis_order": equals_axis,
    }


def _operation_values(crs: CRS | None, cf_attrs: dict[str, Any] | None = None) -> dict[str, object]:
    if crs is None:
        return {
            "projection_method": "",
            "latitude_of_projection_origin": np.nan,
            "standard_parallel": np.nan,
            "central_meridian": np.nan,
            "false_easting": np.nan,
            "false_northing": np.nan,
            "semi_major_axis_m": np.nan,
            "semi_minor_axis_m": np.nan,
            "inverse_flattening": np.nan,
        }
    cf = crs.to_cf()
    params = {p.name.lower(): float(p.value) for p in crs.coordinate_operation.params}
    standard = cf.get("standard_parallel", params.get("latitude of standard parallel", np.nan))
    if isinstance(standard, (list, tuple, np.ndarray)):
        standard = float(standard[0]) if len(standard) else np.nan
    central = cf.get("straight_vertical_longitude_from_pole", params.get("longitude of origin", np.nan))
    lat_origin = cf.get("latitude_of_projection_origin", np.nan)
    if not np.isfinite(float(lat_origin)) and cf_attrs is not None and "latitude_of_projection_origin" in cf_attrs:
        lat_origin = float(cf_attrs["latitude_of_projection_origin"])
    if not np.isfinite(float(lat_origin)) and "Polar Stereographic" in crs.coordinate_operation.method_name:
        lat_origin = 90.0
    return {
        "projection_method": crs.coordinate_operation.method_name,
        "latitude_of_projection_origin": float(lat_origin),
        "standard_parallel": float(standard),
        "central_meridian": float(central),
        "false_easting": float(cf.get("false_easting", params.get("false easting", np.nan))),
        "false_northing": float(cf.get("false_northing", params.get("false northing", np.nan))),
        "semi_major_axis_m": float(crs.ellipsoid.semi_major_metre),
        "semi_minor_axis_m": float(crs.ellipsoid.semi_minor_metre),
        "inverse_flattening": float(crs.ellipsoid.inverse_flattening),
    }


def _documented_contract(values: dict[str, object]) -> dict[str, object]:
    rows: dict[str, object] = {}
    method = str(values["projection_method"])
    rows["projection_method_match"] = DOCUMENTED_G02202_NORTH_CRS["projection_method_contains"] in method
    ok = bool(rows["projection_method_match"])
    for key in [
        "latitude_of_projection_origin",
        "standard_parallel",
        "central_meridian",
        "false_easting",
        "false_northing",
        "semi_major_axis_m",
        "semi_minor_axis_m",
    ]:
        diff = abs(float(values[key]) - float(DOCUMENTED_G02202_NORTH_CRS[key]))
        rows[f"{key}_documented_abs_diff"] = diff
        ok = ok and bool(np.isfinite(diff) and diff <= 1e-9)
    rows["documented_contract_status"] = "PASS" if ok else "FAIL"
    return rows


@lru_cache(maxsize=512)
def read_sic(date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(find_sic_file(date)) as ds:
        var = guess_sic_var(ds)
        da = ds[var]
        arr = np.asarray(da.isel(time=0).values if "time" in da.dims else da.values, dtype=float).squeeze()
        arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
    return arr, x, y


def sic_grid_crs_signature(date: pd.Timestamp) -> dict[str, object]:
    path = find_sic_file(date)
    try:
        with xr.open_dataset(path) as ds:
            var = guess_sic_var(ds)
            da = ds[var]
            x = np.asarray(ds["x"].values, dtype=float)
            y = np.asarray(ds["y"].values, dtype=float)
            shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
            dims = tuple(str(d) for d in da.dims if str(d) in {"y", "x"})
            crs_info = _parse_current_sic_crs(ds, da)
            ref = sic_crs()
            authoritative = crs_info["authoritative_crs"]
            return {
                "date": date.strftime("%Y-%m-%d"),
                "file": rel(path),
                "shape": f"{shape[0]}x{shape[1]}",
                "grid_signature": _grid_signature(shape, dims, x, y),
                "grid_mapping_var": crs_info["grid_mapping_var"],
                "crs_parse_source": crs_info["crs_parse_source"],
                "cf_crs_parse_status": crs_info["cf_crs_parse_status"],
                "wkt_crs_parse_status": crs_info["wkt_crs_parse_status"],
                "cf_wkt_consistency_status": crs_info["cf_wkt_consistency_status"],
                "crs_signature": _crs_signature(authoritative),
                "crs_equals_reference": bool(authoritative.equals(ref)),
                "crs_equals_reference_ignore_axis_order": bool(authoritative.equals(ref, ignore_axis_order=True)),
                "status": "ok",
            }
    except Exception as exc:
        return {
            "date": date.strftime("%Y-%m-%d"),
            "file": rel(path) if path.exists() else "",
            "shape": "",
            "grid_signature": "",
            "grid_mapping_var": "",
            "crs_parse_source": "",
            "cf_crs_parse_status": "",
            "wkt_crs_parse_status": "",
            "cf_wkt_consistency_status": "",
            "crs_signature": "",
            "crs_equals_reference": False,
            "crs_equals_reference_ignore_axis_order": False,
            "status": f"error:{exc}",
        }


def write_cdr_required_grid_crs_audit(events: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    dates = sorted(
        set(pd.to_datetime(events["date"]).dt.normalize())
        | set(pd.to_datetime(events["date"]).dt.normalize() - pd.Timedelta(days=5))
    )
    rows = [sic_grid_crs_signature(pd.Timestamp(d)) for d in dates]
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    diag = write_cdr_cf_wkt_crs_diagnostic(dates, OUT_DIR / "cdr_cf_wkt_crs_diagnostic.csv")
    geo = write_cdr_cf_wkt_geolocation_delta_summary(
        dates,
        OUT_DIR / "cdr_cf_wkt_geolocation_delta_summary.json",
    )
    valid = int(df["status"].eq("ok").sum())
    unique_grid = int(df["grid_signature"].replace("", np.nan).dropna().nunique())
    unique_crs = int(df["crs_signature"].replace("", np.nan).dropna().nunique())
    cf_contract = int(diag["cf_documented_contract_status"].eq("PASS").sum())
    wkt_contract = int(diag["wkt_documented_contract_status"].eq("PASS").sum())
    geo_status = str(geo.get("crs_final_gate", "FAIL"))
    if valid != len(df) or unique_grid != 1 or unique_crs != 1 or cf_contract != len(diag) or wkt_contract != len(diag) or geo_status != "PASS":
        raise ValueError(
            "CDR grid/CRS audit failed: "
            f"valid={valid}/{len(df)}, unique_grid={unique_grid}, unique_crs={unique_crs}, "
            f"cf_contract={cf_contract}/{len(diag)}, wkt_contract={wkt_contract}/{len(diag)}, "
            f"geo_status={geo_status}"
        )
    return df


def write_cdr_cf_wkt_crs_diagnostic(dates: list[pd.Timestamp], out_path: Path) -> pd.DataFrame:
    rows = []
    for date in dates:
        path = find_sic_file(pd.Timestamp(date))
        with xr.open_dataset(path) as ds:
            var = guess_sic_var(ds)
            da = ds[var]
            crs_info = _parse_current_sic_crs(ds, da)
            cf_crs = crs_info["cf_crs"]
            wkt_crs = crs_info["wkt_crs"]
            cf_vals = _operation_values(cf_crs, crs_info["cf_attrs"])
            wkt_vals = _operation_values(wkt_crs)
            cf_contract = _documented_contract(cf_vals)
            wkt_contract = _documented_contract(wkt_vals)
        row = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "file": rel(path),
            "cf_crs_signature": _crs_signature(cf_crs) if cf_crs is not None else "",
            "wkt_crs_signature": _crs_signature(wkt_crs) if wkt_crs is not None else "",
            "crs_equals": crs_info["crs_equals"],
            "crs_equals_ignore_axis_order": crs_info["crs_equals_ignore_axis_order"],
            "cf_documented_contract_status": cf_contract["documented_contract_status"],
            "wkt_documented_contract_status": wkt_contract["documented_contract_status"],
        }
        for prefix, vals in (("cf", cf_vals), ("wkt", wkt_vals)):
            row[f"{prefix}_projection_method"] = vals["projection_method"]
            row[f"{prefix}_latitude_of_projection_origin"] = vals["latitude_of_projection_origin"]
            row[f"{prefix}_standard_parallel"] = vals["standard_parallel"]
            row[f"{prefix}_central_meridian"] = vals["central_meridian"]
            row[f"{prefix}_false_easting"] = vals["false_easting"]
            row[f"{prefix}_false_northing"] = vals["false_northing"]
            row[f"{prefix}_semi_major_axis_m"] = vals["semi_major_axis_m"]
            row[f"{prefix}_semi_minor_axis_m"] = vals["semi_minor_axis_m"]
            row[f"{prefix}_inverse_flattening"] = vals["inverse_flattening"]
            contract = cf_contract if prefix == "cf" else wkt_contract
            for key, value in contract.items():
                if key != "documented_contract_status":
                    row[f"{prefix}_{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_cdr_cf_wkt_geolocation_delta_summary(dates: list[pd.Timestamp], out_path: Path) -> dict[str, object]:
    date = pd.Timestamp(dates[0])
    path = find_sic_file(date)
    with xr.open_dataset(path) as ds:
        var = guess_sic_var(ds)
        da = ds[var]
        crs_info = _parse_current_sic_crs(ds, da)
        cf_crs = crs_info["cf_crs"]
        wkt_crs = crs_info["wkt_crs"]
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
    if cf_crs is None or wkt_crs is None:
        raise ValueError("Both CF and WKT CRS must be parseable for geolocation diagnostic")
    xx, yy = np.meshgrid(x, y)
    cf_transformer = Transformer.from_crs(cf_crs, CRS.from_epsg(4326), always_xy=True)
    wkt_transformer = Transformer.from_crs(wkt_crs, CRS.from_epsg(4326), always_xy=True)
    lon_cf, lat_cf = cf_transformer.transform(xx, yy)
    lon_wkt, lat_wkt = wkt_transformer.transform(xx, yy)
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon_cf, lat_cf, lon_wkt, lat_wkt)
    finite = np.isfinite(dist_m)
    max_idx = np.unravel_index(int(np.nanargmax(np.where(finite, dist_m, np.nan))), dist_m.shape)
    max_delta = float(np.nanmax(dist_m))
    summary = {
        "grid_cell_count": int(dist_m.size),
        "finite_comparison_count": int(finite.sum()),
        "max_geodesic_delta_m": max_delta,
        "p99_geodesic_delta_m": float(np.nanpercentile(dist_m, 99)),
        "median_geodesic_delta_m": float(np.nanmedian(dist_m)),
        "max_abs_lon_delta_deg": float(np.nanmax(np.abs(lon_cf - lon_wkt))),
        "max_abs_lat_delta_deg": float(np.nanmax(np.abs(lat_cf - lat_wkt))),
        "max_delta_x": float(xx[max_idx]),
        "max_delta_y": float(yy[max_idx]),
        "max_delta_lon_cf": float(lon_cf[max_idx]),
        "max_delta_lat_cf": float(lat_cf[max_idx]),
        "max_delta_lon_wkt": float(lon_wkt[max_idx]),
        "max_delta_lat_wkt": float(lat_wkt[max_idx]),
        "crs_equals": bool(cf_crs.equals(wkt_crs)),
        "crs_equals_ignore_axis_order": bool(cf_crs.equals(wkt_crs, ignore_axis_order=True)),
        "crs_interpretation": "",
        "crs_final_gate": "FAIL",
    }
    if max_delta <= 1e-6 and summary["max_abs_lon_delta_deg"] <= 1e-12 and summary["max_abs_lat_delta_deg"] <= 1e-12:
        summary["crs_final_gate"] = "PASS"
        summary["crs_interpretation"] = (
            "CRS.equals false is metadata/representation-level disagreement, "
            "not detectable actual-grid geolocation disagreement"
        )
    else:
        summary["crs_final_gate"] = "HOLD"
        summary["crs_interpretation"] = "Non-zero actual-grid CF/WKT geolocation delta requires review"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@lru_cache(maxsize=512)
def read_sic_with_lonlat(date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(find_sic_file(date)) as raw:
        ds = _add_projected_lat_lon(raw)
        var = guess_sic_var(ds)
        da = ds[var]
        arr = np.asarray(da.isel(time=0).values if "time" in da.dims else da.values, dtype=float).squeeze()
        arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        lat = np.asarray(ds["lat"].values, dtype=float)
    return arr, x, y, lon, lat


@lru_cache(maxsize=1)
def pan_arctic_stage1_mask() -> np.ndarray:
    _, _, _, lon2, lat2 = read_sic_with_lonlat(pd.Timestamp("1989-06-30"))
    return parse_region("pan_arctic").contains(lon2, lat2)


@lru_cache(maxsize=1)
def sic_crs() -> CRS:
    with xr.open_dataset(find_sic_file(pd.Timestamp("1989-06-30"))) as ds:
        attrs = ds["crs"].attrs
        crs_text = attrs.get("spatial_ref") or attrs.get("crs_wkt") or attrs.get("proj4text")
        if crs_text:
            return CRS.from_user_input(crs_text)
        return CRS.from_cf(attrs)


def xy_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(sic_crs(), CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return np.asarray(lon), np.asarray(lat)


def xy_points_to_lonlat(x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(sic_crs(), CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)
