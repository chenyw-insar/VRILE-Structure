"""Region definitions for VRILE analyses.

Primary named Arctic regions use the NSIDC-0780 northern hemisphere sea-ice
regional mask, specifically the 25 km NSIDC polar stereographic north NetCDF
with the new land mask overlaid (``sea_ice_region_surface_mask``).  The
built-in longitude/latitude boxes are retained only as approximate plotting
bounds and as explicit custom/legacy regions.  They should not be treated as
the final scientific region definitions.

Set ``VRILE_NSIDC_REGION_MASK`` to the local NSIDC-0780 PS-N25km NetCDF file,
or place ``NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc`` under one of the
default raw-data locations listed in :func:`_candidate_mask_paths`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable

import numpy as np


NSIDC_0780_MASK_FILENAME = "NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
NSIDC_0780_REGION_VARIABLE = "sea_ice_region_surface_mask"


@dataclass(frozen=True)
class RegionDefinition:
    """Display and provenance contract for potentially ambiguous region names."""

    internal_id: str
    display_label: str
    display_label_zh: str
    definition_type: str
    source: str
    code_or_bounds: str
    formal_description: str


REGION_DEFINITION_REGISTRY: dict[str, RegionDefinition] = {
    "central_arctic": RegionDefinition(
        internal_id="central_arctic",
        display_label="Central Arctic (NSIDC-0780 region 1)",
        display_label_zh="中央北极区（NSIDC-0780第1区）",
        definition_type="official_nsidc0780",
        source="NSIDC-0780 Sea Ice Index Arctic regional mask",
        code_or_bounds="region_code=1",
        formal_description="Official NSIDC-0780 named physical region, code 1.",
    ),
    "central_basin": RegionDefinition(
        internal_id="central_basin",
        display_label="Central Basin (custom 82–90°N)",
        display_label_zh="中央北极盆地（自定义82–90°N纬度带）",
        definition_type="custom_82_90N",
        source="VRILE custom/legacy latitude-bounded region",
        code_or_bounds="lon=-180..180;lat=82..90",
        formal_description="Custom 82–90°N latitude-bounded sensitivity region; not an NSIDC-0780 named region.",
    ),
    "central_arctic_basin": RegionDefinition(
        internal_id="central_arctic_basin",
        display_label="Central Basin (custom 82–90°N)",
        display_label_zh="中央北极盆地（自定义82–90°N纬度带）",
        definition_type="custom_82_90N",
        source="alias of VRILE custom/legacy central_basin",
        code_or_bounds="lon=-180..180;lat=82..90",
        formal_description="Alias of the custom 82–90°N sensitivity region; not an NSIDC-0780 named region.",
    ),
}


def region_display_label(internal_id: str, language: str = "en") -> str:
    """Return the locked display label for a registered region ID."""

    definition = REGION_DEFINITION_REGISTRY.get(internal_id)
    if definition is None:
        return internal_id
    if language == "en":
        return definition.display_label
    if language == "zh":
        return definition.display_label_zh
    raise ValueError("language must be 'en' or 'zh'")

# NSIDC north polar stereographic 25 km grid used by the Sea Ice Index.
_NSIDC_NORTH_SHAPE = (448, 304)
_NSIDC_NORTH_X0_M = -3850000.0
_NSIDC_NORTH_Y0_M = 5850000.0
_NSIDC_NORTH_DX_M = 25000.0


# Region IDs follow NSIDC-0780 Version 1, Northern Hemisphere, Table 3.
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


# Approximate bounds for plotting, labels, and legacy buffer sensitivity only.
# Scientific membership for official named regions is determined from the
# NSIDC mask IDs above, not from these boxes.
PLOTTING_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "sea_of_okhotsk": (135, 165, 45, 62),
    "sea_of_japan": (127, 142, 35, 52),
    "bohai_and_yellow_seas": (117, 126, 36, 41),
    "baltic_sea": (10, 32, 53, 66),
    "bering_sea": (160, -160, 55, 67),
    "gulf_of_alaska": (-160, -135, 55, 62),
    "hudson_bay": (-96, -74, 50, 65),
    "gulf_of_st_lawrence": (-70, -55, 45, 52),
    "baffin_and_labrador_seas": (-80, -45, 50, 80),
    "east_greenland_sea": (-45, 15, 60, 85),
    "barents_sea": (15, 60, 68, 82),
    "kara_sea": (55, 100, 68, 82),
    "laptev_sea": (100, 145, 70, 82),
    "east_siberian_sea": (145, 180, 68, 82),
    "chukchi_sea": (-180, -155, 66, 76),
    "beaufort_sea": (-155, -120, 68, 82),
    "canadian_archipelago": (-125, -75, 68, 82),
    "central_arctic": (-180, 180, 80, 90),
    "pan_arctic": (-180, 180, 30.98, 90),
}


@dataclass(frozen=True)
class Region:
    """Explicit custom longitude/latitude bounding box.

    This class is retained for custom regions and legacy plotting/buffer
    workflows.  Official NSIDC-0780 region membership is represented by
    :class:`NsidcSeaIceIndexRegion`.
    """

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    source: str = "custom_bbox"

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon360 = np.mod(lon, 360.0)
        if abs(self.lon_max - self.lon_min) >= 360.0:
            lon_mask = np.ones_like(lon360, dtype=bool)
        else:
            lo = self.lon_min % 360.0
            hi = self.lon_max % 360.0
            if lo <= hi:
                lon_mask = (lon360 >= lo) & (lon360 <= hi)
            else:
                lon_mask = (lon360 >= lo) | (lon360 <= hi)
        lat_mask = (lat >= self.lat_min) & (lat <= self.lat_max)
        return lon_mask & lat_mask


@dataclass(frozen=True)
class NsidcSeaIceIndexRegion:
    """Region backed by the NSIDC-0780 Arctic regional mask."""

    name: str
    region_ids: tuple[int, ...]
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    source: str = "nsidc_0780_sea_ice_region_surface_mask"

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        region_mask = np.isin(_load_nsidc_region_mask(), self.region_ids)
        lon_arr = np.asarray(lon)
        lat_arr = np.asarray(lat)
        if lon_arr.shape == region_mask.shape and lat_arr.shape == region_mask.shape:
            return region_mask.copy()
        return _sample_nsidc_mask(region_mask, lon_arr, lat_arr)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_mask_paths() -> list[Path]:
    out: list[Path] = []
    env_path = os.environ.get("VRILE_NSIDC_REGION_MASK")
    if env_path:
        out.append(Path(env_path).expanduser())
    root = _repo_root()
    for base in [
        root / "data/raw/nsidc",
        root / "data/raw/nsidc_region_masks",
        root / "data/raw/seaice_index",
        root / "data/raw",
        Path.cwd(),
    ]:
        out.append(base / NSIDC_0780_MASK_FILENAME)
    return out


def _find_nsidc_region_mask() -> Path:
    for path in _candidate_mask_paths():
        if path.exists():
            return path
    candidates = "\n".join(f"  - {p}" for p in _candidate_mask_paths())
    raise FileNotFoundError(
        "NSIDC-0780 Arctic region mask is required for official named regions.\n"
        "Download NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc from NSIDC and either set\n"
        "VRILE_NSIDC_REGION_MASK to its path or place it in one of:\n"
        f"{candidates}\n"
        "Use an explicit lon_min,lon_max,lat_min,lat_max string only for custom regions."
    )


@lru_cache(maxsize=1)
def _load_nsidc_region_mask() -> np.ndarray:
    path = _find_nsidc_region_mask()
    if path.suffix.lower() != ".nc":
        raise ValueError(
            "Official named regions now require the NSIDC-0780 PS-N25km NetCDF mask. "
            f"Got {path}. Use {NSIDC_0780_MASK_FILENAME} or set VRILE_NSIDC_REGION_MASK to that file."
        )
    try:
        import xarray as xr
    except ImportError:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError(
                "xarray or netCDF4 is required to read the NSIDC-0780 NetCDF region mask."
            ) from exc
        with Dataset(path) as ds:
            if NSIDC_0780_REGION_VARIABLE not in ds.variables:
                raise KeyError(f"{NSIDC_0780_REGION_VARIABLE!r} not found in {path}")
            arr = np.asarray(ds.variables[NSIDC_0780_REGION_VARIABLE][:])
    else:
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            if NSIDC_0780_REGION_VARIABLE not in ds:
                raise KeyError(f"{NSIDC_0780_REGION_VARIABLE!r} not found in {path}")
            arr = np.asarray(ds[NSIDC_0780_REGION_VARIABLE].values)
    arr = np.asarray(arr, dtype=np.int16)
    if arr.ndim == 1:
        if arr.size != _NSIDC_NORTH_SHAPE[0] * _NSIDC_NORTH_SHAPE[1]:
            raise ValueError(f"Unexpected NSIDC region mask size in {path}: {arr.size}")
        arr = arr.reshape(_NSIDC_NORTH_SHAPE)
    if arr.shape == (_NSIDC_NORTH_SHAPE[1], _NSIDC_NORTH_SHAPE[0]):
        arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D NSIDC region mask, got shape {arr.shape} from {path}")
    return arr


def _sample_nsidc_mask(mask: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = _ll_to_nsidc_north().transform(lon, lat)
    col = np.rint((x - _NSIDC_NORTH_X0_M) / _NSIDC_NORTH_DX_M).astype(int)
    row = np.rint((_NSIDC_NORTH_Y0_M - y) / _NSIDC_NORTH_DX_M).astype(int)
    inside = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (row >= 0)
        & (row < mask.shape[0])
        & (col >= 0)
        & (col < mask.shape[1])
    )
    out = np.zeros(np.broadcast(lon, lat).shape, dtype=bool)
    if np.any(inside):
        out[inside] = mask[row[inside], col[inside]]
    return out


@lru_cache(maxsize=1)
def _ll_to_nsidc_north():
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "pyproj is required to sample the NSIDC-0780 region mask on arbitrary lon/lat grids. "
            "Install pyproj in the active environment, or run on data already on the NSIDC 25 km grid."
        ) from exc
    return Transformer.from_crs("EPSG:4326", "EPSG:3411", always_xy=True)


def _nsidc_region(name: str, ids: Iterable[int]) -> NsidcSeaIceIndexRegion:
    bounds = PLOTTING_BOUNDS[name]
    return NsidcSeaIceIndexRegion(name, tuple(ids), *bounds)


def _custom_region(name: str, bounds: tuple[float, float, float, float]) -> Region:
    return Region(name, *bounds, source="custom_legacy_bbox")


PAN_ARCTIC_REGION_IDS = tuple(NSIDC_0780_REGION_IDS.values())

REGIONS: dict[str, Region | NsidcSeaIceIndexRegion] = {
    name: _nsidc_region(name, [region_id])
    for name, region_id in NSIDC_0780_REGION_IDS.items()
}

REGIONS.update(
    {
        "pan_arctic": _nsidc_region("pan_arctic", PAN_ARCTIC_REGION_IDS),
        "pan_arctic_reference": _nsidc_region("pan_arctic", PAN_ARCTIC_REGION_IDS),
        "arctic": _nsidc_region("pan_arctic", PAN_ARCTIC_REGION_IDS),
        # Common aliases for official NSIDC-0780 regions.
        "okhotsk": REGIONS["sea_of_okhotsk"],
        "bering": REGIONS["bering_sea"],
        "st_lawrence": REGIONS["gulf_of_st_lawrence"],
        "greenland": REGIONS["east_greenland_sea"],
        "greenland_sea": REGIONS["east_greenland_sea"],
        "east_greenland": REGIONS["east_greenland_sea"],
        "baffin_bay": REGIONS["baffin_and_labrador_seas"],
        "baffin_labrador": REGIONS["baffin_and_labrador_seas"],
        "baffin_labrador_seas": REGIONS["baffin_and_labrador_seas"],
        "japan": REGIONS["sea_of_japan"],
        "bohai_yellow": REGIONS["bohai_and_yellow_seas"],
        "baltic": REGIONS["baltic_sea"],
        "alaska": REGIONS["gulf_of_alaska"],
        "barents": REGIONS["barents_sea"],
        "kara": REGIONS["kara_sea"],
        "laptev": REGIONS["laptev_sea"],
        "east_siberian": REGIONS["east_siberian_sea"],
        "chukchi": REGIONS["chukchi_sea"],
        "beaufort": REGIONS["beaufort_sea"],
        # Custom/legacy boxes retained for sensitivity checks and legacy figures.
        "custom_barents_extended": _custom_region("custom_barents_extended", (0, 80, 65, 85)),
        "barents_extended": _custom_region("custom_barents_extended", (0, 80, 65, 85)),
        "barents_ext": _custom_region("custom_barents_extended", (0, 80, 65, 85)),
        "custom_arctic_atlantic": _custom_region("custom_arctic_atlantic", (-20, 100, 60, 85)),
        "arctic_atlantic": _custom_region("custom_arctic_atlantic", (-20, 100, 60, 85)),
        "custom_greenland_nordic_seas": _custom_region("custom_greenland_nordic_seas", (-45, 30, 60, 85)),
        "greenland_nordic": _custom_region("custom_greenland_nordic_seas", (-45, 30, 60, 85)),
        "greenland_nordic_seas": _custom_region("custom_greenland_nordic_seas", (-45, 30, 60, 85)),
        "nordic_seas": _custom_region("custom_greenland_nordic_seas", (-45, 30, 60, 85)),
        "custom_central_basin": _custom_region("custom_central_basin", (-180, 180, 82, 90)),
        "central_basin": _custom_region("custom_central_basin", (-180, 180, 82, 90)),
        "central_arctic_basin": _custom_region("custom_central_basin", (-180, 180, 82, 90)),
    }
)


def parse_region(value: str | None) -> Region | NsidcSeaIceIndexRegion:
    """Parse a NSIDC named region or an explicit custom bounding box.

    Named Sea Ice Index regions are mask-backed.  Strings of the form
    ``lon_min,lon_max,lat_min,lat_max`` create explicit custom regions.
    """

    if value is None or value == "":
        return REGIONS["pan_arctic"]
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    if key in REGIONS:
        return REGIONS[key]
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        known = ", ".join(sorted(REGIONS))
        raise ValueError(
            f"Unknown region '{value}'. Use one of: {known}, or lon_min,lon_max,lat_min,lat_max for a custom region."
        )
    lon_min, lon_max, lat_min, lat_max = map(float, parts)
    return Region("custom_bbox", lon_min, lon_max, lat_min, lat_max)


def list_regions() -> list[str]:
    """Return a sorted list of available region names."""

    return sorted(REGIONS)
