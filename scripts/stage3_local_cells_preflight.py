#!/usr/bin/env python
"""Preflight the current Stage-1/2 local-event inputs and sparse cell artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.grid import (
    LOCAL_FILTERED,
    LOCAL_MAJOR,
    LOCAL_SEVERE,
    LOCAL_UNIQUE,
    ROOT,
    rel,
    sha256_file,
    sic_grid_crs_signature,
    write_cdr_cf_wkt_crs_diagnostic,
    write_cdr_cf_wkt_geolocation_delta_summary,
)

CELLS_DIR = ROOT / "data/processed/local_event_cells"
INDEX_PATH = CELLS_DIR / "patch_cell_index.csv"
MEMBERSHIP_PATH = CELLS_DIR / "unique_event_patch_membership.csv"
GRID_AUDIT_PATH = CELLS_DIR / "local_event_cell_grid_crs_signature_audit.csv"
VALIDATION_PATH = CELLS_DIR / "local_event_cell_validation.json"
PROVENANCE_PATH = CELLS_DIR / "local_event_cell_provenance.json"


def _fail(message: str) -> None:
    raise SystemExit(f"LOCAL_EVENT_CELL_PREFLIGHT_FAIL: {message}")


def _require_columns(frame: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        _fail(f"{path} missing columns={missing}")


def _read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [LOCAL_FILTERED, LOCAL_UNIQUE, LOCAL_SEVERE, LOCAL_MAJOR]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        _fail(f"missing current Stage-1/2 inputs={missing}")
    filtered = pd.read_csv(LOCAL_FILTERED, parse_dates=["date", "start_date"])
    unique = pd.read_csv(LOCAL_UNIQUE, parse_dates=["event_start", "event_end"])
    severe = pd.read_csv(LOCAL_SEVERE)
    major = pd.read_csv(LOCAL_MAJOR)
    _require_columns(filtered, LOCAL_FILTERED, {"object_id", "date", "start_date", "year", "window_days", "threshold", "object_area_cells"})
    _require_columns(unique, LOCAL_UNIQUE, {"unique_local_event_id", "event_start", "event_end"})
    _require_columns(severe, LOCAL_SEVERE, {"unique_local_event_id"})
    _require_columns(major, LOCAL_MAJOR, {"unique_local_event_id"})
    return filtered, unique, severe, major


def validate_inputs() -> dict[str, object]:
    filtered, unique, severe, major = _read_sources()
    if filtered.empty or unique.empty:
        _fail("current Stage-1/2 local-event inputs are empty")
    if filtered["object_id"].isna().any() or filtered["object_id"].astype(str).duplicated().any():
        _fail("filtered object_id has nulls or duplicates")
    if unique["unique_local_event_id"].isna().any() or unique["unique_local_event_id"].astype(str).duplicated().any():
        _fail("canonical unique_local_event_id has nulls or duplicates")
    broad_ids = set(unique["unique_local_event_id"].astype(str))
    severe_ids = set(severe["unique_local_event_id"].astype(str))
    major_ids = set(major["unique_local_event_id"].astype(str))
    if not major_ids <= severe_ids <= broad_ids:
        _fail("event hierarchy is not major_severe subset severe subset broad")
    if not filtered["date"].dt.month.isin([6, 7, 8]).all():
        _fail("filtered patch dates extend outside JJA")
    result = {
        "status": "PASS",
        "filtered_patch_rows": int(len(filtered)),
        "filtered_object_ids": int(filtered["object_id"].nunique()),
        "broad_event_ids": int(len(broad_ids)),
        "severe_event_ids": int(len(severe_ids)),
        "major_severe_event_ids": int(len(major_ids)),
        "patch_date_min": filtered["date"].min().strftime("%Y-%m-%d"),
        "patch_date_max": filtered["date"].max().strftime("%Y-%m-%d"),
        "source_date_min": min(filtered["date"].min(), filtered["start_date"].min()).strftime("%Y-%m-%d"),
        "source_date_max": max(filtered["date"].max(), filtered["start_date"].max()).strftime("%Y-%m-%d"),
    }
    print("LOCAL_EVENT_CELL_INPUT_PREFLIGHT " + json.dumps(result, sort_keys=True))
    return result


def _validate_npz(index: pd.DataFrame) -> dict[str, int]:
    total_cells = 0
    npz_files = sorted(index["npz_file"].astype(str).unique())
    declared = {str(CELLS_DIR / name) for name in npz_files}
    present = {str(path) for path in CELLS_DIR.glob("*.npz")}
    if declared != present:
        _fail(f"NPZ inventory mismatch missing={sorted(declared-present)} unexpected={sorted(present-declared)}")
    for npz_name, group in index.groupby("npz_file", sort=True):
        data = np.load(CELLS_DIR / str(npz_name))
        required = {"object_index", "y_idx", "x_idx", "delta_sic"}
        if set(data.files) != required:
            _fail(f"{npz_name} arrays={sorted(data.files)} expected={sorted(required)}")
        arrays = [data[name] for name in ("object_index", "y_idx", "x_idx", "delta_sic")]
        lengths = {len(arr) for arr in arrays}
        if len(lengths) != 1:
            _fail(f"{npz_name} array lengths differ")
        n_cells = len(arrays[0])
        ordered = group.sort_values("start_offset")
        expected_offsets = ordered["cell_count"].astype(np.int64).cumsum().shift(fill_value=0)
        if not np.array_equal(ordered["start_offset"].to_numpy(np.int64), expected_offsets.to_numpy(np.int64)):
            _fail(f"{npz_name} start_offset is not contiguous")
        if int(ordered["cell_count"].sum()) != n_cells:
            _fail(f"{npz_name} cell_count sum does not equal array length")
        object_index, y_idx, x_idx, delta = arrays
        if ((y_idx < 0) | (y_idx >= 448) | (x_idx < 0) | (x_idx >= 304)).any():
            _fail(f"{npz_name} contains out-of-grid cell coordinates")
        if not np.isfinite(delta).all():
            _fail(f"{npz_name} contains non-finite delta_sic")
        for row in ordered.itertuples(index=False):
            start = int(row.start_offset)
            stop = start + int(row.cell_count)
            if not np.all(object_index[start:stop] == int(row.object_index)):
                _fail(f"{npz_name} object_index slice mismatch for {row.object_id}")
            packed = y_idx[start:stop].astype(np.int64) * 304 + x_idx[start:stop].astype(np.int64)
            if np.unique(packed).size != packed.size:
                _fail(f"{npz_name} duplicate cells for {row.object_id}")
        total_cells += n_cells
    return {"npz_file_count": len(npz_files), "sparse_cell_rows": int(total_cells)}


def _grid_audit(filtered: pd.DataFrame) -> dict[str, object]:
    dates = sorted(set(filtered["date"].dt.normalize()) | set(filtered["start_date"].dt.normalize()))
    rows = [sic_grid_crs_signature(pd.Timestamp(date)) for date in dates]
    audit = pd.DataFrame(rows)
    audit.to_csv(GRID_AUDIT_PATH, index=False)
    valid = int(audit["status"].eq("ok").sum())
    unique_grid = int(audit.loc[audit["status"].eq("ok"), "grid_signature"].nunique())
    unique_crs = int(audit.loc[audit["status"].eq("ok"), "crs_signature"].nunique())
    reference_equal = bool(audit["crs_equals_reference_ignore_axis_order"].all())
    shape_ok = bool(audit["shape"].eq("448x304").all())
    representative_dates = [pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])]
    diagnostic = write_cdr_cf_wkt_crs_diagnostic(
        representative_dates,
        CELLS_DIR / "local_event_cell_cf_wkt_crs_diagnostic.csv",
    )
    geolocation = write_cdr_cf_wkt_geolocation_delta_summary(
        representative_dates,
        CELLS_DIR / "local_event_cell_cf_wkt_geolocation_delta_summary.json",
    )
    documented_ok = bool(
        diagnostic["cf_documented_contract_status"].eq("PASS").all()
        and diagnostic["wkt_documented_contract_status"].eq("PASS").all()
    )
    geolocation_ok = str(geolocation.get("crs_final_gate", "FAIL")) == "PASS"
    if valid != len(audit) or unique_grid != 1 or unique_crs != 1 or not shape_ok or not documented_ok or not geolocation_ok:
        _fail(
            f"grid/CRS gate valid={valid}/{len(audit)} unique_grid={unique_grid} "
            f"unique_crs={unique_crs} shape_ok={shape_ok} documented_ok={documented_ok} "
            f"geolocation_ok={geolocation_ok}"
        )
    return {
        "required_sic_dates": int(len(audit)),
        "grid_signature": str(audit["grid_signature"].iloc[0]),
        "crs_signature": str(audit["crs_signature"].iloc[0]),
        "grid_shape": str(audit["shape"].iloc[0]),
        "legacy_helper_crs_reference_equal_diagnostic": reference_equal,
        "documented_cf_wkt_contract": "PASS",
        "cf_wkt_geolocation_gate": "PASS",
    }


def _identity(path: Path, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if rows is not None:
        value["rows"] = rows
    return value


def validate_artifacts() -> dict[str, object]:
    inputs = validate_inputs()
    filtered, unique, severe, major = _read_sources()
    if not INDEX_PATH.is_file() or not MEMBERSHIP_PATH.is_file():
        _fail(f"missing producer outputs: {INDEX_PATH} or {MEMBERSHIP_PATH}")
    index = pd.read_csv(INDEX_PATH, parse_dates=["date", "start_date"])
    membership = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["date", "start_date"])
    _require_columns(index, INDEX_PATH, {"object_index", "object_id", "patch_key", "date", "start_date", "year", "npz_file", "start_offset", "cell_count"})
    _require_columns(membership, MEMBERSHIP_PATH, {"unique_local_event_id", "object_id", "patch_key", "date", "start_date"})
    duplicate_counts = {
        "index_object_index": int(index["object_index"].duplicated().sum()),
        "index_object_id": int(index["object_id"].astype(str).duplicated().sum()),
        "index_patch_key": int(index["patch_key"].astype(str).duplicated().sum()),
        "membership_object_id": int(membership["object_id"].astype(str).duplicated().sum()),
        "membership_patch_key": int(membership["patch_key"].astype(str).duplicated().sum()),
    }
    if any(duplicate_counts.values()):
        _fail(f"duplicate gate failed={duplicate_counts}")
    source_objects = set(filtered["object_id"].astype(str))
    if len(index) != len(filtered) or set(index["object_id"].astype(str)) != source_objects:
        _fail("patch index row count/object_id coverage differs from filtered patches")
    if len(membership) != len(filtered) or set(membership["object_id"].astype(str)) != source_objects:
        _fail("membership row count/object_id coverage differs from filtered patches")
    canonical_ids = set(unique["unique_local_event_id"].astype(str))
    membership_ids = set(membership["unique_local_event_id"].astype(str))
    if membership_ids != canonical_ids:
        _fail(f"membership event-ID coverage differs missing={len(canonical_ids-membership_ids)} extra={len(membership_ids-canonical_ids)}")
    severe_ids = set(severe["unique_local_event_id"].astype(str))
    major_ids = set(major["unique_local_event_id"].astype(str))
    if not major_ids <= severe_ids <= membership_ids:
        _fail("membership does not cover major_severe subset severe subset broad")
    for column in ("date", "start_date"):
        if not index[column].equals(filtered.sort_values(["date", "window_days", "threshold", "object_id"]).reset_index(drop=True)[column]):
            _fail(f"patch index {column} sequence differs from producer input order")
    npz_stats = _validate_npz(index)
    grid_stats = _grid_audit(filtered)
    result = {
        **inputs,
        **npz_stats,
        **grid_stats,
        "patch_index_rows": int(len(index)),
        "membership_rows": int(len(membership)),
        "membership_event_ids": int(len(membership_ids)),
        "duplicate_counts": duplicate_counts,
        "artifact_date_min": index["date"].min().strftime("%Y-%m-%d"),
        "artifact_date_max": index["date"].max().strftime("%Y-%m-%d"),
    }
    code_paths = [
        ROOT / "scripts/stage3_local_footprints.py",
        ROOT / "scripts/stage3_local_cells_preflight.py",
        ROOT / "src/vrile/local_objects.py",
        ROOT / "src/vrile/stage3/footprints.py",
        ROOT / "src/vrile/stage3/grid.py",
    ]
    source_rows = {LOCAL_FILTERED: len(filtered), LOCAL_UNIQUE: len(unique), LOCAL_SEVERE: len(severe), LOCAL_MAJOR: len(major)}
    output_paths = [
        INDEX_PATH,
        MEMBERSHIP_PATH,
        GRID_AUDIT_PATH,
        CELLS_DIR / "local_event_cell_cf_wkt_crs_diagnostic.csv",
        CELLS_DIR / "local_event_cell_cf_wkt_geolocation_delta_summary.json",
    ] + sorted(CELLS_DIR.glob("*.npz"))
    provenance = {
        "schema_version": "stage3-local-event-cells-provenance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "scripts/stage3_local_footprints.py",
        "inputs": [_identity(path, source_rows[path]) for path in source_rows],
        "code": [_identity(path) for path in code_paths],
        "outputs": [_identity(path, len(index) if path == INDEX_PATH else (len(membership) if path == MEMBERSHIP_PATH else None)) for path in output_paths],
        "contracts": result,
        "runtime": {"python": platform.python_version(), "executable": sys.executable},
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    result["provenance_sha256"] = sha256_file(PROVENANCE_PATH)
    VALIDATION_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("LOCAL_EVENT_CELL_ARTIFACT_PREFLIGHT " + json.dumps(result, sort_keys=True))
    print(f"WROTE {GRID_AUDIT_PATH}")
    print(f"WROTE {PROVENANCE_PATH}")
    print(f"WROTE {VALIDATION_PATH}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inputs", "artifacts"])
    args = parser.parse_args()
    validate_inputs() if args.mode == "inputs" else validate_artifacts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
