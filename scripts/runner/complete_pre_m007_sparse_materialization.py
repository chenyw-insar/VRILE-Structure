#!/usr/bin/env python3
"""Complete the sparse-cell transition after the known pre-M007 centroid gate.

This is a release-engineering adapter, not a scientific producer. The frozen
materializer writes sparse cells and its reconciliation before raising when
the legacy centroid check encounters rows requiring promoted M007 geometry.
Continuation is allowed only when every non-centroid quantity still satisfies
the frozen reconciliation rules. No sparse cell or status value is rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SRC = RELEASE_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vrile.stage3.footprints import restore_unique_membership


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-exit-status", type=int, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--canonical-unique", type=Path, required=True)
    parser.add_argument('--cells_dir', type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--membership-reconciliation", type=Path, required=True)
    parser.add_argument("--scope-report", type=Path, required=True)
    return parser.parse_args()


def close(left: pd.Series, right: pd.Series, *, rtol: float, atol: float) -> np.ndarray:
    return np.isclose(
        pd.to_numeric(left, errors="coerce").to_numpy(float),
        pd.to_numeric(right, errors="coerce").to_numpy(float),
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HOLD_PRE_M007_SPARSE_TRANSITION:{message}")


def validate_sparse_arrays(index: pd.DataFrame, cells_dir: Path) -> list[str]:
    referenced = sorted(set(index["npz_file"].astype(str)))
    actual = sorted(path.name for path in cells_dir.glob("*.npz"))
    require(actual == referenced, "NPZ_FILE_SET_MISMATCH")
    for name, group in index.groupby("npz_file", sort=True):
        with np.load(cells_dir / str(name), allow_pickle=False) as data:
            required = {"object_index", "y_idx", "x_idx", "delta_sic"}
            require(required == set(data.files), f"NPZ_ARRAY_SET:{name}")
            object_index = data["object_index"]
            y_idx = data["y_idx"]
            x_idx = data["x_idx"]
            delta_sic = data["delta_sic"]
            require(
                len(object_index) == len(y_idx) == len(x_idx) == len(delta_sic),
                f"NPZ_ARRAY_LENGTH:{name}",
            )
            for row in group.itertuples(index=False):
                start = int(row.start_offset)
                stop = start + int(row.cell_count)
                require(stop <= len(object_index), f"NPZ_OFFSET_RANGE:{row.object_id}")
                require(
                    bool(np.all(object_index[start:stop] == int(row.object_index))),
                    f"NPZ_OBJECT_INDEX:{row.object_id}",
                )
    return referenced


def main() -> int:
    args = parse_args()
    require(args.producer_exit_status in {0, 1}, f"UNEXPECTED_PRODUCER_EXIT:{args.producer_exit_status}")
    for path in (args.filtered, args.canonical_unique, args.reconciliation):
        require(path.is_file(), f"MISSING_FILE:{path}")
    require((args.cells_dir / "patch_cell_index.csv").is_file(), "MISSING_PATCH_INDEX")

    filtered = pd.read_csv(args.filtered, parse_dates=["date", "start_date"])
    canonical = pd.read_csv(args.canonical_unique, parse_dates=["event_start", "event_end"])
    index = pd.read_csv(args.cells_dir / "patch_cell_index.csv")
    recon = pd.read_csv(args.reconciliation)

    require(len(filtered) == len(index) == len(recon), "ROW_COVERAGE")
    require(filtered["object_id"].astype(str).is_unique, "FILTERED_OBJECT_ID_UNIQUENESS")
    require(index["object_id"].astype(str).is_unique, "INDEX_OBJECT_ID_UNIQUENESS")
    require(
        set(filtered["object_id"].astype(str)) == set(index["object_id"].astype(str)),
        "OBJECT_ID_SET",
    )
    require(
        set(filtered["object_id"].astype(str)) == set(recon["reported_object_id"].astype(str)),
        "RECONCILIATION_OBJECT_ID_SET",
    )
    require(
        bool((recon["reported_component_rank"] == recon["recomputed_component_rank"]).all()),
        "COMPONENT_RANK",
    )

    checks: dict[str, np.ndarray] = {
        "cell_count": recon["reported_cell_count"].to_numpy(int)
        == recon["recomputed_cell_count"].to_numpy(int),
        "mean_sic_change": close(
            recon["reported_mean_sic_change"], recon["recomputed_mean_sic_change"], rtol=1e-8, atol=5e-6
        ),
        "min_sic_change": close(
            recon["reported_min_sic_change"], recon["recomputed_min_sic_change"], rtol=1e-8, atol=5e-6
        ),
        "cumulative_sic_loss": close(
            recon["reported_cumulative_sic_loss"], recon["recomputed_cumulative_sic_loss"], rtol=1e-6, atol=1e-3
        ),
        "max_loss_lon": close(
            recon["reported_max_loss_lon"], recon["recomputed_max_loss_lon"], rtol=1e-8, atol=5e-6
        ),
        "max_loss_lat": close(
            recon["reported_max_loss_lat"], recon["recomputed_max_loss_lat"], rtol=1e-8, atol=5e-6
        ),
        "legacy_nominal_area_km2": close(
            recon["reported_legacy_nominal_area_km2"],
            recon["recomputed_legacy_nominal_area_km2"],
            rtol=0.0,
            atol=1e-9,
        ),
    }
    mismatch_counts = {name: int((~values).sum()) for name, values in checks.items()}
    require(all(value == 0 for value in mismatch_counts.values()), f"NON_CENTROID_MISMATCH:{mismatch_counts}")

    lon_equal = close(
        recon["reported_centroid_lon"], recon["recomputed_centroid_lon"], rtol=1e-8, atol=5e-6
    )
    lat_equal = close(
        recon["reported_centroid_lat"], recon["recomputed_centroid_lat"], rtol=1e-8, atol=5e-6
    )
    centroid_mismatch = ~(lon_equal & lat_equal)
    recorded_pass = recon["status"].astype(str).eq("PASS").to_numpy()
    require(bool(np.array_equal(recorded_pass, ~centroid_mismatch)), "STATUS_DECOMPOSITION")
    if args.producer_exit_status == 0:
        require(not bool(centroid_mismatch.any()), "ZERO_EXIT_WITH_CENTROID_MISMATCH")
    else:
        require(bool(centroid_mismatch.any()), "NONZERO_EXIT_WITHOUT_CENTROID_MISMATCH")

    npz_files = validate_sparse_arrays(index, args.cells_dir)
    membership, membership_rec = restore_unique_membership(
        filtered,
        canonical,
        args.cells_dir,
        args.membership_reconciliation,
    )
    require(bool(membership_rec["status"].eq("PASS").all()), "UNIQUE_MEMBERSHIP_RECONCILIATION")

    report = {
        "status": "PASS",
        "classification": "RELEASE_ENGINEERING_PRE_M007_TRANSITION_ADAPTER",
        "producer_exit_status": args.producer_exit_status,
        "filtered_rows": int(len(filtered)),
        "index_rows": int(len(index)),
        "reconciliation_rows": int(len(recon)),
        "npz_file_count": int(len(npz_files)),
        "non_centroid_mismatch_counts": mismatch_counts,
        "observed_centroid_longitude_mismatches": int((~lon_equal).sum()),
        "observed_centroid_latitude_mismatches": int((~lat_equal).sum()),
        "observed_centroid_any_mismatches": int(centroid_mismatch.sum()),
        "unique_membership_rows": int(len(membership)),
        "centroid_gate": "DEFERRED_TO_REQUIRED_M007",
        "reconciliation_status_rewritten": False,
        "previous_derived_result_read": False,
        "validation_reference_read": False,
    }
    args.scope_report.parent.mkdir(parents=True, exist_ok=True)
    args.scope_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PRE_M007_SPARSE_CELL_MATERIALIZATION=PASS")
    print(f"PRE_M007_CENTROID_OBSERVED={report['observed_centroid_longitude_mismatches']}/{report['observed_centroid_latitude_mismatches']}")
    print("PRE_M007_NON_CENTROID_RECONCILIATION=PASS")
    print("FINAL_CENTROID_GATE=DEFERRED_TO_REQUIRED_M007")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
