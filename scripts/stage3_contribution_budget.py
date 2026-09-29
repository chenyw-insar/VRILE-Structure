#!/usr/bin/env python
"""Build Batch 3 dual-track local contribution budgets."""

from __future__ import annotations

import sys
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.budget import (
    build_class_partitions,
    cdr_contribution_budget,
    detector_contribution_budget,
    region_budget_accounting_audit,
    region_budgets,
)
from vrile.stage3.grid import LOCAL_MAJOR, LOCAL_SEVERE, LOCAL_UNIQUE, OUT_DIR, ROOT, load_pan_events


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _validation_root() -> Path | None:
    value = os.environ.get("STAGE3_VALIDATION_ROOT", "").strip()
    return ROOT / value if value and not Path(value).is_absolute() else (Path(value) if value else None)


def _check_region_gap_regression(accounting: pd.DataFrame) -> None:
    root = _validation_root()
    if root is None:
        return
    baseline = root / "detector_region_gap_baseline.csv"
    if not baseline.exists():
        return
    old = pd.read_csv(baseline)
    cmp = old.merge(accounting, on="pan_event_id", how="inner")
    amount_ok = np.allclose(cmp["baseline_unassigned_loss_km2"], cmp["detector_unassigned_loss_km2"], rtol=1e-12, atol=1e-6)
    fraction_ok = np.allclose(cmp["baseline_unassigned_fraction"], cmp["detector_unassigned_loss_fraction"], rtol=1e-12, atol=1e-12, equal_nan=True)
    cmp["amount_regression_pass"] = np.isclose(cmp["baseline_unassigned_loss_km2"], cmp["detector_unassigned_loss_km2"], rtol=1e-12, atol=1e-6)
    cmp["fraction_regression_pass"] = np.isclose(cmp["baseline_unassigned_fraction"], cmp["detector_unassigned_loss_fraction"], rtol=1e-12, atol=1e-12, equal_nan=True)
    cmp.to_csv(root / "detector_region_gap_regression_audit.csv", index=False)
    if not amount_ok or not fraction_ok or len(cmp) != 99:
        raise SystemExit("Detector region-gap regression audit failed")


def _array_sha256_int8(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int8).tobytes()).hexdigest()


def _check_class_code_regression(partition_dir: Path, pan: pd.DataFrame) -> None:
    root = _validation_root()
    if root is None:
        return
    baseline = root / "class_code_regression_baseline.csv"
    if not baseline.exists():
        return
    rows = []
    for pan_id in pan["pan_event_id"].astype(str):
        with xr.open_dataset(partition_dir / f"{pan_id}.nc") as ds:
            for track, var in (("detector", "detector_local_class_code"), ("cdr", "cdr_local_class_code")):
                rows.append(
                    {
                        "pan_event_id": pan_id,
                        "track": track,
                        "class_code_sha256_new": _array_sha256_int8(np.asarray(ds[var].values, dtype=np.int8)),
                    }
                )
    new = pd.DataFrame(rows)
    cmp = pd.read_csv(baseline).merge(new, on=["pan_event_id", "track"], how="outer")
    cmp["class_code_regression_pass"] = cmp["class_code_sha256"].astype(str).eq(cmp["class_code_sha256_new"].astype(str))
    cmp.to_csv(root / "class_code_regression_audit.csv", index=False)
    if len(cmp) != 198 or not cmp["class_code_regression_pass"].all():
        raise SystemExit("Class-code regression audit failed")


def _check_contribution_regression(det_budget: pd.DataFrame, cdr_budget: pd.DataFrame) -> None:
    root = _validation_root()
    if root is None:
        return
    checks = [
        ("detector", det_budget, root / "panarctic_detector_contribution_budget_baseline.csv"),
        ("cdr", cdr_budget, root / "panarctic_cdr_contribution_budget_baseline.csv"),
    ]
    rows = []
    for track, current, baseline in checks:
        if not baseline.exists():
            continue
        old = pd.read_csv(baseline)
        key = ["pan_event_id"]
        merged = old.merge(current, on=key, suffixes=("_old", "_new"), how="outer")
        numeric_cols = [c for c in current.columns if c not in {"pan_event_id", "event_date"} and pd.api.types.is_numeric_dtype(current[c])]
        for _, row in merged.iterrows():
            event_ok = True
            max_abs = 0.0
            for col in numeric_cols:
                old_v = float(row[f"{col}_old"])
                new_v = float(row[f"{col}_new"])
                ok = bool(np.isclose(old_v, new_v, rtol=1e-12, atol=1e-6, equal_nan=True))
                event_ok = event_ok and ok
                if np.isfinite(old_v) and np.isfinite(new_v):
                    max_abs = max(max_abs, abs(old_v - new_v))
            rows.append(
                {
                    "track": track,
                    "pan_event_id": row["pan_event_id"],
                    "contribution_regression_pass": event_ok,
                    "max_abs_numeric_delta": max_abs,
                }
            )
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(root / "contribution_regression_audit.csv", index=False)
        if not out["contribution_regression_pass"].all():
            raise SystemExit("Contribution budget regression audit failed")


def main() -> int:
    pan = load_pan_events()
    cells_dir = ROOT / "data/processed/local_event_cells"
    patch_index = pd.read_csv(cells_dir / "patch_cell_index.csv")
    membership = pd.read_csv(cells_dir / "unique_event_patch_membership.csv")
    unique = pd.read_csv(LOCAL_UNIQUE)
    severe = read_optional(LOCAL_SEVERE)
    major = read_optional(LOCAL_MAJOR)
    detector_dir = ROOT / "data/processed/panarctic_detector_extent_fields"
    cdr_dir = ROOT / "data/processed/panarctic_loss_fields"
    transaction_data_root = os.environ.get("VRILE_STAGE3_DATA_ROOT", "").strip()
    partition_dir = (
        Path(transaction_data_root).resolve() / "panarctic_class_partition_fields"
        if transaction_data_root
        else ROOT / "data/processed/panarctic_class_partition_fields"
    )

    align, audit = build_class_partitions(
        pan,
        membership,
        patch_index,
        unique,
        severe,
        major,
        cells_dir,
        detector_dir,
        partition_dir,
        OUT_DIR,
    )
    det_budget = detector_contribution_budget(pan, detector_dir, partition_dir, OUT_DIR)
    cdr_budget = cdr_contribution_budget(pan, cdr_dir, partition_dir, OUT_DIR)
    det_region, cdr_region = region_budgets(pan, detector_dir, cdr_dir, OUT_DIR)
    accounting = region_budget_accounting_audit(det_budget, cdr_budget, det_region, cdr_region, OUT_DIR)
    _check_class_code_regression(partition_dir, pan)
    _check_contribution_regression(det_budget, cdr_budget)
    _check_region_gap_regression(accounting)

    if len(align) != 99 or len(audit) != 198:
        raise SystemExit("Batch 3 alignment/partition audit row count gate failed")
    if not audit["status"].eq("PASS").all():
        raise SystemExit("Batch 3 class partition audit contains FAIL rows")
    if not det_budget["detector_partition_closure_status"].eq("PASS").all():
        raise SystemExit("Track D contribution budget closure failed")
    if not cdr_budget["cdr_partition_closure_status"].eq("PASS").all():
        raise SystemExit("Track S contribution budget closure failed")
    if len(det_region) != 99 * 19 or len(cdr_region) != 99 * 19:
        raise SystemExit("Region budget must contain 19 accounting rows per pan-Arctic event")

    print(f"WROTE {OUT_DIR}")
    print(
        f"alignment_rows={len(align)} partition_audit_rows={len(audit)} "
        f"detector_budget_rows={len(det_budget)} cdr_budget_rows={len(cdr_budget)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
