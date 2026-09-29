#!/usr/bin/env python
"""Batch 3 preflight gates for pan-Arctic/local contribution budgets."""

from __future__ import annotations

import json
import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.grid import OUT_DIR, ROOT, load_cell_area_km2, load_pan_events, sha256_file, write_cdr_required_grid_crs_audit


STRICT_RTOL = 1e-12
STRICT_ATOL = 1e-6


def validation_root() -> Path:
    value = os.environ.get("STAGE3_VALIDATION_ROOT", "").strip()
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    path = OUT_DIR / "validation_runs" / "manual_unspecified"
    return path


def _array_sha256_int32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<i4").tobytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def capture_validation_baselines() -> Path:
    root = validation_root()
    root.mkdir(parents=True, exist_ok=True)
    events = load_pan_events()
    field_dir = ROOT / "data/processed/panarctic_loss_fields"
    hash_rows = []
    for pan_id in events["pan_event_id"].astype(str):
        with xr.open_dataset(field_dir / f"{pan_id}.nc") as ds:
            hash_rows.append(
                {
                    "pan_event_id": pan_id,
                    "sha256": _array_sha256_int32(np.asarray(ds["loss_component_id"].values, dtype=np.int32)),
                }
            )
    pd.DataFrame(hash_rows).to_csv(root / "component_membership_baseline.csv", index=False)

    summary = pd.read_csv(OUT_DIR / "panarctic_loss_field_event_summary.csv")
    numeric_cols = [
        "pan_event_id",
        "resolved_sic_loss_km2eq",
        "cdr_gross_sic_loss_km2eq",
        "cdr_gross_sic_gain_km2eq",
        "cdr_signed_sic_change_km2eq",
    ]
    summary[numeric_cols].to_csv(root / "cdr_numeric_baseline.csv", index=False)

    det_budget = pd.read_csv(OUT_DIR / "panarctic_detector_contribution_budget.csv")
    det_region = pd.read_csv(OUT_DIR / "panarctic_detector_region_budget.csv")
    global_col = "detector_gross_extent_loss_km2" if "detector_gross_extent_loss_km2" in det_budget.columns else "gross_loss"
    region_sum = (
        det_region[det_region["region_code"].between(0, 18)]
        .groupby("pan_event_id")["detector_gross_extent_loss_km2"]
        .sum()
        .rename("baseline_codes_0_18_loss_km2")
        .reset_index()
    )
    gap = det_budget[["pan_event_id", global_col]].rename(columns={global_col: "baseline_global_loss_km2"}).merge(region_sum, on="pan_event_id", how="left")
    gap["baseline_unassigned_loss_km2"] = gap["baseline_global_loss_km2"] - gap["baseline_codes_0_18_loss_km2"]
    gap["baseline_unassigned_fraction"] = np.where(
        gap["baseline_global_loss_km2"] > 0,
        gap["baseline_unassigned_loss_km2"] / gap["baseline_global_loss_km2"],
        np.nan,
    )
    gap.to_csv(root / "detector_region_gap_baseline.csv", index=False)
    print(f"CAPTURED validation baselines in {root}")
    return root


def cdr_component_raster_table_reconciliation() -> pd.DataFrame:
    area_km2, _, _ = load_cell_area_km2()
    comp_table = pd.read_csv(OUT_DIR / "panarctic_loss_components.csv")
    summary = pd.read_csv(OUT_DIR / "panarctic_loss_field_event_summary.csv")
    comp_lookup = comp_table.set_index(["pan_event_id", "component_id"])
    event_updates = []
    rows = []
    for pan_id in sorted(summary["pan_event_id"].astype(str)):
        with xr.open_dataset(ROOT / "data/processed/panarctic_loss_fields" / f"{pan_id}.nc") as ds:
            loss = np.asarray(ds["cdr_sic_loss"].values, dtype=float)
            labels = np.asarray(ds["loss_component_id"].values, dtype=np.int32)
        event_resolved = float(np.nansum(np.where(labels > 0, loss * area_km2, 0.0)))
        event_table = float(comp_table.loc[comp_table["pan_event_id"].astype(str) == pan_id, "integrated_sic_loss_km2eq"].sum())
        event_error = event_resolved - event_table
        event_status = "PASS" if np.isclose(event_error, 0.0, rtol=STRICT_RTOL, atol=STRICT_ATOL) else "FAIL"
        total_loss = float(summary.loc[summary["pan_event_id"].astype(str) == pan_id, "cdr_gross_sic_loss_km2eq"].iloc[0])
        event_updates.append(
            {
                "pan_event_id": pan_id,
                "resolved_sic_loss_km2eq": event_resolved,
                "component_loss_closure_error_km2eq": event_error,
                "component_loss_closure_status": event_status,
                "component_resolved_sic_loss_fraction": event_resolved / total_loss if total_loss > 0 else np.nan,
                "n_loss_components": int(len([v for v in np.unique(labels) if int(v) > 0])),
            }
        )
        for cid in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            mask = labels == cid
            table = comp_lookup.loc[(pan_id, cid)]
            raster_cell_count = int(mask.sum())
            raster_area = float(np.nansum(np.where(mask, area_km2, 0.0)))
            raster_loss = float(np.nansum(np.where(mask, loss * area_km2, 0.0)))
            cell_ok = raster_cell_count == int(table["cell_count"])
            area_ok = np.isclose(raster_area, float(table["area_km2"]), rtol=STRICT_RTOL, atol=STRICT_ATOL)
            loss_ok = np.isclose(raster_loss, float(table["integrated_sic_loss_km2eq"]), rtol=STRICT_RTOL, atol=STRICT_ATOL)
            rows.append(
                {
                    "pan_event_id": pan_id,
                    "component_id": cid,
                    "reported_cell_count": int(table["cell_count"]),
                    "raster_cell_count": raster_cell_count,
                    "reported_area_km2": float(table["area_km2"]),
                    "raster_area_km2": raster_area,
                    "reported_integrated_sic_loss_km2eq": float(table["integrated_sic_loss_km2eq"]),
                    "raster_integrated_sic_loss_km2eq": raster_loss,
                    "cell_count_status": "PASS" if cell_ok else "FAIL",
                    "area_status": "PASS" if area_ok else "FAIL",
                    "loss_status": "PASS" if loss_ok else "FAIL",
                    "component_status": "PASS" if cell_ok and area_ok and loss_ok else "FAIL",
                    "event_resolved_loss_from_raster_km2eq": event_resolved,
                    "event_resolved_loss_from_table_km2eq": event_table,
                    "event_resolved_loss_closure_error_km2eq": event_error,
                    "event_resolved_loss_closure_status": event_status,
                }
            )
    rec = pd.DataFrame(rows)
    rec.to_csv(OUT_DIR / "cdr_component_raster_table_reconciliation.csv", index=False)

    updates = pd.DataFrame(event_updates).set_index("pan_event_id")
    out = summary.copy()
    for col in [
        "resolved_sic_loss_km2eq",
        "component_loss_closure_error_km2eq",
        "component_loss_closure_status",
        "component_resolved_sic_loss_fraction",
        "n_loss_components",
    ]:
        out[col] = out["pan_event_id"].map(updates[col])
    out.to_csv(OUT_DIR / "panarctic_loss_field_event_summary.csv", index=False)
    baseline = validation_root() / "cdr_numeric_baseline.csv"
    if baseline.exists():
        old = pd.read_csv(baseline)
        new = out[
            [
                "pan_event_id",
                "resolved_sic_loss_km2eq",
                "cdr_gross_sic_loss_km2eq",
                "cdr_gross_sic_gain_km2eq",
                "cdr_signed_sic_change_km2eq",
            ]
        ]
        delta = old.merge(new, on="pan_event_id", suffixes=("_old", "_new"))
        delta_rows = []
        for _, row in delta.iterrows():
            delta_rec = {"pan_event_id": row["pan_event_id"]}
            for col in [
                "resolved_sic_loss_km2eq",
                "cdr_gross_sic_loss_km2eq",
                "cdr_gross_sic_gain_km2eq",
                "cdr_signed_sic_change_km2eq",
            ]:
                old_v = float(row[f"{col}_old"])
                new_v = float(row[f"{col}_new"])
                delta_rec[f"{col}_old"] = old_v
                delta_rec[f"{col}_new"] = new_v
                delta_rec[f"{col}_abs_delta"] = abs(new_v - old_v)
            delta_rows.append(delta_rec)
        delta_out = pd.DataFrame(delta_rows)
        expected = set(summary["pan_event_id"].astype(str))
        observed = set(delta_out["pan_event_id"].astype(str))
        delta_numeric = delta_out.drop(columns=["pan_event_id"]).apply(pd.to_numeric, errors="coerce")
        if len(delta_out) != 99 or observed != expected or not np.isfinite(delta_numeric.to_numpy(dtype=float)).all():
            raise SystemExit("CDR serialization numeric-delta audit must contain 99 exact events and finite numeric values")
        delta_out.to_csv(validation_root() / "cdr_serialization_numeric_delta_audit.csv", index=False)
    if not rec["component_status"].eq("PASS").all():
        raise SystemExit("CDR component raster/table reconciliation has FAIL component rows")
    if not rec.groupby("pan_event_id")["event_resolved_loss_closure_status"].first().eq("PASS").all():
        raise SystemExit("CDR event resolved-loss raster/table closure has FAIL rows")
    return rec


def enforce_detector_source_gate() -> str:
    route = _read_json(OUT_DIR / "stage3_route_decision.json").get("route", "unknown")
    if route != "exact_detector_attribution":
        return route
    summary = pd.read_csv(OUT_DIR / "panarctic_detector_source_closure_summary.csv")
    audit = pd.read_csv(OUT_DIR / "panarctic_detector_source_closure_audit.csv")
    events = load_pan_events()
    expected = set(events["pan_event_id"].astype(str))
    observed = list(audit["pan_event_id"].astype(str))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise SystemExit("D1a detector-source closure audit event_id set does not match the 99 primary events")
    gates = summary["engineering_gate"].dropna().astype(str).unique()
    if len(gates) != 1:
        raise SystemExit(f"D1a detector-source closure summary must contain exactly one engineering gate, got {list(gates)}")
    gate = str(gates[0])
    if gate != "PASS":
        raise SystemExit(f"Exact detector attribution route requires D1a engineering gate PASS, got {gate}")
    return route


def refresh_provenance(route: str, component_count: int) -> None:
    method_path = OUT_DIR / "stage3_method_contract.json"
    method = _read_json(method_path)
    method.update(
        {
            "footprint_implementation_status": "verified_batch2r_reconciliation",
            "verified_patch_count": 35559,
            "dual_track_alignment_locked": True,
            "stage3_scope": "Batch 2R PASS plus current Batch 3R3 CRS diagnosis and provenance repair",
            "batch3_scope": "Batch 2R PASS plus current Batch 3R3 CRS diagnosis and provenance repair",
        }
    )
    _write_json(method_path, method)

    foot_path = OUT_DIR / "footprint_strategy_audit.json"
    foot = _read_json(foot_path)
    foot.update(
        {
            "implementation_status": "verified_batch2r_reconciliation",
            "verified_patch_count": 35559,
            "dual_track_alignment_locked": True,
            "stage3_scope": "Batch 2R PASS plus current Batch 3R3 CRS diagnosis and provenance repair",
        }
    )
    _write_json(foot_path, foot)

    batch3_prompt = ROOT / "VRILE_Stage3_Batch3R3_Codex_Prompt.md"
    meta = {
        "active_conda_env": __import__("os").environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "route": route,
        "plan_sha256": sha256_file(ROOT / 'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md'),
        "batch3_prompt_path": "VRILE_Stage3_Batch3R3_Codex_Prompt.md",
        "batch3_prompt_status": "found" if batch3_prompt.is_file() else "not_present_optional_context",
        "batch3_prompt_sha256": sha256_file(batch3_prompt) if batch3_prompt.is_file() else "",
        "validation_run_root": str(validation_root()),
        "component_count": int(component_count),
        "preflight_script_sha256": sha256_file(ROOT / "scripts/stage3_batch3_preflight.py"),
    }
    _write_json(OUT_DIR / "batch3_preflight_metadata.json", meta)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-baseline-only", action="store_true")
    args = p.parse_args(argv)
    if args.capture_baseline_only:
        capture_validation_baselines()
        return 0
    events = load_pan_events()
    cdr_audit = write_cdr_required_grid_crs_audit(events, OUT_DIR / "cdr_required_grid_crs_signature_audit.csv")
    route = enforce_detector_source_gate()
    rec = cdr_component_raster_table_reconciliation()
    refresh_provenance(route, len(rec))
    print(f"WROTE {OUT_DIR}")
    print(
        "cdr_required_dates="
        f"{len(cdr_audit)} component_rows={len(rec)} "
        f"component_pass={int(rec['component_status'].eq('PASS').sum())} "
        f"event_closure_pass={int(rec.groupby('pan_event_id')['event_resolved_loss_closure_status'].first().eq('PASS').sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
