#!/usr/bin/env python
"""Compute Batch 3 spatial composition metrics."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vrile.stage3.grid import OUT_DIR, ROOT, load_pan_events, sha256_file
from vrile.stage3.metrics import spatial_composition_metrics


REQUIRED_METRIC_COLUMNS = [
    "pan_event_id",
    "event_date",
    "effective_component_number_sic",
    "largest_component_resolved_sic_loss_share",
    "top3_component_resolved_sic_loss_share",
    "n_loss_components",
    "top2_component_centroid_distance_km",
    "weighted_component_spatial_spread_km",
    "component_resolved_sic_loss_fraction",
    "resolved_sic_loss_km2eq",
    "effective_named_region_number_extent_loss",
    "largest_named_region_extent_loss_share",
    "top3_named_region_extent_loss_share",
    "non_region_ocean_extent_loss_fraction",
    "effective_named_region_number_sic_loss",
    "largest_named_region_sic_loss_share",
    "top3_named_region_sic_loss_share",
    "non_region_ocean_sic_loss_fraction",
]


def _validation_root() -> Path | None:
    value = os.environ.get("STAGE3_VALIDATION_ROOT", "").strip()
    return ROOT / value if value and not Path(value).is_absolute() else (Path(value) if value else None)


def _count_pass(path: Path, col: str) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    return int(df[col].eq("PASS").sum()) if col in df.columns else 0


def _file_record(role: str, rel_path: str, source: str) -> dict | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "provenance_role": role,
        "relative_path": rel_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": h,
        "source_or_generator": source,
    }


def write_provenance_manifest(vroot: Path | None) -> None:
    if vroot is None:
        return
    records = []
    for rel_path in [
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/component_membership_baseline.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/cdr_numeric_baseline.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/detector_region_gap_baseline.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/class_code_regression_baseline.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/panarctic_detector_contribution_budget_baseline.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/panarctic_cdr_contribution_budget_baseline.csv",
        "outputs/panarctic_local_contribution/batch2r_run_metadata.json",
    ]:
        rec = _file_record("baseline_input", rel_path, "pre-existing verified baseline")
        if rec:
            records.append(rec)
    for rel_path in [
        "scripts/stage3_batch3_preflight.py",
        "scripts/stage3_contribution_budget.py",
        "scripts/stage3_spatial_composition_metrics.py",
        "src/vrile/stage3/grid.py",
        "src/vrile/stage3/budget.py",
    ]:
        rec = _file_record("code", rel_path, "current workspace")
        if rec:
            records.append(rec)
    for rel_path in [
        'method_static/provenance/AGENTS.md',
        'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md',
        "VRILE_Stage3_Batch3R3_Codex_Prompt.md",
    ]:
        rec = _file_record("context", rel_path, "current context or previous full-run log")
        if rec:
            records.append(rec)
    for log_path in sorted((ROOT / "logs").glob("stage3_panarctic_local_contribution_*.log"), reverse=True)[:2]:
        rel_path = str(log_path.relative_to(ROOT))
        rec = _file_record("context", rel_path, "Stage 3 full-run log")
        if rec:
            records.append(rec)
    for rel_path in [
        "outputs/panarctic_local_contribution/cdr_required_grid_crs_signature_audit.csv",
        "outputs/panarctic_local_contribution/cdr_cf_wkt_crs_diagnostic.csv",
        "outputs/panarctic_local_contribution/cdr_cf_wkt_geolocation_delta_summary.json",
        "outputs/panarctic_local_contribution/panarctic_detector_source_closure_audit.csv",
        "outputs/panarctic_local_contribution/panarctic_detector_source_closure_summary.csv",
        "outputs/panarctic_local_contribution/track_alignment_audit.csv",
        "outputs/panarctic_local_contribution/class_partition_overlap_audit.csv",
        "outputs/panarctic_local_contribution/stage3_method_contract.json",
        "outputs/panarctic_local_contribution/footprint_strategy_audit.json",
        "outputs/panarctic_local_contribution/batch3_run_metadata.json",
        "outputs/panarctic_local_contribution/batch3_preflight_metadata.json",
        "outputs/panarctic_local_contribution/panarctic_detector_contribution_budget.csv",
        "outputs/panarctic_local_contribution/panarctic_cdr_contribution_budget.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/class_code_regression_audit.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/contribution_regression_audit.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/detector_region_gap_regression_audit.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/cdr_serialization_numeric_delta_audit.csv",
        "outputs/panarctic_local_contribution/validation_runs/" + vroot.name + "/component_membership_hash_comparison.csv",
    ]:
        rec = _file_record("generated", rel_path, "current Stage 3 validation/output")
        if rec:
            records.append(rec)
    pd.DataFrame(records).to_csv(vroot / "provenance_manifest.csv", index=False)


def main() -> int:
    components = pd.read_csv(OUT_DIR / "panarctic_loss_components.csv")
    event_summary = pd.read_csv(OUT_DIR / "panarctic_loss_field_event_summary.csv")
    detector_region = pd.read_csv(OUT_DIR / "panarctic_detector_region_budget.csv")
    cdr_region = pd.read_csv(OUT_DIR / "panarctic_cdr_region_budget.csv")
    detector_budget = pd.read_csv(OUT_DIR / "panarctic_detector_contribution_budget.csv")
    cdr_budget = pd.read_csv(OUT_DIR / "panarctic_cdr_contribution_budget.csv")
    metrics = spatial_composition_metrics(components, event_summary, detector_region, cdr_region, detector_budget, cdr_budget)
    forbidden = {"largest_component_resolved_loss_share", "top3_component_resolved_loss_share"}
    if forbidden & set(metrics.columns):
        raise SystemExit("Spatial metrics contain forbidden legacy component share aliases")
    missing = [c for c in REQUIRED_METRIC_COLUMNS if c not in metrics.columns]
    if missing:
        raise SystemExit(f"Spatial metrics missing required canonical columns: {missing}")
    metrics = metrics[REQUIRED_METRIC_COLUMNS]
    metrics.to_csv(OUT_DIR / "panarctic_spatial_composition_metrics.csv", index=False)

    audit = pd.read_csv(OUT_DIR / "class_partition_overlap_audit.csv")
    rec = pd.read_csv(OUT_DIR / "cdr_component_raster_table_reconciliation.csv")
    route = json.loads((OUT_DIR / "stage3_route_decision.json").read_text(encoding="utf-8")).get("route", "unknown")
    vroot = _validation_root()
    cmp_count = 0
    if vroot is not None and (vroot / "component_membership_hash_comparison.csv").exists():
        cmp = pd.read_csv(vroot / "component_membership_hash_comparison.csv")
        cmp_count = int(cmp["hash_match"].sum())
    files_for_hash = [
        "scripts/stage3_batch3_preflight.py",
        "scripts/stage3_cdr_loss_fields.py",
        "scripts/stage3_contribution_budget.py",
        "scripts/stage3_spatial_composition_metrics.py",
    ]
    package_files = [
        "src/vrile/stage3/grid.py",
        "src/vrile/stage3/fields.py",
        "src/vrile/stage3/budget.py",
        "src/vrile/stage3/metrics.py",
    ]
    batch3_prompt = ROOT / "VRILE_Stage3_Batch3R3_Codex_Prompt.md"
    meta = {
        "active_conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "route": route,
        "event_count": int(len(load_pan_events())),
        "component_count": int(len(components)),
        "detector_partition_pass_count": int(((audit["track"] == "detector") & audit["status"].eq("PASS")).sum()),
        "cdr_partition_pass_count": int(((audit["track"] == "cdr") & audit["status"].eq("PASS")).sum()),
        "strict_component_pass_count": int(rec["component_status"].eq("PASS").sum()),
        "strict_event_resolved_loss_pass_count": int(rec.groupby("pan_event_id")["event_resolved_loss_closure_status"].first().eq("PASS").sum()),
        "component_membership_hash_match_count": cmp_count,
        "stage3_scope": "Batch 2R PASS plus current Batch 3R3 CRS diagnosis and provenance repair",
        "plan_sha256": sha256_file(ROOT / 'method_static/provenance/VRILE_Stage3_PanArctic_Local_Contribution_Experiment_Plan.md'),
        "batch3_prompt_path": "VRILE_Stage3_Batch3R3_Codex_Prompt.md",
        "batch3_prompt_status": "found" if batch3_prompt.is_file() else "not_present_optional_context",
        "batch3_prompt_sha256": sha256_file(batch3_prompt) if batch3_prompt.is_file() else "",
        "batch2r_metadata_sha256": sha256_file(OUT_DIR / "batch2r_run_metadata.json") if (OUT_DIR / "batch2r_run_metadata.json").exists() else "",
        "validation_run_root": str(vroot) if vroot is not None else "",
        "batch3_script_sha256s": {p: sha256_file(ROOT / p) for p in files_for_hash},
        "stage3_package_sha256s": {p: sha256_file(ROOT / p) for p in package_files},
    }
    (OUT_DIR / "batch3_run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    write_provenance_manifest(vroot)

    if len(metrics) != 99 or metrics["pan_event_id"].nunique() != 99:
        raise SystemExit("Spatial composition metrics must contain exactly 99 unique pan_event_id")
    print(f"WROTE {OUT_DIR / 'panarctic_spatial_composition_metrics.csv'}")
    print(f"events={len(metrics)} components={len(components)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
