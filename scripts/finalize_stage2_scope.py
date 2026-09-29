#!/usr/bin/env python3
"""Finalize Stage-2 scope completion after both corrected floor015 runs pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("VRILE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PIPE = ROOT / '.'
DEFINITIONS = PIPE / 'method_static'
REGION_MECHANISMS = DEFINITIONS / "METHOD_STATIC_DEFINITION_STAGE2_REGION_MECHANISMS.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().eq("true")
    return series.astype(bool)


def copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise RuntimeError(f"copy hash mismatch: {source} -> {target}")


def compare_required_runs(audit: Path) -> pd.DataFrame:
    rows = []
    required = [
        "event_clusters/synoptic_event_clusters_severe.csv",
        "event_clusters/synoptic_event_clusters_major.csv",
        "atmospheric_forcing/cyclone_proximity/cyclone_proximity_tests.csv",
        "atmospheric_forcing/cyclone_proximity/cyclone_proximity_evidence_status.csv",
        "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_samples.csv",
        "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_qa.csv",
        "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_tests.csv",
        "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_evidence_status.csv",
        "scope_completion/corrected_stage2_raw_evidence_full_770.csv",
        "scope_completion/corrected_stage2_region_evidence_summary.csv",
        "scope_completion/C010_current_status_summary.csv",
    ]
    for relative in required:
        first, second = audit / "run1" / relative, audit / "run2" / relative
        first_sha, second_sha = sha256(first), sha256(second)
        rows.append({
            "artifact": relative,
            "run1_sha256": first_sha,
            "run2_sha256": second_sha,
            "byte_identical": first_sha == second_sha,
        })
    for relative in ["floor015_stage2_raw_evidence.csv", "floor015_stage2_region_summary.csv"]:
        first, second = audit / "floor015_run1" / relative, audit / "floor015_run2" / relative
        first_sha, second_sha = sha256(first), sha256(second)
        rows.append({
            "artifact": f"floor015/{relative}",
            "run1_sha256": first_sha,
            "run2_sha256": second_sha,
            "byte_identical": first_sha == second_sha,
        })
    result = pd.DataFrame(rows)
    if not result.byte_identical.all():
        raise RuntimeError("HOLD_REPRODUCIBILITY: required outputs differ")
    return result


def supported_location(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        (pd.to_numeric(frame[f"{prefix}_n_units"], errors="coerce") >= 3)
        & (pd.to_numeric(frame[f"{prefix}_p_FDR"], errors="coerce") < 0.05)
        & as_bool(frame[f"{prefix}_ci_excludes_zero"])
        & (pd.to_numeric(frame[f"{prefix}_standardized_mean_difference_zero"], errors="coerce").abs() >= 0.2)
    )


def stable_specificity(floor_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    location = floor_raw[floor_raw.diagnostic_type.eq("location_buffer")].copy()
    location["primary_location_supported"] = supported_location(location, "primary")
    location["floor015_location_supported"] = supported_location(location, "floor015")
    wrong = floor_raw[floor_raw.diagnostic_type.eq("wrong_region")].copy()
    wrong_summary = (
        wrong.groupby(["source_region", "buffer_km", "variable"], as_index=False)
        .agg(
            primary_wrong_4of4=("primary_control_passed", lambda x: bool(as_bool(x).all()) and len(x) == 4),
            floor015_wrong_4of4=("floor015_control_passed", lambda x: bool(as_bool(x).all()) and len(x) == 4),
            comparison_count=("wrong_region", "size"),
        )
        .rename(columns={"source_region": "region"})
    )
    year = floor_raw[floor_raw.diagnostic_type.eq("year_block")].copy()
    for prefix in ("primary", "floor015"):
        lo = pd.to_numeric(year[f"{prefix}_ci_low"], errors="coerce")
        hi = pd.to_numeric(year[f"{prefix}_ci_high"], errors="coerce")
        year[f"{prefix}_year_confirmed"] = (
            year[f"{prefix}_block_bootstrap_status"].astype(str).eq("ok_event_level_year_resampling")
            & (lo * hi > 0)
        )
    year = year[["region", "buffer_km", "variable", "primary_year_confirmed", "floor015_year_confirmed"]]
    checks = location.merge(wrong_summary, on=["region", "buffer_km", "variable"], validate="one_to_one")
    checks = checks.merge(year, on=["region", "buffer_km", "variable"], validate="one_to_one")
    checks["stable_spatial_specificity"] = checks[
        [
            "primary_location_supported", "floor015_location_supported",
            "primary_wrong_4of4", "floor015_wrong_4of4",
            "primary_year_confirmed", "floor015_year_confirmed",
        ]
    ].all(axis=1)
    checks["atmospheric_diagnostic"] = checks.variable.astype(str).str.startswith("era5_")
    atmospheric = checks[checks.atmospheric_diagnostic]
    region_variable = atmospheric.groupby(["region", "variable"], as_index=False).stable_spatial_specificity.any()
    counts = region_variable.groupby("variable").stable_spatial_specificity.sum()
    maximum = int(counts.max()) if len(counts) else 0
    return checks, maximum


def one_status(floor_raw: pd.DataFrame, region: str, variable: str) -> dict[str, object]:
    location = floor_raw[
        floor_raw.diagnostic_type.eq("location_buffer")
        & floor_raw.region.eq(region)
        & floor_raw.variable.eq(variable)
    ].copy()
    location["primary_supported"] = supported_location(location, "primary")
    location["floor015_supported"] = supported_location(location, "floor015")
    lead = floor_raw[
        floor_raw.diagnostic_type.eq("lead_lag")
        & floor_raw.region.eq(region)
        & floor_raw.variable.eq(variable)
    ]
    year = floor_raw[
        floor_raw.diagnostic_type.eq("year_block")
        & floor_raw.region.eq(region)
        & floor_raw.variable.eq(variable)
    ].copy()
    for prefix in ("primary", "floor015"):
        lo = pd.to_numeric(year[f"{prefix}_ci_low"], errors="coerce")
        hi = pd.to_numeric(year[f"{prefix}_ci_high"], errors="coerce")
        year[f"{prefix}_supported"] = lo * hi > 0
    return {
        "primary_location_buffers": location.loc[location.primary_supported, "buffer_km"].astype(int).tolist(),
        "floor015_location_buffers": location.loc[location.floor015_supported, "buffer_km"].astype(int).tolist(),
        "primary_lead_lag_groups": lead.loc[as_bool(lead.primary_significant_background_test), "lag_group"].tolist(),
        "floor015_lead_lag_groups": lead.loc[as_bool(lead.floor015_significant_background_test), "lag_group"].tolist(),
        "primary_year_buffers": year.loc[year.primary_supported, "buffer_km"].astype(int).tolist(),
        "floor015_year_buffers": year.loc[year.floor015_supported, "buffer_km"].astype(int).tolist(),
    }


def write_claim_outputs(audit: Path, floor_raw: pd.DataFrame, floor_summary: pd.DataFrame) -> dict[str, object]:
    checks, maximum = stable_specificity(floor_raw)
    checks.to_csv(audit / "stable_spatial_specificity_claim_gate.csv", index=False)
    method_definitions = pd.read_csv(REGION_MECHANISMS)
    if len(method_definitions) != 40:
        raise RuntimeError(f"Stage-2 method-definition rows={len(method_definitions)}, expected 40")
    corrected_summary = pd.read_csv(audit / "corrected_stage2_region_evidence_summary.csv")
    corrected_robust = int(corrected_summary.manuscript_status.eq("SUPPORTED_WITH_ROBUSTNESS").sum())
    headline = "PRESERVED" if maximum < 5 else "HOLD_MAIN_STAGE2_CLAIM_REVIEW"
    zero_statement = "REVISE" if corrected_robust > 0 else "MAY_REMAIN"
    beaufort = one_status(floor_raw, "beaufort_sea", "era5_wspd10_buffer")
    laptev = one_status(floor_raw, "laptev_sea", "era5_v10_buffer")
    triage = pd.DataFrame([
        {
            "claim_or_example": "major median cumulative_loss",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": "CURRENT_RUN_VALUE_NOT_COMPUTED_BY_THIS_FINALIZER",
            "manuscript_use_check": "not directly cited in current manuscript/RESULTS.md",
            "proposed_disposition": "NUMERIC_UPDATE_ONLY",
            "notes": "duration/hierarchy direction unchanged; author signoff required before prose update",
        },
        {
            "claim_or_example": "major median max_area_km2",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": "CURRENT_RUN_VALUE_NOT_COMPUTED_BY_THIS_FINALIZER",
            "manuscript_use_check": "not directly cited in current manuscript/RESULTS.md",
            "proposed_disposition": "NUMERIC_UPDATE_ONLY",
            "notes": "direction unchanged; author signoff required before prose update",
        },
        {
            "claim_or_example": "Central Arctic MSL 500 km",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": "CURRENT_RUN_VALUE_NOT_COMPUTED_BY_THIS_FINALIZER",
            "manuscript_use_check": "not directly cited in current manuscript/RESULTS.md",
            "proposed_disposition": "NO_CLAIM_IMPACT",
            "notes": "unsupported in both universes; sign flip retained in evidence",
        },
        {
            "claim_or_example": "Results 3.6 Beaufort wspd10 example",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": json.dumps(beaufort, sort_keys=True),
            "manuscript_use_check": "directly cited in Results 3.6 and Fig.6",
            "proposed_disposition": "REVISE_OR_RETIRE_EXAMPLE",
            "notes": "old example must not remain unchanged",
        },
        {
            "claim_or_example": "Results 3.6 Laptev v10 example",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": json.dumps(laptev, sort_keys=True),
            "manuscript_use_check": "directly cited in Results 3.6 and Fig.6",
            "proposed_disposition": "PENDING_AUTHOR_DECISION",
            "notes": "candidate preserved example; corrected floor015 evidence included",
        },
        {
            "claim_or_example": "Stage 2 heterogeneous/non-universal headline",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": f"max stable regions for one atmospheric diagnostic={maximum}",
            "manuscript_use_check": "headline claim",
            "proposed_disposition": "NO_CLAIM_IMPACT" if headline == "PRESERVED" else "SCIENTIFIC_CLAIM_CHANGED",
            "notes": f"predefined headline gate={headline}",
        },
        {
            "claim_or_example": "zero SUPPORTED_WITH_ROBUSTNESS row statement",
            "frozen_evidence": "POST_GENERATION_VALIDATION_REFERENCE_NOT_READ",
            "corrected_evidence": f"count={corrected_robust}",
            "manuscript_use_check": "stronger supporting statement",
            "proposed_disposition": "CLAIM_WORDING_REVISION_REQUIRED" if zero_statement == "REVISE" else "NO_CLAIM_IMPACT",
            "notes": f"zero-row statement status={zero_statement}; broader headline assessed separately",
        },
    ])
    triage.to_csv(audit / "stage2_claim_impact_triage.csv", index=False)
    review_lines = [
        "# Stage 2 author claim review sheet — 2026-08-20",
        "",
        "This sheet records evidence for human signoff. `not_supported` does not mean that a mechanism is absent.",
        "",
        "Allowed author decisions: `APPROVE_UNCHANGED`, `APPROVE_NUMERIC_UPDATE`, `APPROVE_REVISED_WORDING`, `REMOVE_FROM_MAIN_RESULTS`, `HOLD_FOR_FURTHER_ANALYSIS`.",
        "",
        "| Claim / Results subsection | Frozen meaning | Corrected evidence | Primary/floor015 status | Proposed disposition | Author decision | Author notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in triage.to_dict("records"):
        review_lines.append(
            f"| {row['claim_or_example']} | {row['frozen_evidence']} | {str(row['corrected_evidence']).replace('|','/')} | "
            f"{str(row['notes']).replace('|','/')} | {row['proposed_disposition']} |  |  |"
        )
    (audit / "STAGE2_AUTHOR_CLAIM_REVIEW_SHEET_20260820.md").write_text("\n".join(review_lines) + "\n")
    handoff = f"""# Stage 3 post-claim-review handoff — 2026-08-20

No Stage 3 analysis was executed in this scope-completion run.

## VERIFIED_INVARIANT

- Fig. 2 pan-Arctic spatial composition: carried forward from the accepted corrected-catalog Stage 2 rebuild invariant check.

## REBUILD_REQUIRED

- All Stage 3 products depending directly or indirectly on corrected unique-event membership.
- Broad/severe/major partition-dependent products.
- Track D / Track S aligned membership, local contribution, and overlap products.
- Matched controls and reverse major-event analysis.

## Claim-review gate

- Main Stage 2 headline gate: {headline}.
- Beaufort wspd10 example: REVISE_OR_RETIRE_EXAMPLE.
- Laptev v10 example: PENDING_AUTHOR_DECISION.
- Stage 3 execution remains blocked until author claim signoff.
"""
    (audit / "STAGE3_POST_CLAIM_REVIEW_HANDOFF_20260820.md").write_text(handoff)
    return {
        "max_stable_regions": maximum,
        "headline_status": headline,
        "historical_reference_used_during_generation": False,
        "corrected_supported_with_robustness": corrected_robust,
        "zero_row_statement_status": zero_statement,
        "beaufort": beaufort,
        "laptev": laptev,
        "triage_counts": triage.proposed_disposition.value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit_dir', required=True)
    args = parser.parse_args()
    audit = Path(args.audit_dir)
    reproducibility = compare_required_runs(audit)
    reproducibility.to_csv(audit / "stage2_scope_clean_rerun_comparison.csv", index=False)
    source_scope = audit / "run1/scope_completion"
    for name in [
        "corrected_stage2_raw_evidence_full_770.csv",
        "corrected_stage2_region_evidence_summary.csv",
        "C010_current_status_summary.csv",
        "stage2_scope_completeness_audit.csv",
        "scope_completion_validation.json",
    ]:
        copy_exact(source_scope / name, audit / name)
    copy_exact(
        audit / "floor015_run1/floor015_stage2_region_summary.csv",
        audit / "stage2_floor015_corrected_robustness.csv",
    )
    floor_raw = pd.read_csv(audit / "floor015_run1/floor015_stage2_raw_evidence.csv")
    floor_summary = pd.read_csv(audit / "floor015_run1/floor015_stage2_region_summary.csv")
    if len(floor_raw) != 770:
        raise RuntimeError(f"floor015 formal row count={len(floor_raw)}, expected 770")
    claim = write_claim_outputs(audit, floor_raw, floor_summary)
    severe = pd.read_csv(audit / "run1/event_clusters/synoptic_event_clusters_severe.csv")
    major = pd.read_csv(audit / "run1/event_clusters/synoptic_event_clusters_major.csv")
    cluster = pd.DataFrame([
        {"catalog": "severe", "corrected_cluster_count": len(severe), "run1_sha256": sha256(audit / "run1/event_clusters/synoptic_event_clusters_severe.csv"), "run2_sha256": sha256(audit / "run2/event_clusters/synoptic_event_clusters_severe.csv"), "member_set_QA": "PASS_BYTE_IDENTICAL"},
        {"catalog": "major", "corrected_cluster_count": len(major), "run1_sha256": sha256(audit / "run1/event_clusters/synoptic_event_clusters_major.csv"), "run2_sha256": sha256(audit / "run2/event_clusters/synoptic_event_clusters_major.csv"), "member_set_QA": "PASS_BYTE_IDENTICAL"},
    ])
    cluster.to_csv(audit / "corrected_event_cluster_regression.csv", index=False)
    c010 = pd.read_csv(audit / "C010_current_status_summary.csv")
    c010_status = (
        "CURRENT_RUN_ALL_REGIONS_NOT_SUPPORTED"
        if c010.current_regional_status.astype(str).eq("not_supported").all()
        else "HOLD_C010_CLAIM_REVIEW"
    )
    final = {
        "scope_status": "PASS",
        "C010_STATUS": c010_status,
        "floor015_status": "PASS",
        "reproducibility_status": "PASS",
        "historical_reference_used_during_generation": False,
        "claim_review_state": "READY_FOR_AUTHOR_CLAIM_SIGNOFF" if claim["headline_status"] == "PRESERVED" else "HOLD_MAIN_STAGE2_CLAIM_REVIEW",
        "claim": claim,
    }
    (audit / "final_decision_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
