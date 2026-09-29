#!/usr/bin/env python3
"""Complete corrected formal Stage-2 scope with the two C010 families.

Generation consumes method-static definitions and current-run products only.
Historical/frozen products belong exclusively to post-generation validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("VRILE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PIPE = ROOT / '.'
DEFINITIONS = PIPE / 'method_static'
EVIDENCE_ROWS = DEFINITIONS / "METHOD_STATIC_DEFINITION_STAGE2_EVIDENCE_ROWS.csv"
REGION_MECHANISMS = DEFINITIONS / "METHOD_STATIC_DEFINITION_STAGE2_REGION_MECHANISMS.csv"
EXPECTED_COUNTS = {
    "location_buffer": 90,
    "lead_lag": 90,
    "wrong_region_control": 360,
    "year_block_robustness": 90,
    "ice_edge_relative_wind": 120,
    "cyclone_proximity": 20,
}
RAW_OUTPUT_COLUMNS = [
    "evidence_id", "diagnostic_family", "analysis_tier", "region", "wrong_region",
    "mechanism", "variable", "buffer_km", "lag_group", "test_lags",
    "background_lags", "event_population", "statistical_unit", "n_units", "effect",
    "standardized_effect", "ci_low", "ci_high", "p_raw", "p_fdr",
    "supported_or_significant", "workflow_label", "bootstrap_resamples", "fdr_scope",
    "evaluated", "status", "reason", "source_file", "source_row_key",
]
REGION_SUMMARY_COLUMNS = [
    "region", "mechanism", "variable_definition", "event_population", "statistical_unit",
    "primary_effect_ci_p_fdr", "primary_supported", "pre_event_support",
    "event_time_support", "post_event_support", "wrong_region_specificity",
    "year_block_confirmation", "supplementary_support", "data_status",
    "manuscript_status", "manuscript_safe_statement",
    "between_region_contrast_status", "source_files",
]

sys.path.insert(0, str(PIPE / "scripts"))
import stage2_experiment_core as stage2  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def unique(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    found = frame[mask]
    if len(found) != 1:
        raise RuntimeError(f"{label}: expected one source row, got {len(found)}")
    return found.iloc[0]


def c010_tables(base: Path) -> dict[str, pd.DataFrame]:
    return {
        "cyclone": pd.read_csv(base / "atmospheric_forcing/cyclone_proximity/cyclone_proximity_tests.csv"),
        "cyclone_status": pd.read_csv(base / "atmospheric_forcing/cyclone_proximity/cyclone_proximity_evidence_status.csv"),
        "edge": pd.read_csv(base / "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_tests.csv"),
        "edge_status": pd.read_csv(base / "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_evidence_status.csv"),
    }


def map_c010(template: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = template.copy()
    for index, row in output.iterrows():
        region, variable = str(row.region), str(row.variable)
        if row.diagnostic_family == "cyclone_proximity":
            source = unique(
                tables["cyclone"],
                tables["cyclone"].region.astype(str).eq(region)
                & tables["cyclone"].variable.astype(str).eq(variable),
                str(row.evidence_id),
            )
            region_status = unique(
                tables["cyclone_status"],
                tables["cyclone_status"].region.astype(str).eq(region),
                f"{row.evidence_id}:status",
            )
            expected = stage2.CYCLONE_EXPECTED_DIRECTION.get(variable)
            support = bool(
                int(source.n_units) >= 3
                and float(source.p_FDR) < 0.05
                and as_bool(source.ci_excludes_zero)
                and expected is not None
                and stage2.difference_matches_direction(float(source.mean_difference), expected)
            )
            values = {
                "n_units": int(source.n_units),
                "effect": source.mean_difference,
                "standardized_effect": source.standardized_mean_difference_zero,
                "ci_low": source.ci_low,
                "ci_high": source.ci_high,
                "p_raw": source.p_raw,
                "p_fdr": source.p_FDR,
                "supported_or_significant": support,
                "workflow_label": f"region_evidence_status={region_status.evidence_status}",
                "status": region_status.evidence_status,
                "reason": region_status.reason,
                "evaluated": True,
            }
        elif row.diagnostic_family == "ice_edge_relative_wind":
            source = unique(
                tables["edge"],
                tables["edge"].analysis_tier.astype(str).eq(str(row.analysis_tier))
                & tables["edge"].region.astype(str).eq(region)
                & tables["edge"].variable.astype(str).eq(variable)
                & tables["edge"].lag_group.astype(str).eq(str(row.lag_group)),
                str(row.evidence_id),
            )
            region_status = unique(
                tables["edge_status"],
                tables["edge_status"].region.astype(str).eq(region),
                f"{row.evidence_id}:status",
            )
            support = str(source.mechanism_evidence_status) == "statistically_supported_dynamic_indicator"
            values = {
                "n_units": int(source.n_units),
                "effect": source.mean_difference,
                "standardized_effect": source.standardized_mean_difference_zero,
                "ci_low": source.ci_low,
                "ci_high": source.ci_high,
                "p_raw": source.p_raw,
                "p_fdr": source.p_FDR,
                "supported_or_significant": support,
                "workflow_label": source.mechanism_evidence_status,
                "status": region_status.evidence_status,
                "reason": region_status.reason,
                "evaluated": True,
            }
        else:
            raise RuntimeError(f"unexpected C010 family: {row.diagnostic_family}")
        for column, value in values.items():
            output.at[index, column] = value
    output = output.reindex(columns=RAW_OUTPUT_COLUMNS)
    output["n_units"] = pd.to_numeric(output["n_units"], errors="raise").astype("Int64")
    return output


def fmt(value) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.12g}"


def corrected_region_summary(
    raw: pd.DataFrame,
    base630: Path,
    c010: dict[str, pd.DataFrame],
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    wrong_summary = pd.read_csv(base630 / "controls/wrong_region_control_pair_summary.csv")
    primary_variables = raw[raw.diagnostic_family.eq("location_buffer")]
    for (region, variable), primary in primary_variables.groupby(["region", "variable"], sort=True):
        template = definitions[
            (definitions.region == region) & (definitions.variable_definition == variable)
        ].iloc[0]
        primary = primary.sort_values("buffer_km")
        lead = raw[(raw.diagnostic_family == "lead_lag") & (raw.region == region) & (raw.variable == variable)]
        year = raw[(raw.diagnostic_family == "year_block_robustness") & (raw.region == region) & (raw.variable == variable)].sort_values("buffer_km")
        wrong = wrong_summary[(wrong_summary.source_region == region) & (wrong_summary.variable == variable)].sort_values("buffer_km")
        primary_support = primary.supported_or_significant.map(as_bool)
        lead_support = lead.supported_or_significant.map(as_bool)
        primary_text = ";".join(
            f"{int(item.buffer_km)}km:effect={fmt(item.effect)},CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}],p_FDR={fmt(item.p_fdr)}"
            for item in primary.itertuples(index=False)
        )
        temporal = {}
        for lag_group in ["pre-event", "event-time", "post-event"]:
            item = lead[lead.lag_group.eq(lag_group)].iloc[0]
            temporal[lag_group] = f"significant={as_bool(item.supported_or_significant)};effect={fmt(item.effect)};CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}];p_FDR={fmt(item.p_fdr)}"
        wrong_text = ";".join(
            f"{int(item.buffer_km)}km:{int(item.n_passed_wrong_regions)}/4;all={as_bool(item.all_wrong_regions_passed)}"
            for item in wrong.itertuples(index=False)
        )
        year_text = ";".join(
            f"{int(item.buffer_km)}km:confirmed={as_bool(item.supported_or_significant)};CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}]"
            for item in year.itertuples(index=False)
        )
        specificity = any(as_bool(item.all_wrong_regions_passed) for item in wrong.itertuples(index=False))
        year_confirm = any(as_bool(value) for value in year.supported_or_significant)
        temporal_any = bool(lead_support.any())
        primary_any = bool(primary_support.any())
        robust = primary_any and temporal_any and specificity and year_confirm
        manuscript_status = "SUPPORTED_WITH_ROBUSTNESS" if robust else ("SUPPORTED_PRIMARY_ONLY" if primary_any else "NOT_SUPPORTED")
        supported_buffers = ",".join(str(int(value)) for value in primary.loc[primary_support, "buffer_km"]) or "none"
        lead_groups = ",".join(lead.loc[lead_support, "lag_group"].astype(str)) or "none"
        safe = (
            f"Under the uniform five-region workflow, {region} {template.mechanism} had primary location-buffer support at "
            f"{supported_buffers} km; significant temporal contrast groups: {lead_groups}; 4/4 wrong-region specificity: "
            f"{'passed' if specificity else 'not passed'}; year-block confirmation for a primary-supported buffer: "
            f"{'yes' if year_confirm and primary_any else 'no'}. This is an association and not a direct between-region or causal contrast."
        )
        rows.append({
            "region": region,
            "mechanism": template.mechanism,
            "variable_definition": variable,
            "event_population": template.event_population,
            "statistical_unit": template.statistical_unit,
            "primary_effect_ci_p_fdr": primary_text,
            "primary_supported": primary_any,
            "pre_event_support": temporal["pre-event"],
            "event_time_support": temporal["event-time"],
            "post_event_support": temporal["post-event"],
            "wrong_region_specificity": wrong_text,
            "year_block_confirmation": year_text,
            "supplementary_support": "not_applicable",
            "data_status": "COMPLETE_SYMMETRIC",
            "manuscript_status": manuscript_status,
            "manuscript_safe_statement": safe,
            "between_region_contrast_status": template.between_region_contrast_status,
            "source_files": template.source_files,
        })
    for family, tests_key, status_key in [
        ("cyclone_proximity", "cyclone", "cyclone_status"),
        ("ice_edge_relative_wind", "edge", "edge_status"),
    ]:
        _tests, statuses = c010[tests_key], c010[status_key]
        for region in stage2.FOCUS_REGIONS:
            template = definitions[(definitions.region == region) & (definitions.mechanism == family)].iloc[0]
            status = statuses[statuses.region.eq(region)].iloc[0]
            group = raw[(raw.diagnostic_family == family) & (raw.region == region)]
            if family == "cyclone_proximity":
                effect_text = ";".join(
                    f"{item.variable}:effect={fmt(item.effect)},CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}],p_FDR={fmt(item.p_fdr)}"
                    for item in group.itertuples(index=False)
                )
                temporal = "not_applicable_combined_-1_0_1"
                safe_prefix = f"Cyclone-proximity diagnostics were uniformly evaluated for {region} and classified {status.evidence_status}"
            else:
                primary_out = group[(group.analysis_tier == "primary") & (group.variable == "outward_edge_normal_wind_ms")]
                effect_text = ";".join(
                    f"{item.lag_group}:effect={fmt(item.effect)},CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}],p_FDR={fmt(item.p_fdr)}"
                    for item in primary_out.itertuples(index=False)
                )
                by_lag = {
                    item.lag_group: f"status={item.workflow_label};effect={fmt(item.effect)};CI=[{fmt(item.ci_low)},{fmt(item.ci_high)}];p_FDR={fmt(item.p_fdr)}"
                    for item in primary_out.itertuples(index=False)
                }
                temporal = None
                safe_prefix = f"Ice-edge-relative-wind diagnostics were uniformly evaluated for {region} and classified {status.evidence_status}"
            supported = str(status.evidence_status) == "supported"
            safe = safe_prefix + ("." if supported else "; not_supported does not establish mechanism absence.")
            rows.append({
                "region": region,
                "mechanism": family,
                "variable_definition": template.variable_definition,
                "event_population": template.event_population,
                "statistical_unit": template.statistical_unit,
                "primary_effect_ci_p_fdr": effect_text,
                "primary_supported": supported,
                "pre_event_support": temporal if temporal else by_lag["pre-event"],
                "event_time_support": temporal if temporal else by_lag["event-time"],
                "post_event_support": temporal if temporal else by_lag["post-event"],
                "wrong_region_specificity": "not_evaluated_supplementary",
                "year_block_confirmation": "not_evaluated_supplementary",
                "supplementary_support": f"status={status.evidence_status};reason={status.reason}",
                "data_status": "COMPLETE_SYMMETRIC",
                "manuscript_status": "SUPPORTED_PRIMARY_ONLY" if supported else "NOT_SUPPORTED",
                "manuscript_safe_statement": safe,
                "between_region_contrast_status": template.between_region_contrast_status,
                "source_files": template.source_files,
            })
    result = pd.DataFrame(rows).reindex(columns=REGION_SUMMARY_COLUMNS)
    if len(result) != 40:
        raise RuntimeError(f"corrected region summary rows={len(result)}")
    return result.sort_values(["region", "mechanism", "variable_definition"]).reset_index(drop=True)


def resolve_630(base630: Path) -> tuple[Path, Path]:
    if base630.is_file():
        if base630.parent.name != "formal_evidence":
            raise ValueError("a --base630 file must be inside its current-run formal_evidence directory")
        return base630, base630.parent.parent
    candidates = [
        base630 / "formal_evidence/formal_stage2_raw_evidence.csv",
        base630 / "formal_stage2_raw_evidence.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, base630
    raise FileNotFoundError(
        "current-run 630-row evidence not found under --base630; no historical fallback is permitted"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument("--base630", required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    base630 = Path(args.base630)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    definitions = pd.read_csv(EVIDENCE_ROWS)
    region_definitions = pd.read_csv(REGION_MECHANISMS)
    existing630_path, base630_run = resolve_630(base630)
    existing630 = pd.read_csv(existing630_path)
    c010_template = definitions[
        definitions.diagnostic_family.isin(["cyclone_proximity", "ice_edge_relative_wind"])
    ].copy().reset_index(drop=True)
    corrected_tables = c010_tables(run)
    corrected_c010 = map_c010(c010_template, corrected_tables)
    combined = pd.concat([existing630, corrected_c010], ignore_index=True)
    combined = definitions[["evidence_id"]].merge(combined, on="evidence_id", how="left", validate="one_to_one")
    combined = combined.reindex(columns=RAW_OUTPUT_COLUMNS)
    combined["n_units"] = pd.to_numeric(combined["n_units"], errors="raise").astype("Int64")
    counts = combined.diagnostic_family.value_counts().to_dict()
    gates = {
        "rows": len(combined) == 770,
        "unique_evidence_id": combined.evidence_id.nunique() == 770,
        "duplicates": int(combined.evidence_id.duplicated().sum()) == 0,
        "method_definition_evidence_ID_set_exact": set(combined.evidence_id) == set(definitions.evidence_id),
        "all_five_focus_regions": set(stage2.FOCUS_REGIONS).issubset(set(combined.region)),
        "schema_compatible": list(combined.columns) == RAW_OUTPUT_COLUMNS,
        "family_counts": counts == EXPECTED_COUNTS,
    }
    if not all(gates.values()):
        raise RuntimeError(f"HOLD_STAGE2_FORMAL_SCOPE_INCOMPLETE: {gates}")
    combined.to_csv(out / "corrected_stage2_raw_evidence_full_770.csv", index=False)
    summary = corrected_region_summary(combined, base630_run, corrected_tables, region_definitions)
    summary.to_csv(out / "corrected_stage2_region_evidence_summary.csv", index=False)

    status_rows = []
    for family, status_name in [
        ("cyclone_proximity", "cyclone_status"),
        ("ice_edge_relative_wind", "edge_status"),
    ]:
        statuses = corrected_tables[status_name].set_index("region")
        for region in stage2.FOCUS_REGIONS:
            status_rows.append({
                "diagnostic_family": family,
                "region": region,
                "current_regional_status": statuses.loc[region, "evidence_status"],
                "reason": statuses.loc[region, "reason"],
            })
    current_status = pd.DataFrame(status_rows)
    c010_closed = bool(
        current_status.groupby("diagnostic_family").current_regional_status.apply(
            lambda values: values.eq("not_supported").all()
        ).all()
    )
    current_status["C010_STATUS"] = "CURRENT_RUN_ALL_REGIONS_NOT_SUPPORTED" if c010_closed else "HOLD_C010_CLAIM_REVIEW"
    current_status.to_csv(out / "C010_current_status_summary.csv", index=False)
    pd.DataFrame([
        {"gate": key, "value": value, "status": "PASS" if value else "FAIL"}
        for key, value in gates.items()
    ]).to_csv(out / "stage2_scope_completeness_audit.csv", index=False)
    validation = {
        "status": "PASS",
        "gates": gates,
        "family_counts": counts,
        "C010_STATUS": "CURRENT_RUN_ALL_REGIONS_NOT_SUPPORTED" if c010_closed else "HOLD_C010_CLAIM_REVIEW",
        "historical_reference_used_during_generation": False,
        "method_definition_rows": len(definitions),
        "outputs": {
            path.name: {"sha256": sha256(path), "rows": len(pd.read_csv(path))}
            for path in sorted(out.glob("*.csv"))
        },
    }
    (out / "scope_completion_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
