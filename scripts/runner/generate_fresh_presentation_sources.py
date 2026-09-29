#!/usr/bin/env python3
"""Generate the 14 current F1-F4 presentation sidecars from fresh run outputs.

This is a release-engineering composition layer.  It imports the accepted
M025 F1/F2/F3 preparation APIs and reproduces the M024/M026/M028 table-export
semantics without reading a prior presentation delivery, verified-run tree,
golden result, or canonical accepted output.  It does not render figures and
does not alter scientific values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = RELEASE_ROOT / 'scripts/presentation'
F3_CONFIG = RELEASE_ROOT / 'figure_specs/figure3.json'
DEFAULT_CLAIM_MAP = RELEASE_ROOT / "method_static/stage2_evidence_change_claim_map.csv"

F1_DIR = Path('F1')
F2_DIR = Path('F2')
F34_DIR = Path('F3_F4')
F4_DIR = Path('F4')

OUTPUT_PATHS = (
    F1_DIR / "F1_R3_annual_mean_context.csv",
    F1_DIR / "F1_R3_observed_window.csv",
    F1_DIR / "F1_R3_scale_payload.json",
    F2_DIR / "F2_R3_event_display.csv",
    F2_DIR / "F2_R3_year_halfmonth_counts.csv",
    F34_DIR / "F3_revision_round3_formal_evidence_matrix.csv",
    F34_DIR / "F3_revision_round3_interpretation_effect_rows.csv",
    F34_DIR / "F3_revision_round3_kara_registered_changes.csv",
    F34_DIR / "F4A_REPRESENTATIVE_COMPONENT_CELLS.csv",
    F34_DIR / "F4C_CORRESPONDENCE_DEFINITION_DATA.csv",
    F34_DIR / "F4S_ABSOLUTE_DOMAIN_CONTRAST_SOURCE_DATA.csv",
    F34_DIR / "F4S_MULTIREGION_SOURCE_DATA.csv",
    F4_DIR / "F4B_POPULATION_SOURCE_DATA.csv",
    F4_DIR / "F4D_NORMALIZED_SOURCE_DATA.csv",
)

M021_METRICS = "panarctic_spatial_composition_metrics.csv"
M021_COMPONENTS = "panarctic_loss_components.csv"
M021_REVERSE = "reverse_test_summary.csv"
M023_CROSS = "CROSS_SCALE_CORRESPONDENCE_CROSSTAB.csv"
M023_NORMALIZATION = "D4D_DOMAIN_NORMALIZATION_SUMMARY.csv"
M023_FOCAL = "D4D_FOCAL_STRENGTH_SENSITIVITY.csv"
M023_OPPORTUNITY = "REVERSE_DOMAIN_OPPORTUNITY_METRICS.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} unavailable: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} unavailable: {resolved}")
    return resolved


def reject_prior_presentation_inputs(paths: list[Path]) -> None:
    forbidden = ("/presentation_support/current_source_delivery/", "/verified_run1/")
    for path in paths:
        text = path.resolve().as_posix()
        if any(token in text for token in forbidden):
            raise ValueError(f"prior presentation/golden input forbidden: {text}")


def load_accepted_modules():
    if not SCRIPT_DIR.is_dir():
        raise FileNotFoundError(f"accepted presentation API directory unavailable: {SCRIPT_DIR}")
    sys.path.insert(0, str(SCRIPT_DIR))
    f12 = importlib.import_module("revision_round1_f1_f2")
    f3 = importlib.import_module("revision_round1_f3")
    return f12, f3


def prepare_f1_f2(args: argparse.Namespace, f12: Any) -> dict[Path, Any]:
    daily = require_file(args.daily_sie, "fresh Stage-1 daily SIE")
    broad = require_file(args.broad_events, "fresh post-M007 broad events")
    severe = require_file(args.severe_events, "fresh severe events")
    major = require_file(args.major_events, "fresh major events")
    mask = require_file(args.nsidc_region_mask, "raw NSIDC region mask")

    f1 = f12.prepare_f1(daily)
    if f1["metadata"]["selected_year"] != 2005 or f1["metadata"]["selected_date"] != "2005-07-21":
        raise ValueError("F1 deterministic example identity changed")

    f12.F2_HALF_MONTH_LABELS = ("Jun1", "Jun15", "Jul1", "Jul15", "Aug1", "Aug15")
    f2 = f12.prepare_f2(
        broad,
        severe,
        major,
        mask,
        expected_counts={"broad": 9530, "severe": 2427, "major_severe": 554},
    )
    annual = f1["annual"].copy()
    window = f1["window"].copy()
    window["date"] = window["date"].dt.strftime("%Y-%m-%d")
    events = f2["events"].copy()
    events["event_start"] = events["event_start"].dt.strftime("%Y-%m-%d")
    return {
        F1_DIR / "F1_R3_annual_mean_context.csv": annual,
        F1_DIR / "F1_R3_observed_window.csv": window,
        F1_DIR / "F1_R3_scale_payload.json": f1["scale"],
        F2_DIR / "F2_R3_event_display.csv": events,
        F2_DIR / "F2_R3_year_halfmonth_counts.csv": f2["timing"],
    }


def prepare_f3(args: argparse.Namespace, f3: Any) -> dict[Path, pd.DataFrame]:
    raw_path = require_file(args.stage2_raw_770, "fresh M010 corrected raw 770")
    summary_path = require_file(args.stage2_region_summary_40, "fresh M010 region summary 40")
    paired_path = require_file(args.floor015_paired_770, "fresh floor015 paired 770")
    claim_path = require_file(args.claim_map, "method-static Stage-2 claim map")

    config = json.loads(F3_CONFIG.read_text(encoding="utf-8"))
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    claim_map = pd.read_csv(claim_path, keep_default_na=False)
    expected_rows = int(config["expected_raw_evidence_rows"])
    if len(raw) != expected_rows or raw["evidence_id"].nunique() != expected_rows:
        raise ValueError("F3 raw-evidence identity gate failed")
    if len(paired) != expected_rows:
        raise ValueError("F3 paired primary/floor015 row gate failed")
    if len(summary) != int(config["expected_region_summary_rows"]):
        raise ValueError("F3 region-summary row gate failed")
    expected_pairs = {(r, m) for r in config["region_keys"] for m in config["mechanisms"]}
    actual_pairs = set(zip(summary["region"], summary["mechanism"]))
    if actual_pairs != expected_pairs or summary.duplicated(["region", "mechanism"]).any():
        raise ValueError("F3 complete 8-by-5 region/mechanism gate failed")
    if not set(summary["manuscript_status"]).issubset(set(config["allowed_statuses"])):
        raise ValueError("F3 unexpected formal status")

    kara = current_kara_rows(claim_map, raw, paired)
    region_order = {value: index for index, value in enumerate(config["region_keys"])}
    mechanism_order = {value: index for index, value in enumerate(config["mechanisms"])}
    formal = summary[["region", "mechanism", "manuscript_status", "primary_supported", "manuscript_safe_statement"]].copy()
    formal["region_order"] = formal["region"].map(region_order)
    formal["mechanism_order"] = formal["mechanism"].map(mechanism_order)
    formal = formal.sort_values(["mechanism_order", "region_order"]).reset_index(drop=True)

    laptev = f3._effect_rows(paired, "laptev_sea", "era5_v10_buffer")
    beaufort = f3._effect_rows(paired, "beaufort_sea", "era5_wspd10_buffer")
    kara_wrong = paired[
        paired["diagnostic_type"].eq("wrong_region")
        & paired["source_region"].eq("kara_sea")
        & paired["variable"].eq("era5_v10_buffer")
        & paired["buffer_km"].eq(100)
        & paired["wrong_region"].eq("beaufort_sea")
    ]
    if len(kara_wrong) != 1 or kara_wrong.iloc[0]["floor015_control_status"] != "source_primary_not_supported":
        raise ValueError("F3 Kara not-evaluable specificity gate failed")

    payload = f3.F3Payload(
        input_paths={
            "corrected_raw_evidence_770.csv": raw_path,
            "corrected_region_summary.csv": summary_path,
            "primary_floor015_paired_raw_770.csv": paired_path,
            "stage2_evidence_change_claim_map.csv": claim_path,
        },
        raw=raw,
        summary=summary,
        paired=paired,
        trend=pd.DataFrame(),
        claim_map=claim_map,
        formal_matrix=formal,
        laptev_effects=laptev,
        beaufort_effects=beaufort,
        kara_claims=kara.reset_index(drop=True),
    )
    frames = f3.panel_data_frames(payload)
    effects = frames[f3.PANEL_DATA_FILENAMES["interpretation_effects"]].copy()
    effects["case"] = effects["case"].replace({
        "Laptev v10": "Laptev Sea — meridional wind (v10)",
        "Beaufort wspd10": "Beaufort Sea — wind speed (wspd10)",
    })
    return {
        F34_DIR / "F3_revision_round3_formal_evidence_matrix.csv": frames[f3.PANEL_DATA_FILENAMES["formal_matrix"]],
        F34_DIR / "F3_revision_round3_interpretation_effect_rows.csv": effects,
        F34_DIR / "F3_revision_round3_kara_registered_changes.csv": frames[f3.PANEL_DATA_FILENAMES["kara_changes"]],
    }


def current_kara_rows(selection: pd.DataFrame, raw: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Current table extraction using the accepted floor015 support helper.

    Only registration keys/labels may be static. No historical numerical input,
    fallback or new scientific threshold. M010 and paired primary must agree.
    """
    static_columns = [
        'change_id', 'change_class', 'region', 'diagnostic_type', 'variable', 'buffer',
        'lag/control', 'claim / Results subsection',
        'required factual action or proposed interpretive disposition', 'author_review_row_id',
    ]
    if list(selection.columns) != static_columns:
        raise ValueError('Kara selection schema must not contain historical result columns')
    if len(selection) != 3 or selection.change_id.duplicated().any() or set(selection.author_review_row_id) != {'I003'}:
        raise ValueError('registered Kara selection identity failed')
    if set(selection.change_class) != {'KARA_FLOOR015_SENSITIVITY'}:
        raise ValueError('unregistered Kara change class')
    sys.path.insert(0, str(RELEASE_ROOT / 'src'))
    from vrile.robustness.downstream_floor015 import _stage2_region_summary
    rows = []
    for _, choice in selection.sort_values('change_id').iterrows():
        family = choice['diagnostic_type']
        if family not in {'location_buffer', 'lead_lag', 'wrong_region'}:
            raise ValueError('unregistered Kara diagnostic')
        region_key = 'source_region' if family == 'wrong_region' else 'region'
        selected = paired.loc[(paired.diagnostic_type == family) & (paired[region_key] == choice.region) & (paired.variable == choice.variable)]
        formal_family = 'wrong_region_control' if family == 'wrong_region' else family
        formal = raw.loc[(raw.diagnostic_family == formal_family) & (raw.region == choice.region) & (raw.variable == choice.variable)]
        if choice['buffer'] != '':
            radius = float(choice['buffer'])
            selected = selected.loc[selected.buffer_km.eq(radius)]
            formal = formal.loc[formal.buffer_km.eq(radius)]
        if family == 'lead_lag':
            selected = selected.loc[selected.lag_group.eq(choice['lag/control'])]
            formal = formal.loc[formal.lag_group.eq(choice['lag/control'])]
        elif family == 'wrong_region':
            selected = selected.loc[selected.wrong_region.eq(choice['lag/control'])]
            formal = formal.loc[formal.wrong_region.eq(choice['lag/control'])]
        if len(selected) != 1 or len(formal) != 1:
            raise ValueError('Kara current upstream key missing/duplicate')
        source = selected.iloc[0]
        status = _stage2_region_summary(selected)
        status = status.loc[(status.region == choice.region) & (status.diagnostic_type == family)].iloc[0]
        corrected = formal.iloc[0]
        if corrected.p_fdr != source.primary_p_FDR or bool(corrected.supported_or_significant) != bool(status.primary_supported_rows):
            raise ValueError('Kara M010 versus current primary-paired provenance mismatch')
        result = choice.to_dict()
        result['F0 state/value'] = 'not applicable to corrected-floor comparison'
        for scenario, column in [('primary', 'corrected state/value'), ('floor015', 'floor015 state/value')]:
            p = source[f'{scenario}_p_FDR']
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Kara invalid current p/FDR')
            supported = bool(status[f'{scenario}_supported_rows'])
            control = source[f'{scenario}_control_passed']
            if pd.notna(control) and not isinstance(control, (bool, np.bool_)):
                raise ValueError('Kara invalid current control flag')
            result[column] = f'support={supported};p_FDR={p};control={control}'
        rows.append(result)
    order = static_columns[:7] + ['F0 state/value', 'corrected state/value', 'floor015 state/value'] + static_columns[7:]
    return pd.DataFrame(rows)[order]


def representative_selection(population: pd.DataFrame) -> pd.Series:
    columns = [
        "effective_component_number_sic",
        "largest_component_resolved_sic_loss_share",
        "top3_component_resolved_sic_loss_share",
        "weighted_component_spatial_spread_km",
    ]
    if len(population) != 99 or population["pan_event_id"].nunique() != 99:
        raise ValueError("F4 representative-event universe mismatch")
    medians = population[columns].median()
    iqrs = population[columns].quantile(0.75) - population[columns].quantile(0.25)
    if (iqrs <= 0).any():
        raise ValueError("F4 representative-event IQR gate failed")
    audit = population[["pan_event_id", "event_date", *columns]].copy()
    audit["standardized_median_distance"] = np.sqrt((((population[columns] - medians) / iqrs) ** 2).sum(axis=1))
    audit = audit.sort_values(["standardized_median_distance", "event_date", "pan_event_id"], kind="mergesort")
    selected = audit.iloc[0]
    if selected["pan_event_id"] != "PAN000062" or str(selected["event_date"])[:10] != "2016-07-12":
        raise ValueError("F4 representative-event identity mismatch")
    return selected


def representative_cells(components_path: Path, field_path: Path, mask_path: Path) -> pd.DataFrame:
    components = pd.read_csv(components_path)
    components = components[components["pan_event_id"].eq("PAN000062")].copy()
    components["component_loss_rank"] = components["integrated_sic_loss_km2eq"].rank(method="first", ascending=False).astype(int)
    with xr.open_dataset(field_path) as dataset, xr.open_dataset(mask_path) as masks:
        loss = dataset["cdr_sic_loss"].load().values
        labels = dataset["loss_component_id"].load().values.astype(int)
        x = dataset["x"].load().values
        y = dataset["y"].load().values
        # Load the raw surface mask to enforce the accepted M026 spatial-grid contract.
        surface = masks["sea_ice_region_surface_mask"].load().values
    if surface.shape != labels.shape:
        raise ValueError("F4 raw NSIDC mask and M021 field grid differ")
    yy, xx = np.meshgrid(y, x, indexing="ij")
    component_map = components.set_index("component_id")
    keep = labels > 0
    ids = labels[keep]
    if not set(np.unique(ids)).issubset(set(component_map.index)):
        raise ValueError("F4 field component IDs missing from fresh M021 component table")
    cells = pd.DataFrame({
        "pan_event_id": "PAN000062",
        "event_date": "2016-07-12",
        "x_m": xx[keep],
        "y_m": yy[keep],
        "loss_component_id": ids,
        "cdr_sic_loss": loss[keep],
        "component_sic_loss_km2eq": [component_map.loc[value, "integrated_sic_loss_km2eq"] for value in ids],
        "component_loss_rank": [component_map.loc[value, "component_loss_rank"] for value in ids],
    })
    cells["is_largest_component"] = cells["component_loss_rank"].eq(1)
    if len(cells) != 2777 or cells["loss_component_id"].nunique() != 38:
        raise ValueError("F4 representative component-cell gate failed")
    return cells


def normalized_table(normalization: pd.DataFrame, focal: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "local_loss_fraction_diff": "focal",
        "residual_loss_fraction_diff": "residual",
        "other_region_loss_fraction_diff": "other_region",
    }
    rows: list[dict[str, Any]] = []
    domain_rows = normalization[normalization["family"].eq("domainwise_initial_ice")]
    if set(domain_rows["metric"]) != set(mapping):
        raise ValueError("M023 domain-normalization metric set mismatch")
    for row in domain_rows.itertuples(index=False):
        rows.append({
            "series": "raw_normalized",
            "domain": mapping[row.metric],
            "estimate": row.mean_difference,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
            "n_cases": row.n_cases,
            "source_reference": f"M023_CURRENT_RUN/{M023_NORMALIZATION}",
        })
    base = focal[focal["model"].eq("BASE")]
    if set(base["outcome"]) != {"residual_loss_fraction_diff", "other_region_loss_fraction_diff"}:
        raise ValueError("M023 focal-strength BASE outcome set mismatch")
    for row in base.itertuples(index=False):
        rows.append({
            "series": "focal_strength_adjusted",
            "domain": "residual" if row.outcome.startswith("residual") else "other_region",
            "estimate": row.intercept_point_estimate,
            "ci_low": row.bootstrap_ci_low,
            "ci_high": row.bootstrap_ci_high,
            "n_cases": row.n_cases,
            "source_reference": f"M023_CURRENT_RUN/{M023_FOCAL}",
        })
    result = pd.DataFrame(rows)
    if len(result) != 5 or set(result["n_cases"].astype(int)) != {276}:
        raise ValueError("F4 normalized table row/case gate failed")
    return result


def validate_cross_table(cross: pd.DataFrame) -> None:
    expected_core = {"both": 85, "temporal_only": 19, "footprint_only": 199, "neither": 251}
    expected_margins = {"temporal_margin": 104, "footprint_margin": 284}
    if len(cross) != 6 or cross["cell_label"].nunique() != 6:
        raise ValueError("M023 X1/X2 crosstab six-row shape mismatch")
    values = dict(zip(cross["cell_label"], cross["n_events"].astype(int)))
    actual_core = {key: values.get(key) for key in expected_core}
    actual_margins = {key: values.get(key) for key in expected_margins}
    if actual_core != expected_core:
        raise ValueError(f"M023 X1/X2 crosstab core mismatch: {actual_core}")
    if actual_margins != expected_margins:
        raise ValueError(f"M023 X1/X2 crosstab margin mismatch: {actual_margins}")


def prepare_f4(args: argparse.Namespace) -> dict[Path, pd.DataFrame]:
    m021 = require_dir(args.m021_products, "fresh M021 products")
    field_root = require_dir(args.stage3_loss_fields, "fresh M021 loss fields")
    m023 = require_dir(args.m023_root, "fresh M023 outputs")
    sensitivity_path = require_file(args.sensitivity_matrix, "fresh Stage-3 sensitivity matrix")
    mask_path = require_file(args.nsidc_region_mask, "raw NSIDC region mask")

    metrics_path = require_file(m021 / M021_METRICS, "fresh M021 composition metrics")
    components_path = require_file(m021 / M021_COMPONENTS, "fresh M021 components")
    reverse_path = require_file(m021 / M021_REVERSE, "fresh M021 reverse summary")
    field_path = require_file(field_root / "PAN000062.nc", "fresh M021 representative field")
    cross_path = require_file(m023 / M023_CROSS, "fresh M023 crosstab")
    normalization_path = require_file(m023 / M023_NORMALIZATION, "fresh M023 normalization")
    focal_path = require_file(m023 / M023_FOCAL, "fresh M023 focal sensitivity")
    opportunity_path = require_file(m023 / M023_OPPORTUNITY, "fresh M023 opportunity metrics")

    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 99 or metrics["pan_event_id"].nunique() != 99:
        raise ValueError("M021 pan-event metric universe mismatch")
    population_columns = [
        "pan_event_id", "event_date", "effective_component_number_sic",
        "largest_component_resolved_sic_loss_share", "top3_component_resolved_sic_loss_share",
        "weighted_component_spatial_spread_km",
    ]
    population = metrics[population_columns].copy()
    population["source_reference"] = f"M021_CURRENT_RUN/{M021_METRICS}"
    representative_selection(population)
    cells = representative_cells(components_path, field_path, mask_path)

    multiregion_columns = [
        "pan_event_id", "event_date", "effective_named_region_number_sic_loss",
        "largest_named_region_sic_loss_share", "top3_named_region_sic_loss_share",
    ]
    multiregion = metrics[multiregion_columns].copy()
    multiregion["stage3_region_framework"] = "18_named_regions"
    multiregion["source_reference"] = f"M021_CURRENT_RUN/{M021_METRICS}"

    sensitivity = pd.read_csv(sensitivity_path)
    required_sensitivity = {"sic_floor", "strict_min_cells", "n_events_total", "n_events_with_components", "fraction_events_with_ge3_components"}
    if len(sensitivity) != 9 or not required_sensitivity.issubset(sensitivity.columns):
        raise ValueError("fresh Stage-3 sensitivity 9-setting contract failed")
    if not (sensitivity["n_events_total"].eq(99) & sensitivity["n_events_with_components"].eq(99) & sensitivity["fraction_events_with_ge3_components"].eq(1.0)).all():
        raise ValueError("fresh Stage-3 sensitivity parity gate failed")

    cross = pd.read_csv(cross_path, dtype=str)
    validate_cross_table(cross)

    normalized = normalized_table(pd.read_csv(normalization_path), pd.read_csv(focal_path))
    reverse = pd.read_csv(reverse_path)
    loss_metrics = ["local_sic_loss_km2eq", "residual_sic_loss_km2eq", "other_region_sic_loss_km2eq"]
    absolute = reverse[reverse["metric"].isin(loss_metrics)].copy()
    absolute["display_label"] = absolute["metric"].map({
        "local_sic_loss_km2eq": "Focal local SIC loss",
        "residual_sic_loss_km2eq": "Residual SIC loss",
        "other_region_sic_loss_km2eq": "Other-region SIC loss",
    })
    absolute["source_reference"] = f"M021_CURRENT_RUN/{M021_REVERSE}"
    absolute = absolute[[
        "metric", "display_label", "n_cases", "mean_difference", "median_difference",
        "ci_low", "ci_high", "p_raw", "p_fdr", "status", "alternative",
        "bootstrap_resamples", "signflip_resamples", "source_reference",
    ]]
    if len(absolute) != 3 or not absolute["n_cases"].eq(276).all():
        raise ValueError("M021 reverse-loss contrast gate failed")
    opportunity = pd.read_csv(opportunity_path)
    absolute["median_initial_ice_equivalent_opportunity_km2eq"] = [
        opportunity["local_initial_ice_equivalent_area_km2eq"].median(),
        opportunity["residual_initial_ice_equivalent_area_km2eq"].median(),
        opportunity["other_region_initial_ice_equivalent_area_km2eq"].median(),
    ]
    return {
        F34_DIR / "F4A_REPRESENTATIVE_COMPONENT_CELLS.csv": cells,
        F34_DIR / "F4C_CORRESPONDENCE_DEFINITION_DATA.csv": cross,
        F34_DIR / "F4S_ABSOLUTE_DOMAIN_CONTRAST_SOURCE_DATA.csv": absolute,
        F34_DIR / "F4S_MULTIREGION_SOURCE_DATA.csv": multiregion,
        F4_DIR / "F4B_POPULATION_SOURCE_DATA.csv": population,
        F4_DIR / "F4D_NORMALIZED_SOURCE_DATA.csv": normalized,
    }


def fixture_test() -> int:
    normalization = pd.DataFrame([
        {"family": "domainwise_initial_ice", "metric": metric, "n_cases": 276, "mean_difference": value, "ci_low": value - 0.001, "ci_high": value + 0.001}
        for metric, value in (("local_loss_fraction_diff", 0.04), ("residual_loss_fraction_diff", 0.008), ("other_region_loss_fraction_diff", 0.007))
    ])
    focal = pd.DataFrame([
        {"outcome": "residual_loss_fraction_diff", "model": "BASE", "n_cases": 276, "intercept_point_estimate": 0.0084, "bootstrap_ci_low": 0.006, "bootstrap_ci_high": 0.011},
        {"outcome": "other_region_loss_fraction_diff", "model": "BASE", "n_cases": 276, "intercept_point_estimate": 0.0066, "bootstrap_ci_low": 0.003, "bootstrap_ci_high": 0.010},
    ])
    result = normalized_table(normalization, focal)
    if list(result["domain"]) != ["focal", "residual", "other_region", "residual", "other_region"]:
        raise AssertionError("normalized source-table fixture order changed")
    if len(OUTPUT_PATHS) != 14 or len(set(OUTPUT_PATHS)) != 14:
        raise AssertionError("presentation sidecar inventory is not exactly 14 unique paths")
    cross = pd.DataFrame([
        {"cell_label": "both", "n_events": "85"},
        {"cell_label": "temporal_only", "n_events": "19"},
        {"cell_label": "footprint_only", "n_events": "199"},
        {"cell_label": "neither", "n_events": "251"},
        {"cell_label": "temporal_margin", "n_events": "104"},
        {"cell_label": "footprint_margin", "n_events": "284"},
    ])
    validate_cross_table(cross)
    print("FRESH_PRESENTATION_SOURCE_FIXTURE = PASS")
    print("PRESENTATION_SIDECAR_PATHS = 14/14")
    print("M023_CROSSTAB_SIX_ROW_FIXTURE = PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-test", action="store_true", help="run bounded source-table fixture tests only")
    parser.add_argument("--daily-sie", type=Path)
    parser.add_argument("--broad-events", type=Path)
    parser.add_argument("--severe-events", type=Path)
    parser.add_argument("--major-events", type=Path)
    parser.add_argument("--stage2-raw-770", type=Path)
    parser.add_argument("--stage2-region-summary-40", type=Path)
    parser.add_argument("--floor015-paired-770", type=Path)
    parser.add_argument("--claim-map", type=Path, default=DEFAULT_CLAIM_MAP)
    parser.add_argument("--m021-products", type=Path)
    parser.add_argument("--stage3-loss-fields", type=Path)
    parser.add_argument("--sensitivity-matrix", type=Path)
    parser.add_argument('--m023_dir', type=Path, dest='m023_root')
    parser.add_argument("--nsidc-region-mask", type=Path)
    parser.add_argument('--output_dir', type=Path, dest='output_root')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fixture_test:
        return fixture_test()
    required = (
        "daily_sie", "broad_events", "severe_events", "major_events",
        "stage2_raw_770", "stage2_region_summary_40", "floor015_paired_770",
        "m021_products", "stage3_loss_fields", "sensitivity_matrix", "m023_root",
        "nsidc_region_mask", "output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required generation arguments: {', '.join(missing)}")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists; refusing overwrite: {output_root}")
    generation_inputs = [
        args.daily_sie, args.broad_events, args.severe_events, args.major_events,
        args.stage2_raw_770, args.stage2_region_summary_40, args.floor015_paired_770,
        args.m021_products, args.stage3_loss_fields, args.sensitivity_matrix,
        args.m023_root, args.nsidc_region_mask,
    ]
    reject_prior_presentation_inputs(generation_inputs)
    f12, f3 = load_accepted_modules()
    payloads: dict[Path, Any] = {}
    payloads.update(prepare_f1_f2(args, f12))
    payloads.update(prepare_f3(args, f3))
    payloads.update(prepare_f4(args))
    if set(payloads) != set(OUTPUT_PATHS):
        raise RuntimeError(f"sidecar inventory mismatch: {set(OUTPUT_PATHS) ^ set(payloads)}")

    output_root.mkdir(parents=True)
    for relative_path in OUTPUT_PATHS:
        value = payloads[relative_path]
        destination = output_root / relative_path
        if relative_path.suffix == ".json":
            write_json(destination, value)
        else:
            write_csv(destination, value)
    rows = [
        {"relative_path": path.as_posix(), "sha256": sha256(output_root / path), "size_bytes": (output_root / path).stat().st_size}
        for path in OUTPUT_PATHS
    ]
    manifest = output_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "sha256", "size_bytes"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    ledger_lines = [f"{row['sha256']}  {row['relative_path']}" for row in rows]
    ledger_lines.append(f"{sha256(manifest)}  manifest.csv")
    ledger = output_root / "sha256.txt"
    ledger.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
    print("FRESH_PRESENTATION_SOURCE_CLOSURE = PASS")
    print("PRESENTATION_SIDECARS_GENERATED = 14/14")
    print("PREVIOUS_PRESENTATION_RESULTS_USED = NO")
    print(f"OUTPUT_ROOT = {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
