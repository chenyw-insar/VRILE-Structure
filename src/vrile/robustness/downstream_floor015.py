"""Floor-0.15 downstream robustness closure in an isolated transaction.

This module deliberately reuses the repaired toolbox's active scientific
functions.  It never redirects a primary writer into the primary output tree:
all generated data live below ``outputs/robustness_lowsic_v1/downstream_floor015``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from vrile.local_objects import detect_connected_components
from vrile.robustness.lowsic_local import reconstruct_local_catalogs
from vrile.stage3 import controls as stage3_controls
from vrile.stage3.budget import build_class_partitions, cdr_contribution_budget, detector_contribution_budget
from vrile.stage3.grid import (
    load_cell_area_km2,
    load_pan_events,
    read_sic,
    valid_ocean_mask,
)
from vrile.stage3.overlap import run_overlap


OUTPUT_REL = Path("outputs/robustness_lowsic_v1/downstream_floor015")
FOCUS_REGIONS = ["beaufort_sea", "laptev_sea", "kara_sea", "barents_sea", "central_arctic"]
LEVEL_PATHS = {
    "broad": "outputs/local_vrile_enhanced/unique_local_events.csv",
    "severe": "outputs/local_vrile_severe/severe_unique_local_events.csv",
    "major_severe": "outputs/local_vrile_major_severe/major_severe_events_union.csv",
}
EVENT_FIELDS = [
    "dominant_region",
    "event_start",
    "event_end",
    "duration_days",
    "n_daily_patches",
    "cumulative_loss",
    "max_area_km2",
    "track_centroid_lon",
    "track_centroid_lat",
]
SEED_STAGE3 = 20260709
BOOTSTRAP_RESAMPLES = 10_000
SIGNFLIP_RESAMPLES = 10_000


def _utc_now() -> str:
    """Return a timezone-explicit UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    """Calculate one file SHA-256 without loading large files at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stage2_module(root: Path):
    """Load the active Stage-2 implementation so its functions can be reused."""

    path = root / "scripts/stage2_experiment_core.py"
    name = "vrile_floor015_active_stage2"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load active Stage-2 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _primary_hash_snapshot(root: Path) -> pd.DataFrame:
    """Freeze the primary files that this audit reads, before any computation."""

    paths = {
        root / "outputs/panarctic_local_contribution/frozen_baselines/stage3_scientific_output_hashes.csv",
        root / "outputs/robustness_lowsic_v1/claim_propagation_table.csv",
        root / "data/processed/local_event_cells/unique_event_patch_membership.csv",
        root / "data/processed/local_event_cells/patch_cell_index.csv",
        root / "outputs/spatiotemporal_matching/matching_rule_sensitivity.csv",
        root / "outputs/location_buffer_mechanism/location_buffer_evidence_table.csv",
        root / "outputs/lead_lag/lead_lag_background_tests.csv",
        root / "outputs/controls/wrong_region_control_tests.csv",
        root / "outputs/controls/year_block_bootstrap_summary.csv",
        root / "outputs/atmospheric_forcing/cyclone_proximity/cyclone_proximity_tests.csv",
        root / "outputs/atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_tests.csv",
        root / "outputs/panarctic_local_contribution/panarctic_local_event_overlap.csv",
        root / "outputs/panarctic_local_contribution/panarctic_local_component_overlap_pairs.csv",
        root / "outputs/panarctic_local_contribution/panarctic_detector_contribution_budget.csv",
        root / "outputs/panarctic_local_contribution/panarctic_cdr_contribution_budget.csv",
        root / "outputs/panarctic_local_contribution/panarctic_matched_control_summary.csv",
        root / "outputs/panarctic_local_contribution/panarctic_matched_control_anchors.csv",
        root / "outputs/panarctic_local_contribution/panarctic_control_test_summary.csv",
        root / "outputs/panarctic_local_contribution/panarctic_control_count_sensitivity_summary.csv",
        root / "outputs/panarctic_local_contribution/panarctic_primary_supported_year_block_bootstrap.csv",
    }
    paths.update(root / value for value in LEVEL_PATHS.values())
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Missing primary audit input: {path}")
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _verify_primary_snapshot(root: Path, before: pd.DataFrame) -> pd.DataFrame:
    """Verify that no primary file read by the audit changed during execution."""

    rows = []
    for row in before.itertuples(index=False):
        path = root / row.path
        size = path.stat().st_size if path.exists() else -1
        sha = _sha256(path) if path.exists() else ""
        rows.append(
            {
                "path": row.path,
                "expected_size_bytes": int(row.size_bytes),
                "actual_size_bytes": int(size),
                "expected_sha256": row.sha256,
                "actual_sha256": sha,
                "status": "PASS" if size == int(row.size_bytes) and sha == row.sha256 else "FAIL",
            }
        )
    result = pd.DataFrame(rows)
    if not result["status"].eq("PASS").all():
        raise RuntimeError("Primary/frozen output immutability gate failed")
    return result


def _event_membership_signatures(
    patches: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    """Build ID-independent patch signatures for every reconstructed event."""

    patch = patches.set_index("object_id")
    output: dict[str, tuple[str, ...]] = {}
    for event_id, group in membership.groupby("unique_local_event_id", sort=False):
        values = []
        for object_id in group["object_id"].astype(str):
            row = patch.loc[object_id]
            values.append(
                "|".join(
                    [
                        pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d"),
                        str(row.get("region", "")),
                        str(int(row["object_area_cells"])),
                        f"{float(row['centroid_lon']):.8f}",
                        f"{float(row['centroid_lat']):.8f}",
                        f"{float(row['cumulative_sic_loss']):.8f}",
                    ]
                )
            )
        output[str(event_id)] = tuple(sorted(values))
    return output


def _frozen_membership_signatures(root: Path) -> dict[str, tuple[str, ...]]:
    """Build the same ID-independent membership signature from primary tables."""

    patches = pd.read_csv(root / "outputs/local_vrile_enhanced/local_objects_filtered.csv")
    membership = pd.read_csv(root / "data/processed/local_event_cells/unique_event_patch_membership.csv")
    return _event_membership_signatures(patches, membership)


def _equivalence_gate(
    root: Path,
    catalogs: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Require event-level and membership equivalence before downstream work."""

    reconstructed = catalogs["none"]
    frozen_members = _frozen_membership_signatures(root)
    reconstructed_members = _event_membership_signatures(
        reconstructed["patches"], reconstructed["membership"]
    )
    rows: list[dict[str, object]] = []
    total_unmatched_frozen = 0
    total_unmatched_reconstructed = 0
    total_field_mismatches = 0
    frozen_by_signature = {value: key for key, value in frozen_members.items()}
    rebuilt_by_signature = {value: key for key, value in reconstructed_members.items()}
    if len(frozen_by_signature) != len(frozen_members) or len(rebuilt_by_signature) != len(reconstructed_members):
        raise RuntimeError("STOP_LOCAL_RECONSTRUCTION_NON_EQUIVALENCE: duplicate canonical membership signature")
    for level, relative in LEVEL_PATHS.items():
        frozen = pd.read_csv(root / relative)
        rebuilt = reconstructed[level].copy()
        frozen_ids = set(frozen["unique_local_event_id"].astype(str))
        rebuilt_ids = set(rebuilt["unique_local_event_id"].astype(str))
        frozen_signatures = {frozen_members[event_id] for event_id in frozen_ids}
        rebuilt_signatures = {reconstructed_members[event_id] for event_id in rebuilt_ids}
        unmatched_frozen = frozen_signatures - rebuilt_signatures
        unmatched_rebuilt = rebuilt_signatures - frozen_signatures
        total_unmatched_frozen += len(unmatched_frozen)
        total_unmatched_reconstructed += len(unmatched_rebuilt)
        frozen_lookup = frozen.set_index("unique_local_event_id")
        rebuilt_lookup = rebuilt.set_index("unique_local_event_id")
        for signature in sorted(frozen_signatures & rebuilt_signatures):
            frozen_event_id = frozen_by_signature[signature]
            rebuilt_event_id = rebuilt_by_signature[signature]
            left = frozen_lookup.loc[frozen_event_id]
            right = rebuilt_lookup.loc[rebuilt_event_id]
            mismatched = []
            for field in EVENT_FIELDS:
                if field in {"dominant_region", "event_start", "event_end"}:
                    equal = str(left[field]) == str(right[field])
                elif field in {"duration_days", "n_daily_patches"}:
                    equal = int(left[field]) == int(right[field])
                else:
                    equal = bool(
                        np.isclose(
                            float(left[field]), float(right[field]), rtol=1e-8, atol=1e-6, equal_nan=True
                        )
                    )
                if not equal:
                    mismatched.append(field)
            membership_equal = frozen_members[frozen_event_id] == reconstructed_members[rebuilt_event_id]
            if not membership_equal:
                mismatched.append("patch_event_membership")
            total_field_mismatches += len(mismatched)
            rows.append(
                {
                    "record_type": "event",
                    "severity_level": level,
                    "frozen_event_id": frozen_event_id,
                    "reconstructed_event_id": rebuilt_event_id,
                    "matched_one_to_one": True,
                    "field_mismatch_count": len(mismatched),
                    "mismatched_fields": ";".join(mismatched),
                    "membership_exact_match": membership_equal,
                    "status": "PASS" if not mismatched else "FAIL",
                }
            )
        rows.append(
            {
                "record_type": "level_summary",
                "severity_level": level,
                "frozen_event_id": "",
                "reconstructed_event_id": "",
                "matched_one_to_one": not unmatched_frozen and not unmatched_rebuilt,
                "field_mismatch_count": sum(
                    int(row["field_mismatch_count"])
                    for row in rows
                    if row["record_type"] == "event" and row["severity_level"] == level
                ),
                "mismatched_fields": "",
                "membership_exact_match": all(
                    bool(row["membership_exact_match"])
                    for row in rows
                    if row["record_type"] == "event" and row["severity_level"] == level
                ),
                "unmatched_frozen_events": len(unmatched_frozen),
                "unmatched_reconstructed_events": len(unmatched_rebuilt),
                "status": "PASS"
                if not unmatched_frozen
                and not unmatched_rebuilt
                and all(
                    row["status"] == "PASS"
                    for row in rows
                    if row["record_type"] == "event" and row["severity_level"] == level
                )
                else "FAIL",
            }
        )
    rows.append(
        {
            "record_type": "global_gate_summary",
            "severity_level": "all",
            "matched_one_to_one": not total_unmatched_frozen and not total_unmatched_reconstructed,
            "field_mismatch_count": total_field_mismatches,
            "membership_exact_match": total_field_mismatches == 0,
            "unmatched_frozen_events": total_unmatched_frozen,
            "unmatched_reconstructed_events": total_unmatched_reconstructed,
            "status": "PASS" if not total_unmatched_frozen and not total_unmatched_reconstructed and not total_field_mismatches else "FAIL",
        }
    )
    return pd.DataFrame(rows)


def _enrich_catalogs(root: Path, catalogs: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Attach production overlap flags needed by the unchanged clustering code."""

    pan_dates = set(
        pd.to_datetime(pd.read_csv(root / "outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv")["date"])
        .dt.normalize()
    )
    regional_path = root / "outputs/regional_vrile_enhanced/regional_vrile_events_unique.csv"
    regional_available = regional_path.is_file()
    if regional_available:
        regional = pd.read_csv(regional_path)
        regional_dates = set(
            zip(regional["region"].astype(str), pd.to_datetime(regional["event_date"]).dt.normalize())
        )
    else:
        regional_dates = set()
    for floor_catalog in catalogs.values():
        patches = floor_catalog["patches"].copy()
        patches["overlap_with_panarctic_vrile"] = pd.to_datetime(patches["date"]).dt.normalize().isin(pan_dates)
        patches["overlap_with_regional_vrile"] = [
            (str(region), pd.Timestamp(date).normalize()) in regional_dates
            for region, date in patches[["region", "date"]].itertuples(index=False, name=None)
        ]
        membership = floor_catalog["membership"]
        lookup = patches.set_index("object_id")
        for level in ("broad", "severe", "major_severe"):
            frame = floor_catalog[level].copy()
            pan_flags = {}
            regional_flags = {}
            for event_id, group in membership.groupby("unique_local_event_id", sort=False):
                members = lookup.loc[group["object_id"].astype(str)]
                pan_flags[str(event_id)] = bool(members["overlap_with_panarctic_vrile"].any())
                regional_flags[str(event_id)] = bool(members["overlap_with_regional_vrile"].any())
            ids = frame["unique_local_event_id"].astype(str)
            frame["overlap_with_panarctic_vrile"] = ids.map(pan_flags).fillna(False)
            frame["overlap_with_regional_vrile_v1"] = ids.map(regional_flags).fillna(False)
            frame["overlap_with_regional_response_diagnostic"] = frame["overlap_with_regional_vrile_v1"]
            if not regional_available:
                frame["regional_overlap_annotation_status"] = "UNAVAILABLE_NOT_EVALUATED"
            frame["event_catalog_role"] = "floor015_robustness_local_spatial_event"
            frame["event_date"] = frame["event_start"]
            floor_catalog[level] = frame
        floor_catalog["patches"] = patches


def _catalog_summary(catalogs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Summarize primary-equivalent and floor-0.15 reconstructed catalogs."""

    rows = []
    for floor in ("none", "0.15"):
        for level in ("broad", "severe", "major_severe"):
            frame = catalogs[floor][level]
            for region in ["all", *sorted(frame["dominant_region"].dropna().astype(str).unique())]:
                sub = frame if region == "all" else frame[frame["dominant_region"] == region]
                rows.append(
                    {
                        "sic_floor": floor,
                        "severity_level": level,
                        "region": region,
                        "event_count": len(sub),
                        "unique_start_dates": pd.to_datetime(sub["event_start"]).nunique(),
                    }
                )
    long = pd.DataFrame(rows)
    primary = long[long["sic_floor"] == "none"].drop(columns="sic_floor").rename(
        columns={"event_count": "primary_event_count", "unique_start_dates": "primary_unique_start_dates"}
    )
    floor = long[long["sic_floor"] == "0.15"].drop(columns="sic_floor").rename(
        columns={"event_count": "floor015_event_count", "unique_start_dates": "floor015_unique_start_dates"}
    )
    result = primary.merge(floor, on=["severity_level", "region"], how="outer", validate="one_to_one").fillna(0)
    result["absolute_change"] = result["floor015_event_count"] - result["primary_event_count"]
    result["relative_change"] = result["absolute_change"] / result["primary_event_count"].replace(0, np.nan)
    return result


def _write_catalog_inputs(
    work_dir: Path,
    catalogs: dict[str, dict[str, pd.DataFrame]],
    stage2,
) -> dict[str, Path]:
    """Persist floor-0.15 catalogs and unchanged-rule synoptic clusters."""

    catalog_dir = work_dir / "catalogs"
    cluster_dir = work_dir / "clusters"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    floor = catalogs["0.15"]
    paths: dict[str, Path] = {}
    for level in ("broad", "severe", "major_severe"):
        path = catalog_dir / f"floor015_{level}_events.csv"
        floor[level].to_csv(path, index=False)
        paths[level] = path
    for level, source in (("severe", floor["severe"]), ("major", floor["major_severe"])):
        frame = source.copy()
        frame["event_date"] = pd.to_datetime(frame["event_start"])
        clusters = stage2.cluster_events(frame, gap_days=5, distance_km=500)
        if "regional_overlap_annotation_status" in source.columns:
            clusters["matches_regional"] = pd.NA
            clusters["matches_regional_status"] = "UNAVAILABLE_NOT_EVALUATED"
        path = cluster_dir / f"floor015_{level}_clusters.csv"
        clusters.to_csv(path, index=False)
        paths[f"{level}_clusters"] = path
    return paths


def _stage2_args(stage2, paths: dict[str, Path], workers: int) -> argparse.Namespace:
    """Build the production Stage-2 argument contract with alternate catalogs."""

    parser = argparse.ArgumentParser(add_help=False)
    stage2.add_args(parser)
    args = parser.parse_args([])
    args.quick = False
    args.workers = int(workers)
    args.parallel = workers > 1
    args.disable_parallel = False
    args.backend = "process"
    args.severe_events = str(paths["severe"])
    args.major_events = str(paths["major_severe"])
    args.severe_clusters = str(paths["severe_clusters"])
    args.major_clusters = str(paths["major_clusters"])
    return args


def _matching_frame(local: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Convert a reconstructed local catalog to the active matching schema."""

    return pd.DataFrame(
        {
            "event_id": local["unique_local_event_id"].astype(str),
            "definition": definition,
            "region": local["dominant_region"].astype(str),
            "event_date": pd.to_datetime(local["event_start"]),
            "start_date": pd.to_datetime(local["event_start"]),
            "end_date": pd.to_datetime(local["event_end"]),
            "centroid_lon": pd.to_numeric(local["track_centroid_lon"], errors="coerce"),
            "centroid_lat": pd.to_numeric(local["track_centroid_lat"], errors="coerce"),
        }
    )


def _bidirectional_matching(
    stage2,
    root: Path,
    catalogs: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Recalculate C003/C004/C022 with unchanged temporal and centroid rules."""

    pan = stage2.load_panarctic(
        root / "outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv",
        root / "outputs/reproduce_sie/vrile_locations.csv",
    )
    scenarios = {}
    for scenario, label in (("primary", "none"), ("floor015", "0.15")):
        scenarios[scenario] = {
            "panarctic": pan,
            "severe": _matching_frame(catalogs[label]["severe"], "severe"),
            "major_severe": _matching_frame(catalogs[label]["major_severe"], "major_severe"),
        }
    pairs = [
        ("C003", "panarctic", "severe", "pan-Arctic -> severe"),
        ("C004", "panarctic", "major_severe", "pan-Arctic -> major_severe"),
        ("C022", "major_severe", "panarctic", "major_severe -> pan-Arctic"),
    ]
    rules = [
        ("time-only_pm3d", None),
        ("time+centroid_300km_pm3d", 300),
        ("time+centroid_500km_pm3d", 500),
    ]
    rows = []
    for claim_id, source, target, direction in pairs:
        for rule, distance in rules:
            values: dict[str, object] = {}
            for scenario, definitions in scenarios.items():
                numerator, distances = stage2.match_one(
                    definitions[source], definitions[target], 3, False, distance
                )
                denominator = len(definitions[source])
                values.update(
                    {
                        f"{scenario}_numerator": int(numerator),
                        f"{scenario}_denominator": int(denominator),
                        f"{scenario}_fraction": numerator / denominator if denominator else np.nan,
                        f"{scenario}_median_distance_km": float(np.median(distances)) if distances else np.nan,
                    }
                )
            rows.append(
                {
                    "claim_id": claim_id,
                    "direction": direction,
                    "source_event_unit": source,
                    "target_event_unit": target,
                    "matching_rule": rule,
                    "temporal_tolerance_days": 3,
                    **values,
                    "direction_preserved": bool(values["primary_fraction"] > 0) == bool(values["floor015_fraction"] > 0),
                    "absolute_fraction_change": float(values["floor015_fraction"] - values["primary_fraction"]),
                }
            )
    result = pd.DataFrame(rows)
    expected_primary = {
        ("C003", "time-only_pm3d"): (99, 99),
        ("C003", "time+centroid_300km_pm3d"): (42, 99),
        ("C003", "time+centroid_500km_pm3d"): (54, 99),
        ("C004", "time-only_pm3d"): (70, 99),
        ("C004", "time+centroid_300km_pm3d"): (18, 99),
        ("C004", "time+centroid_500km_pm3d"): (21, 99),
        ("C022", "time-only_pm3d"): (105, 546),
        ("C022", "time+centroid_300km_pm3d"): (18, 546),
        ("C022", "time+centroid_500km_pm3d"): (22, 546),
    }
    for row in result.itertuples(index=False):
        expected = expected_primary[(row.claim_id, row.matching_rule)]
        if (row.primary_numerator, row.primary_denominator) != expected:
            raise RuntimeError(
                "Primary matching reproduction failed: "
                f"{row.claim_id}/{row.matching_rule} observed="
                f"{row.primary_numerator}/{row.primary_denominator} expected={expected[0]}/{expected[1]}"
            )
    return result


STAGE2_TABLE_SPECS = {
    "location_buffer": (
        "location_buffer_mechanism/location_buffer_evidence_table.csv",
        ["region", "buffer_km", "variable"],
        90,
    ),
    "lead_lag": (
        "lead_lag/lead_lag_background_tests.csv",
        ["region", "variable", "lag_group"],
        90,
    ),
    "wrong_region": (
        "controls/wrong_region_control_tests.csv",
        ["source_region", "wrong_region", "buffer_km", "variable"],
        360,
    ),
    "year_block": (
        "controls/year_block_bootstrap_summary.csv",
        ["region", "buffer_km", "variable"],
        90,
    ),
    "cyclone": (
        "atmospheric_forcing/cyclone_proximity/cyclone_proximity_tests.csv",
        ["region", "variable"],
        20,
    ),
    "ice_edge": (
        "atmospheric_forcing/ice_edge_relative_wind/ice_edge_relative_wind_tests.csv",
        ["analysis_tier", "region", "variable", "lag_group"],
        120,
    ),
}


def _pair_evidence_table(
    diagnostic: str,
    primary: pd.DataFrame,
    floor: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Pair complete primary and floor tables without dropping unsupported rows."""

    if primary.duplicated(keys).any() or floor.duplicated(keys).any():
        raise RuntimeError(f"Duplicate Stage-2 evidence key in {diagnostic}")
    common = [column for column in primary.columns if column in floor.columns and column not in keys]
    left = primary[keys + common].rename(columns={column: f"primary_{column}" for column in common})
    right = floor[keys + common].rename(columns={column: f"floor015_{column}" for column in common})
    paired = left.merge(right, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        missing = paired.loc[~paired["_merge"].eq("both"), keys + ["_merge"]]
        raise RuntimeError(f"Stage-2 row-set mismatch for {diagnostic}: {missing.head().to_dict('records')}")
    paired = paired.drop(columns="_merge")
    paired.insert(0, "diagnostic_type", diagnostic)
    return paired


def _run_stage2_diagnostics(
    stage2,
    root: Path,
    output_dir: Path,
    paths: dict[str, Path],
    workers: int,
    primary_output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Floor-only execution organization; original scientific body below unchanged."""
    from vrile.robustness.floor015_executor import installed
    with installed(stage2, root, output_dir):
        return _run_stage2_diagnostics_body(stage2, root, output_dir, paths, workers, primary_output_root)


def _run_stage2_diagnostics_body(
    stage2,
    root: Path,
    output_dir: Path,
    paths: dict[str, Path],
    workers: int,
    primary_output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only the six catalog-dependent formal Stage-2 diagnostics."""

    stage_dir = output_dir / "stage2_work"
    args = _stage2_args(stage2, paths, workers)
    locations = stage_dir / "location_buffer_mechanism"
    args.out_dir = str(locations)
    stage2.stage6_centroid_buffer(args)
    lead = stage_dir / "lead_lag"
    args.location_buffer_values = str(locations / "location_buffer_lag_values.csv")
    args.out_dir = str(lead)
    stage2.stage7_lead_lag(args)
    controls = stage_dir / "controls"
    args.wrong_region_location_buffer_values = str(locations / "wrong_region_location_buffer_lag_values.csv")
    args.location_buffer_evidence = str(locations / "location_buffer_evidence_table.csv")
    args.out_dir = str(controls)
    stage2.stage8_controls(args)
    cyclone = stage_dir / "atmospheric_forcing/cyclone_proximity"
    args.out_dir = str(cyclone)
    stage2.stage9_cyclone(args)
    ice_edge = stage_dir / "atmospheric_forcing/ice_edge_relative_wind"
    args.out_dir = str(ice_edge)
    stage2.stage10_ice_edge(args)

    primary_output_root = primary_output_root or (root / "outputs")
    paired_tables = []
    for diagnostic, (relative, keys, expected_rows) in STAGE2_TABLE_SPECS.items():
        primary = pd.read_csv(primary_output_root / relative)
        floor = pd.read_csv(stage_dir / relative)
        if diagnostic in {"cyclone", "ice_edge"}:
            status_name = (
                "cyclone_proximity_evidence_status.csv"
                if diagnostic == "cyclone"
                else "ice_edge_relative_wind_evidence_status.csv"
            )
            primary_status = pd.read_csv((primary_output_root / relative).parent / status_name).set_index("region")["evidence_status"]
            floor_status = pd.read_csv((stage_dir / relative).parent / status_name).set_index("region")["evidence_status"]
            primary["formal_evidence_status"] = primary["region"].map(primary_status)
            floor["formal_evidence_status"] = floor["region"].map(floor_status)
        if len(primary) != expected_rows or len(floor) != expected_rows:
            raise RuntimeError(
                f"Stage-2 completeness gate failed for {diagnostic}: "
                f"primary={len(primary)} floor015={len(floor)} expected={expected_rows}"
            )
        paired_tables.append(_pair_evidence_table(diagnostic, primary, floor, keys))
    raw = pd.concat(paired_tables, ignore_index=True, sort=False)
    summary = _stage2_region_summary(raw)
    summary = pd.concat(
        [summary, _wrong_region_four_of_four_summary(primary_output_root, stage_dir)],
        ignore_index=True,
    )
    return raw, summary


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read persisted booleans without treating the string 'False' as true."""

    values = frame[column]
    if values.dtype == object:
        return values.astype(str).str.lower().eq("true")
    return values.astype(bool)


def _stage2_region_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize support-pattern preservation by focus region and diagnostic."""

    rows = []
    for diagnostic in STAGE2_TABLE_SPECS:
        sub = raw[raw["diagnostic_type"] == diagnostic].copy()
        region_col = "source_region" if diagnostic == "wrong_region" else "region"
        for region in FOCUS_REGIONS:
            group = sub[sub[region_col] == region].copy()
            if diagnostic == "location_buffer":
                status_cols = []
                for scenario in ("primary", "floor015"):
                    status = (
                        (pd.to_numeric(group[f"{scenario}_n_units"], errors="coerce") >= 3)
                        & (pd.to_numeric(group[f"{scenario}_p_FDR"], errors="coerce") < 0.05)
                        & _bool_series(group, f"{scenario}_ci_excludes_zero")
                        & (pd.to_numeric(group[f"{scenario}_standardized_mean_difference_zero"], errors="coerce").abs() >= 0.2)
                    )
                    status_cols.append((scenario, status))
            elif diagnostic == "lead_lag":
                status_cols = [
                    (scenario, _bool_series(group, f"{scenario}_significant_background_test"))
                    for scenario in ("primary", "floor015")
                ]
            elif diagnostic == "wrong_region":
                status_cols = [
                    (scenario, _bool_series(group, f"{scenario}_control_passed"))
                    for scenario in ("primary", "floor015")
                ]
            elif diagnostic == "year_block":
                status_cols = [
                    (
                        scenario,
                        group[f"{scenario}_block_bootstrap_status"].astype(str).eq("ok_event_level_year_resampling")
                        & (pd.to_numeric(group[f"{scenario}_ci_low"], errors="coerce") * pd.to_numeric(group[f"{scenario}_ci_high"], errors="coerce") > 0),
                    )
                    for scenario in ("primary", "floor015")
                ]
            elif diagnostic in {"cyclone", "ice_edge"}:
                status_cols = [
                    (
                        scenario,
                        group[f"{scenario}_formal_evidence_status"].astype(str).eq("supported"),
                    )
                    for scenario in ("primary", "floor015")
                ]
            support = {scenario: int(values.sum()) for scenario, values in status_cols}
            row_preserved = bool(np.array_equal(status_cols[0][1].to_numpy(), status_cols[1][1].to_numpy()))
            rows.append(
                {
                    "region": region,
                    "diagnostic_type": diagnostic,
                    "primary_test_rows": len(group),
                    "floor015_test_rows": len(group),
                    "primary_supported_rows": support["primary"],
                    "floor015_supported_rows": support["floor015"],
                    "support_status_preserved_row_by_row": row_preserved,
                    "manuscript_interpretation_preserved": row_preserved,
                    "status": "ROBUSTNESS_PASS" if row_preserved else "ROBUSTNESS_CHANGED_REQUIRES_MANUSCRIPT_REVISION",
                }
            )
    return pd.DataFrame(rows)


def _wrong_region_four_of_four_summary(primary_output_root: Path, stage_dir: Path) -> pd.DataFrame:
    """Compare the formal 4-of-4 wrong-region gate separately from raw tests."""

    primary = pd.read_csv(primary_output_root / "controls/wrong_region_control_pair_summary.csv")
    floor = pd.read_csv(stage_dir / "controls/wrong_region_control_pair_summary.csv")
    keys = ["source_region", "buffer_km", "variable"]
    paired = primary.merge(floor, on=keys, suffixes=("_primary", "_floor015"), validate="one_to_one")
    rows = []
    for region in FOCUS_REGIONS:
        group = paired[paired["source_region"] == region]
        primary_status = _bool_series(group, "all_wrong_regions_passed_primary")
        floor_status = _bool_series(group, "all_wrong_regions_passed_floor015")
        preserved = bool(np.array_equal(primary_status.to_numpy(), floor_status.to_numpy()))
        rows.append(
            {
                "region": region,
                "diagnostic_type": "wrong_region_4of4_gate",
                "primary_test_rows": len(group),
                "floor015_test_rows": len(group),
                "primary_supported_rows": int(primary_status.sum()),
                "floor015_supported_rows": int(floor_status.sum()),
                "support_status_preserved_row_by_row": preserved,
                "manuscript_interpretation_preserved": preserved,
                "status": "ROBUSTNESS_PASS" if preserved else "ROBUSTNESS_CHANGED_REQUIRES_MANUSCRIPT_REVISION",
            }
        )
    return pd.DataFrame(rows)


def _build_floor015_cdr_fields(
    pan_events: pd.DataFrame,
    field_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Track-S fields with only the prespecified start-SIC floor changed."""

    area, x, y = load_cell_area_km2()
    ocean = valid_ocean_mask()
    field_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    components: list[dict[str, object]] = []
    for event in pan_events.to_dict("records"):
        pan_id = str(event["pan_event_id"])
        event_date = pd.Timestamp(event["date"]).normalize()
        start_date = event_date - pd.Timedelta(days=5)
        start, _, _ = read_sic(start_date)
        end, _, _ = read_sic(event_date)
        valid = ocean & np.isfinite(start) & np.isfinite(end) & (start >= 0.15)
        signed = np.where(valid, end - start, np.nan)
        loss = np.where(np.isfinite(signed), np.maximum(0.0, -signed), np.nan)
        gain = np.where(np.isfinite(signed), np.maximum(0.0, signed), np.nan)
        signed_store = signed.astype(np.float32)
        loss_store = loss.astype(np.float32)
        gain_store = gain.astype(np.float32)
        signed_sum = signed_store.astype(float)
        loss_sum = loss_store.astype(float)
        gain_sum = gain_store.astype(float)
        comps = detect_connected_components(
            signed,
            valid,
            -0.1,
            4,
            binary_closing=True,
            strict_min_cells=True,
        )
        labels = np.zeros_like(signed, dtype=np.int32)
        resolved = 0.0
        for component_id, component in enumerate(comps, start=1):
            labels[component.mask] = component_id
            integrated = float(
                np.nansum(np.where(component.mask, loss_sum * area, 0.0), dtype=np.float64)
            )
            resolved += integrated
            components.append(
                {
                    "pan_event_id": pan_id,
                    "component_id": component_id,
                    "cell_count": component.cell_count,
                    "integrated_sic_loss_km2eq": integrated,
                }
            )
        total_loss = float(np.nansum(loss_sum * area, dtype=np.float64))
        total_gain = float(np.nansum(gain_sum * area, dtype=np.float64))
        signed_integral = float(np.nansum(signed_sum * area, dtype=np.float64))
        budget_error = total_gain - total_loss - signed_integral
        xr.Dataset(
            {
                "cdr_signed_sic_change": (("y", "x"), signed_store),
                "cdr_sic_loss": (("y", "x"), loss_store),
                "cdr_sic_gain": (("y", "x"), gain_store),
                "loss_component_id": (("y", "x"), labels),
            },
            coords={"x": x, "y": y},
            attrs={
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "robustness_definition": "valid_ocean AND finite(start,end) AND start_SIC>=0.15; delta_SIC<=-0.10; strict cell_count>4",
            },
        ).to_netcdf(field_dir / f"{pan_id}.nc")
        summaries.append(
            {
                "pan_event_id": pan_id,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "cdr_window_start": start_date.strftime("%Y-%m-%d"),
                "cdr_window_end": event_date.strftime("%Y-%m-%d"),
                "cdr_gross_sic_loss_km2eq": total_loss,
                "cdr_gross_sic_gain_km2eq": total_gain,
                "cdr_signed_sic_change_km2eq": signed_integral,
                "cdr_budget_closure_error_km2eq": budget_error,
                "cdr_budget_closure_status": "PASS" if np.isclose(budget_error, 0, rtol=1e-12, atol=1e-6) else "FAIL",
                "resolved_sic_loss_km2eq": resolved,
                "component_resolved_sic_loss_fraction": resolved / total_loss if total_loss > 0 else np.nan,
                "n_loss_components": len(comps),
            }
        )
    summary = pd.DataFrame(summaries)
    component_table = pd.DataFrame(components)
    if len(summary) != 99 or not summary["cdr_budget_closure_status"].eq("PASS").all():
        raise RuntimeError("Floor-0.15 Track-S field gate failed")
    return summary, component_table


def _stage3_overlap_and_composition(
    root: Path,
    output_dir: Path,
    catalogs: dict[str, dict[str, pd.DataFrame]],
    local_cells_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recalculate C012/C013/C014 in an isolated Stage-3 data root."""

    pan_events = load_pan_events()
    work = output_dir / "stage3_work"
    processed = work / "processed"
    cdr_dir = processed / "panarctic_loss_fields_floor015"
    partition_dir = processed / "panarctic_class_partition_fields"
    detector_dir = root / "data/processed/panarctic_detector_extent_fields"
    stage_out = work / "outputs"
    stage_out.mkdir(parents=True, exist_ok=True)
    floor = catalogs["0.15"]
    patch_index = pd.read_csv(local_cells_dir / "patch_cell_index.csv")
    cdr_summary, components = _build_floor015_cdr_fields(pan_events, cdr_dir)
    cdr_summary.to_csv(stage_out / "floor015_trackS_field_summary.csv", index=False)
    components.to_csv(stage_out / "floor015_trackS_components.csv", index=False)
    event, relations, pairs, alignment = run_overlap(
        pan_events,
        floor["membership"],
        patch_index,
        floor["broad"],
        floor["severe"],
        floor["major_severe"],
        local_cells_dir,
        cdr_dir,
        detector_dir,
    )
    event.to_csv(stage_out / "floor015_overlap_event_level.csv", index=False)
    relations.to_csv(stage_out / "floor015_overlap_relations.csv", index=False)
    pairs.to_csv(stage_out / "floor015_overlap_component_pairs.csv", index=False)
    alignment.to_csv(stage_out / "floor015_overlap_alignment_audit.csv", index=False)
    if len(event) != 99 or alignment["future_patch_excluded_count"].lt(0).any():
        raise RuntimeError("Floor-0.15 overlap/no-look-ahead gate failed")
    primary_relations = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_local_event_overlap.csv"
    )
    primary_pairs = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_local_component_overlap_pairs.csv"
    )
    overlap_summary = pd.DataFrame(
        [
            {
                "metric": "pan_event_x_local_event_relations",
                "primary_value": len(primary_relations),
                "floor015_value": len(relations),
                "absolute_change": len(relations) - len(primary_relations),
                "direction_preserved": len(relations) > 0,
            },
            {
                "metric": "component_overlap_pairs",
                "primary_value": len(primary_pairs),
                "floor015_value": len(pairs),
                "absolute_change": len(pairs) - len(primary_pairs),
                "direction_preserved": len(pairs) > 0,
            },
        ]
    )

    partition_dir.mkdir(parents=True, exist_ok=True)
    build_class_partitions(
        pan_events,
        floor["membership"],
        patch_index,
        floor["broad"],
        floor["severe"],
        floor["major_severe"],
        local_cells_dir,
        detector_dir,
        partition_dir,
        stage_out,
    )
    detector_budget = detector_contribution_budget(pan_events, detector_dir, partition_dir, stage_out)
    cdr_budget = cdr_contribution_budget(pan_events, cdr_dir, partition_dir, stage_out)
    primary_detector = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_detector_contribution_budget.csv"
    )
    primary_cdr = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_cdr_contribution_budget.csv"
    )
    track_d = _pair_budget(primary_detector, detector_budget, "detector")
    track_s = _pair_budget(primary_cdr, cdr_budget, "cdr")
    if not detector_budget["detector_partition_closure_status"].eq("PASS").all():
        raise RuntimeError("Floor-0.15 Track-D partition closure failed")
    if not cdr_budget["cdr_partition_closure_status"].eq("PASS").all():
        raise RuntimeError("Floor-0.15 Track-S partition closure failed")
    return overlap_summary, track_d, track_s


def _pair_budget(primary: pd.DataFrame, floor: pd.DataFrame, track: str) -> pd.DataFrame:
    """Pair all 99 event budgets and add manuscript composition summaries."""

    paired = primary.merge(floor, on=["pan_event_id", "event_date"], suffixes=("_primary", "_floor015"), validate="one_to_one")
    if len(paired) != 99:
        raise RuntimeError(f"{track} budget pairing lost pan events")
    paired["record_type"] = "event"
    fraction_columns = [
        column for column in primary.columns
        if "loss_fraction" in column and column in floor.columns
    ]
    summary = {"record_type": "median_summary", "pan_event_id": "ALL", "event_date": ""}
    for column in fraction_columns:
        summary[f"{column}_primary"] = float(pd.to_numeric(primary[column], errors="coerce").median())
        summary[f"{column}_floor015"] = float(pd.to_numeric(floor[column], errors="coerce").median())
        summary[f"{column}_absolute_change"] = summary[f"{column}_floor015"] - summary[f"{column}_primary"]
    return pd.concat([paired, pd.DataFrame([summary])], ignore_index=True, sort=False)


def _run_matched_controls(
    root: Path,
    output_dir: Path,
    catalogs: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-evaluate fixed case/control dates after the floor-0.15 perturbation."""

    pan_events = load_pan_events()
    floor = catalogs["0.15"]
    control_summary = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_matched_control_summary.csv"
    )
    control_anchors = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_matched_control_anchors.csv"
    )
    processed = output_dir / "stage3_work/processed"
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def floor_window_fields(start_date, end_date):
        key = (pd.Timestamp(start_date).strftime("%Y-%m-%d"), pd.Timestamp(end_date).strftime("%Y-%m-%d"))
        if key not in cache:
            start, _, _ = read_sic(pd.Timestamp(start_date))
            end, _, _ = read_sic(pd.Timestamp(end_date))
            valid = valid_ocean_mask() & np.isfinite(start) & np.isfinite(end) & (start >= 0.15)
            signed = np.where(valid, end - start, np.nan)
            cache[key] = (
                signed,
                np.where(np.isfinite(signed), np.maximum(0.0, -signed), np.nan),
                np.where(np.isfinite(signed), np.maximum(0.0, signed), np.nan),
            )
        return cache[key]

    old_data_root = os.environ.get("VRILE_STAGE3_DATA_ROOT")
    old_window = stage3_controls.cdr_window_fields
    os.environ["VRILE_STAGE3_DATA_ROOT"] = str(processed)
    stage3_controls.cdr_window_fields = floor_window_fields
    try:
        metrics, differences, regional, synchronization = stage3_controls.compute_control_outputs(
            pan_events,
            control_summary,
            control_anchors,
            floor["membership"],
            floor["major_severe"],
            root=root,
        )
    finally:
        stage3_controls.cdr_window_fields = old_window
        if old_data_root is None:
            os.environ.pop("VRILE_STAGE3_DATA_ROOT", None)
        else:
            os.environ["VRILE_STAGE3_DATA_ROOT"] = old_data_root
    work = output_dir / "stage3_work/outputs"
    metrics.to_csv(work / "floor015_control_window_metrics.csv", index=False)
    differences.to_csv(work / "floor015_control_event_differences.csv", index=False)
    regional.to_csv(work / "floor015_control_regional_diagnostics.csv", index=False)
    synchronization.to_csv(work / "floor015_control_synchronization.csv", index=False)
    tests = stage3_controls.infer_control_tests(
        differences,
        seed=SEED_STAGE3,
        n_bootstrap=BOOTSTRAP_RESAMPLES,
        n_signflip=SIGNFLIP_RESAMPLES,
    )
    sensitivity_events, sensitivity = stage3_controls.control_count_sensitivity(
        metrics,
        tests,
        counts=(10, 15, 20),
        seed=SEED_STAGE3,
        n_bootstrap=BOOTSTRAP_RESAMPLES,
        n_signflip=SIGNFLIP_RESAMPLES,
        primary_differences=differences,
    )
    year_block = stage3_controls.year_block_bootstrap_confirmation(
        differences,
        tests,
        seed=SEED_STAGE3,
        n_bootstrap=BOOTSTRAP_RESAMPLES,
    )
    tests.to_csv(work / "floor015_control_test_summary.csv", index=False)
    sensitivity_events.to_csv(work / "floor015_control_count_event_differences.csv", index=False)
    sensitivity.to_csv(work / "floor015_control_count_sensitivity.csv", index=False)
    year_block.to_csv(work / "floor015_control_year_block.csv", index=False)
    primary_tests = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_control_test_summary.csv"
    )
    primary_sensitivity = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_control_count_sensitivity_summary.csv"
    )
    primary_year = pd.read_csv(
        root / "outputs/panarctic_local_contribution/panarctic_primary_supported_year_block_bootstrap.csv"
    )
    paired_tests = _pair_named_table(primary_tests, tests, ["family", "metric", "alternative"], "primary_test")
    paired_sensitivity = _pair_named_table(
        primary_sensitivity,
        sensitivity,
        ["family", "metric", "alternative", "control_count"],
        "control_count_sensitivity",
    )
    paired_year = _pair_named_table(
        primary_year,
        year_block,
        ["metric"],
        "year_block_confirmation",
        require_exact_set=False,
    )
    comparison = pd.concat([paired_tests, paired_sensitivity, paired_year], ignore_index=True, sort=False)
    return comparison, tests, sensitivity


def _pair_named_table(
    primary: pd.DataFrame,
    floor: pd.DataFrame,
    keys: list[str],
    record_type: str,
    require_exact_set: bool = True,
) -> pd.DataFrame:
    """Pair two formal tables on their authoritative test key."""

    common = [column for column in primary.columns if column in floor.columns and column not in keys]
    paired = primary[keys + common].rename(
        columns={column: f"primary_{column}" for column in common}
    ).merge(
        floor[keys + common].rename(columns={column: f"floor015_{column}" for column in common}),
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if require_exact_set and not paired["_merge"].eq("both").all():
        raise RuntimeError(f"Matched-control table mismatch: {record_type}")
    paired = paired.rename(columns={"_merge": "row_set_status"})
    paired.insert(0, "record_type", record_type)
    if "primary_status" in paired and "floor015_status" in paired:
        paired["support_status_preserved"] = paired["primary_status"].astype(str).eq(
            paired["floor015_status"].astype(str)
        )
    if "primary_mean_difference" in paired and "floor015_mean_difference" in paired:
        paired["direction_preserved"] = np.sign(
            pd.to_numeric(paired["primary_mean_difference"], errors="coerce")
        ).eq(np.sign(pd.to_numeric(paired["floor015_mean_difference"], errors="coerce")))
    if "primary_year_block_confirmed" in paired and "floor015_year_block_confirmed" in paired:
        paired["support_status_preserved"] = _bool_series(
            paired, "primary_year_block_confirmed"
        ).eq(_bool_series(paired, "floor015_year_block_confirmed"))
    return paired


def _correct_propagation(root: Path) -> pd.DataFrame:
    """Correct C013/C014/C019 while preserving the historical table untouched."""

    table = pd.read_csv(root / "outputs/robustness_lowsic_v1/claim_propagation_table.csv")
    ledger = pd.read_csv(root / "VRILE_PAPER_EVIDENCE_PACKAGE/08_claim_evidence_ledger.csv")
    topics = ledger.set_index("claim_id")["topic"].astype(str).to_dict()
    edits = {
        "C013": {
            "dependency": "frozen Track-D detector field + floor-specific local class partitions",
            "propagation_status": "INPUT_CHANGED_RECALC_REQUIRED_FOR_ROBUSTNESS",
            "reason": "Track-D extent is frozen, but broad/severe/major_severe class partitions depend on the floor-sensitive local catalogs.",
        },
        "C012": {
            "dependency": "floor-specific Track-S components + floor-specific local catalog overlap relations",
            "propagation_status": "INPUT_CHANGED_RECALC_REQUIRED_FOR_ROBUSTNESS",
            "reason": "Both inputs change under the floor; the required floor-0.15 overlap rerun was completed in this closure.",
        },
        "C014": {
            "dependency": "floor-specific Track-S loss field + floor-specific local class partitions",
            "propagation_status": "INPUT_CHANGED_RECALC_REQUIRED_FOR_ROBUSTNESS",
            "reason": "Both the Track-S eligible loss field and local class partitions change under start-SIC floor 0.15.",
        },
        "C019": {
            "dependency": "reverse matched-control compensation analysis",
            "propagation_status": "NOT_RERUN_EXCLUDED_FROM_MAIN_ROBUSTNESS_BASELINE",
            "reason": "Final Claim Ledger defines C019 as reverse compensation; it is secondary and was not rerun in this main-claim closure.",
        },
        "C020": {
            "dependency": "ice-motion kinematic budget",
            "propagation_status": "METHOD_OR_DATA_GATED",
            "reason": "Final Claim Ledger defines C020 as the ice-motion DATA_GATED_NOT_RUN limitation.",
        },
    }
    for claim_id, values in edits.items():
        selector = table["claim_id"].astype(str).eq(claim_id)
        if selector.sum() != 1:
            raise RuntimeError(f"Propagation mapping is not unique for {claim_id}")
        for column, value in values.items():
            table.loc[selector, column] = value
    table["final_claim_ledger_topic"] = table["claim_id"].map(topics)
    claim_rows = table["claim_id"].astype(str).str.fullmatch(r"C\d{3}")
    if table.loc[claim_rows, "final_claim_ledger_topic"].isna().any():
        raise RuntimeError("Propagation table contains a claim absent from Final Claim Ledger")
    if topics.get("C019") != "reverse compensation" or topics.get("C020") != "ice-motion budget":
        raise RuntimeError("Final Claim Ledger C019/C020 mapping differs from the audited contract")
    return table


def _claim_comparison(
    matching: pd.DataFrame,
    stage2_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    track_d: pd.DataFrame,
    track_s: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Classify only direction/status/interpretation preservation, without new cutoffs."""

    rows: list[dict[str, object]] = []
    for claim_id in ("C003", "C004", "C022"):
        sub = matching[matching["claim_id"] == claim_id]
        preserved = bool(sub["direction_preserved"].all())
        rows.append(_claim_row(claim_id, preserved, "bidirectional matching direction remains non-zero under all three defined rules"))
    for region in FOCUS_REGIONS:
        sub = stage2_summary[stage2_summary["region"] == region]
        preserved = bool(sub["support_status_preserved_row_by_row"].all())
        rows.append(
            _claim_row(
                f"Stage2:{region}",
                preserved,
                "all six formal raw diagnostic support patterns compared row-by-row",
            )
        )
    rows.append(
        _claim_row(
            "C012",
            bool(overlap["direction_preserved"].all()),
            "pan/local relations and component-overlap pairs remain present; no-look-ahead gate passed",
        )
    )
    for claim_id, table, prefix in (("C013", track_d, "detector"), ("C014", track_s, "cdr")):
        summary = table[table["record_type"] == "median_summary"].iloc[0]
        major_column = f"major_{prefix}_extent_loss_fraction" if prefix == "detector" else "major_cdr_sic_loss_fraction"
        residual_column = f"residual_{prefix}_extent_loss_fraction" if prefix == "detector" else "residual_cdr_sic_loss_fraction"
        if prefix == "detector":
            major_column = "major_detector_extent_loss_fraction"
            residual_column = "residual_detector_extent_loss_fraction"
        preserved = bool(
            np.sign(summary[f"{major_column}_primary"]) == np.sign(summary[f"{major_column}_floor015"])
            and np.sign(summary[f"{residual_column}_primary"]) == np.sign(summary[f"{residual_column}_floor015"])
        )
        rows.append(_claim_row(claim_id, preserved, "descriptive major/residual composition directions preserved and all partition closures passed"))

    test_rows = controls[controls["record_type"] == "primary_test"].set_index("metric")
    sensitivity = controls[controls["record_type"] == "control_count_sensitivity"].copy()
    c015_test = test_rows.loc["residual_compensation_ratio"]
    c015_sens = sensitivity[sensitivity["metric"] == "residual_compensation_ratio"]
    c015_year = controls[
        (controls["record_type"] == "year_block_confirmation")
        & (controls["metric"] == "residual_compensation_ratio")
    ]
    year_preserved = bool(
        len(c015_year) == 1
        and str(c015_year.iloc[0].get("row_set_status", "")) == "both"
        and bool(c015_year.iloc[0].get("support_status_preserved", False))
    )
    c015_preserved = bool(
        c015_test.get("support_status_preserved", False)
        and c015_test.get("direction_preserved", False)
        and _bool_series(c015_sens, "support_status_preserved").all()
        and _bool_series(c015_sens, "direction_preserved").all()
        and year_preserved
    )
    rows.append(_claim_row("C015", c015_preserved, "fixed-control residual compensation direction/status, 10/15/20 sensitivity, and year-block confirmation compared"))
    c016_metrics = ["residual_sic_loss_km2eq", "other_region_sic_loss_km2eq"]
    c016_test = test_rows.loc[c016_metrics]
    c016_sens = sensitivity[sensitivity["metric"].isin(c016_metrics)]
    c016_preserved = bool(
        _bool_series(c016_test, "support_status_preserved").all()
        and _bool_series(c016_test, "direction_preserved").all()
        and _bool_series(c016_sens, "support_status_preserved").all()
        and _bool_series(c016_sens, "direction_preserved").all()
    )
    rows.append(_claim_row("C016", c016_preserved, "fixed-control co-loss direction/status and 10/15/20 control-count pattern compared"))
    rows.extend(
        [
            {
                "claim_id": "C017",
                "direction_preserved": "NOT_TESTED",
                "support_status_preserved": "NOT_APPLICABLE",
                "scientific_interpretation_preserved": "NOT_TESTED",
                "robustness_status": "NOT_RERUN_SECONDARY_DESCRIPTIVE",
                "reason": "Reverse descriptive rerun was optional and excluded from this main-claim closure.",
            },
            {
                "claim_id": "C018",
                "direction_preserved": "NOT_TESTED",
                "support_status_preserved": "HOLD",
                "scientific_interpretation_preserved": "HOLD",
                "robustness_status": "HOLD_PENDING_STATISTICAL_REVIEW",
                "reason": "Reverse inference remains HOLD; no dependence-aware inference was introduced.",
            },
            {
                "claim_id": "C019",
                "direction_preserved": "NOT_TESTED",
                "support_status_preserved": "NOT_TESTED",
                "scientific_interpretation_preserved": "NOT_TESTED",
                "robustness_status": "NOT_RERUN_EXCLUDED_FROM_MAIN_ROBUSTNESS_BASELINE",
                "reason": "C019 is reverse compensation, secondary and not central.",
            },
            {
                "claim_id": "C020",
                "direction_preserved": "NOT_APPLICABLE",
                "support_status_preserved": "DATA_GATED_NOT_RUN",
                "scientific_interpretation_preserved": True,
                "robustness_status": "METHOD_OR_DATA_GATED",
                "reason": "Ice-motion kinematic budget remains DATA_GATED_NOT_RUN.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _claim_row(claim_id: str, preserved: bool, reason: str) -> dict[str, object]:
    """Build one main-claim direction/status/interpretation decision row."""

    return {
        "claim_id": claim_id,
        "direction_preserved": preserved,
        "support_status_preserved": preserved,
        "scientific_interpretation_preserved": preserved,
        "robustness_status": "ROBUSTNESS_PASS" if preserved else "ROBUSTNESS_CHANGED_REQUIRES_MANUSCRIPT_REVISION",
        "reason": reason,
    }


def _write_manifest(output_dir: Path) -> pd.DataFrame:
    """Hash every closure artifact except the self-referential manifest."""

    manifest_path = output_dir / "DOWNSTREAM_FLOOR015_MANIFEST.csv"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest["path"].duplicated().any():
        raise RuntimeError("Downstream manifest contains duplicate paths")
    manifest.to_csv(manifest_path, index=False)
    return manifest


def _write_report(
    output_dir: Path,
    equivalence: pd.DataFrame,
    catalogs: pd.DataFrame,
    matching: pd.DataFrame,
    stage2: pd.DataFrame,
    overlap: pd.DataFrame,
    track_d: pd.DataFrame,
    track_s: pd.DataFrame,
    controls: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    """Write a concise evidence-backed closure report from generated tables."""

    gate = equivalence[equivalence["record_type"] == "global_gate_summary"].iloc[0]
    count_rows = catalogs[catalogs["region"] == "all"].set_index("severity_level")
    lines = [
        "# VRILE start-SIC floor=0.15 downstream robustness closure",
        "",
        "## Execution contract",
        "",
        "Only the start-SIC floor was changed. Local catalogs were fully reconstructed; Track S used `start SIC >= 0.15` with the primary strict component rule `cell_count > 4`. Track D fields and pan-event case/control assignments were held fixed. No floor-0.30 downstream run was performed.",
        "",
        "## Local reconstruction identity gate",
        "",
        f"Status: **{gate['status']}**. Unmatched frozen events={int(gate.get('unmatched_frozen_events', 0))}; unmatched reconstructed events={int(gate.get('unmatched_reconstructed_events', 0))}; scientifically relevant field mismatches={int(gate['field_mismatch_count'])}.",
        "",
        "## Floor-0.15 local catalog",
        "",
    ]
    for level, row in count_rows.iterrows():
        lines.append(
            f"- {level}: {int(row.primary_event_count)} -> {int(row.floor015_event_count)} events (change {int(row.absolute_change):+d})."
        )
    lines.extend(["", "## Main claim results", ""])
    for row in claims.itertuples(index=False):
        lines.append(f"- {row.claim_id}: **{row.robustness_status}** — {row.reason}")
    lines.extend(["", "## Required questions", ""])
    for claim_id in ("C003", "C004", "C022", "C012", "C013", "C014", "C015", "C016", "C018"):
        row = claims[claims["claim_id"] == claim_id].iloc[0]
        lines.append(f"- {claim_id}: {row['robustness_status']}.")
    region_rows = claims[claims["claim_id"].astype(str).str.startswith("Stage2:")]
    lines.append(
        "- Five-region Stage 2 raw pattern: "
        + ("ROBUSTNESS_PASS." if region_rows["robustness_status"].eq("ROBUSTNESS_PASS").all() else "CHANGED; see floor015_stage2_region_summary.csv.")
    )
    changed = claims[claims["robustness_status"] == "ROBUSTNESS_CHANGED_REQUIRES_MANUSCRIPT_REVISION"]["claim_id"].astype(str).tolist()
    lines.append(f"- Claims requiring manuscript wording revision: {', '.join(changed) if changed else 'none'}.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- primary outputs modified = NO",
            "- primary scientific code modified = NO",
            "- full pipeline = NO",
            "- floor0.30 downstream rerun = NO",
            "- Final Paper Evidence Bundle modified = NO",
        ]
    )
    (output_dir / "DOWNSTREAM_FLOOR015_ROBUSTNESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2_floor015_only(
    root: Path,
    *,
    output_dir: Path,
    primary_output_root: Path,
    workers: int = 12,
) -> None:
    """Rebuild only the corrected floor-0.15 catalog and formal Stage-2 chain.

    This deliberately stops before every Stage-3 overlap, composition,
    control, or reverse-analysis function.  ``primary_output_root`` is an
    isolated verified view of the corrected primary Stage-2 evidence.
    """

    root = root.resolve()
    output_dir = output_dir.resolve()
    primary_output_root = primary_output_root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-2-only floor015 output: {output_dir}")
    output_dir.mkdir(parents=True)
    primary_lock_paths = [
        root / "outputs/local_vrile_enhanced/unique_local_events.csv",
        root / "outputs/local_vrile_severe/severe_unique_local_events.csv",
        root / "outputs/local_vrile_major_severe/major_severe_events_union.csv",
        root / "data/processed/local_event_cells/unique_event_patch_membership.csv",
    ]
    primary_lock = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "universe": "CORRECTED_LIVE_LOCAL_CATALOG",
            }
            for path in primary_lock_paths
        ]
    )
    primary_lock.to_csv(output_dir / "stage2_primary_input_hash_lock.csv", index=False)
    local_cells = output_dir / "stage2_work/processed/local_event_cells_floor015"
    catalogs, _ = reconstruct_local_catalogs(
        root,
        quick=False,
        floors=(0.15,),
        sparse_cell_dirs={"0.15": local_cells},
    )
    _enrich_catalogs(root, catalogs)
    floor = catalogs["0.15"]
    pd.DataFrame(
        [
            {
                "sic_floor": "0.15",
                "severity_level": level,
                "region": region,
                "event_count": len(frame if region == "all" else frame[frame["dominant_region"] == region]),
                "canonical_longitude_geometry": True,
            }
            for level in ("broad", "severe", "major_severe")
            for frame in [floor[level]]
            for region in ["all", *sorted(frame["dominant_region"].dropna().astype(str).unique())]
        ]
    ).to_csv(output_dir / "floor015_local_catalog_summary.csv", index=False)
    floor["patches"].groupby("longitude_geometry_class", sort=True).size().rename(
        "patch_count"
    ).reset_index().to_csv(
        output_dir / "floor015_patch_geometry_classification.csv", index=False
    )
    stage2_module = _load_stage2_module(root)
    paths = _write_catalog_inputs(output_dir / "stage2_work", catalogs, stage2_module)
    raw, summary = _run_stage2_diagnostics(
        stage2_module,
        root,
        output_dir,
        paths,
        workers,
        primary_output_root=primary_output_root,
    )
    raw.to_csv(output_dir / "floor015_stage2_raw_evidence.csv", index=False)
    summary.to_csv(output_dir / "floor015_stage2_region_summary.csv", index=False)
    metadata = {
        "scope": "CORRECTED_FLOOR015_FORMAL_STAGE2_ONLY",
        "canonical_longitude_geometry_reached": True,
        "primary_corrected_catalog_hash_lock": "stage2_primary_input_hash_lock.csv",
        "floor_none_raw_redetection": False,
        "stage3_executed": False,
        "raw_rows": int(len(raw)),
        "region_summary_rows": int(len(summary)),
        "primary_output_root": str(primary_output_root),
        "output_dir": str(output_dir),
    }
    (output_dir / "stage2_floor015_scope_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"STAGE2_FLOOR015_ONLY_PASS output={output_dir} rows={len(raw)}")


def run_downstream_floor015(root: Path, *, workers: int = 12) -> None:
    """Execute all gates and atomically publish the isolated closure directory."""

    root = root.resolve()
    final_dir = root / OUTPUT_REL
    transaction = root / "outputs/robustness_lowsic_v1" / f"_downstream_floor015_transaction_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    transaction.mkdir(parents=True, exist_ok=False)
    before = _primary_hash_snapshot(root)
    before.to_csv(transaction / "primary_input_hashes_before.csv", index=False)
    local_cells = transaction / "stage3_work/processed/local_event_cells_floor015"
    print("GATE_START local_floor_none_reconstruction_identity")
    catalogs, _ = reconstruct_local_catalogs(
        root,
        quick=False,
        floors=(None, 0.15),
        sparse_cell_dirs={"0.15": local_cells},
    )
    equivalence = _equivalence_gate(root, catalogs)
    equivalence.to_csv(transaction / "local_primary_reconstruction_equivalence.csv", index=False)
    gate = equivalence[equivalence["record_type"] == "global_gate_summary"].iloc[0]
    if gate["status"] != "PASS":
        raise RuntimeError("STOP_LOCAL_RECONSTRUCTION_NON_EQUIVALENCE")
    print("GATE_PASS local_floor_none_reconstruction_identity")
    _enrich_catalogs(root, catalogs)
    catalog_summary = _catalog_summary(catalogs)
    catalog_summary.to_csv(transaction / "floor015_local_catalog_summary.csv", index=False)
    stage2_module = _load_stage2_module(root)
    paths = _write_catalog_inputs(transaction / "stage2_work", catalogs, stage2_module)
    matching = _bidirectional_matching(stage2_module, root, catalogs)
    matching.to_csv(transaction / "floor015_bidirectional_matching.csv", index=False)
    print("STAGE2_FLOOR015_START diagnostics=location-buffer,lead-lag,wrong-region,year-block,cyclone,ice-edge")
    stage2_raw, stage2_summary = _run_stage2_diagnostics(
        stage2_module, root, transaction, paths, workers
    )
    stage2_raw.to_csv(transaction / "floor015_stage2_raw_evidence.csv", index=False)
    stage2_summary.to_csv(transaction / "floor015_stage2_region_summary.csv", index=False)
    print("STAGE2_FLOOR015_DONE rows=" + str(len(stage2_raw)))
    overlap, track_d, track_s = _stage3_overlap_and_composition(
        root, transaction, catalogs, local_cells
    )
    overlap.to_csv(transaction / "floor015_overlap_summary.csv", index=False)
    track_d.to_csv(transaction / "floor015_trackD_composition.csv", index=False)
    track_s.to_csv(transaction / "floor015_trackS_composition.csv", index=False)
    controls, control_tests, control_sensitivity = _run_matched_controls(root, transaction, catalogs)
    controls.to_csv(transaction / "floor015_panevent_matched_control.csv", index=False)
    propagation = _correct_propagation(root)
    propagation.to_csv(transaction / "claim_propagation_table_corrected.csv", index=False)
    claims = _claim_comparison(
        matching, stage2_summary, overlap, track_d, track_s, controls
    )
    claims.to_csv(transaction / "floor015_claim_robustness_comparison.csv", index=False)
    _write_report(
        transaction,
        equivalence,
        catalog_summary,
        matching,
        stage2_summary,
        overlap,
        track_d,
        track_s,
        controls,
        claims,
    )
    verification = _verify_primary_snapshot(root, before)
    verification.to_csv(transaction / "primary_input_hash_verification.csv", index=False)
    manifest = _write_manifest(transaction)
    for row in manifest.itertuples(index=False):
        path = transaction / row.path
        if path.stat().st_size != int(row.size_bytes) or _sha256(path) != row.sha256:
            raise RuntimeError(f"Manifest verification failed: {row.path}")
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing closure directory: {final_dir}")
    transaction.replace(final_dir)
    print(f"DOWNSTREAM_FLOOR015_PASS output={final_dir}")
    print(f"manifest_files={len(manifest)}")
