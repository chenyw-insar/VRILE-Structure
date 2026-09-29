#!/usr/bin/env python
"""Audit-only corrected Stage 3 rebuild using the locked production methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

CANONICAL = Path(os.environ.get("VRILE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PIPE = CANONICAL / '.'
RAW_DATA_ROOT = Path(os.environ.get("VRILE_RAW_DATA_ROOT", PIPE / "data/raw")).expanduser().resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("VRILE_PROCESSED_DATA_ROOT", PIPE / "data/processed")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("VRILE_OUTPUT_ROOT", PIPE / "outputs")).expanduser().resolve()
sys.path.insert(0, str(PIPE / "src"))

from vrile.stage3.budget import (  # noqa: E402
    build_class_partitions,
    cdr_contribution_budget,
    detector_contribution_budget,
    region_budget_accounting_audit,
    region_budgets,
)
import vrile.stage3.controls as controls_module  # noqa: E402
from vrile.stage3.controls import (  # noqa: E402
    compute_control_outputs,
    control_count_sensitivity,
    infer_control_tests,
    matching_quality_audit,
    matching_reconstruction_audit,
    reconstruct_matching_independently,
    selected_control_anchors,
    validate_control_gates,
    year_block_bootstrap_confirmation,
)
from vrile.stage3.grid import load_pan_events  # noqa: E402
from vrile.stage3.overlap import run_overlap  # noqa: E402
from vrile.stage3.reverse import (  # noqa: E402
    classify_major_events,
    compute_reverse_window_metrics,
    control_count_robustness,
    infer_reverse_tests,
    reverse_differences,
    select_reverse_controls,
    year_block_confirmation,
)

SEED_CONTROL = 20260709
SEED_REVERSE = 20260710
RESAMPLES = 10_000


# This is an I/O cache only. The production reader and all scientific window
# calculations remain unchanged; arrays are never mutated by the callers.
_production_read_sic = controls_module.read_sic


@lru_cache(maxsize=None)
def _cached_read_sic_iso(date_iso: str):
    return _production_read_sic(pd.Timestamp(date_iso))


def _cached_read_sic(date):
    return _cached_read_sic_iso(pd.Timestamp(date).normalize().strftime("%Y-%m-%d"))


controls_module.read_sic = _cached_read_sic


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != CANONICAL.resolve():
        raise SystemExit(f"HOLD_CANONICAL_PROJECT_ROOT: cwd={Path.cwd()}")
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}")
    products = out / "products"
    validation = out / "validation"
    processed = out / "data" / "processed"
    partition_dir = processed / "panarctic_class_partition_fields"
    products.mkdir(parents=True)
    validation.mkdir(parents=True)
    os.environ["VRILE_STAGE3_DATA_ROOT"] = str(processed)

    pan_path = OUTPUT_ROOT / "reproduce_sie/vrile_events_unique_both_jja_5p.csv"
    # RC4.6 generation is deliberately independent of prior evidence products.
    # A frozen source-evidence manifest may be supplied only for a later
    # comparison run; ordinary raw-to-final generation relies on the current
    # run's complete 99-event identity/cardinality gate below.
    manifest_value = os.environ.get("VRILE_SOURCE_EVIDENCE_MANIFEST", "").strip()
    registered = None
    if manifest_value:
        manifest_path = Path(manifest_value).expanduser().resolve()
        manifest = pd.read_csv(manifest_path)
        registered = manifest.loc[
            manifest["path"].eq("outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv"), "sha256"
        ].iloc[0]
    pan = load_pan_events()
    if len(pan) != 99 or pan["pan_event_id"].nunique() != 99:
        raise SystemExit("HOLD_PANARCTIC_EVENT_INVARIANT_FAILURE")
    if registered is not None and sha256(pan_path) != registered:
        raise SystemExit("HOLD_PANARCTIC_EVENT_REFERENCE_HASH_FAILURE")

    cells_dir = PROCESSED_DATA_ROOT / "local_event_cells"
    paths = {
        "pan_events": pan_path,
        "membership": cells_dir / "unique_event_patch_membership.csv",
        "patch_index": cells_dir / "patch_cell_index.csv",
        "unique": OUTPUT_ROOT / "local_vrile_enhanced/unique_local_events.csv",
        "severe": OUTPUT_ROOT / "local_vrile_severe/severe_unique_local_events.csv",
        "major": OUTPUT_ROOT / "local_vrile_major_severe/major_severe_events_union.csv",
        "daily_sie": OUTPUT_ROOT / "reproduce_sie/daily_sie_processed.csv",
    }
    patch_index = pd.read_csv(paths["patch_index"])
    membership = pd.read_csv(paths["membership"])
    unique = pd.read_csv(paths["unique"])
    severe = pd.read_csv(paths["severe"])
    major = pd.read_csv(paths["major"])
    daily_sie = pd.read_csv(paths["daily_sie"])
    if (len(unique), len(severe), len(major)) != (9530, 2427, 554):
        raise SystemExit("Corrected local universe count mismatch")
    if any("archive" in str(p).lower() or "stale" in str(p).lower() for p in paths.values()):
        raise SystemExit("Stale/historical path used as active input")

    # Pan-only spatial composition products are invariant to local geometry. Copy
    # the current authoritative products into each isolated build and verify them.
    live = Path(os.environ.get("VRILE_STAGE3_PRODUCTS_ROOT", OUTPUT_ROOT / "panarctic_local_contribution")).expanduser().resolve()
    invariant_names = [
        "panarctic_loss_components.csv",
        "panarctic_loss_field_event_summary.csv",
        "panarctic_spatial_composition_metrics.csv",
        "cdr_component_raster_table_reconciliation.csv",
    ]
    for name in invariant_names:
        shutil.copy2(live / name, products / name)
    comp_qa = pd.read_csv(products / "cdr_component_raster_table_reconciliation.csv")
    metrics_invariant = pd.read_csv(products / "panarctic_spatial_composition_metrics.csv")
    if len(comp_qa) != 3463 or not comp_qa["component_status"].eq("PASS").all():
        raise SystemExit("Component raster/table closure failed")
    if len(metrics_invariant) != 99 or metrics_invariant["pan_event_id"].nunique() != 99:
        raise SystemExit("Spatial composition event closure failed")

    detector_dir = PROCESSED_DATA_ROOT / "panarctic_detector_extent_fields"
    cdr_dir = PROCESSED_DATA_ROOT / "panarctic_loss_fields"

    align, class_audit = build_class_partitions(
        pan, membership, patch_index, unique, severe, major,
        cells_dir, detector_dir, partition_dir, products,
    )
    det_budget = detector_contribution_budget(pan, detector_dir, partition_dir, products)
    cdr_budget = cdr_contribution_budget(pan, cdr_dir, partition_dir, products)
    det_region, cdr_region = region_budgets(pan, detector_dir, cdr_dir, products)
    accounting = region_budget_accounting_audit(det_budget, cdr_budget, det_region, cdr_region, products)

    event_df, relation_df, pair_df, alignment_df = run_overlap(
        pan, membership, patch_index, unique, severe, major,
        cells_dir, cdr_dir, detector_dir,
    )
    for frame, name in [
        (event_df, "panarctic_event_overlap_summary.csv"),
        (relation_df, "panarctic_local_event_overlap.csv"),
        (pair_df, "panarctic_local_component_overlap_pairs.csv"),
        (alignment_df, "time_alignment_audit.csv"),
    ]:
        write_csv(frame, products / name)

    membership_prepped = membership.copy()
    for col in ["date", "start_date"]:
        membership_prepped[col] = pd.to_datetime(membership_prepped[col]).dt.normalize()
    membership_prepped["unique_local_event_id"] = membership_prepped["unique_local_event_id"].astype(str)
    membership_prepped["object_id"] = membership_prepped["object_id"].astype(str)

    control_summary, control_anchors = selected_control_anchors(pan, daily_sie)
    window_metrics, differences, regional, synchronization = compute_control_outputs(
        pan, control_summary, control_anchors, membership_prepped, major, root=PIPE,
    )
    control_tests = infer_control_tests(
        differences, seed=SEED_CONTROL, n_bootstrap=RESAMPLES, n_signflip=RESAMPLES,
    )
    control_gates = validate_control_gates(pan, control_summary, control_anchors, regional, differences)
    recon_summary, recon_anchors = reconstruct_matching_independently(pan, daily_sie)
    recon_audit = matching_reconstruction_audit(
        control_summary, control_anchors, recon_summary, recon_anchors,
    )
    control_quality = matching_quality_audit(control_summary, control_anchors)
    sensitivity_diffs, sensitivity = control_count_sensitivity(
        window_metrics, control_tests, counts=(10, 15, 20), seed=SEED_CONTROL,
        n_bootstrap=RESAMPLES, n_signflip=RESAMPLES, primary_differences=differences,
    )
    control_yearblock = year_block_bootstrap_confirmation(
        differences, control_tests, seed=SEED_CONTROL, n_bootstrap=RESAMPLES,
    )
    control_outputs = {
        "panarctic_matched_control_summary.csv": control_summary,
        "panarctic_matched_control_anchors.csv": control_anchors,
        "panarctic_control_window_metrics.csv": window_metrics,
        "panarctic_coloss_event_differences.csv": differences,
        "panarctic_regional_coloss_matrix.csv": regional,
        "panarctic_synchronization_metrics.csv": synchronization,
        "panarctic_control_test_summary.csv": control_tests,
        "panarctic_coloss_control_gate_audit.csv": control_gates,
        "panarctic_control_matching_reconstruction_audit.csv": recon_audit,
        "panarctic_control_matching_quality_audit.csv": control_quality,
        "panarctic_control_count_sensitivity_event_differences.csv": sensitivity_diffs,
        "panarctic_control_count_sensitivity_summary.csv": sensitivity,
        "panarctic_primary_supported_year_block_bootstrap.csv": control_yearblock,
    }
    for name, frame in control_outputs.items():
        write_csv(frame, products / name)

    # The corrected promoted table preserves event_start but omits the legacy
    # redundant event_date alias. The accepted reverse-analysis anchor was
    # event_date == event_start; restore only that schema alias for the existing
    # production reverse functions.
    major_reverse = major.copy()
    major_reverse["event_date"] = major_reverse["event_start"]
    classification = classify_major_events(major_reverse, relation_df, pan, daily_sie)
    reverse_matching, reverse_anchors, reverse_quality = select_reverse_controls(classification)
    reverse_metrics = compute_reverse_window_metrics(
        classification, reverse_anchors, membership, patch_index, cells_dir,
    )
    reverse_diff = reverse_differences(reverse_metrics, reverse_anchors, reverse_matching)
    reverse_tests = infer_reverse_tests(
        reverse_diff, seed=SEED_REVERSE, n_bootstrap=RESAMPLES, n_signflip=RESAMPLES,
    )
    reverse_sens_diff, reverse_sens = control_count_robustness(
        reverse_metrics, reverse_anchors, reverse_matching, reverse_tests,
        seed=SEED_REVERSE, n_bootstrap=RESAMPLES, n_signflip=RESAMPLES,
    )
    reverse_yearblock = year_block_confirmation(
        reverse_diff, reverse_tests, seed=SEED_REVERSE, n_bootstrap=RESAMPLES,
    )
    reverse_outputs = {
        "reverse_major_event_classification.csv": classification,
        "reverse_matched_control_summary.csv": reverse_matching,
        "reverse_matched_control_anchors.csv": reverse_anchors,
        "reverse_control_matching_quality_audit.csv": reverse_quality,
        "reverse_window_metrics.csv": reverse_metrics,
        "reverse_event_differences.csv": reverse_diff,
        "reverse_test_summary.csv": reverse_tests,
        "reverse_control_count_sensitivity_event_differences.csv": reverse_sens_diff,
        "reverse_control_count_sensitivity_summary.csv": reverse_sens,
        "reverse_primary_supported_year_block_bootstrap.csv": reverse_yearblock,
    }
    for name, frame in reverse_outputs.items():
        write_csv(frame, products / name)

    # Independent event-specific audit of the patch rows actually represented
    # by each exported pan-event x ULE relation. Count closure against the
    # exported time_aligned_patch_count prevents a tautological membership-only
    # check from passing without tracing the consumed relation payload.
    no_lookahead_ok = True
    included_patch_event_relations = 0
    max_included_days_after_t = -10**9
    for pan_row in pan.itertuples(index=False):
        pan_id = str(pan_row.pan_event_id)
        event_date = pd.Timestamp(pan_row.date).normalize()
        window_start = event_date - pd.Timedelta(days=5)
        rel = relation_df[relation_df["pan_event_id"].eq(pan_id)]
        rel_counts = rel.set_index("unique_local_event_id")["time_aligned_patch_count"].astype(int).to_dict()
        included = membership_prepped[
            membership_prepped["unique_local_event_id"].isin(rel_counts)
            & (membership_prepped["start_date"] <= event_date)
            & (membership_prepped["date"] >= window_start)
            & (membership_prepped["date"] <= event_date)
        ]
        actual_counts = included.groupby("unique_local_event_id")["object_id"].nunique().to_dict()
        count_exact = set(actual_counts) == set(rel_counts) and all(actual_counts[uid] == rel_counts[uid] for uid in rel_counts)
        future_count = int((included["date"] > event_date).sum())
        event_max = int((included["date"].max() - event_date).days) if not included.empty else 0
        included_patch_event_relations += int(len(included))
        max_included_days_after_t = max(max_included_days_after_t, event_max)
        no_lookahead_ok = no_lookahead_ok and count_exact and future_count == 0 and event_max <= 0

    qa_rows = [
        ("canonical_root", str(CANONICAL.resolve()) == str(Path.cwd().resolve()), str(Path.cwd())),
        ("pan_event_count", len(pan) == 99 and pan["pan_event_id"].nunique() == 99, f"{len(pan)}"),
        (
            "pan_source_current_run_identity",
            registered is None or sha256(pan_path) == registered,
            "current_run_99_unique" if registered is None else sha256(pan_path),
        ),
        ("corrected_local_counts", (len(unique), len(severe), len(major)) == (9530, 2427, 554), f"{len(unique)}/{len(severe)}/{len(major)}"),
        ("no_stale_active_inputs", True, "0"),
        ("component_accounting_closure", len(comp_qa) == 3463 and comp_qa["component_status"].eq("PASS").all(), f"{comp_qa['component_status'].eq('PASS').sum()}/{len(comp_qa)}"),
        ("pan_overlap_closure", len(event_df) == 99 and event_df["pan_event_id"].nunique() == 99, f"{len(event_df)}"),
        ("no_future_patches_included", no_lookahead_ok, f"audited_relations={included_patch_event_relations}; max_days_after_T={max_included_days_after_t}"),
        ("class_partition_closure", len(class_audit) == 198 and class_audit["status"].eq("PASS").all(), f"{class_audit['status'].eq('PASS').sum()}/{len(class_audit)}"),
        ("contribution_budget_closure", det_budget["detector_partition_closure_status"].eq("PASS").all() and cdr_budget["cdr_partition_closure_status"].eq("PASS").all(), "detector+CDR"),
        ("regional_matrix_99x18", len(regional) == 1782 and regional["region_code"].nunique() == 18, f"{len(regional)} rows"),
        ("control_gates", control_gates["status"].eq("PASS").all(), f"{control_gates['status'].eq('PASS').sum()}/{len(control_gates)}"),
        ("control_reconstruction", recon_audit["status"].eq("PASS").all(), f"{recon_audit['status'].eq('PASS').sum()}/{len(recon_audit)}"),
        ("reverse_universe_closure", len(classification) == 554 and classification["unique_local_event_id"].nunique() == 554, f"{len(classification)}"),
        ("reverse_linkage_closure", int(classification["pan_linked"].sum()) + int((~classification["pan_linked"]).sum()) == 554, "linked+unlinked=554"),
        ("reverse_matching_closure", len(reverse_matching) == int(classification["pan_linked"].sum()), f"{len(reverse_matching)} cases"),
        ("duplicate_relation_keys", not relation_df.duplicated(["pan_event_id", "unique_local_event_id"]).any(), str(int(relation_df.duplicated(["pan_event_id", "unique_local_event_id"]).sum()))),
    ]
    qa = pd.DataFrame(
        [{"gate": gate, "status": "PASS" if ok else "FAIL", "detail": detail} for gate, ok, detail in qa_rows]
    )
    write_csv(qa, validation / "stage3_corrected_rebuild_qa.csv")
    if not qa["status"].eq("PASS").all():
        raise SystemExit("Stage3 corrected rebuild QA failure")

    def manifest_path_text(path: Path) -> str:
        try:
            return str(path.relative_to(CANONICAL))
        except ValueError:
            return str(path)

    input_manifest = pd.DataFrame([
        {"role": role, "path": manifest_path_text(path), "sha256": sha256(path), "active_input": "YES"}
        for role, path in paths.items()
    ])
    write_csv(input_manifest, validation / "input_hashes.csv")
    output_manifest = pd.DataFrame([
        {"path": str(path.relative_to(out)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(products.rglob("*")) if path.is_file()
    ])
    write_csv(output_manifest, validation / "output_hashes.csv")
    status_counts = {str(k): int(v) for k, v in reverse_matching["matching_status"].value_counts().items()}
    summary = {
        "active_project_root": str(CANONICAL),
        "pan_event_count": 99,
        "pan_source_sha256": sha256(pan_path),
        "corrected_local_universe": {"broad": 9530, "severe": 2427, "major": 554},
        "component_closure": {"pass": int(comp_qa["component_status"].eq("PASS").sum()), "rows": len(comp_qa)},
        "regional_matrix_rows": len(regional),
        "control_matching_status_counts": {str(k): int(v) for k, v in control_summary["matching_status"].value_counts().items()},
        "reverse": {
            "pan_linked": int(classification["pan_linked"].sum()),
            "non_pan_linked": int((~classification["pan_linked"]).sum()),
            "matching_status_counts": status_counts,
        },
        "status": "PASS",
    }
    (validation / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
