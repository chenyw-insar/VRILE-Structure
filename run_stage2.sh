#!/usr/bin/env bash
set -euo pipefail

# Public Stage 2 entrypoint. The complete final route is visible here: fresh
# M009 families, 630-row assembly, M010, two floor015 runs, and the required
# same-buffer stable-specificity finalizer.

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runner/entrypoint.sh
source "$SOURCE_ROOT/scripts/runner/entrypoint.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run_stage2.sh --run_dir DIR

Executes the accepted final Stage 2 topology:
  ERA5 cyclone derivative -> two independent M009 runs -> 630-row gates
  -> M010 770-to-40 closure -> two floor015 branches
  -> same-buffer stable-specificity finalizer -> parity checks.
EOF
  runner_common_options
  exit 0
fi

runner_prepare run_stage2.sh NO NO "$@"
require_marker 10_stage1_m007
require_science_confirmation
validate_source_isolation

[[ -z "$VRILE_ACCEPTED_COMPARISON_ROOT" ]] || \
  hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
unset VRILE_ACCEPTED_COMPARISON_ROOT
STAGE2_AUDIT="$VRILE_OUTPUT_ROOT/stage2_scope_audit"
export STAGE2_AUDIT

# The floor015 wrapper uses independently imported spawned workers. Each opens
# its own NetCDF/HDF5 files; scientific I/O is not shared between threads.
# Four workers passed the fixed two-repeat sample gate. This is scheduling only;
# both complete floor015 branches and all scientific settings remain unchanged.
FLOOR015_SAFE_WORKERS=4

# Part A: M009 -> 630 rows -> M010, repeated independently twice.
stage_trap 20_stage2_m010
require_absent "$STAGE2_AUDIT"
require_absent "$VRILE_PROCESSED_DATA_ROOT/cyclone_tracks/era5_derived"

cd "$VRILE_PROJECT_ROOT"
run_gmt_python 20_cyclone_tracks \
  "$PIPE/scripts/stage2_build_era5_cyclone_tracks.py" \
  --start-year 1989 --end-year 2025 --months jja \
  --input-glob "$VRILE_RAW_DATA_ROOT/era5/single_levels/*.nc" \
  --out_dir "$VRILE_PROCESSED_DATA_ROOT/cyclone_tracks/era5_derived" \
  --detection-region=-180,180,50,90 --overwrite

run_primary_stage2() {
  local run_name="$1"
  local run="$STAGE2_AUDIT/$run_name"
  mkdir -p "$run"

  # M009 accepted final subroutes. Optional historical subcommands are not run.
  run_gmt_python "21_${run_name}_clusters" \
    "$PIPE/scripts/stage2_experiment_core.py" clusters \
    --severe-events "$VRILE_OUTPUT_ROOT/local_vrile_severe/severe_unique_local_events.csv" \
    --major-events "$VRILE_OUTPUT_ROOT/local_vrile_major_severe/major_severe_events_union.csv" \
    --out_dir "$run/event_clusters"

  run_gmt_python "22_${run_name}_location_buffer" \
    "$PIPE/scripts/stage2_experiment_core.py" location_buffer \
    --major-events "$VRILE_OUTPUT_ROOT/local_vrile_major_severe/major_severe_events_union.csv" \
    --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
    --era5-single-glob "$VRILE_RAW_DATA_ROOT/era5/single_levels/*.nc" \
    --out_dir "$run/location_buffer_mechanism" \
    --parallel --backend process --workers "$VRILE_WORKERS"

  run_gmt_python "23_${run_name}_lead_lag" \
    "$PIPE/scripts/stage2_experiment_core.py" lead_lag \
    --major-events "$VRILE_OUTPUT_ROOT/local_vrile_major_severe/major_severe_events_union.csv" \
    --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
    --era5-single-glob "$VRILE_RAW_DATA_ROOT/era5/single_levels/*.nc" \
    --location-buffer-values "$run/location_buffer_mechanism/location_buffer_lag_values.csv" \
    --out_dir "$run/lead_lag"

  run_gmt_python "24_${run_name}_controls" \
    "$PIPE/scripts/stage2_experiment_core.py" controls \
    --major-events "$VRILE_OUTPUT_ROOT/local_vrile_major_severe/major_severe_events_union.csv" \
    --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
    --era5-single-glob "$VRILE_RAW_DATA_ROOT/era5/single_levels/*.nc" \
    --location-buffer-values "$run/location_buffer_mechanism/location_buffer_lag_values.csv" \
    --wrong-region-location-buffer-values "$run/location_buffer_mechanism/wrong_region_location_buffer_lag_values.csv" \
    --location-buffer-evidence "$run/location_buffer_mechanism/location_buffer_evidence_table.csv" \
    --out_dir "$run/controls" --parallel --backend process --workers "$VRILE_WORKERS"

  run_gmt_python "25_${run_name}_cyclone" \
    "$PIPE/scripts/stage2_experiment_core.py" cyclone \
    --major-clusters "$run/event_clusters/synoptic_event_clusters_major.csv" \
    --cyclone-tracks "$VRILE_PROCESSED_DATA_ROOT/cyclone_tracks/era5_derived/cyclone_track_points.csv" \
    --out_dir "$run/atmospheric_forcing/cyclone_proximity" \
    --parallel --backend thread --workers "$VRILE_WORKERS"

  run_gmt_python "26_${run_name}_ice_edge" \
    "$PIPE/scripts/stage2_experiment_core.py" ice_edge \
    --major-clusters "$run/event_clusters/synoptic_event_clusters_major.csv" \
    --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
    --era5-single-glob "$VRILE_RAW_DATA_ROOT/era5/single_levels/*.nc" \
    --out_dir "$run/atmospheric_forcing/ice_edge_relative_wind" \
    --parallel --backend process --workers "$VRILE_WORKERS"

  # Required promoted Stage-2 assembly precursor. Historical/hash parity is
  # handled by separate maintainer tools after generation, not this route.
  run_gmt_python "27_${run_name}_assemble_630" \
    "$VRILE_PROJECT_ROOT/scripts/assemble_formal_stage2.py" \
    --run_dir "$run"
  mkdir -p "$VRILE_OUTPUT_ROOT/stage2_formal_evidence"
  cp "$run/formal_evidence/formal_stage2_raw_evidence.csv" \
    "$VRILE_OUTPUT_ROOT/stage2_formal_evidence/formal_stage2_raw_evidence.csv"

  # Mandatory M010 770-row evidence and 40-cell regional summary.
  run_gmt_python "28_${run_name}_m010" \
    "$VRILE_PROJECT_ROOT/scripts/complete_stage2_scope.py" \
    --run_dir "$run" --base630 "$run" --out_dir "$run/scope_completion"
}

run_primary_stage2 run1
run_primary_stage2 run2
stage_success 20_stage2_m010

# Part B: independent floor015 branches and the same-buffer finalizer.
stage_trap 30_floor015_stable
require_dir "$STAGE2_AUDIT/run1/scope_completion"
require_dir "$STAGE2_AUDIT/run2/scope_completion"
require_absent "$STAGE2_AUDIT/floor015_run1"
require_absent "$STAGE2_AUDIT/floor015_run2"
require_absent "$STAGE2_AUDIT/stable_spatial_specificity_claim_gate.csv"

run_gmt_python 30_floor015_run1 \
  "$PIPE/scripts/stage2_robustness_floor015_downstream.py" \
  --project-root "$PIPE" --stage2-only \
  --out_dir "$STAGE2_AUDIT/floor015_run1" \
  --primary_output_dir "$STAGE2_AUDIT/run1" --workers "$FLOOR015_SAFE_WORKERS"
run_gmt_python 31_floor015_assemble_run1 \
  "$VRILE_PROJECT_ROOT/scripts/assemble_floor015_stage2.py" \
  --floor-run "$STAGE2_AUDIT/floor015_run1" --primary_output_dir "$STAGE2_AUDIT/run1"

run_gmt_python 32_floor015_run2 \
  "$PIPE/scripts/stage2_robustness_floor015_downstream.py" \
  --project-root "$PIPE" --stage2-only \
  --out_dir "$STAGE2_AUDIT/floor015_run2" \
  --primary_output_dir "$STAGE2_AUDIT/run2" --workers "$FLOOR015_SAFE_WORKERS"
run_gmt_python 33_floor015_assemble_run2 \
  "$VRILE_PROJECT_ROOT/scripts/assemble_floor015_stage2.py" \
  --floor-run "$STAGE2_AUDIT/floor015_run2" --primary_output_dir "$STAGE2_AUDIT/run2"

run_gmt_python 34_stable_specificity_finalizer \
  "$VRILE_PROJECT_ROOT/scripts/finalize_stage2_scope.py" \
  --audit_dir "$STAGE2_AUDIT"

stage_success 30_floor015_stable
echo "STAGE2_CURRENT_RUN_GENERATION=PASS"
