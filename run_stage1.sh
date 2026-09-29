#!/usr/bin/env bash
set -euo pipefail

# Public Stage 1 entrypoint. This file intentionally shows the complete final
# Stage 1 order, including the cross-stage sparse-cell materializer required by
# the promoted M007 corrected-catalogue rebuild.

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runner/entrypoint.sh
source "$SOURCE_ROOT/scripts/runner/entrypoint.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run_stage1.sh --run_dir DIR [--raw_dir DIR] [options]

Creates a fresh isolated run when needed, performs preflight, then executes:
  pan-Arctic detector -> SIC locations -> local patches -> consolidation
  -> guarded sparse-cell materialization -> promoted M007 -> parity check.
EOF
  runner_common_options
  exit 0
fi

runner_prepare run_stage1.sh YES YES "$@"
stage_trap 10_stage1_m007
require_marker 02_preflight
require_science_confirmation
validate_source_isolation

[[ -z "$VRILE_ACCEPTED_COMPARISON_ROOT" ]] || \
  hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
unset VRILE_ACCEPTED_COMPARISON_ROOT

for path in \
  "$VRILE_OUTPUT_ROOT/reproduce_sie" \
  "$VRILE_OUTPUT_ROOT/local_vrile" \
  "$VRILE_OUTPUT_ROOT/local_vrile_enhanced" \
  "$VRILE_OUTPUT_ROOT/local_vrile_severe" \
  "$VRILE_OUTPUT_ROOT/local_vrile_major_severe" \
  "$VRILE_OUTPUT_ROOT/spatiotemporal_matching" \
  "$VRILE_PROCESSED_DATA_ROOT/local_event_cells"; do
  require_absent "$path"
done

cd "$VRILE_PROJECT_ROOT"

# 1. Pan-Arctic event detection and locations.
run_science_python 10_panarctic_detector \
  "$PIPE/scripts/stage1_reproduce_sie_detection.py" \
  --input "$VRILE_RAW_DATA_ROOT/nsidc_sie/N_seaice_extent_daily_v4.0.csv" \
  --out_dir "$VRILE_OUTPUT_ROOT/reproduce_sie" \
  --months jja --method both --unique-gap-days 1 --start-year 1989 --end-year 2025

run_science_python 11_panarctic_locations \
  "$PIPE/scripts/stage1_reproduce_sic_locations.py" \
  --events "$VRILE_OUTPUT_ROOT/reproduce_sie/vrile_events_unique_both_jja_5p.csv" \
  --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
  --output "$VRILE_OUTPUT_ROOT/reproduce_sie/vrile_locations.csv" \
  --region pan_arctic --window-days 5 --loss-threshold=-0.10 \
  --min-object-cells 4 --parallel --backend process --workers "$VRILE_WORKERS"

# 2. Broad local patches and consolidated events.
run_science_python 12_local_patch_detection \
  "$PIPE/scripts/stage1_detect_local_sic_loss_objects.py" \
  --sic_dir "$VRILE_RAW_DATA_ROOT/nsidc_sic" \
  --panarctic-events "$VRILE_OUTPUT_ROOT/reproduce_sie/vrile_events_unique_both_jja_5p.csv" \
  --regional-events "$VRILE_VALIDATION_ROOT/regional_annotation_not_supplied.csv" \
  --out_dir "$VRILE_OUTPUT_ROOT/local_vrile" \
  --start-year 1989 --end-year 2025 --months jja \
  --window-days 5 --thresholds=-0.10 --min-cells 4 \
  --parallel --backend process --workers "$VRILE_WORKERS"

run_science_python 13_local_consolidation \
  "$PIPE/scripts/stage1_consolidate_local_objects.py" \
  --local-objects "$VRILE_OUTPUT_ROOT/local_vrile/local_objects_all_jja.csv" \
  --out_dir "$VRILE_OUTPUT_ROOT/local_vrile_enhanced" \
  --months jja --start-year 1989 --end-year 2025 \
  --primary-window-days 5 --primary-sic-change-threshold=-0.10 \
  --primary-min-cells 50 --merge-days 2 --merge-distance-km 300

# 3. Cross-stage prerequisite: materialize sparse cells with the frozen guard.
# This is not the final catalogue producer; execution must continue to M007.
# The unchanged writer retains its legacy pre-M007 centroid gate. It writes
# the complete sparse-cell intermediate before reporting those promoted
# longitude-geometry differences. Capture the exit, then permit continuation
# only after the release adapter proves all non-centroid fields and arrays are
# internally reconciled. No prior-result signature is consulted here.
export STAGE3_TRANSACTION_ROOT="$PIPE/_stage3_transaction_researcher_stage1"
export VRILE_STAGE3_OUTPUT_ROOT="$STAGE3_TRANSACTION_ROOT/candidate/outputs/panarctic_local_contribution"
require_absent "$STAGE3_TRANSACTION_ROOT"
sparse_log="$VRILE_LOG_ROOT/14_sparse_materialization.log"
require_absent "$sparse_log"
python3 "$RUNNER_LIB/run_evidence.py" start --run_dir "$VRILE_RUN_ROOT" --label 14_sparse_materialization -- \
  conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B "$PIPE/scripts/stage3_local_footprints.py" --allow-regenerate-frozen
printf 'START=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$sparse_log"
printf 'COMMAND= conda run --no-capture-output -n %q python -B %q --allow-regenerate-frozen\n' \
  "$VRILE_SCIENCE_ENV" "$PIPE/scripts/stage3_local_footprints.py" | tee -a "$sparse_log"
set +e
conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B \
  "$PIPE/scripts/stage3_local_footprints.py" --allow-regenerate-frozen \
  2>&1 | tee -a "$sparse_log"
sparse_statuses=("${PIPESTATUS[@]}")
sparse_exit="${sparse_statuses[0]}"
set -e
printf 'EXIT_STATUS=%s\nDONE=%s\n' "$sparse_exit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$sparse_log"
python3 "$RUNNER_LIB/run_evidence.py" end --run_dir "$VRILE_RUN_ROOT" --label 14_sparse_materialization \
  --command-exit "$sparse_exit" --tee-exit "${sparse_statuses[1]}"
[[ "${sparse_statuses[1]}" == 0 ]] || hold "SPARSE_LOG_TEE_FAILURE"
if [[ "$sparse_exit" -ne 0 ]]; then
  [[ "$sparse_exit" -eq 1 ]] || hold "UNEXPECTED_SPARSE_MATERIALIZER_EXIT:$sparse_exit"
  grep -Fq 'ValueError: footprint_detector_reconciliation.csv contains FAIL rows' "$sparse_log" || \
    hold "UNEXPECTED_SPARSE_MATERIALIZER_FAILURE"
fi

run_science_python 14b_complete_pre_m007_sparse_transition \
  "$RUNNER_LIB/complete_pre_m007_sparse_materialization.py" \
  --producer-exit-status "$sparse_exit" \
  --filtered "$VRILE_OUTPUT_ROOT/local_vrile_enhanced/local_objects_filtered.csv" \
  --canonical-unique "$VRILE_OUTPUT_ROOT/local_vrile_enhanced/unique_local_events.csv" \
  --cells_dir "$VRILE_PROCESSED_DATA_ROOT/local_event_cells" \
  --reconciliation "$VRILE_STAGE3_OUTPUT_ROOT/footprint_detector_reconciliation.csv" \
  --membership-reconciliation "$VRILE_STAGE3_OUTPUT_ROOT/unique_event_membership_reconciliation.csv" \
  --scope-report "$VRILE_VALIDATION_ROOT/pre_m007_sparse_transition.json"

# 4. Mandatory M007 corrected longitude/catalogue rebuild. This writes the
# final broad, severe and major (machine label major_severe) catalogues.
run_science_python 15_m007_corrected_rebuild \
  "$PIPE/scripts/stage1_rebuild_local_catalog_from_frozen_cells.py" \
  --patch-index "$VRILE_PROCESSED_DATA_ROOT/local_event_cells/patch_cell_index.csv" \
  --cells_dir "$VRILE_PROCESSED_DATA_ROOT/local_event_cells" \
  --patch-attributes "$VRILE_OUTPUT_ROOT/local_vrile_enhanced/local_objects_filtered.csv" \
  --grid "$VRILE_RAW_DATA_ROOT/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc" \
  --panarctic-events "$VRILE_OUTPUT_ROOT/reproduce_sie/vrile_events_unique_both_jja_5p.csv" \
  --panarctic-locations "$VRILE_OUTPUT_ROOT/reproduce_sie/vrile_locations.csv" \
  --out_dir "$VRILE_PROJECT_ROOT" --promotion-layout

# Verify the actual producer-to-consumer handoff before marking Stage 1 done.
# M007 metadata is now in the private working project, not the run directory.
run_science_python 15b_validate_m007_handoff \
  "$RUNNER_LIB/check_local_catalog_handoff.py" --run_dir "$VRILE_RUN_ROOT"

stage_success 10_stage1_m007
echo "STAGE1_CURRENT_RUN_GENERATION=PASS"
