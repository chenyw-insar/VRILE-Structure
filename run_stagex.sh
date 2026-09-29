#!/usr/bin/env bash
set -euo pipefail

# Public Stage X entrypoint.  Presentation source tables are regenerated from
# the current isolated run (Stage 1/M007, M010, floor015, M021 and M023) plus
# the external read-only NSIDC mask.  Prior bundled/golden presentation tables
# are not computational inputs.

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runner/entrypoint.sh
source "$SOURCE_ROOT/scripts/runner/entrypoint.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run_stagex.sh --run_dir DIR

Builds current scientific Figures 2–6 and S1–S4, with clean independent panels.
Methods flowchart (Figure 1) is excluded. Shared current-run sidecars feed
the display-only saved-result adapter and the single scripts/figures/render_current.py entry.
The external plotting venv is checked against config/plotting-runtime.json.
VRILE_PLOT_PYTHON may override it but cannot bypass the exact PyMuPDF version gate.
GMT uses the environment selected by --gmt-env.
For plotting only from existing inputs, use that Python entry with --input_dir,
--output_dir and --figure. Its --dry-run never starts science or rendering.
EOF
  runner_common_options
  exit 0
fi

runner_prepare run_stagex.sh NO NO "$@"
stage_trap 60_current_presentation
require_marker 50_m023_cross_scale
require_science_confirmation
validate_source_isolation
[[ -z "$VRILE_ACCEPTED_COMPARISON_ROOT" ]] || \
  hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
unset VRILE_ACCEPTED_COMPARISON_ROOT

BUILD_ROOT="$VRILE_FIGURE_ROOT/current_presentation"
SOURCE_STAGE="$BUILD_ROOT/source_data"
SHARED_PREPARER="$VRILE_PROJECT_ROOT/scripts/presentation/prepare_shared_gmt_tables.py"
G2_BUILD="$BUILD_ROOT/shared_tables"
PLOT_CODE="$VRILE_PROJECT_ROOT/scripts/figures"
source "$RUNNER_LIB/plotting_runtime.sh"
resolve_plotting_runtime "$VRILE_RUN_ROOT/environment/plotting_runtime_stagex.json"
run_logged 68_plot_environment python3 -B "$RUNNER_LIB/plotting_runtime.py" --python "$PLOT_PYTHON"
CURRENT_OUTPUTS="$VRILE_FIGURE_ROOT/final"
require_absent "$BUILD_ROOT"
require_absent "$CURRENT_OUTPUTS"
mkdir -p "$G2_BUILD/inputs"

{
  echo "PRESENTATION_SOURCE_MODE=FRESH_CURRENT_RUN_SOURCE_CLOSURE"
  echo "FRESH_RUN_SCIENCE_OUTPUTS_CONSUMED_BY_PRESENTATION=YES"
  echo "PRIOR_PRESENTATION_OR_GOLDEN_INPUTS_CONSUMED=NO"
  echo "PRESENTATION_GENERATION_SCOPE=CURRENT_RUN_F2_F6_S1_S4"
  echo "GOLDEN_PARITY_VALIDATION_SCOPE=SEPARATE_MAINTAINER_AUDIT_ONLY"
} | tee "$VRILE_VALIDATION_ROOT/presentation_source_mode.txt"

DAILY_SIE="$VRILE_OUTPUT_ROOT/reproduce_sie/daily_sie_processed.csv"
BROAD_EVENTS="$VRILE_OUTPUT_ROOT/local_vrile_enhanced/unique_local_events.csv"
SEVERE_EVENTS="$VRILE_OUTPUT_ROOT/local_vrile_severe/severe_unique_local_events.csv"
MAJOR_EVENTS="$VRILE_OUTPUT_ROOT/local_vrile_major_severe/major_severe_events_union.csv"
STAGE2_RUN1="$VRILE_OUTPUT_ROOT/stage2_scope_audit/run1"
M010_RAW="$STAGE2_RUN1/scope_completion/corrected_stage2_raw_evidence_full_770.csv"
M010_SUMMARY="$STAGE2_RUN1/scope_completion/corrected_stage2_region_evidence_summary.csv"
FLOOR015_PAIRED="$VRILE_OUTPUT_ROOT/stage2_scope_audit/floor015_run1/floor015_stage2_raw_evidence.csv"
M021_PRODUCTS="$VRILE_OUTPUT_ROOT/m021_corrected_stage3/products"
STAGE3_FIELDS="$VRILE_PROCESSED_DATA_ROOT/panarctic_loss_fields"
SENSITIVITY_MATRIX="$VRILE_OUTPUT_ROOT/stage3_component_sensitivity/stage3_component_sensitivity_matrix.csv"
M023_CURRENT="$VRILE_OUTPUT_ROOT/m023_cross_scale"
external_mask_source="$VRILE_RAW_DATA_ROOT/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"

for current_input in \
  "$DAILY_SIE" "$BROAD_EVENTS" "$SEVERE_EVENTS" "$MAJOR_EVENTS" \
  "$M010_RAW" "$M010_SUMMARY" "$FLOOR015_PAIRED" \
  "$SENSITIVITY_MATRIX" "$external_mask_source"; do
  require_file "$current_input"
done
require_dir "$M021_PRODUCTS"
require_dir "$STAGE3_FIELDS"
require_dir "$M023_CURRENT"

run_science_python 69_fresh_presentation_sources \
  "$RUNNER_LIB/generate_fresh_presentation_sources.py" \
  --daily-sie "$DAILY_SIE" \
  --broad-events "$BROAD_EVENTS" \
  --severe-events "$SEVERE_EVENTS" \
  --major-events "$MAJOR_EVENTS" \
  --stage2-raw-770 "$M010_RAW" \
  --stage2-region-summary-40 "$M010_SUMMARY" \
  --floor015-paired-770 "$FLOOR015_PAIRED" \
  --m021-products "$M021_PRODUCTS" \
  --stage3-loss-fields "$STAGE3_FIELDS" \
  --sensitivity-matrix "$SENSITIVITY_MATRIX" \
  --m023_dir "$M023_CURRENT" \
  --nsidc-region-mask "$external_mask_source" \
  --output_dir "$SOURCE_STAGE"

require_file "$SOURCE_STAGE/manifest.csv"
require_file "$SOURCE_STAGE/sha256.txt"
[[ "$(tail -n +2 "$SOURCE_STAGE/manifest.csv" | wc -l)" -eq 14 ]] || \
  hold "FRESH_PRESENTATION_SIDECAR_COUNT"
(
  cd "$SOURCE_STAGE"
  sha256sum -c --strict sha256.txt
) | tee "$VRILE_LOG_ROOT/69_fresh_presentation_source_hashes.log"

external_mask_target="$SOURCE_STAGE/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
mkdir -p "$(dirname "$external_mask_target")"
ln -s "$external_mask_source" "$external_mask_target"
validate_link_target "$external_mask_target" "$external_mask_source"

# Retain the shared, already-tested source table transformation, not old renderers.
run_science_python 70_g2_prepare "$RUNNER_LIB/prepare_current_presentation_inputs.py" \
  --preparer "$SHARED_PREPARER" \
  --input_dir "$SOURCE_STAGE" --source-manifest "$SOURCE_STAGE/manifest.csv" \
  --raw-mask "$external_mask_source" --output_dir "$G2_BUILD/inputs"
require_file "$G2_BUILD/inputs/GMT_INPUT_TABLE_MANIFEST.csv"
run_science_python 71_current_plot_inputs "$PLOT_CODE/bundle_saved_results.py" \
  --prepared_dir "$G2_BUILD/inputs" --sidecar_dir "$SOURCE_STAGE" \
  --results_dir "$VRILE_OUTPUT_ROOT" --output_dir "$BUILD_ROOT/current_inputs"
run_logged 72_current_scientific_figures conda run --no-capture-output -n "$VRILE_GMT_ENV" \
  "$PLOT_PYTHON" -B "$PLOT_CODE/render_current.py" --figure all \
  --input_dir "$BUILD_ROOT/current_inputs" --output_dir "$CURRENT_OUTPUTS" \
  --timeout "${VRILE_FIGURE_TIMEOUT_SECONDS:-120}"
run_logged 73_current_figure_structure conda run --no-capture-output -n "$VRILE_GMT_ENV" \
  "$PLOT_PYTHON" -B "$PLOT_CODE/check_outputs.py" --output_dir "$CURRENT_OUTPUTS"
echo "SVG_EXPORT_STATUS=VECTOR_PYMUPDF; PNG_DPI=600" > "$VRILE_VALIDATION_ROOT/svg_export_status.txt"
stage_success 60_current_presentation
echo "CURRENT_F2_F6_S1_S4_PRESENTATION_GENERATION=PASS"
