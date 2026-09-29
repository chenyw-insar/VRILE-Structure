#!/usr/bin/env bash
set -euo pipefail

# Public Stage 3 entrypoint. It includes the guarded base component/footprint
# chain, promoted M021, M022 no-look-ahead validation, and final M023
# cross-scale/matching/normalization/focal-strength closure.

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runner/entrypoint.sh
source "$SOURCE_ROOT/scripts/runner/entrypoint.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run_stage3.sh --run_dir DIR

Executes the accepted final Stage 3 topology:
  guarded base fields/components -> M021 -> M022 -> component parity
  -> M023 X1/X2, matching, normalization and focal-strength parity.
EOF
  runner_common_options
  exit 0
fi

runner_prepare run_stage3.sh NO NO "$@"
require_marker 30_floor015_stable
require_science_confirmation
validate_source_isolation
[[ -z "$VRILE_ACCEPTED_COMPARISON_ROOT" ]] || \
  hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
unset VRILE_ACCEPTED_COMPARISON_ROOT
unset VRILE_SOURCE_EVIDENCE_MANIFEST VRILE_RUN_TERMINOLOGY_AUDIT

# Part A: nine guarded base calls, M021 and M022.
stage_trap 40_stage3_m021_m022
export STAGE3_TRANSACTION_ROOT="$PIPE/_stage3_transaction_researcher_final"
export VRILE_STAGE3_OUTPUT_ROOT="$STAGE3_TRANSACTION_ROOT/candidate/outputs/panarctic_local_contribution"
export VRILE_STAGE3_DATA_ROOT="$VRILE_PROCESSED_DATA_ROOT"
export STAGE3_VALIDATION_ROOT="$STAGE3_TRANSACTION_ROOT/validation"
M021_ROOT="$VRILE_OUTPUT_ROOT/m021_corrected_stage3"
M022_ROOT="$VRILE_VALIDATION_ROOT/m022_no_lookahead"
export M021_ROOT M022_ROOT
require_absent "$STAGE3_TRANSACTION_ROOT"
require_absent "$M021_ROOT"
require_absent "$M022_ROOT"
require_absent "$VRILE_PROCESSED_DATA_ROOT/panarctic_detector_extent_fields"
require_absent "$VRILE_PROCESSED_DATA_ROOT/panarctic_loss_fields"
mkdir -p "$STAGE3_VALIDATION_ROOT"

cd "$VRILE_PROJECT_ROOT"
run_science_python 40_stage3_cells_inputs \
  "$PIPE/scripts/stage3_local_cells_preflight.py" --allow-regenerate-frozen inputs
run_science_python 41_stage3_foundation \
  "$PIPE/scripts/stage3_panarctic_local_contribution.py" --allow-regenerate-frozen
run_science_python 42_stage3_detector_fields \
  "$PIPE/scripts/stage3_detector_extent_fields.py" --allow-regenerate-frozen \
  --parallel --backend process --workers "$VRILE_WORKERS"
run_science_python 43_stage3_cells_artifacts \
  "$PIPE/scripts/stage3_local_cells_preflight.py" --allow-regenerate-frozen artifacts
run_science_python 44_stage3_cdr_fields \
  "$PIPE/scripts/stage3_cdr_loss_fields.py" --allow-regenerate-frozen \
  --parallel --backend process --workers "$VRILE_WORKERS"
run_science_python 45_stage3_base_overlap \
  "$PIPE/scripts/stage3_raster_overlap.py" --allow-regenerate-frozen
run_science_python 46_stage3_component_preflight \
  "$PIPE/scripts/stage3_batch3_preflight.py" --allow-regenerate-frozen
run_science_python 47_stage3_budgets \
  "$PIPE/scripts/stage3_contribution_budget.py" --allow-regenerate-frozen
run_science_python 48_stage3_composition \
  "$PIPE/scripts/stage3_spatial_composition_metrics.py" --allow-regenerate-frozen

export STAGE3_BASE_PRODUCTS="$VRILE_STAGE3_OUTPUT_ROOT"
export VRILE_STAGE3_PRODUCTS_ROOT="$STAGE3_BASE_PRODUCTS"
run_science_python 49_m021_corrected_stage3 \
  "$VRILE_PROJECT_ROOT/scripts/run_corrected_stage3_20260822.py" \
  --output_dir "$M021_ROOT"

export VRILE_STAGE3_PRODUCTS_ROOT="$M021_ROOT/products"
export VRILE_STAGE3_NO_LOOKAHEAD_OUT="$M022_ROOT"
run_science_python 50_m022_no_lookahead \
  "$VRILE_PROJECT_ROOT/scripts/check_stage3_no_lookahead.py"

run_science_python 51_stage3_component_sensitivity \
  "$RUNNER_LIB/generate_stage3_component_sensitivity.py" \
  --stage3_products_dir "$M021_ROOT/products" \
  --raw_dir "$VRILE_RAW_DATA_ROOT" \
  --output_dir "$VRILE_OUTPUT_ROOT/stage3_component_sensitivity"

stage_success 40_stage3_m021_m022

# Part B: mandatory M023 final cross-scale closure.
stage_trap 50_m023_cross_scale
M023_ROOT="$VRILE_OUTPUT_ROOT/m023_cross_scale"
require_dir "$M021_ROOT/products"
require_absent "$M023_ROOT"
export VRILE_STAGE3_PRODUCTS_ROOT="$M021_ROOT/products"

run_gmt_python 60_m023_cross_scale \
  "$VRILE_PROJECT_ROOT/scripts/stagex_cross_scale.py" \
  --output_dir "$M023_ROOT"
stage_success 50_m023_cross_scale
echo "STAGE3_CURRENT_RUN_GENERATION_THROUGH_M023=PASS"
