#!/usr/bin/env bash
set -euo pipefail

# Public complete-route entrypoint. The sequence is intentionally explicit;
# each called stage remains independently readable and fail-fast.

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runner/entrypoint.sh
source "$SOURCE_ROOT/scripts/runner/entrypoint.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./run_all.sh --run_dir DIR [--raw_dir DIR] [options]
       ./run_all.sh --status [--run_dir DIR]

Runs exactly:
  run_stage1.sh -> run_stage2.sh -> run_stage3.sh -> run_stagex.sh

Produces scientific results and Figures 2-6, S1-S4. Required runtime checks are
automatic. This generation route never reads accepted/frozen result products.

A new run is initialized and preflighted when --run_dir does not yet exist.
EOF
  runner_common_options
  exit 0
fi

if [[ "${1:-}" == --status ]]; then
  shift
  exec bash "$RUNNER_LIB_SOURCE/status.sh" "$@"
fi

runner_prepare run_all.sh YES YES "$@"
require_science_confirmation
validate_source_isolation
source "$RUNNER_LIB/plotting_runtime.sh"
resolve_plotting_runtime "$VRILE_RUN_ROOT/environment/plotting_runtime_run_all.json"
[[ -z "$VRILE_ACCEPTED_COMPARISON_ROOT" ]] || \
  hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_RUN_ALL"

echo "RUN_ALL_STEP_1=STAGE1"
bash "$VRILE_PROJECT_ROOT/run_stage1.sh" --run_dir "$VRILE_RUN_ROOT"

echo "RUN_ALL_STEP_2=STAGE2"
bash "$VRILE_PROJECT_ROOT/run_stage2.sh" --run_dir "$VRILE_RUN_ROOT"

echo "RUN_ALL_STEP_3=STAGE3"
bash "$VRILE_PROJECT_ROOT/run_stage3.sh" --run_dir "$VRILE_RUN_ROOT"

echo "RUN_ALL_STEP_4=STAGEX"
bash "$VRILE_PROJECT_ROOT/run_stagex.sh" --run_dir "$VRILE_RUN_ROOT"

echo "RELEASE_ID=$(python3 -B "$RUNNER_LIB/release_identity.py" --project-root "$VRILE_PROJECT_ROOT" --field release_id)"
echo "RAW_TO_FINAL_GENERATION=PASS"
echo "RESULTS_DIR=$VRILE_OUTPUT_ROOT"
echo "FIGURES_DIR=$VRILE_FIGURE_ROOT/final"
