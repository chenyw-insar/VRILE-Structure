#!/usr/bin/env bash
set -euo pipefail
# Read-only checkpoints: no initialization, config sourcing, imports or writes.
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${VRILE_RUN_DIR:-${VRILE_RUN_ROOT:-$package_dir/outputs}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir) [[ $# -ge 2 && -n "$2" ]] || exit 2; run_dir="$2"; shift 2 ;;
    *) echo "HOLD_UNKNOWN_STATUS_OPTION:$1" >&2; exit 2 ;;
  esac
done
[[ -r "$run_dir/state/00_initialized.pass" ]] || { echo "HOLD_RUN_NOT_INITIALIZED:$run_dir" >&2; exit 2; }
for stage in 00_initialized 01_environments 02_preflight 10_stage1_m007 \
  20_stage2_m010 30_floor015_stable 40_stage3_m021_m022 \
  50_m023_cross_scale 60_current_presentation; do
  if [[ -f "$run_dir/state/$stage.failed" ]]; then
    printf '%s=FAIL\n' "$stage"
  elif [[ -f "$run_dir/state/$stage.pass" ]]; then
    printf '%s=PASS_RECORDED\n' "$stage"
  else
    printf '%s=PENDING\n' "$stage"
  fi
done
