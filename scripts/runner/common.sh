#!/usr/bin/env bash
set -euo pipefail

RUNNER_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hold() {
  echo "HOLD_$*" >&2
  exit 2
}

check_raw_access_policy() {
  local raw_options
  raw_options="$(findmnt -no OPTIONS -T "$VRILE_RAW_DATA_ROOT" 2>/dev/null || true)"
  printf 'raw_root=%s\nmount_options=%s\nallow_writable_raw=%s\n' \
    "$VRILE_RAW_DATA_ROOT" "$raw_options" "$VRILE_ALLOW_WRITABLE_RAW" \
    > "$VRILE_VALIDATION_ROOT/raw_data_mount.txt"
  if [[ ",$raw_options," != *,ro,* ]]; then
    if [[ "$VRILE_ALLOW_WRITABLE_RAW" != YES ]]; then
      hold "RAW_NOT_READ_ONLY:use_a_read_only_mount_or_--allow-writable-raw_before_science"
    fi
    echo "WARNING_RAW_NOT_READ_ONLY:continuing_in_manual_mode;toolbox_does_not_write_RAW;OS_read_only_protection_not_verified" >&2
  fi
}

load_run_config() {
  local config="${VRILE_RUN_CONFIG:-}"
  [[ -n "$config" && -r "$config" ]] || hold "RUN_CONFIG_MISSING:${config:-unset}"
  # shellcheck disable=SC1090
  source "$config"
  : "${VRILE_RUN_ROOT:?}"
  : "${VRILE_PROJECT_ROOT:?}"
  : "${VRILE_RAW_DATA_ROOT:?}"
  # Read-only fallback for historical initialized-run metadata.
  VRILE_SCIENCE_ENV="${VRILE_SCIENCE_ENV:-${VRILE_TEST_ENV:-}}"
  : "${VRILE_SCIENCE_ENV:?}"
  : "${VRILE_GMT_ENV:?}"
  : "${VRILE_WORKERS:?}"
  export VRILE_SCIENCE_ENV VRILE_GMT_ENV

  VRILE_RUN_ROOT="$(realpath -m -- "$VRILE_RUN_ROOT")"
  VRILE_PROJECT_ROOT="$(realpath -m -- "$VRILE_PROJECT_ROOT")"
  VRILE_RAW_DATA_ROOT="$(realpath -m -- "$VRILE_RAW_DATA_ROOT")"
  PIPE="$VRILE_PROJECT_ROOT"
  VRILE_OUTPUT_ROOT="$VRILE_RUN_ROOT/results"
  VRILE_PROCESSED_DATA_ROOT="$VRILE_RUN_ROOT/data/processed"
  VRILE_LOG_ROOT="$VRILE_RUN_ROOT/logs"
  VRILE_VALIDATION_ROOT="$VRILE_RUN_ROOT/validation"
  VRILE_STATE_ROOT="$VRILE_RUN_ROOT/state"
  VRILE_FIGURE_ROOT="$VRILE_RUN_ROOT/figures"
  VRILE_ACCEPTED_COMPARISON_ROOT="${VRILE_ACCEPTED_COMPARISON_ROOT:-}"
  VRILE_NSIDC_REGION_MASK="$VRILE_RAW_DATA_ROOT/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
  VRILE_ALLOW_WRITABLE_RAW="${VRILE_ALLOW_WRITABLE_RAW:-NO}"
  RUNNER_LIB="$VRILE_PROJECT_ROOT/scripts/runner"

  export VRILE_RUN_ROOT VRILE_PROJECT_ROOT VRILE_RAW_DATA_ROOT PIPE
  export VRILE_OUTPUT_ROOT VRILE_PROCESSED_DATA_ROOT VRILE_LOG_ROOT
  export VRILE_VALIDATION_ROOT VRILE_STATE_ROOT VRILE_FIGURE_ROOT
  export VRILE_ACCEPTED_COMPARISON_ROOT VRILE_NSIDC_REGION_MASK
  export RUNNER_LIB
  export VRILE_TRACE_ROOT="$VRILE_RUN_ROOT/evidence/python_reads"
  mkdir -p "$VRILE_TRACE_ROOT"
  export PYTHONPATH="$PIPE/src:$RUNNER_LIB/runtime_trace" PYTHONDONTWRITEBYTECODE=1
  export PYTHONPYCACHEPREFIX="$VRILE_RUN_ROOT/environment/pycache"
  export MPLCONFIGDIR="$VRILE_RUN_ROOT/environment/matplotlib"
  export XDG_CACHE_HOME="$VRILE_RUN_ROOT/environment/cache"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  mkdir -p "$VRILE_LOG_ROOT" "$VRILE_VALIDATION_ROOT" "$VRILE_STATE_ROOT" \
    "$VRILE_FIGURE_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
}

verify_release_file_hashes() {
  local root="$1"
  local report="$2"
  local expected_list="$VRILE_RUN_ROOT/environment/expected_release_files.txt"
  local observed_list="$VRILE_RUN_ROOT/environment/observed_release_files.txt"
  require_file "$root/sha256.txt"
  (
    cd "$root"
    sha256sum -c --strict sha256.txt
  ) > "$report"
  (
    cd "$root"
    { awk '{sub(/^[^ ]+  /, ""); print} END {print "sha256.txt"}' sha256.txt; } | LC_ALL=C sort
  ) > "$expected_list"
  (
    cd "$root"
    find . -type f -printf '%P\n' | LC_ALL=C sort
  ) > "$observed_list"
  cmp -s "$expected_list" "$observed_list" || hold "WORKING_COPY_FILE_SET_MISMATCH"
}

verify_release_import_origin() {
  local env_name="$1"
  local report="$2"
  VRILE_EXPECTED_SOURCE_ROOT="$PIPE/src" \
    conda run --no-capture-output -n "$env_name" python -B -c \
    'import json, os; from pathlib import Path; import vrile, vrile.io, vrile.stage3.fields; expected=Path(os.environ["VRILE_EXPECTED_SOURCE_ROOT"]).resolve(); files=[Path(vrile.__file__).resolve(),Path(vrile.io.__file__).resolve(),Path(vrile.stage3.fields.__file__).resolve()]; bad=[str(p) for p in files if expected not in p.parents]; print(json.dumps({"environment":os.environ.get("CONDA_DEFAULT_ENV"),"expected_source_root":str(expected),"imported_files":[str(p) for p in files]},indent=2)); raise SystemExit(1 if bad else 0)' \
    > "$report"
}

require_file() {
  [[ -f "$1" ]] || hold "MISSING_FILE:$1"
}

require_dir() {
  [[ -d "$1" ]] || hold "MISSING_DIRECTORY:$1"
}

require_absent() {
  [[ ! -e "$1" ]] || hold "REFUSE_OVERWRITE:$1"
}

require_marker() {
  require_file "$VRILE_STATE_ROOT/$1.pass"
}

mark_pass() {
  local stage="$1"
  printf 'status=PASS\ncompleted_utc=%s\nproject_root=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VRILE_PROJECT_ROOT" \
    > "$VRILE_STATE_ROOT/$stage.pass"
  rm -f "$VRILE_STATE_ROOT/$stage.failed"
}

mark_failed() {
  local stage="$1"
  rm -f "$VRILE_STATE_ROOT/$stage.pass"
  printf 'status=FAIL\nfailed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$VRILE_STATE_ROOT/$stage.failed"
}

require_science_confirmation() {
  [[ "${VRILE_CONFIRM_SCIENCE_RUN:-NO}" == "YES" ]] || \
    hold "SCIENCE_RUN_NOT_CONFIRMED:set_VRILE_CONFIRM_SCIENCE_RUN=YES"
}

run_logged() {
  local label="$1"
  shift
  local log="$VRILE_LOG_ROOT/$label.log"
  require_absent "$log"
  python3 "$RUNNER_LIB/run_evidence.py" start --run_dir "$VRILE_RUN_ROOT" --label "$label" -- "$@"
  printf 'START=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  printf 'COMMAND=' | tee -a "$log"
  printf ' %q' "$@" | tee -a "$log"
  printf '\n' | tee -a "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf 'DONE=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
  python3 "$RUNNER_LIB/run_evidence.py" end --run_dir "$VRILE_RUN_ROOT" --label "$label" \
    --command-exit "${statuses[0]}" --tee-exit "${statuses[1]}"
  [[ "${statuses[0]}" == 0 ]] || return "${statuses[0]}"
  [[ "${statuses[1]}" == 0 ]] || return "${statuses[1]}"
}

run_science_python() {
  local label="$1"
  shift
  run_logged "$label" conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B "$@"
}

run_gmt_python() {
  local label="$1"
  shift
  run_logged "$label" conda run --no-capture-output -n "$VRILE_GMT_ENV" python -B "$@"
}

validate_link_target() {
  local link="$1"
  local target="$2"
  [[ -L "$link" ]] || hold "EXPECTED_SYMLINK:$link"
  [[ "$(realpath -m -- "$link")" == "$(realpath -m -- "$target")" ]] || \
    hold "SYMLINK_TARGET_MISMATCH:$link"
}

validate_source_isolation() {
  require_dir "$VRILE_PROJECT_ROOT"
  require_dir "$PIPE/src/vrile"
  [[ "$(realpath -m -- "$(dirname "$VRILE_PROJECT_ROOT")")" == \
    "$(realpath -m -- "$VRILE_RUN_ROOT/work")" ]] || \
    hold "PROJECT_ROOT_NOT_ISOLATED_WORKING_COPY"
  [[ "$(realpath -m -- "$VRILE_PROJECT_ROOT")" != \
    "$(realpath -m -- "$VRILE_RAW_DATA_ROOT")" ]] || \
    hold "PROJECT_ROOT_EQUALS_RAW_DATA_ROOT"
  case "$VRILE_OUTPUT_ROOT/" in
    "$VRILE_PROJECT_ROOT/"*) hold "OUTPUT_ROOT_INSIDE_RELEASE_SOURCE" ;;
  esac
  case "$VRILE_RUN_ROOT/" in
    "$VRILE_RAW_DATA_ROOT/"*) hold "RUN_ROOT_INSIDE_RAW_DATA" ;;
  esac
  validate_link_target "$VRILE_RUN_ROOT/data/raw" "$VRILE_RAW_DATA_ROOT"
  validate_link_target "$PIPE/data" "$VRILE_RUN_ROOT/data"
  validate_link_target "$PIPE/outputs" "$VRILE_OUTPUT_ROOT"
}

stage_trap() {
  local stage="$1"
  rm -f "$VRILE_STATE_ROOT/$stage.pass" "$VRILE_STATE_ROOT/$stage.failed"
  trap 'mark_failed '"$stage" EXIT
}

stage_success() {
  local stage="$1"
  mark_pass "$stage"
  trap - EXIT
}
