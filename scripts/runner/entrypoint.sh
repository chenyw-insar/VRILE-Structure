#!/usr/bin/env bash
set -euo pipefail

# Private CLI/bootstrap support for the five generation entrypoints. Scientific stage
# commands remain visible in the root scripts; this file only handles the
# isolated run copy and common path/environment options.

RUNNER_LIB_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PACKAGE_ROOT="$(cd "$RUNNER_LIB_SOURCE/../.." && pwd)"
# shellcheck source=common.sh
source "$RUNNER_LIB_SOURCE/common.sh"

runner_common_options() {
  printf 'Release: %s\n' "$(python3 -B "$RUNNER_LIB_SOURCE/release_identity.py" --project-root "$SOURCE_PACKAGE_ROOT" --field release_id)"
  cat <<'EOF'
Common options:
  --run_dir DIR                  Run directory; default: package/outputs/.
  --raw_dir DIR                  External RAW inputs; default: package/raw/.
  --science-env NAME             Scientific environment; default: vrile.
  --gmt-env NAME                 GMT environment; default: gmt.
  --workers N                    Engineering parallelism; default: 12.
  --create-missing-envs          Create missing environments from exact locks.
  --init-only                    Initialize paths only (run_all/run_stage1); no science.
  --preflight-only               Check inputs/environments only (run_all/run_stage1); no science.
  --require-readonly-raw          Require an OS read-only RAW mount (strict mode).
  --allow-writable-raw            Warning-only RAW policy (the manual default).

Any new/empty run directory outside RAW and sealed code paths is supported,
including ./outputs_manual/. Without an -only option, generation starts computation.
Existing results are never reset. RAW policy can change only before science.
EOF
}

runner_prepare() {
  local public_script="$1"
  local allow_initialize="$2"
  local auto_preflight="$3"
  shift 3
  export VRILE_EXECUTION_PHASE=GENERATION

  local requested_run_root="${VRILE_RUN_DIR:-${VRILE_RUN_ROOT:-$SOURCE_PACKAGE_ROOT/outputs}}"
  local requested_raw_root="${VRILE_RAW_DIR:-${VRILE_RAW_DATA_ROOT:-}}"
  local requested_accepted_root="${VRILE_REFERENCE_DIR:-${VRILE_ACCEPTED_COMPARISON_ROOT:-}}"
  local requested_science_env="${VRILE_SCIENCE_ENV:-${VRILE_TEST_ENV:-vrile}}"
  local requested_gmt_env="${VRILE_GMT_ENV:-gmt}"
  local requested_workers="${VRILE_WORKERS:-12}"
  local allow_writable_raw="${VRILE_ALLOW_WRITABLE_RAW:-YES}"
  local init_only="${VRILE_RUNNER_INIT_ONLY:-NO}"
  local preflight_only="${VRILE_RUNNER_PREFLIGHT_ONLY:-NO}"
  local science_env_explicit=NO
  local gmt_env_explicit=NO
  local workers_explicit=NO
  local writable_raw_explicit=NO
  [[ -z "${VRILE_TEST_ENV+x}" ]] || science_env_explicit=YES
  [[ -z "${VRILE_SCIENCE_ENV+x}" ]] || science_env_explicit=YES
  [[ -z "${VRILE_GMT_ENV+x}" ]] || gmt_env_explicit=YES
  [[ -z "${VRILE_WORKERS+x}" ]] || workers_explicit=YES
  [[ -z "${VRILE_ALLOW_WRITABLE_RAW+x}" ]] || writable_raw_explicit=YES

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run_dir)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:--run_dir"
        requested_run_root="$2"; shift 2 ;;
      --raw_dir)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:--raw_dir"
        requested_raw_root="$2"; shift 2 ;;
      --reference_dir)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:--reference_dir"
        requested_accepted_root="$2"; shift 2 ;;
      --science-env|--test-env)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:$1"
        requested_science_env="$2"; science_env_explicit=YES; shift 2 ;;
      --gmt-env)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:--gmt-env"
        requested_gmt_env="$2"; gmt_env_explicit=YES; shift 2 ;;
      --workers)
        [[ $# -ge 2 && -n "$2" ]] || hold "OPTION_VALUE_REQUIRED:--workers"
        requested_workers="$2"; workers_explicit=YES; shift 2 ;;
      --create-missing-envs) export VRILE_CREATE_MISSING_ENVS=YES; shift ;;
      --allow-writable-raw)
        allow_writable_raw=YES; writable_raw_explicit=YES; shift ;;
      --require-readonly-raw)
        allow_writable_raw=NO; writable_raw_explicit=YES; shift ;;
      --init-only) init_only=YES; shift ;;
      --preflight-only) preflight_only=YES; shift ;;
      *) hold "UNKNOWN_RUNNER_OPTION:$1" ;;
    esac
  done

  [[ -n "$requested_run_root" ]] || hold "RUN_DIR_REQUIRED"
  [[ "$allow_writable_raw" == YES || "$allow_writable_raw" == NO ]] || hold "INVALID_RAW_POLICY:use_YES_or_NO"
  [[ "$init_only" != YES || "$allow_initialize" == YES ]] || hold "INIT_ONLY_USE_RUN_ALL_OR_RUN_STAGE1"
  [[ "$preflight_only" != YES || "$allow_initialize" == YES ]] || hold "PREFLIGHT_ONLY_USE_RUN_ALL_OR_RUN_STAGE1"
  [[ "$init_only" != YES || "$preflight_only" != YES ]] || hold "CHOOSE_ONE_ONLY_MODE"
  if [[ -n "$requested_accepted_root" ]]; then
    hold "VALIDATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
  fi
  local terminology_audit_requested="${VRILE_RUN_TERMINOLOGY_AUDIT:-NO}"
  if [[ "${terminology_audit_requested^^}" == YES ]]; then
    hold "OPTIONAL_DOCUMENTATION_REFERENCE_NOT_ALLOWED_DURING_GENERATION"
  fi
  requested_run_root="$(realpath -m -- "$requested_run_root")"
  local config="$requested_run_root/state/runner.env"

  if [[ ! -r "$config" ]]; then
    [[ "$allow_initialize" == "YES" ]] || \
      hold "RUN_NOT_INITIALIZED:first_run_run_stage1.sh_or_run_all.sh"
    requested_raw_root="${requested_raw_root:-$SOURCE_PACKAGE_ROOT/raw}"
    local init_args=(
      --source_dir "$SOURCE_PACKAGE_ROOT"
      --run_dir "$requested_run_root"
      --raw_dir "$requested_raw_root"
      --science-env "$requested_science_env"
      --gmt-env "$requested_gmt_env"
      --workers "$requested_workers"
    )
    [[ "$allow_writable_raw" != "YES" ]] || init_args+=(--allow-writable-raw)
    [[ "$allow_writable_raw" != "NO" ]] || init_args+=(--require-readonly-raw)
    bash "$RUNNER_LIB_SOURCE/initialize_run.sh" "${init_args[@]}"
  fi

  local policy_args=(existing --source_dir "$SOURCE_PACKAGE_ROOT" --run_dir "$requested_run_root")
  python3 -B "$RUNNER_LIB_SOURCE/run_policy.py" "${policy_args[@]}" || exit 2

  export VRILE_RUN_CONFIG="$config"
  load_run_config
  export VRILE_CONFIRM_SCIENCE_RUN="${VRILE_CONFIRM_SCIENCE_RUN:-YES}"

  if [[ -n "$requested_raw_root" && \
        "$(realpath -m -- "$requested_raw_root")" != "$VRILE_RAW_DATA_ROOT" ]]; then
    hold "RAW_DATA_ROOT_DIFFERS_FROM_INITIALIZED_RUN"
  fi
  if [[ -n "$requested_accepted_root" ]]; then
    VRILE_ACCEPTED_COMPARISON_ROOT="$(realpath -m -- "$requested_accepted_root")"
    export VRILE_ACCEPTED_COMPARISON_ROOT
  fi
  if [[ "$science_env_explicit" == "YES" && "$requested_science_env" != "$VRILE_SCIENCE_ENV" ]]; then
    hold "SCIENCE_ENV_DIFFERS_FROM_INITIALIZED_RUN"
  fi
  if [[ "$gmt_env_explicit" == "YES" && "$requested_gmt_env" != "$VRILE_GMT_ENV" ]]; then
    hold "GMT_ENV_DIFFERS_FROM_INITIALIZED_RUN"
  fi
  if [[ "$workers_explicit" == "YES" && "$requested_workers" != "$VRILE_WORKERS" ]]; then
    hold "WORKER_COUNT_DIFFERS_FROM_INITIALIZED_RUN"
  fi
  if [[ "$writable_raw_explicit" == YES && "$allow_writable_raw" != "$VRILE_ALLOW_WRITABLE_RAW" ]]; then
    python3 -B "$RUNNER_LIB_SOURCE/run_policy.py" "${policy_args[@]}" \
      --allow-writable "$allow_writable_raw" || exit 2
    VRILE_ALLOW_WRITABLE_RAW="$allow_writable_raw"
  fi

  # Refuse a repeat before stage_trap can replace existing success/failure records.
  local target_stage=""
  case "$public_script" in
    run_stage1.sh) target_stage=10_stage1_m007 ;;
    run_stage2.sh) target_stage=20_stage2_m010 ;;
    run_stage3.sh) target_stage=40_stage3_m021_m022 ;;
    run_stagex.sh) target_stage=60_current_presentation ;;
  esac
  if [[ -n "$target_stage" && ( -e "$VRILE_STATE_ROOT/$target_stage.pass" || -e "$VRILE_STATE_ROOT/$target_stage.failed" ) ]]; then
    hold "STAGE_ALREADY_STARTED:preserving_existing_records;use_a_new_run_dir_for_a_fresh_run"
  fi
  if [[ "$public_script" == run_all.sh && -d "$VRILE_RUN_ROOT/evidence/commands" ]] && \
    [[ -n "$(find "$VRILE_RUN_ROOT/evidence/commands" -maxdepth 1 -type f -print -quit)" ]]; then
    hold "GENERATION_ALREADY_STARTED:run_all_does_not_restart_existing_science;use_the_next_unstarted_stage_or_a_new_run_dir"
  fi

  # Continue with the runner bytes inside the isolated working copy. This is a
  # release-engineering re-exec only; all scientific source remains unchanged
  # except for the documented release-path adaptations.
  if [[ "${VRILE_RUNNER_REEXEC:-NO}" != "YES" && \
        "$(realpath -m -- "$SOURCE_PACKAGE_ROOT")" != "$VRILE_PROJECT_ROOT" ]]; then
    export VRILE_RUNNER_REEXEC=YES
    export VRILE_RUNNER_INIT_ONLY="$init_only"
    export VRILE_RUNNER_PREFLIGHT_ONLY="$preflight_only"
    exec bash "$VRILE_PROJECT_ROOT/$public_script" --run_dir "$VRILE_RUN_ROOT"
  fi

  if [[ "$init_only" == YES ]]; then
    check_raw_access_policy
    echo "INITIALIZATION_ONLY=PASS;SCIENTIFIC_PIPELINE_EXECUTED=NO"
    echo "RUN_DIR=$VRILE_RUN_ROOT"
    echo "NEXT=./$public_script --run_dir $VRILE_RUN_ROOT"
    exit 0
  fi

  check_raw_access_policy

  if [[ "$auto_preflight" == "YES" && ! -f "$VRILE_STATE_ROOT/02_preflight.pass" ]]; then
    VRILE_RUN_CONFIG="$config" bash "$RUNNER_LIB/preflight.sh"
  fi
  if [[ "$preflight_only" == YES ]]; then
    require_marker 01_environments
    require_marker 02_preflight
    echo "PREFLIGHT_ONLY=PASS;SCIENTIFIC_PIPELINE_EXECUTED=NO"
    exit 0
  fi
}
