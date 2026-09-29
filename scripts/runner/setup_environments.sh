#!/usr/bin/env bash
set -euo pipefail
# Private runner library; not a researcher-facing entrypoint.
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_run_config
stage_trap 01_environments

ensure_environment_exists() {
  local env_name="$1"
  local lock_path="$2"
  local pip_overlay="${3:-}"
  if conda env list | awk -v name="$env_name" '$1 == name {found=1} END {exit !found}'; then
    return
  fi
  [[ "${VRILE_CREATE_MISSING_ENVS:-NO}" == "YES" ]] || \
    hold "CONDA_ENVIRONMENT_MISSING:${env_name}:use_--create-missing-envs"
  conda create --yes --name "$env_name" --file "$lock_path"
  if [[ -n "$pip_overlay" ]]; then
    conda run --no-capture-output -n "$env_name" python -B -m pip install \
      --requirement "$pip_overlay"
  fi
}

verify_explicit_environment() {
  local env_name="$1"
  local lock_path="$2"
  local expected="$VRILE_RUN_ROOT/environment/${env_name}_expected_explicit.txt"
  local observed="$VRILE_RUN_ROOT/environment/${env_name}_observed_explicit.txt"
  sed -n '/^https\?:\/\//p' "$lock_path" | LC_ALL=C sort > "$expected"
  conda list --explicit --md5 -n "$env_name" | sed -n '/^https\?:\/\//p' | LC_ALL=C sort > "$observed"
  cmp -s "$expected" "$observed" || hold "${env_name}_EXPLICIT_LOCK_MISMATCH"
}

verify_science_pip_overlay() {
  local observed="$VRILE_RUN_ROOT/environment/science_conda_list.json"
  conda list --json -n "$VRILE_SCIENCE_ENV" > "$observed"
  conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B \
    "$RUNNER_LIB/check_environment.py" \
    --requirements "$VRILE_PROJECT_ROOT/config/requirements-lock-pip.txt" \
    --conda-list "$observed"
}

ensure_environment_exists "$VRILE_SCIENCE_ENV" \
  "$VRILE_PROJECT_ROOT/config/environment-lock-linux-64.explicit.txt" \
  "$VRILE_PROJECT_ROOT/config/requirements-lock-pip.txt"
ensure_environment_exists "$VRILE_GMT_ENV" \
  "$VRILE_PROJECT_ROOT/config/gmt-lock-linux-64.explicit.txt"
verify_explicit_environment "$VRILE_SCIENCE_ENV" "$VRILE_PROJECT_ROOT/config/environment-lock-linux-64.explicit.txt"
verify_science_pip_overlay
verify_explicit_environment "$VRILE_GMT_ENV" "$VRILE_PROJECT_ROOT/config/gmt-lock-linux-64.explicit.txt"

conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B -c \
  'import platform,numpy,pandas,scipy; print(platform.python_version(),numpy.__version__,pandas.__version__,scipy.__version__)' \
  | tee "$VRILE_RUN_ROOT/environment/science_versions.txt"
conda run --no-capture-output -n "$VRILE_GMT_ENV" python -B -c \
  'import platform,numpy,pandas,scipy; print(platform.python_version(),numpy.__version__,pandas.__version__,scipy.__version__)' \
  | tee "$VRILE_RUN_ROOT/environment/gmt_versions.txt"
conda run --no-capture-output -n "$VRILE_GMT_ENV" gmt --version \
  | tee "$VRILE_RUN_ROOT/environment/gmt_version.txt"
conda run --no-capture-output -n "$VRILE_GMT_ENV" gs --version \
  | tee "$VRILE_RUN_ROOT/environment/ghostscript_version.txt"

# Minimal plotting venv is separate from both exact scientific environment locks.
plot_setup_args=(--gmt-env "$VRILE_GMT_ENV" --report "$VRILE_RUN_ROOT/environment/plotting_runtime_setup.json")
[[ "${VRILE_CREATE_MISSING_ENVS:-NO}" != YES ]] || plot_setup_args+=(--create)
bash "$RUNNER_LIB/setup_plotting_runtime.sh" "${plot_setup_args[@]}"

stage_success 01_environments
echo "RUNTIME_ENVIRONMENT_LOCKS=PASS"
