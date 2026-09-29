#!/usr/bin/env bash
set -euo pipefail
toolbox_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
helper="$toolbox_dir/scripts/runner/plotting_runtime.py"
create=NO
base_python=""
report=""
gmt_env="${VRILE_GMT_ENV:-gmt}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/runner/setup_plotting_runtime.sh [--create] [--gmt-env NAME] [--base_python PATH] [--report FILE]

Default: verify the exact plotting contract; no install and no science.
--create explicitly creates a missing minimal external plotting venv, then
installs only config/plotting-requirements.txt (PyMuPDF exact version, binary only).
Base Python comes from --gmt-env (default gmt); existing Conda environments remain unchanged.
Default location: ${XDG_DATA_HOME:-$HOME/.local/share}/vrile/plotting-pymupdf-1.28.2/.
VRILE_PLOT_PYTHON is an advanced override and must pass the identical exact gate.
An existing but invalid environment is never repaired or overwritten silently.
EOF
      exit 0 ;;
    --create) create=YES; shift ;;
    --gmt-env) [[ $# -ge 2 && -n "$2" ]] || exit 2; gmt_env="$2"; shift 2 ;;
    --base_python) [[ $# -ge 2 ]] || exit 2; base_python="$2"; shift 2 ;;
    --report) [[ $# -ge 2 ]] || exit 2; report="$2"; shift 2 ;;
    *) echo "HOLD_UNKNOWN_PLOTTING_SETUP_OPTION:$1" >&2; exit 2 ;;
  esac
done
if [[ -z "${VRILE_PLOT_PYTHON:-}" ]]; then
  plot_python="$(python3 -B "$helper" --default-python)"
  plot_env_dir="$(dirname "$(dirname "$plot_python")")"
  if [[ ! -e "$plot_env_dir" && "$create" == YES ]]; then
    if [[ -z "$base_python" ]]; then
      base_python="$(conda run --no-capture-output -n "$gmt_env" python -I -B -c 'import sys; print(sys.executable)')"
    fi
    [[ -x "$base_python" ]] || { echo "HOLD_PLOTTING_BASE_PYTHON_INVALID:$base_python" >&2; exit 2; }
    "$base_python" -I -B -m venv "$plot_env_dir"
    "$plot_python" -I -B -m pip install --only-binary=:all: --no-deps \
      --requirement "$toolbox_dir/config/plotting-requirements.txt"
  fi
fi
args=()
[[ -z "$report" ]] || args+=(--report "$report")
python3 -B "$helper" "${args[@]}"
