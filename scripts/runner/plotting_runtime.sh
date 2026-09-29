#!/usr/bin/env bash
# Shared exact contract for generation preflight, Stage X and post-generation validation.
resolve_plotting_runtime() {
  local report="$1"
  PLOT_PYTHON="$(python3 -B "$RUNNER_LIB/plotting_runtime.py" --print-python --report "$report")" || \
    hold "PLOTTING_RUNTIME_CONTRACT:see_setup_plotting_runtime.sh_--help"
  export PLOT_PYTHON
}
