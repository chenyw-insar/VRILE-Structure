#!/usr/bin/env bash
set -euo pipefail
# Private runner library; called by run_stage1.sh and run_all.sh.
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_run_config
stage_trap 02_preflight
require_marker 00_initialized
validate_source_isolation
verify_release_file_hashes "$VRILE_PROJECT_ROOT" \
  "$VRILE_VALIDATION_ROOT/working_copy_integrity_before_preflight.txt"

required_files=(
  "$VRILE_RAW_DATA_ROOT/nsidc_sie/N_seaice_extent_daily_v4.0.csv"
  "$VRILE_RAW_DATA_ROOT/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc"
  "$VRILE_RAW_DATA_ROOT/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc"
  "$PIPE/scripts/stage1_reproduce_sie_detection.py"
  "$PIPE/scripts/stage1_reproduce_sic_locations.py"
  "$PIPE/scripts/stage1_detect_local_sic_loss_objects.py"
  "$PIPE/scripts/stage1_consolidate_local_objects.py"
  "$PIPE/scripts/stage3_local_footprints.py"
  "$PIPE/scripts/stage1_rebuild_local_catalog_from_frozen_cells.py"
  "$RUNNER_LIB/complete_pre_m007_sparse_materialization.py"
  "$PIPE/scripts/stage2_build_era5_cyclone_tracks.py"
  "$PIPE/scripts/stage2_experiment_core.py"
  "$PIPE/scripts/stage2_robustness_floor015_downstream.py"
  "$PIPE/scripts/stage3_local_cells_preflight.py"
  "$PIPE/scripts/stage3_panarctic_local_contribution.py"
  "$PIPE/scripts/stage3_detector_extent_fields.py"
  "$PIPE/scripts/stage3_cdr_loss_fields.py"
  "$PIPE/scripts/stage3_raster_overlap.py"
  "$PIPE/scripts/stage3_batch3_preflight.py"
  "$PIPE/scripts/stage3_contribution_budget.py"
  "$PIPE/scripts/stage3_spatial_composition_metrics.py"
  "$VRILE_PROJECT_ROOT/scripts/assemble_formal_stage2.py"
  "$VRILE_PROJECT_ROOT/scripts/complete_stage2_scope.py"
  "$VRILE_PROJECT_ROOT/scripts/assemble_floor015_stage2.py"
  "$VRILE_PROJECT_ROOT/scripts/finalize_stage2_scope.py"
  "$PIPE/method_static/METHOD_STATIC_DEFINITION_STAGE2_EVIDENCE_ROWS.csv"
  "$PIPE/method_static/METHOD_STATIC_DEFINITION_STAGE2_REGION_MECHANISMS.csv"
  "$PIPE/method_static/METHOD_STATIC_DEFINITION_STAGE2_SYMMETRY.csv"
  "$VRILE_PROJECT_ROOT/scripts/run_corrected_stage3_20260822.py"
  "$VRILE_PROJECT_ROOT/scripts/check_stage3_no_lookahead.py"
  "$VRILE_PROJECT_ROOT/scripts/stagex_cross_scale.py"
  "$RUNNER_LIB/generate_fresh_presentation_sources.py"
  "$VRILE_PROJECT_ROOT/method_static/stage2_evidence_change_claim_map.csv"
  "$RUNNER_LIB/prepare_current_presentation_inputs.py"
  "$VRILE_PROJECT_ROOT/scripts/presentation/prepare_shared_gmt_tables.py"
  "$VRILE_PROJECT_ROOT/scripts/figures/render_current.py"
  "$VRILE_PROJECT_ROOT/scripts/figures/bundle_saved_results.py"
  "$VRILE_PROJECT_ROOT/scripts/figures/check_outputs.py"
  "$RUNNER_LIB/generate_stage3_component_sensitivity.py"
  "$RUNNER_LIB/check_environment.py"
  "$RUNNER_LIB/check_local_catalog_handoff.py"
)
for path in "${required_files[@]}"; do require_file "$path"; done

cell_area_sha="$(sha256sum "$VRILE_RAW_DATA_ROOT/nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc" | awk '{print $1}')"
region_mask_sha="$(sha256sum "$VRILE_RAW_DATA_ROOT/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc" | awk '{print $1}')"
[[ "$cell_area_sha" == "8197f5d4a7d4d3080fc60a5632adf60df20dce5a8490935f8de28f6322cbc095" ]] || \
  hold "NSIDC0771_CELL_AREA_HASH_MISMATCH"
[[ "$region_mask_sha" == "5cc42dba4e57e7f55abe448e010100367fb679435024c7fcf37a29d55c29ca92" ]] || \
  hold "NSIDC0780_REGION_MASK_HASH_MISMATCH"
printf 'NSIDC0771_CellArea_PS_N25km_v1.1.nc  %s\nNSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc  %s\n' \
  "$cell_area_sha" "$region_mask_sha" \
  > "$VRILE_VALIDATION_ROOT/external_static_input_hashes.txt"

required_dirs=(
  "$VRILE_RAW_DATA_ROOT/nsidc_sic"
  "$VRILE_RAW_DATA_ROOT/nsidc_g02135_geotiff/north/daily/geotiff"
  "$VRILE_RAW_DATA_ROOT/era5/single_levels"
)
for path in "${required_dirs[@]}"; do require_dir "$path"; done
find "$VRILE_RAW_DATA_ROOT/nsidc_sic" -type f -name 'sic_psn25_*.nc' -print -quit | grep -q . || hold "NSIDC_SIC_FILES_MISSING"
find "$VRILE_RAW_DATA_ROOT/nsidc_g02135_geotiff/north/daily/geotiff" -type f -name 'N_*_extent_v*.tif' -print -quit | grep -q . || hold "G02135_FILES_MISSING"
find "$VRILE_RAW_DATA_ROOT/era5/single_levels" -maxdepth 1 -type f -name '*.nc' -print -quit | grep -q . || hold "ERA5_FILES_MISSING"

missing_years=()
for year in $(seq 1989 2025); do
  find "$VRILE_RAW_DATA_ROOT/nsidc_sic/$year" -type f -name 'sic_psn25_*.nc' -print -quit 2>/dev/null | grep -q . || \
    missing_years+=("nsidc_sic:$year")
  find "$VRILE_RAW_DATA_ROOT/nsidc_g02135_geotiff/north/daily/geotiff/$year" -type f -name 'N_*_extent_v*.tif' -print -quit 2>/dev/null | grep -q . || \
    missing_years+=("g02135:$year")
  find "$VRILE_RAW_DATA_ROOT/era5/single_levels" -maxdepth 1 -type f -name "*${year}*.nc" -print -quit | grep -q . || \
    missing_years+=("era5_single_levels:$year")
done
if ((${#missing_years[@]})); then
  printf '%s\n' "${missing_years[@]}" > "$VRILE_VALIDATION_ROOT/missing_required_year_coverage.txt"
  hold "RAW_YEAR_COVERAGE_MISSING:see_$VRILE_VALIDATION_ROOT/missing_required_year_coverage.txt"
fi
printf 'coverage_years=1989-2025\nnsidc_sic=PASS\ng02135=PASS\nera5_single_levels=PASS\n' \
  > "$VRILE_VALIDATION_ROOT/required_year_coverage.txt"

check_raw_access_policy

bash "$RUNNER_LIB/setup_environments.sh"

# Same exact vector-runtime contract used by Stage X.
source "$RUNNER_LIB/plotting_runtime.sh"
resolve_plotting_runtime "$VRILE_RUN_ROOT/environment/plotting_runtime_preflight.json"

# The final Stage-2 code selects the lexicographically first filename that
# contains each year. Validate that exact selected file before expensive work.
# This is a read-only schema/time-coordinate check; it produces no science.
conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B - \
  "$VRILE_RAW_DATA_ROOT/era5/single_levels" \
  "$VRILE_VALIDATION_ROOT/era5_annual_input_contract.json" <<'PY'
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

root = Path(sys.argv[1])
report = Path(sys.argv[2])
all_files = sorted(root.glob("*.nc"))
failures = []
selected = []

# Every file discovered by the cyclone builder for 1989-2025 must carry MSL.
for path in all_files:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", path.name)]
    if not years or not 1989 <= years[-1] <= 2025:
        continue
    with xr.open_dataset(path) as dataset:
        if "msl" not in dataset:
            failures.append(f"cyclone-discovered file lacks msl: {path.name}")

for year in range(1989, 2026):
    matches = sorted(root.glob(f"*{year}*.nc"))
    if not matches:
        failures.append(f"no filename containing year {year}")
        continue
    if len(matches) != 1:
        failures.append(f"expected exactly one filename containing year {year}; found {len(matches)}")
    path = matches[0]
    record = {"year": year, "selected_first_file": path.name, "candidate_file_count": len(matches)}
    with xr.open_dataset(path) as dataset:
        missing_variables = sorted({"msl", "u10", "v10"} - set(dataset.variables))
        missing_coordinates = sorted({"valid_time", "latitude", "longitude"} - set(dataset.variables))
        if missing_variables:
            failures.append(f"{path.name}: missing variables {missing_variables}")
        if missing_coordinates:
            failures.append(f"{path.name}: missing coordinates {missing_coordinates}")
        if not missing_coordinates:
            times = np.asarray(dataset["valid_time"].values).astype("datetime64[D]")
            expected = np.arange(
                np.datetime64(f"{year}-05-22"),
                np.datetime64(f"{year}-08-26"),
                dtype="datetime64[D]",
            )
            observed = set(times.tolist())
            missing_dates = [str(value) for value in expected if value.tolist() not in observed]
            record.update(
                {
                    "first_date": str(times.min()) if times.size else "EMPTY",
                    "last_date": str(times.max()) if times.size else "EMPTY",
                    "accepted_lag_envelope_dates_required": int(expected.size),
                    "accepted_lag_envelope_dates_missing": len(missing_dates),
                }
            )
            if missing_dates:
                failures.append(
                    f"{path.name}: missing {len(missing_dates)} accepted-lag-envelope dates; first={missing_dates[0]}"
                )
    selected.append(record)

report.write_text(
    json.dumps(
        {
            "status": "PASS" if not failures else "FAIL",
            "selection_semantics": "lexicographically_first_filename_containing_year",
            "required_file_cardinality": "exactly one discoverable NetCDF per year",
            "required_variables": ["msl", "u10", "v10"],
            "required_coordinates": ["valid_time", "latitude", "longitude"],
            "required_daily_coverage": "1989-2025 May 22 through August 25",
            "coverage_basis": "accepted major event_start range June 1-August 20 plus M009 lags -10 through +5 days",
            "selected_files": selected,
            "failures": failures,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
if failures:
    raise SystemExit("HOLD_ERA5_ANNUAL_INPUT_CONTRACT:" + failures[0])
PY

verify_release_import_origin "$VRILE_SCIENCE_ENV" \
  "$VRILE_VALIDATION_ROOT/science_environment_import_origin.json"
verify_release_import_origin "$VRILE_GMT_ENV" \
  "$VRILE_VALIDATION_ROOT/gmt_environment_import_origin.json"

python_sources=()
while IFS= read -r -d '' path; do python_sources+=("$path"); done < <(find "$VRILE_PROJECT_ROOT" -type f -name '*.py' -print0)
conda run --no-capture-output -n "$VRILE_SCIENCE_ENV" python -B -m py_compile "${python_sources[@]}"
while IFS= read -r -d '' path; do bash -n "$path"; done < <(find "$VRILE_PROJECT_ROOT" -type f -name '*.sh' -print0)

df -h "$VRILE_RUN_ROOT" > "$VRILE_VALIDATION_ROOT/free_space_before.txt"
stage_success 02_preflight
echo "RELEASE_EXECUTABLE_SOURCE_ISOLATION=PASS"
echo "EXTERNAL_STATIC_INPUT_HASHES=PASS"
echo "RAW_INPUT_YEAR_COVERAGE_1989_2025=PASS"
echo "ERA5_ANNUAL_INPUT_CONTRACT=PASS"
echo "RESEARCHER_REPRODUCTION_PREFLIGHT=PASS"
