# Plot again from saved inputs

The full run generates Figures **2, 3, 4, 5, 6, S1, S2, S3 and S4** automatically.
Use these commands only when you already have a completed run and want a new
figure export. Figure 1 is a separate Methods flowchart.

## 1. Check the plotting environment

Use the GMT environment name you chose during installation:

```bash
GMT_ENV=gmt
bash scripts/runner/setup_plotting_runtime.sh --gmt-env "$GMT_ENV"
PLOT_PYTHON=$(python3 -B scripts/runner/plotting_runtime.py --print-python)
```

The exact PyMuPDF **1.28.2** requirement applies to both full generation and
standalone plotting. If the plotting venv is missing, run the setup command
with `--create`. The venv is separate from the Conda environments. The advanced
`VRILE_PLOT_PYTHON` override must pass the same version check.

## 2. Select saved inputs and a new output directory

Replace the example paths with your completed run and a new destination:

```bash
RUN_DIR=/path/to/completed_run
INPUT_DIR="$RUN_DIR/figures/current_presentation/current_inputs"
OUTPUT_DIR=/path/to/new_plot_outputs

"$PLOT_PYTHON" -B scripts/figures/render_current.py \
  --figure all --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --dry-run
```

`--dry-run` checks the small input contract and lists required files. It creates
no directories, launches no subprocesses and produces no figures or science.
The input bundle contains one `Figure_KEY/` directory per figure.

## 3. Render and check files

```bash
conda run --no-capture-output -n "$GMT_ENV" \
  "$PLOT_PYTHON" -B scripts/figures/render_current.py \
  --figure all --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --timeout 120

conda run --no-capture-output -n "$GMT_ENV" \
  "$PLOT_PYTHON" -B scripts/figures/check_outputs.py \
  --output_dir "$OUTPUT_DIR"
```

To render one figure, replace `all` with its key and add the same `--figure KEY`
to the checker. Choose a new destination for each render. Existing or partial
outputs are preserved; the renderer will not overwrite or silently retry them.
`--timeout` limits each figure job, including vector assembly and PNG exports.
If using GMT/Ghostscript outside Conda, `--gmt_bin_dir DIR` supplies their directory.

## 4. Find the exports

Whole figures are at `OUTPUT_DIR/Figure_KEY/figures/Figure_KEY.{pdf,svg,png}`.
Clean panels are at `OUTPUT_DIR/Figure_KEY/panels/Figure_KEY/*_clean.{pdf,svg,png}`.
PNG exports use 600 dpi. Logs and a render manifest accompany the exports.
The checker verifies the generated files, formats and panel structure.

Plotting reads existing estimates, intervals and saved example metadata. It does
not read RAW, rerun scientific stages, fit models or replace missing inputs.
Display operations include unit conversion, ECDFs, descriptive quartiles, point
placement, ranking and within-event/within-size subtraction. Fixed display
limits remain enforced; out-of-range populations stop with an explicit error.

## Optional: rebuild a plotting-input bundle

If the completed run retains its shared tables, source tables and results, the
following command creates a new bundle using only those saved products:

```bash
BUILD_DIR="$RUN_DIR/figures/current_presentation"
NEW_INPUT_DIR=/path/to/new_plot_inputs
"$PLOT_PYTHON" -B scripts/figures/bundle_saved_results.py \
  --prepared_dir "$BUILD_DIR/shared_tables/inputs" \
  --sidecar_dir "$BUILD_DIR/source_data" \
  --results_dir "$RUN_DIR/results" --output_dir "$NEW_INPUT_DIR"
```

The adapter preserves the saved representative-event identity, detector example
metadata and sample sizes. It does not synthesize scientific results. Use
`NEW_INPUT_DIR` as `INPUT_DIR` for the dry-run and render commands above.
