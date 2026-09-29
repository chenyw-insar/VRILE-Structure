# Install and run

Use Linux/WSL on x86-64, Bash, Conda and system `python3` on PATH. Start in the
extracted toolbox directory. Plan at least 300 GiB of free run space in addition
to RAW storage; actual use depends on your data and filesystem. Full generation
is a long CPU/disk workload: keep the machine awake and use a persistent terminal.

## 1. Install

Choose two different, new environment names. These names are user choices;
the dependency locks are fixed.

```bash
SCI_ENV=vrile
GMT_ENV=gmt
conda create -y -n "$SCI_ENV" --file config/environment-lock-linux-64.explicit.txt
conda run -n "$SCI_ENV" python -m pip install --no-deps -r config/requirements-lock-pip.txt
conda create -y -n "$GMT_ENV" --file config/gmt-lock-linux-64.explicit.txt
bash scripts/runner/setup_plotting_runtime.sh --create --gmt-env "$GMT_ENV"
```

If an environment already exists, use it only if it matches the supplied lock;
otherwise choose another name. The runner checks both Conda locks and the science
pip overlay. It never replaces an existing mismatching environment.

```bash
./run_all.sh --help
conda run -n "$SCI_ENV" python -c 'import numpy,pandas,scipy; print(numpy.__version__,pandas.__version__,scipy.__version__)'
conda run -n "$GMT_ENV" gmt --version
conda run -n "$GMT_ENV" gs --version
bash scripts/runner/setup_plotting_runtime.sh --gmt-env "$GMT_ENV"
```

Both environments support scientific computation. The GMT environment also
provides mapping tools. Vector figure assembly uses a separate minimal venv
with **PyMuPDF 1.28.2**; the setup command creates it without changing either
Conda environment. `--create-missing-envs` is an optional runner convenience
for creating missing environments from these same locks.

## 2. Prepare RAW

Follow [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for the complete layout,
provider links and required coverage. Prepare the original NSIDC extent, SIC,
extent GeoTIFF, cell-area and region-mask products plus ERA5 single levels.
The toolbox does not download data or fill missing inputs.

For the default input path in a fresh extraction, replace only the empty `raw/`
placeholder. Substitute your actual data directory for `/path/to/raw`:

```bash
rmdir raw
ln -s /path/to/raw raw
```

If `raw/` is already populated or linked, leave it in place. Alternatively,
pass `--raw_dir /path/to/raw` when starting your run. RAW is always read-only
input to the toolbox. An OS read-only mount is recommended; add
`--require-readonly-raw` to require it. By default, writable storage produces
a warning and the toolbox leaves the data and permissions unchanged.

## 3. Run

With the default `raw/` and a new/empty `outputs/` directory:

```bash
./run_all.sh --science-env "$SCI_ENV" --gmt-env "$GMT_ENV"
```

Or choose your own paths:

```bash
RAW_DIR=/path/to/raw
RUN_DIR=/path/to/new_run
./run_all.sh --science-env "$SCI_ENV" --gmt-env "$GMT_ENV" \
  --raw_dir "$RAW_DIR" --run_dir "$RUN_DIR"
```

Choose one route. A new run gets an isolated copy of the toolbox. Its RAW path,
environment names and worker count are recorded at initialization. The run
directory must be new/empty and outside RAW and source directories; a separate
package-local directory such as `outputs_manual/` is supported. Use the same
path and environment choices in later commands for that run.

For an optional check before generation, add `--preflight-only` to your chosen
command:

```bash
./run_all.sh --preflight-only --science-env "$SCI_ENV" --gmt-env "$GMT_ENV" \
  --raw_dir "$RAW_DIR" --run_dir "$RUN_DIR"
```

Preflight initializes the run if needed, checks source integrity, environments,
plotting runtime and RAW inputs, and writes diagnostic reports and checkpoints
inside the run. It reads RAW without changing it and generates no scientific
results or figures. Start that prepared run with the same command after removing
`--preflight-only`.

`--init-only` is a lighter alternative: it creates the run layout and identity
without performing the complete environment/RAW checks or starting computation.
`--workers N` sets engineering parallelism (default 12) when creating a run.

The workflow automatically completes event detection, atmospheric analysis,
the fixed floor015 branches, component-resolved loss, cross-scale analysis and
figure production. Required scientific handoff and consistency checks run within
these stages. Existing scientific runs are never reset or automatically restarted.
Never start multiple runners against one run directory.

## 4. Find results

For the default run:

```bash
./run_all.sh --status
ls outputs/results
ls outputs/figures/final
ls outputs/logs
tail -n 30 outputs/logs/73_current_figure_structure.log
```

For a custom run:

```bash
./run_all.sh --status --run_dir "$RUN_DIR"
ls "$RUN_DIR/results"
ls "$RUN_DIR/figures/final"
ls "$RUN_DIR/logs"
```

Successful generation prints `RAW_TO_FINAL_GENERATION=PASS`. The status command
shows all nine checkpoints as `PASS_RECORDED`; it only reads saved checkpoint
state. Final figures are at
`figures/final/Figure_KEY/figures/Figure_KEY.{pdf,svg,png}` within the run.
Clean panels are at `figures/final/Figure_KEY/panels/Figure_KEY/`.
The keys are `2`, `3`, `4`, `5`, `6`, `S1`, `S2`, `S3`, `S4`.
Figure 1 is a separate Methods flowchart and is not generated here.

Keep the complete run directory, including `work/`, `state/`, `environment/`,
`evidence/`, `logs/`, results and figures. Runtime diagnostic reports are under
`validation/`. To plot again from saved inputs in a new directory, follow
[PLOTTING.md](PLOTTING.md).

## If a command stops

```bash
./run_all.sh --status --run_dir "$RUN_DIR"
ls -lt "$RUN_DIR/logs"
ls "$RUN_DIR/validation"
```

For the default run, set `RUN_DIR=outputs` first. Read the reported error and
the failed command's log. Missing-file, year-coverage or schema errors point to
RAW preparation. Environment-lock mismatches require a separate environment
created from the supplied lock. Plotting errors should first be checked with
`bash scripts/runner/setup_plotting_runtime.sh --gmt-env "$GMT_ENV"`.

Preserve a failed run. Do not delete checkpoints, edit its source copy or insert
previous results to continue. Resolve the cause before selecting a new run path.

## Optional: run one stage at a time

These commands are an alternative to `run_all.sh`. Use them in order, waiting
for each stage to finish successfully; each stage must be unstarted:

```bash
./run_stage1.sh --run_dir "$RUN_DIR" --raw_dir "$RAW_DIR" \
  --science-env "$SCI_ENV" --gmt-env "$GMT_ENV"
./run_stage2.sh --run_dir "$RUN_DIR"
./run_stage3.sh --run_dir "$RUN_DIR"
./run_stagex.sh --run_dir "$RUN_DIR"
```

Do not repeat these stages after a completed `run_all.sh`.
