# VRILE-Structure v1.0.0

VRILE-Structure: A Pan-Arctic Toolkit for Very Rapid Sea-Ice Loss Event
Detection, Regional Analysis, and Spatial Decomposition

Detect very rapid sea-ice loss events, analyse regional conditions and decompose
pan-Arctic event structure through a reproducible RAW-to-final workflow. Generate
catalogues, atmospheric diagnostics and scientific Figures 2–6 and S1–S4 from
external RAW data for the 1989–2025 study period.
Use Linux/WSL (x86-64), Bash and Conda. Run these commands from the extracted
toolbox directory; see [RUNBOOK.md](RUNBOOK.md) for details.

## 1. Install

Choose two new environment names. `vrile` and `gmt` are defaults; you may change
either value. Keep the supplied dependency versions.

```bash
SCI_ENV=vrile
GMT_ENV=gmt
conda create -y -n "$SCI_ENV" --file config/environment-lock-linux-64.explicit.txt
conda run -n "$SCI_ENV" python -m pip install --no-deps -r config/requirements-lock-pip.txt
conda create -y -n "$GMT_ENV" --file config/gmt-lock-linux-64.explicit.txt
bash scripts/runner/setup_plotting_runtime.sh --create --gmt-env "$GMT_ENV"
```

## 2. Prepare RAW

Obtain the inputs listed in [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).
Replace `/path/to/raw` with your prepared data directory. In a fresh extraction,
replace the empty `raw/` placeholder with a link:

```bash
rmdir raw
ln -s /path/to/raw raw
```

If `raw/` is already populated or linked, keep it and use `--raw_dir` below.
The toolbox reads RAW data without changing it. RAW and previous results are
not included in the download.

## 3. Run

```bash
./run_all.sh --science-env "$SCI_ENV" --gmt-env "$GMT_ENV"
```

The default run directory is `outputs/`. To choose your own new/empty directory:

```bash
./run_all.sh --science-env "$SCI_ENV" --gmt-env "$GMT_ENV" \
  --raw_dir /path/to/raw --run_dir /path/to/new_run
```

Choose one command. Required input, environment and scientific consistency checks
run automatically. Existing runs are protected from overwrite.

## 4. Find results

```bash
./run_all.sh --status
ls outputs/results
ls outputs/figures/final
ls outputs/logs
```

For a custom run, add `--run_dir /path/to/new_run` to the status command and
replace `outputs` in the paths. Figures include PDF, SVG and 600 dpi PNG plus
independent clean panels. Preserve the complete run directory and logs.
[PLOTTING.md](PLOTTING.md) covers plotting again from saved inputs.

## Release, license and contact

Version: **1.0.0**. Release date: **2026-09-30**.
Author: **Yaowen Chen**. Contact:
[chenyaowen@hhu.edu.cn](mailto:chenyaowen@hhu.edu.cn).
Author homepage: <https://github.com/chenyw-insar> (not a software repository).

License: **MIT** for project-owned software; see [LICENSE](LICENSE).
Third-party software and data retain their original terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use VRILE-Structure in scientific work, please cite the software and the
associated paper once the paper becomes available.

Chen, Yaowen (2026). *VRILE-Structure: A Pan-Arctic Toolkit for Very Rapid Sea-Ice
Loss Event Detection, Regional Analysis, and Spatial Decomposition* (Version
1.0.0) [Software]. Machine-readable metadata: [CITATION.cff](CITATION.cff).

Associated paper: The companion VRILE manuscript is currently in preparation.
Please cite the published paper once bibliographic information becomes available.
No software DOI or formal software repository URL is currently available.

## Method lineage

Method lineage includes Cavallo, S. M., Frank, M. C. and Bitz, C. M. (2025),
“Sea ice loss in association with Arctic cyclones,” *Communications Earth &
Environment*, 6, 44, [doi:10.1038/s43247-025-02022-9](https://doi.org/10.1038/s43247-025-02022-9).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
