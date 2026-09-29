# Prepare external RAW inputs

Create one data directory with the following layout. Replace `YEAR` with each
year from **1989 through 2025** and retain the provider filenames:

```text
raw/
├── nsidc_sie/N_seaice_extent_daily_v4.0.csv
├── nsidc_sic/YEAR/sic_psn25_YYYYMMDD_*.nc
├── nsidc_g02135_geotiff/north/daily/geotiff/YEAR/N_*_extent_v*.tif
├── nsidc_ancillary/NSIDC0771_CellArea_PS_N25km_v1.1.nc
├── nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc
└── era5/single_levels/one_annual_file_containing_YEAR.nc
```

Supply the complete daily input coverage for the study's summer calculations,
including required event windows. Keep original grids, metadata and data bytes.
The runner checks required directories, annual presence, exact ancillary hashes
and the ERA5 schema/time coverage. Later scientific stages check the particular
daily files they consume; a preflight pass does not certify every data value.

## Data sources and acknowledgement

| Input | Provider guidance |
| --- | --- |
| Daily sea-ice extent CSV and northern daily extent GeoTIFF | [NSIDC Sea Ice Index, G02135 Version 4](https://nsidc.org/data/g02135/versions/4) |
| Northern 25-km daily SIC NetCDF, with `cdr_seaice_conc` | [NOAA/NSIDC passive-microwave SIC climate data record, G02202 Version 6](https://nsidc.org/data/g02202/versions/6); the reproduction input filenames use `v06r00`; retain the exact product version and file metadata |
| `NSIDC0771_CellArea_PS_N25km_v1.1.nc` | [NSIDC-0771 polar stereographic ancillary grids](https://nsidc.org/data/nsidc-0771/versions/1) |
| `NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc` | [NSIDC-0780 regional masks](https://nsidc.org/data/nsidc-0780/versions/1) |
| ERA5 single-level atmospheric fields | [ECMWF/Copernicus ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) |

Follow the provider's access terms and cite the version, subset and access date
you actually used. A provider catalogue link is not permission to substitute a
new product version. The toolbox includes no downloader and does not redistribute
these datasets. Missing data must be obtained from the provider.

## ERA5 file contract

Each year must have exactly one discoverable NetCDF file directly under
`era5/single_levels/`, with that year in its filename. Each annual file must
contain `msl`, `u10`, `v10` and coordinates `valid_time`, `latitude`, `longitude`.
Required daily coverage is **May 22–August 25 for every year 1989–2025**.
Use the source temporal sampling required by the scientific inputs; this
date-coverage check does not resample data. Duplicate annual files or missing
variables/coordinates cause an explicit stop.

## Exact ancillary inputs

The two ancillary inputs have fixed SHA256 requirements:

```text
8197f5d4a7d4d3080fc60a5632adf60df20dce5a8490935f8de28f6322cbc095  NSIDC0771_CellArea_PS_N25km_v1.1.nc
5cc42dba4e57e7f55abe448e010100367fb679435024c7fcf37a29d55c29ca92  NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc
```

If a downloaded file differs, preserve it and investigate the version/format
before running; do not alter the toolbox's hash checks. Link this prepared
directory as `raw/` or pass it with `--raw_dir`, as shown in [RUNBOOK.md](RUNBOOK.md).
