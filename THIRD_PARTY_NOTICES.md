# Attribution and external dependencies

## Method lineage

Parts of this toolbox's method lineage were informed by and adapted from the
VRILE workflow and associated original toolbox described by Cavallo, S. M.,
Frank, M. C. and Bitz, C. M. (2025), “Sea ice loss in association with Arctic
cyclones,” *Communications Earth & Environment*, 6, 44,
[doi:10.1038/s43247-025-02022-9](https://doi.org/10.1038/s43247-025-02022-9).
This acknowledgement does not identify those authors as the authors of this
release or imply their endorsement of its project-specific extensions and repairs.
The article and original external toolbox are not bundled.

## Software dependencies

The toolbox uses Python and scientific/geospatial libraries including NumPy,
pandas, SciPy, xarray, Matplotlib, netCDF4, Rasterio, GDAL/PROJ, and related
packages, plus GMT and Ghostscript. Figure assembly uses PyMuPDF/MuPDF.
Dependency packages are installed separately and retain their own licenses,
copyright notices and citation guidance.

The complete pinned installation lists are:

- `config/environment-lock-linux-64.explicit.txt`: science Conda packages.
- `config/requirements-lock-pip.txt`: science pip overlay.
- `config/gmt-lock-linux-64.explicit.txt`: GMT Conda packages.
- `config/plotting-requirements.txt`: exact standalone plotting dependency.

These lists identify dependency versions; the project license does not relicense
those packages. Preserve upstream notices when redistributing dependencies.

## External data

The workflow uses NOAA/NSIDC sea-ice products, NSIDC ancillary grids and regional
masks, and ECMWF/Copernicus ERA5 data. Obtain and cite the actual product versions
and subsets used, following the providers' guidance linked in
[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md). RAW data, masks and grids are not
redistributed in this toolbox.

## Project metadata

[LICENSE](LICENSE) contains the MIT License, Copyright (c) 2026 Yaowen Chen.
It applies only to project-owned material the author has authority to license.
Software citation metadata are provided in [CITATION.cff](CITATION.cff).
