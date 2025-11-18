# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LandOpti is a land use optimization project focused on analyzing and optimizing land use patterns using geospatial data from China's Land Cover Dataset (CLCD). The project uses Gurobi optimization solver and Python geospatial libraries.

## Environment Setup

The project uses conda for dependency management:

```bash
# Create the environment
conda env create -f environment.yaml

# Activate the environment
conda activate lu_opti
```

**Key Dependencies:**
- Python 3.13
- Gurobi 13.0.0 (optimization solver - requires license)
- Geospatial: rasterio, geopandas, rioxarray, xarray
- Data processing: numpy, pandas, dask
- Data formats: h5py, h5netcdf, netcdf4
- Visualization: matplotlib

## Data Structure

The project works with CLCD (China Land Cover Dataset) data:

- **Location:** `data/LUMAP/CLCD_v01_2020_albert_province/`
- **Format:** GeoTIFF files (.tif) for each Chinese province
- **Naming convention:** `CLCD_v01_2020_albert_{province}.tif`
- **Classification system:** Documented in `data/LUMAP/CLCD_classificationsystem.xlsx`
- **Coverage:** 34 provinces/regions including Hong Kong, Macao, and Taiwan

## Architecture Notes

**Current State:** The repository is in early development with only the environment configuration in place. The main data processing script `dataprep.py` is currently empty.

**Expected Architecture:**
- Data preparation/preprocessing likely handled in `dataprep.py`
- Geospatial data will be processed using rasterio/rioxarray for reading TIF files
- Optimization models will use Gurobi solver
- Large datasets may leverage dask for parallel/distributed computing
- NetCDF/HDF5 formats may be used for intermediate data storage

## Gurobi License

Gurobi requires a valid license to run optimization models. Ensure you have:
- Academic license (free for universities)
- Commercial license
- Or WLS (Web License Service) configured

License setup: `grbgetkey` command with your license key.

## Working with Geospatial Data

When reading CLCD raster data:
```python
import rasterio
import rioxarray

# Using rasterio
with rasterio.open('data/LUMAP/CLCD_v01_2020_albert_province/CLCD_v01_2020_albert_{province}.tif') as src:
    data = src.read()

# Using rioxarray (recommended for xarray integration)
ds = rioxarray.open_rasterio('data/LUMAP/...')
```

The classification system in `CLCD_classificationsystem.xlsx` should be referenced for interpreting pixel values in the raster data.
