# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LandOpti is a land-use optimization system that finds optimal crop allocation patterns (Rice, Maize, Soybean) using Gurobi mixed-integer programming. It reads geospatial land cover rasters, formulates a MIP with economic and policy constraints, solves it, and provides batch parameter sweeps with an interactive HTML explorer.

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate lu_opti
```

**Key Dependencies:** Python 3.13, gurobipy 13.0.0, rasterio, rioxarray, xarray, geopandas, matplotlib, dask/joblib.

Gurobi requires a license (`grbgetkey <KEY>`). Free for academic use.

## Pipeline Architecture

Five sequential steps — each step imports from `step_2_setup_cost.py` for shared parameters:

| Script | Purpose |
|--------|---------|
| `step_1_dataprep.py` | Downsample full-res GeoTIFF 10× via rasterio streaming I/O |
| `step_2_setup_cost.py` | Define land-use classes (0=Rice, 1=Maize, 2=Soybean, 3=Urban), economic params, area constraints; load raster into xarray; compute `net_benefit` matrix [N_CELLS × 3] |
| `step_3_gurobi_optimize.py` | Build and solve single-scenario MIP (binary x[i,j] variables); export optimized GeoTIFF + JSON summary |
| `step_4_visualise_results.py` | Side-by-side matplotlib comparison of initial vs. optimized maps |
| `step_5_batch_precompute.py` | Sweep 1,500-scenario parameter grid with joblib parallelization; output indexed PNGs + JSON for `explorer.html` |

**Dependency flow:** Steps 3–5 import variables from step 2. Step 1 produces the downsampled raster that step 2 loads.

## Key Data Structures

- **`net_benefit`** (numpy array, shape [N_CELLS, 3]): Per-cell net profit for each crop, computed in `step_2_setup_cost.py`. This is the objective coefficient matrix used by the optimizer.
- **`crop_mask`** / **`urban_mask`**: Boolean arrays separating optimizable crop cells from fixed urban cells.
- **Decision variables**: Binary `x[i,j]` — 1 if cell i is assigned crop j. Problem size: ~609K variables, ~203K constraints.

## Economic Parameters (defaults in step_2)

- Revenue (¥/cell/yr): Rice 1200, Maize 1000, Soybean 1100
- Cost (¥/cell/yr): Rice 550, Maize 400, Soybean 450
- Transition cost between different crops: 350 ¥/cell (same-crop = 0)
- Area targets: Rice 30–40%, Maize 30–40%, Soybean 25–35%

## Data Structure

- **Input raster:** `data/LUMAP/CDL2019_clip.tif` (full-res), `CDL2019_clip_RES_100.tif` (downsampled)
- **Urban data:** `data/LUMAP/town.tif`, `town_distance.tif`
- **Shapefiles:** `data/SHP/ROI_rec.*`, `ROI_town.*`
- **Outputs:** `data/result.json` (single run), `data/web/` (batch results + PNGs for explorer)
- **Land-use classes:** 0=Rice, 1=Maize, 2=Soybean, 3=Urban (fixed)

## Running

```bash
# Single optimization
python step_1_dataprep.py          # only needed once
python step_3_gurobi_optimize.py   # solves in ~4 seconds
python step_4_visualise_results.py

# Batch sweep
python step_5_batch_precompute.py --jobs 4          # full 1,500 scenarios
python step_5_batch_precompute.py --test --jobs 2   # 8-scenario test
python step_5_batch_precompute.py --dry-run          # show grid only

# Interactive explorer (no server needed)
xdg-open explorer.html
```

## Code Conventions

- Steps share state by importing from `step_2_setup_cost.py` directly (shared namespace pattern).
- Rasterio streaming reads are used to avoid loading full rasters into memory.
- Batch workers use `Params.OutputFlag=0` and single Gurobi thread to avoid over-subscription.
- Output rasters preserve CRS and transform metadata from the input via rioxarray.
