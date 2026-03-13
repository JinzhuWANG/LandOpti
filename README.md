# LandOpti

A land-use optimization system that finds optimal crop allocation patterns using mathematical programming. The system reads geospatial land cover rasters, formulates a mixed-integer program (MIP) with economic and policy constraints, and solves it with the [Gurobi](https://www.gurobi.com/) optimizer. A batch pipeline sweeps across parameter combinations, and an interactive HTML explorer lets stakeholders browse the results without re-running models.

![Land-use comparison](data/web/initial.png)

## Problem Description

Given a region with existing crop allocations (Rice, Maize, Soybean) and fixed urban cells, the optimizer decides which crop each cell should grow to **maximize net profit** (revenue − operating cost − transition cost) subject to:

- **Assignment:** every crop cell is assigned exactly one crop
- **Area targets:** each crop must occupy a specified fraction of total cropland (e.g., Rice 30–40%)
- **Transition costs:** switching a cell from one crop to another incurs a per-cell penalty

The default problem instance has **203,148 crop cells** (499 × 499 grid, excluding 44,150 urban cells), producing **609,444 binary decision variables** and **203,154 constraints**. Gurobi solves this in ~4 seconds.

## Pipeline

The project is organized as a five-step sequential pipeline:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step_1_dataprep.py` | Reads full-resolution GeoTIFF and downsamples 10× using rasterio streaming I/O (nearest-neighbor resampling to preserve discrete classes) |
| 2 | `step_2_setup_cost.py` | Defines land-use classes, economic parameters (revenue, cost, transition cost), area constraints, and loads the raster into an xarray |
| 3 | `step_3_gurobi_optimize.py` | Builds and solves the Gurobi MIP for a single scenario; exports optimized GeoTIFF and JSON summary |
| 4 | `step_4_visualise_results.py` | Renders a side-by-side matplotlib comparison of initial vs. optimized land-use maps |
| 5 | `step_5_batch_precompute.py` | Sweeps a parameter grid (profit × transition cost × area constraint mode) with joblib parallelization; outputs indexed PNGs + JSON for the web explorer |

**Dependency flow:**

```
step_1 (dataprep) → GeoTIFF artifacts
                         ↓
step_2 (setup)  ←  imported by steps 3–5
                         ↓
step_3 (optimize)    step_5 (batch)
       ↓                    ↓
step_4 (visualise)   explorer.html
```

## Default Economic Parameters

| Crop | Revenue (¥/cell/yr) | Cost (¥/cell/yr) | Net Profit (¥/cell/yr) |
|------|---------------------|-------------------|------------------------|
| Rice | 1,200 | 550 | 650 |
| Maize | 1,000 | 400 | 600 |
| Soybean | 1,100 | 450 | 650 |

Transition costs between different crops default to **350 ¥/cell** (same-crop transitions are free).

## Batch Parameter Grid

The batch pipeline (`step_5`) explores:

| Dimension | Values | Count |
|-----------|--------|-------|
| Rice profit | 450, 550, 650, 750, 850 | 5 |
| Maize profit | 400, 500, 600, 700, 800 | 5 |
| Soybean profit | 450, 550, 650, 750, 850 | 5 |
| Transition cost scale | 0.0, 0.5, 1.0, 1.5 | 4 |
| Area constraint mode | loose, medium, tight | 3 |

**Total: 1,500 scenarios** (use `--test` for an 8-scenario subset).

```bash
# Full run (1,500 scenarios, parallel)
python step_5_batch_precompute.py --jobs 4

# Quick test (8 scenarios)
python step_5_batch_precompute.py --test --jobs 2

# Dry run (show grid, don't solve)
python step_5_batch_precompute.py --dry-run
```

## Interactive Explorer

Open `explorer.html` in a browser to explore pre-computed scenarios. The explorer provides:

- Sliders for crop profit parameters and area constraint mode selection
- Side-by-side initial vs. optimized land-use maps
- Bar charts showing land allocation against target ranges
- Statistics: objective value, cells changed, solve time, feasibility status

> The explorer reads from `data/web/index.json` and `data/web/maps/` — no server required.

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (free for academic use)

### Setup

```bash
# Clone the repository
git clone git@github.com:JinzhuWANG/LandOpti.git
cd LandOpti

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate lu_opti

# Configure Gurobi license (if not already done)
grbgetkey <YOUR_LICENSE_KEY>
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.13 | Runtime |
| gurobipy | 13.0.0 | MIP solver |
| rasterio | 1.4.3 | Raster I/O |
| rioxarray | 0.20.0 | xarray + rasterio integration |
| xarray | 2025.10 | N-dimensional arrays |
| geopandas | 1.1.1 | Vector geospatial data |
| matplotlib | 3.10 | Visualization |
| joblib | (via dask) | Parallel batch processing |

## Quick Start

```bash
# 1. Prepare data (downsample raster)
python step_1_dataprep.py

# 2. Run single optimization
python step_3_gurobi_optimize.py

# 3. Visualize results
python step_4_visualise_results.py

# 4. (Optional) Run batch parameter sweep
python step_5_batch_precompute.py --test --jobs 2

# 5. (Optional) Open interactive explorer
open explorer.html   # or xdg-open on Linux
```

## Data

| Path | Description |
|------|-------------|
| `data/LUMAP/CDL2019_clip.tif` | Full-resolution land cover raster |
| `data/LUMAP/CDL2019_clip_RES_100.tif` | 10× downsampled raster |
| `data/LUMAP/town.tif` | Urban area mask |
| `data/LUMAP/town_distance.tif` | Distance-to-town raster |
| `data/SHP/` | Region of interest shapefiles |
| `data/result.json` | Single-run optimization summary |
| `data/web/` | Pre-computed batch results for the explorer |

**Land-use classes:** 0 = Rice, 1 = Maize, 2 = Soybean, 3 = Urban (fixed)

## Project Structure

```
LandOpti/
├── step_1_dataprep.py          # Data preparation & downsampling
├── step_2_setup_cost.py        # Parameters, costs, constraints
├── step_3_gurobi_optimize.py   # Single-scenario Gurobi solver
├── step_4_visualise_results.py # Matplotlib comparison plots
├── step_5_batch_precompute.py  # Batch parameter sweep
├── explorer.html               # Interactive web explorer
├── environment.yaml            # Conda environment spec
├── data/                       # Input rasters, shapefiles, results
└── Visulisation/               # Presentation materials
```

## License

This project uses the Gurobi optimization solver, which requires a separate license. See [Gurobi Academic Licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) for free academic access.
