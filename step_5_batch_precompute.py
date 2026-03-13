"""
Step 5: Batch pre-compute Gurobi optimisation across a parameter grid.

Generates PNG maps and an index.json for the interactive explorer.
Uses joblib to parallelise solves across CPU cores.

Usage:
    python step_5_batch_precompute.py              # full run (~1500 scenarios)
    python step_5_batch_precompute.py --dry-run     # print grid size and exit
    python step_5_batch_precompute.py --test         # run only 8 scenarios for testing
    python step_5_batch_precompute.py --jobs 4       # limit to 4 parallel workers
"""

import argparse
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for multiprocessing

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap

from step_2_setup_cost import (
    CROPS,
    N_CROPS,
    crop_indices,
    initial_map,
    lumap,
    lumap_flat,
    lumap_vals,
    n_urban,
    trans_arr as BASE_TRANS_ARR,
)

# ── Parameter grid ────────────────────────────────────────────────────────────

PROFIT_RICE   = [450, 550, 650, 750, 850]
PROFIT_MAIZE  = [400, 500, 600, 700, 800]
PROFIT_SOY    = [450, 550, 650, 750, 850]
TRANS_SCALES  = [0.0, 0.5, 1.0, 1.5]
AREA_MODES    = ["loose", "medium", "tight"]

# Area constraint definitions
AREA_TARGETS = {
    "loose":  {"min": {"Rice": 0.15, "Maize": 0.15, "SoyBean": 0.15},
               "max": {"Rice": 0.55, "Maize": 0.55, "SoyBean": 0.55}},
    "medium": {"min": {"Rice": 0.30, "Maize": 0.30, "SoyBean": 0.25},
               "max": {"Rice": 0.40, "Maize": 0.40, "SoyBean": 0.35}},
    "tight":  {"min": {"Rice": 0.33, "Maize": 0.33, "SoyBean": 0.28},
               "max": {"Rice": 0.37, "Maize": 0.37, "SoyBean": 0.32}},
}

N_CELLS = len(initial_map)

# ── PNG rendering setup ──────────────────────────────────────────────────────
COLORS = ["#2ca02c", "#FFD700", "#FF8C00", "#808080", "#FFFFFF"]
CMAP = ListedColormap(COLORS)
RASTER_SHAPE = lumap_vals.shape

# Pre-build the fixed part of the raster (urban + nodata) once
_BASE_RASTER = lumap_flat.copy().astype(float)
_BASE_RASTER[_BASE_RASTER == -1] = 4  # NoData → index 4 (white)
# Crop cells will be overwritten per scenario


def build_full_raster(opt_map):
    """Reconstruct full raster from crop-cell assignments."""
    full = _BASE_RASTER.copy()
    full[crop_indices] = opt_map
    return full.reshape(RASTER_SHAPE)


def save_png(raster, path):
    """Save a raster as a small indexed-color PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=125)
    ax.imshow(raster, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=125)
    plt.close(fig)


def run_one_scenario(idx, p_rice, p_maize, p_soy, t_scale, a_mode, maps_dir,
                     initial_map_arr, base_trans_arr, crop_indices_arr,
                     base_raster_arr, raster_shape, n_cells, n_crops, crops):
    """Solve one scenario and save its PNG. Runs in a worker process."""
    t0 = time.time()

    profit_dict = {"Rice": p_rice, "Maize": p_maize, "SoyBean": p_soy}
    targets = AREA_TARGETS[a_mode]
    params = {
        "rice_profit": p_rice,
        "maize_profit": p_maize,
        "soy_profit": p_soy,
        "trans_scale": t_scale,
        "area_mode": a_mode,
    }

    # Compute net benefit
    profit_arr = np.array([profit_dict[c] for c in crops], dtype=float)
    trans_scaled = base_trans_arr * t_scale
    net_benefit = np.zeros((n_cells, n_crops), dtype=float)
    for i in range(n_cells):
        orig = initial_map_arr[i]
        for j in range(n_crops):
            net_benefit[i, j] = profit_arr[j] - trans_scaled[orig, j]

    # Build and solve Gurobi model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("batch", env=env)
    m.Params.LogToConsole = 0
    m.Params.Threads = 1  # each worker uses 1 thread to avoid oversubscription

    x = m.addVars(n_cells, n_crops, vtype=GRB.BINARY, name="x")

    obj = gp.quicksum(
        net_benefit[i, j] * x[i, j]
        for i in range(n_cells)
        for j in range(n_crops)
    )
    m.setObjective(obj, GRB.MAXIMIZE)

    for i in range(n_cells):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n_crops)) == 1)

    for j, crop in enumerate(crops):
        total_j = gp.quicksum(x[i, j] for i in range(n_cells))
        m.addConstr(total_j >= targets["min"][crop] * n_cells, name=f"min_{crop}")
        m.addConstr(total_j <= targets["max"][crop] * n_cells, name=f"max_{crop}")

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        env.dispose()
        elapsed = time.time() - t0
        return {
            "id": idx, "feasible": False, "params": params,
            "solve_time": round(elapsed, 2),
        }

    # Extract solution
    opt_map = np.zeros(n_cells, dtype=int)
    for i in range(n_cells):
        for j in range(n_crops):
            if x[i, j].X > 0.5:
                opt_map[i] = j

    changes = int(np.sum(opt_map != initial_map_arr))
    final_counts = {c: int(np.sum(opt_map == k)) for k, c in enumerate(crops)}
    obj_val = round(m.ObjVal)

    m.dispose()
    env.dispose()

    # Save PNG
    full = base_raster_arr.copy()
    full[crop_indices_arr] = opt_map
    raster = full.reshape(raster_shape)
    png_name = f"s_{idx:05d}.png"
    save_png(raster, os.path.join(maps_dir, png_name))

    elapsed = time.time() - t0
    return {
        "id": idx, "feasible": True, "params": params,
        "obj_val": obj_val,
        "final_counts": final_counts,
        "cells_changed": changes,
        "pct_changed": round(100 * changes / n_cells, 1),
        "solve_time": round(elapsed, 2),
        "png": png_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print grid size and exit")
    parser.add_argument("--test", action="store_true", help="Run only 8 scenarios")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers (-1 = all cores, default)")
    args = parser.parse_args()

    # Build grid
    if args.test:
        grid = list(itertools.product(
            [550, 750], [500, 700], [650], [0.5, 1.0], ["medium"]
        ))
    else:
        grid = list(itertools.product(
            PROFIT_RICE, PROFIT_MAIZE, PROFIT_SOY, TRANS_SCALES, AREA_MODES
        ))

    print(f"Parameter grid: {len(grid)} scenarios")
    if args.dry_run:
        print("Axes:")
        print(f"  Rice profit:   {PROFIT_RICE}")
        print(f"  Maize profit:  {PROFIT_MAIZE}")
        print(f"  SoyBean profit:{PROFIT_SOY}")
        print(f"  Trans scale:   {TRANS_SCALES}")
        print(f"  Area mode:     {AREA_MODES}")
        return

    # Output directories
    web_dir = "data/web"
    maps_dir = os.path.join(web_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    # Save initial map PNG
    initial_raster = build_full_raster(initial_map)
    save_png(initial_raster, os.path.join(web_dir, "initial.png"))
    print("Saved initial.png")

    # Prepare serializable arrays for workers
    initial_map_arr = np.array(initial_map)
    base_trans_arr = np.array(BASE_TRANS_ARR)
    crop_indices_arr = np.array(crop_indices)
    base_raster_arr = np.array(_BASE_RASTER)
    raster_shape = tuple(RASTER_SHAPE)
    crops = list(CROPS)

    n_jobs = args.jobs
    print(f"Running with joblib (n_jobs={n_jobs})...")
    t_total = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_scenario)(
            idx, p_rice, p_maize, p_soy, t_scale, a_mode, maps_dir,
            initial_map_arr, base_trans_arr, crop_indices_arr,
            base_raster_arr, raster_shape, N_CELLS, N_CROPS, crops,
        )
        for idx, (p_rice, p_maize, p_soy, t_scale, a_mode) in enumerate(grid)
    )

    total_time = time.time() - t_total

    # Sort by id to maintain grid order
    scenarios = sorted(results, key=lambda s: s["id"])

    # Print summary
    n_feasible = sum(1 for s in scenarios if s["feasible"])
    n_infeasible = len(scenarios) - n_feasible
    print(f"\nDone: {len(grid)} scenarios in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Feasible: {n_feasible}, Infeasible: {n_infeasible}")

    # Save index
    index = {
        "grid": {
            "rice_profit": PROFIT_RICE if not args.test else [550, 750],
            "maize_profit": PROFIT_MAIZE if not args.test else [500, 700],
            "soy_profit": PROFIT_SOY if not args.test else [650],
            "trans_scale": TRANS_SCALES if not args.test else [0.5, 1.0],
            "area_mode": AREA_MODES if not args.test else ["medium"],
        },
        "area_targets": AREA_TARGETS,
        "base_trans_cost": BASE_TRANS_ARR.tolist(),
        "crops": CROPS,
        "n_crop_cells": N_CELLS,
        "n_urban_cells": n_urban,
        "raster_shape": list(RASTER_SHAPE),
        "initial_counts": {c: int(np.sum(initial_map == k)) for k, c in enumerate(CROPS)},
        "scenarios": scenarios,
    }

    index_path = os.path.join(web_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
