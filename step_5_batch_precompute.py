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
    dist_crop,
    initial_map,
    lumap,
    lumap_flat,
    lumap_vals,
    n_urban,
    trans_arr as BASE_TRANS_ARR,
)

# ── Parameter grid ────────────────────────────────────────────────────────────

PROFIT_RICE   = [300, 450, 600, 750, 900]
PROFIT_MAIZE  = [200, 350, 500, 650, 800]
PROFIT_SOY    = [250, 400, 550, 700, 850]
TRANS_SCALES  = [0.5, 1.0, 1.5, 2.0]
AREA_MODES    = ["loose", "medium", "tight"]
TREE_TARGETS  = [0.05, 0.15, 0.20]

# Area constraint definitions (Tree min/max are set per-scenario from TREE_TARGETS)
AREA_TARGETS = {
    "loose":  {"min": {"Rice": 0.10, "Maize": 0.10, "SoyBean": 0.10},
               "max": {"Rice": 0.55, "Maize": 0.55, "SoyBean": 0.55}},
    "medium": {"min": {"Rice": 0.25, "Maize": 0.25, "SoyBean": 0.20},
               "max": {"Rice": 0.35, "Maize": 0.35, "SoyBean": 0.30}},
    "tight":  {"min": {"Rice": 0.28, "Maize": 0.28, "SoyBean": 0.23},
               "max": {"Rice": 0.32, "Maize": 0.32, "SoyBean": 0.27}},
}

N_CELLS = len(initial_map)

# ── PNG rendering setup ──────────────────────────────────────────────────────
COLORS = ["#00BFFF", "#FFA500", "#2ca02c", "#FF0000", "#006400", "#FFFFFF"]
CMAP = ListedColormap(COLORS)
RASTER_SHAPE = lumap_vals.shape

# Pre-build the fixed part of the raster (urban + nodata) once
_BASE_RASTER = lumap_flat.copy().astype(float)
_BASE_RASTER[_BASE_RASTER == -1] = 5  # NoData → index 5 (white)
# Crop cells will be overwritten per scenario


def build_full_raster(opt_map):
    """Reconstruct full raster from crop-cell assignments."""
    # Remap crop index 3 (Tree) to raster value 4 to avoid conflict with Urban (3)
    raster_vals = opt_map.copy()
    raster_vals[raster_vals == 3] = 4
    full = _BASE_RASTER.copy()
    full[crop_indices] = raster_vals
    return full.reshape(RASTER_SHAPE)


def save_png(raster, path):
    """Save a raster as a small indexed-color PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=125)
    ax.imshow(raster, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=125)
    plt.close(fig)


def run_one_scenario(idx, p_rice, p_maize, p_soy, t_scale, a_mode, tree_tgt,
                     maps_dir,
                     initial_map_arr, base_trans_arr, crop_indices_arr,
                     base_raster_arr, raster_shape, n_cells, n_crops, crops,
                     dist_crop_arr):
    """Solve one scenario and save its PNG. Runs in a worker process."""
    t0 = time.time()

    profit_dict = {"Rice": p_rice, "Maize": p_maize, "SoyBean": p_soy, "Tree": 0}
    targets = {
        "min": {**AREA_TARGETS[a_mode]["min"], "Tree": tree_tgt},
        "max": {**AREA_TARGETS[a_mode]["max"], "Tree": min(tree_tgt + 0.10, 0.30)},
    }
    params = {
        "rice_profit": p_rice,
        "maize_profit": p_maize,
        "soy_profit": p_soy,
        "trans_scale": t_scale,
        "area_mode": a_mode,
        "tree_target": tree_tgt,
    }

    # Compute net benefit (Tree has no distance cost)
    profit_arr = np.array([profit_dict[c] for c in crops], dtype=float)
    trans_scaled = base_trans_arr * t_scale
    dist_matrix = np.tile(dist_crop_arr[:, np.newaxis], (1, n_crops))
    tree_idx = crops.index("Tree")
    dist_matrix[:, tree_idx] = 0.0
    net_benefit = profit_arr[np.newaxis, :] - trans_scaled[initial_map_arr, :] - dist_matrix

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

    # Save PNG — remap crop index 3 (Tree) to raster value 4 (Urban is 3)
    opt_raster_vals = opt_map.copy()
    opt_raster_vals[opt_raster_vals == 3] = 4
    full = base_raster_arr.copy()
    full[crop_indices_arr] = opt_raster_vals
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
            [450, 750], [350, 650], [400, 700], [0.5, 1.0], ["medium"], [0.05, 0.15]
        ))
    else:
        grid = list(itertools.product(
            PROFIT_RICE, PROFIT_MAIZE, PROFIT_SOY, TRANS_SCALES, AREA_MODES, TREE_TARGETS
        ))

    print(f"Parameter grid: {len(grid)} scenarios")
    if args.dry_run:
        print("Axes:")
        print(f"  Rice profit:   {PROFIT_RICE}")
        print(f"  Maize profit:  {PROFIT_MAIZE}")
        print(f"  SoyBean profit:{PROFIT_SOY}")
        print(f"  Trans scale:   {TRANS_SCALES}")
        print(f"  Area mode:     {AREA_MODES}")
        print(f"  Tree target:   {TREE_TARGETS}")
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
    dist_crop_arr = np.array(dist_crop)

    n_jobs = args.jobs
    print(f"Running with joblib (n_jobs={n_jobs})...")
    t_total = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_scenario)(
            idx, p_rice, p_maize, p_soy, t_scale, a_mode, tree_tgt,
            maps_dir,
            initial_map_arr, base_trans_arr, crop_indices_arr,
            base_raster_arr, raster_shape, N_CELLS, N_CROPS, crops,
            dist_crop_arr,
        )
        for idx, (p_rice, p_maize, p_soy, t_scale, a_mode, tree_tgt) in enumerate(grid)
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
    # Build area_targets with Tree info for each tree_target level
    area_targets_out = {}
    for mode in (AREA_MODES if not args.test else ["medium"]):
        area_targets_out[mode] = {
            "min": {**AREA_TARGETS[mode]["min"]},
            "max": {**AREA_TARGETS[mode]["max"]},
        }
    index = {
        "grid": {
            "rice_profit": PROFIT_RICE if not args.test else [550, 750],
            "maize_profit": PROFIT_MAIZE if not args.test else [500, 700],
            "soy_profit": PROFIT_SOY if not args.test else [650],
            "trans_scale": TRANS_SCALES if not args.test else [0.5, 1.0],  # no 0.0
            "area_mode": AREA_MODES if not args.test else ["medium"],
            "tree_target": TREE_TARGETS if not args.test else [0.05, 0.15],
        },
        "area_targets": area_targets_out,
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
