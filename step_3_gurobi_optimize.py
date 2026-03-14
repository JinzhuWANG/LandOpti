"""
Step 3: Build and solve the Gurobi land-use optimisation model.

Objective: Maximise total net benefit (profit minus transition cost)
Subject to:
    - Each crop cell is assigned to exactly one crop
    - Area targets (min/max fraction) for each crop
    - Urban cells remain fixed (handled by excluding them from the model)

Reads setup from step_2_setup_cost.py via exec (all variables shared).
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from step_2_setup_cost import (
    COST,
    CROPS,
    N_CELLS,
    N_CROPS,
    PROFIT,
    REVENUE,
    AREA_TARGET,
    TRANS_COST,
    crop_indices,
    dist_crop,
    initial_map,
    lumap,
    lumap_flat,
    lumap_vals,
    n_urban,
    net_benefit,
    trans_arr,
)

# ── Build Gurobi Model ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Building Gurobi model...")
t0 = time.time()

m = gp.Model("LandUseOptimisation")
m.Params.LogToConsole = 1

# Decision variables: x[i, j] = 1 if crop cell i is assigned to crop j
x = m.addVars(N_CELLS, N_CROPS, vtype=GRB.BINARY, name="x")

# Objective: maximise total net benefit
obj = gp.quicksum(
    net_benefit[i, j] * x[i, j]
    for i in range(N_CELLS)
    for j in range(N_CROPS)
)
m.setObjective(obj, GRB.MAXIMIZE)

# Constraint 1: each cell assigned to exactly one crop
for i in range(N_CELLS):
    m.addConstr(
        gp.quicksum(x[i, j] for j in range(N_CROPS)) == 1,
        name=f"assign_{i}"
    )

# Constraint 2: area targets (fraction of crop cells, excluding urban)
for j, crop in enumerate(CROPS):
    total_j = gp.quicksum(x[i, j] for i in range(N_CELLS))
    lo, hi = AREA_TARGET[crop]
    m.addConstr(total_j >= lo * N_CELLS, name=f"min_{crop}")
    m.addConstr(total_j <= hi * N_CELLS, name=f"max_{crop}")

build_time = time.time() - t0
print(f"Model built in {build_time:.2f}s — {m.NumVars} vars, {m.NumConstrs} constraints")

# ── Solve ────────────────────────────────────────────────────────────────────
print("\nSolving...")
t1 = time.time()
m.optimize()
solve_time = time.time() - t1

# ── Extract results ──────────────────────────────────────────────────────────
if m.Status == GRB.OPTIMAL:
    print(f"\nOptimal solution found in {solve_time:.2f}s")

    # Extract optimal assignment for each crop cell
    opt_map = np.zeros(N_CELLS, dtype=int)
    for i in range(N_CELLS):
        for j in range(N_CROPS):
            if x[i, j].X > 0.5:
                opt_map[i] = j

    # Count changes
    changes = int(np.sum(opt_map != initial_map))

    # Final counts (Tree cells that were originally non-Tree are also counted as changes)
    final_counts = {c: int(np.sum(opt_map == k)) for k, c in enumerate(CROPS)}

    # Cost breakdown
    total_revenue = sum(REVENUE[CROPS[j]] * final_counts[CROPS[j]] for j in range(N_CROPS))
    total_cost = sum(COST[CROPS[j]] * final_counts[CROPS[j]] for j in range(N_CROPS))
    total_trans = 0.0
    for i in range(N_CELLS):
        orig, new = initial_map[i], opt_map[i]
        total_trans += trans_arr[orig, new]
    total_dist = float(np.sum(dist_crop))

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\nObjective (net profit): {m.ObjVal:,.0f}")
    print(f"  Revenue:         {total_revenue:,.0f}")
    print(f"  Operating cost:  -{total_cost:,.0f}")
    print(f"  Transition cost: -{total_trans:,.0f}")
    print(f"  Distance cost:   -{total_dist:,.0f}")
    print(f"\nCells changed: {changes} / {N_CELLS} ({100 * changes / N_CELLS:.1f}%)")

    print("\nFinal land use (crop cells):")
    for c, cnt in final_counts.items():
        print(f"  {c}: {cnt} cells ({100 * cnt / N_CELLS:.1f}%)")

    print("\nTarget compliance:")
    for j, crop in enumerate(CROPS):
        frac = final_counts[crop] / N_CELLS
        lo, hi = AREA_TARGET[crop]
        print(f"  {crop}: {frac:.1%}  (target: {lo:.0%} - {hi:.0%})")

    # ── Write optimised map back to raster ───────────────────────────────────
    # Reconstruct full raster: urban cells keep value 3, crop cells get new assignment
    # Remap crop index 3 (Tree) to raster value 4 to avoid conflict with Urban (3)
    opt_raster_vals = opt_map.copy()
    opt_raster_vals[opt_raster_vals == 3] = 4  # Tree → raster class 4
    opt_full = lumap_flat.copy()
    opt_full[crop_indices] = opt_raster_vals

    opt_raster_2d = opt_full.reshape(lumap_vals.shape)
    opt_xr = lumap.copy(data=opt_raster_2d.astype(float))
    # Set cells that were originally NaN back to NaN
    opt_xr = opt_xr.where(~np.isnan(lumap.values))

    out_path = "data/LUMAP/CDL2019_clip_RES_100_optimised.tif"
    opt_xr.rio.to_raster(out_path, compress="LZW")
    print(f"\nOptimised land-use map saved to: {out_path}")

    # ── Save summary to JSON ─────────────────────────────────────────────────
    import json
    result = {
        "crops": CROPS,
        "revenue": REVENUE,
        "cost": COST,
        "profit": PROFIT,
        "trans_cost": {f"{f}->{t}": v for (f, t), v in TRANS_COST.items()},
        "targets": {crop: {"min": lo, "max": hi} for crop, (lo, hi) in AREA_TARGET.items()},
        "initial_counts": {c: int(np.sum(initial_map == k)) for k, c in enumerate(CROPS)},
        "final_counts": final_counts,
        "n_crop_cells": N_CELLS,
        "n_urban_cells": n_urban,
        "cells_changed": changes,
        "obj_val": round(m.ObjVal),
        "total_revenue": round(total_revenue),
        "total_cost": round(total_cost),
        "total_trans": round(total_trans),
        "total_dist": round(total_dist),
        "solve_time": round(solve_time, 3),
        "build_time": round(build_time, 3),
        "n_vars": m.NumVars,
        "n_constrs": m.NumConstrs,
    }
    with open("data/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Summary saved to data/result.json")

else:
    print(f"Model status: {m.Status} (not optimal)")
    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible — check target constraints.")
        m.computeIIS()
        m.write("data/infeasible.ilp")
        print("IIS written to data/infeasible.ilp")
