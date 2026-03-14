"""
Step 2: Define cost/revenue parameters and transition costs for land use optimisation.

Land use classes (raster cell values):
    0 = Rice
    1 = Maize
    2 = Soy Bean
    3 = Urban (fixed, not re-assignable)
    4 = Tree (optimisable, no distance cost)
"""

import numpy as np
import xarray as xr
import rioxarray as rxr

# ── Load the land-use map ────────────────────────────────────────────────────
lumap = rxr.open_rasterio('data/LUMAP/CDL2019_clip_RES_100_ext.tif', masked=True).squeeze()
distance_cost = rxr.open_rasterio('data/LUMAP/town_distance.tif', masked=True).squeeze() * 100

# ── Land-use class definitions ───────────────────────────────────────────────
CROPS = ["Rice", "Maize", "SoyBean", "Tree"]
LANDUSES = ["Rice", "Maize", "SoyBean", "Urban", "Tree"]
N_CROPS = len(CROPS)       # 4 (optimisable classes)
N_LANDUSES = len(LANDUSES) # 5 (including urban)

CROP_IDX = {c: i for i, c in enumerate(CROPS)}     # Rice:0, Maize:1, SoyBean:2
LANDUSE_IDX = {c: i for i, c in enumerate(LANDUSES)}

# ── Revenue and cost per cell (yuan/cell/year) ───────────────────────────────
REVENUE = {"Rice": 1200, "Maize": 1000, "SoyBean": 1100, "Tree": 0}
COST    = {"Rice": 550,  "Maize": 400,  "SoyBean": 450,  "Tree": 0}
PROFIT  = {c: REVENUE[c] - COST[c] for c in CROPS}
# => Rice: 650, Maize: 600, SoyBean: 650, Tree: 0

# ── Transition costs (from -> to), 0 if same crop ────────────────────────────
# Only defined for crop-to-crop transitions (Urban cells are fixed)
TRANS_COST = {
    ("Rice",    "Rice"):    0,
    ("Rice",    "Maize"):   350,
    ("Rice",    "SoyBean"): 300,
    ("Rice",    "Tree"):    500,
    ("Maize",   "Rice"):    380,
    ("Maize",   "Maize"):   0,
    ("Maize",   "SoyBean"): 280,
    ("Maize",   "Tree"):    500,
    ("SoyBean", "Rice"):    360,
    ("SoyBean", "Maize"):   300,
    ("SoyBean", "SoyBean"): 0,
    ("SoyBean", "Tree"):    500,
    ("Tree",    "Rice"):    600,
    ("Tree",    "Maize"):   600,
    ("Tree",    "SoyBean"): 600,
    ("Tree",    "Tree"):    0,
}

# ── Policy area targets (fractions of total CROP cells, excluding urban) ─────
AREA_TARGET = {
    "Rice": (0.25, 0.35),
    "Maize": (0.25, 0.35),
    "SoyBean": (0.20, 0.30),
    "Tree": (0.05, 0.15),
}

# ── Flatten the raster and separate urban vs crop cells ──────────────────────
lumap_vals = lumap.values
# Replace NaN with -1 for masking
lumap_flat = np.where(np.isnan(lumap_vals), -1, lumap_vals).astype(int).ravel()

# Identify valid cells (non-NaN)
valid_mask = lumap_flat >= 0

# Separate urban (class 3) and crop cells (classes 0, 1, 2)
urban_mask = lumap_flat == 3
crop_mask = valid_mask & ~urban_mask

# Get indices and initial values of crop cells only
crop_indices = np.where(crop_mask)[0]
# Remap raster values to crop indices: 0→0(Rice), 1→1(Maize), 2→2(SoyBean), 4→3(Tree)
_raw_map = lumap_flat[crop_indices]
initial_map = np.where(_raw_map == 4, 3, _raw_map)
N_CELLS = len(crop_indices)

n_urban = int(np.sum(urban_mask))
n_total_valid = int(np.sum(valid_mask))

print(f"Raster shape: {lumap_vals.shape}")
print(f"Total valid cells: {n_total_valid}")
print(f"  Crop cells: {N_CELLS}")
print(f"  Urban cells: {n_urban}")

# ── Build transition cost matrix ─────────────────────────────────────────────
trans_arr = np.zeros((N_CROPS, N_CROPS), dtype=float)
for (f, t), v in TRANS_COST.items():
    trans_arr[CROP_IDX[f], CROP_IDX[t]] = v

# ── Flatten distance cost to match crop cells ─────────────────────────────
dist_vals = distance_cost.values
dist_flat = np.where(np.isnan(dist_vals), 0, dist_vals).ravel()
dist_crop = dist_flat[crop_indices]  # shape (N_CELLS,)

# ── Compute net benefit array: net_benefit[i, j] = profit[j] - trans_cost[initial[i], j] - dist_cost[i]
# Tree (crop index 3) has no distance cost; all other crops incur distance cost
profit_arr = np.array([PROFIT[c] for c in CROPS], dtype=float)  # shape (4,)
dist_matrix = np.tile(dist_crop[:, np.newaxis], (1, N_CROPS))  # shape (N_CELLS, 4)
dist_matrix[:, CROP_IDX["Tree"]] = 0.0  # Tree has no distance cost
net_benefit = profit_arr[np.newaxis, :] - trans_arr[initial_map, :] - dist_matrix

# ── Print initial land-use summary ───────────────────────────────────────────
print("\nInitial land use (crop cells only):")
for j, crop in enumerate(CROPS):
    cnt = int(np.sum(initial_map == j))
    print(f"  {crop}: {cnt} cells ({100 * cnt / N_CELLS:.1f}%)")

print(f"\nProfit per cell: {PROFIT}")
print(f"Transition cost matrix:\n{trans_arr}")
print(f"\nNet benefit array shape: {net_benefit.shape}")
print(f"Setup complete — ready for optimisation.")
