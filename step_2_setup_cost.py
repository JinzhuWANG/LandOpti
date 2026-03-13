"""
Step 2: Define cost/revenue parameters and transition costs for land use optimisation.

Land use classes (raster cell values):
    0 = Rice
    1 = Maize
    2 = Soy Bean
    3 = Urban (fixed, not re-assignable)
"""

import numpy as np
import xarray as xr
import rioxarray as rxr

# ── Load the land-use map ────────────────────────────────────────────────────
lumap = rxr.open_rasterio('data/LUMAP/CDL2019_clip_RES_100_ext.tif', masked=True).squeeze()

# ── Land-use class definitions ───────────────────────────────────────────────
CROPS = ["Rice", "Maize", "SoyBean"]
LANDUSES = ["Rice", "Maize", "SoyBean", "Urban"]
N_CROPS = len(CROPS)       # 3 (optimisable classes)
N_LANDUSES = len(LANDUSES) # 4 (including urban)

CROP_IDX = {c: i for i, c in enumerate(CROPS)}     # Rice:0, Maize:1, SoyBean:2
LANDUSE_IDX = {c: i for i, c in enumerate(LANDUSES)}

# ── Revenue and cost per cell (yuan/cell/year) ───────────────────────────────
REVENUE = {"Rice": 1200, "Maize": 1000, "SoyBean": 1100}
COST    = {"Rice": 550,  "Maize": 400,  "SoyBean": 450}
PROFIT  = {c: REVENUE[c] - COST[c] for c in CROPS}
# => Rice: 650, Maize: 600, SoyBean: 650  (narrow gaps discourage monoculture)

# ── Transition costs (from -> to), 0 if same crop ────────────────────────────
# Only defined for crop-to-crop transitions (Urban cells are fixed)
TRANS_COST = {
    ("Rice",    "Rice"):    0,
    ("Rice",    "Maize"):   350,
    ("Rice",    "SoyBean"): 300,
    ("Maize",   "Rice"):    380,
    ("Maize",   "Maize"):   0,
    ("Maize",   "SoyBean"): 280,
    ("SoyBean", "Rice"):    360,
    ("SoyBean", "Maize"):   300,
    ("SoyBean", "SoyBean"): 0,
}

# ── Policy area targets (fractions of total CROP cells, excluding urban) ─────
TARGET_MIN = {"Rice": 0.30, "Maize": 0.30, "SoyBean": 0.25}
TARGET_MAX = {"Rice": 0.40, "Maize": 0.40, "SoyBean": 0.35}

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
initial_map = lumap_flat[crop_indices]  # values 0, 1, or 2
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

# ── Compute net benefit array: net_benefit[i, j] = profit[j] - trans_cost[initial[i], j]
profit_arr = np.array([PROFIT[c] for c in CROPS], dtype=float)  # shape (3,)
net_benefit = np.zeros((N_CELLS, N_CROPS), dtype=float)
for i in range(N_CELLS):
    orig = initial_map[i]
    for j in range(N_CROPS):
        net_benefit[i, j] = profit_arr[j] - trans_arr[orig, j]

# ── Print initial land-use summary ───────────────────────────────────────────
print("\nInitial land use (crop cells only):")
for j, crop in enumerate(CROPS):
    cnt = int(np.sum(initial_map == j))
    print(f"  {crop}: {cnt} cells ({100 * cnt / N_CELLS:.1f}%)")

print(f"\nProfit per cell: {PROFIT}")
print(f"Transition cost matrix:\n{trans_arr}")
print(f"\nNet benefit array shape: {net_benefit.shape}")
print(f"Setup complete — ready for optimisation.")
