"""
Step 4: Visualise initial vs optimised land-use maps side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rioxarray as rxr

# ── Load rasters ─────────────────────────────────────────────────────────────
initial = rxr.open_rasterio('data/LUMAP/CDL2019_clip_RES_100_ext.tif', masked=True).squeeze()
optimised = rxr.open_rasterio('data/LUMAP/CDL2019_clip_RES_100_optimised.tif', masked=True).squeeze()

# ── Colour map: 0=Rice(green), 1=Maize(gold), 2=SoyBean(orange), 3=Urban(grey) ──
colors = ["#2ca02c", "#FFD700", "#FF8C00", "#808080"]
cmap = ListedColormap(colors)
labels = ["Rice", "Maize", "Soy Bean", "Urban"]
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, data, title in [
    (axes[0], initial, "Initial Land Use"),
    (axes[1], optimised, "Optimised Land Use"),
]:
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=11, frameon=False)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("data/landuse_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to data/landuse_comparison.png")
