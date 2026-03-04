
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio


# Config
RESFACTOR = 10


# Read TIFF — downsample during read via out_shape (never loads full raster)
with rasterio.open("data/LUMAP/CDL2019_clip.tif") as src:
    full_h, full_w = src.height, src.width
    out_h = full_h // RESFACTOR
    out_w = full_w // RESFACTOR
    data = src.read(
        1,
        out_shape=(out_h, out_w),
        resampling=rasterio.enums.Resampling.nearest,
        out_dtype=np.uint8,
    )
    # Derive the new transform directly from rasterio
    new_transform = src.transform * src.transform.scale(
        full_w / out_w,
        full_h / out_h,
    )
    crs = src.crs

# Wrap in xarray DataArray to keep the rest of the pipeline unchanged
ys = new_transform.f + new_transform.e * (np.arange(out_h) + 0.5)
xs = new_transform.c + new_transform.a * (np.arange(out_w) + 0.5)
xr_crop = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
xr_crop = xr_crop.rio.write_crs(crs)
xr_crop = xr_crop.rio.write_transform(new_transform)



xr_crop.rio.to_raster(
    'data/LUMAP/CDL2019_clip_RES_100.tif',
    compress='LZW'
)


