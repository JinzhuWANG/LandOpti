
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

from affine import Affine


# Config
RESFACTOR = 10


# Read TIFF
LUCC_lookup = pd.read_excel(
        'data/LUMAP/CLCD_classificationsystem.xlsx', 
        sheet_name='Sheet1'
    ).set_index('ID')['Class'].to_dict()

xr_chongqing = rxr.open_rasterio(
    'data/LUMAP/CLCD_v01_2020_albert_province/CLCD_v01_2020_albert_chongqing.tif', 
    masked=True,
    chunks='auto'
    ).sel(band=1, drop=True
    ).isel(x=slice(1, None, RESFACTOR), y=slice(1, None, RESFACTOR)
    ).astype(np.uint8
    )

# Update transform
trans = list(xr_chongqing.rio.transform())
cell_size_x = trans[0] * RESFACTOR
cell_size_y = trans[4] * RESFACTOR
trans[2] = min(xr_chongqing.x.values) + (cell_size_x / 2)
trans[5] = max(xr_chongqing.y.values) - (cell_size_y / 2)
trans[0] = cell_size_x
trans[4] = -cell_size_y
xr_chongqing.rio.write_transform(Affine(*trans), inplace=True)



xr_chongqing.rio.to_raster(
    'data/LUMAP/CLCD_v01_2020_albert_province/CLCD_v01_2020_albert_chongqing_coarse10.tif',
    compress='LZW'
)


