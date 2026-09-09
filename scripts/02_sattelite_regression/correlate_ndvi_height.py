"""
Correlate 2022 NDVI (Sentinel-2, from Earth Engine) against 2022 canopy
height (SA DEW/Green Adelaide LiDAR-derived raster).

This is the actual analysis Artur suggested: is there a detectable
relationship between NDVI and canopy height, using two real 2022
datasets that need no SAPN data or geolocation.

SETUP:
    pip install rasterio numpy matplotlib scipy

INPUTS (edit the two paths below):
    CANOPY_HEIGHT_TIF -- the real UrbanCanopyHeight2022.tif (0.5m resolution,
                          GDA2020 MGA Zone 54 / EPSG:7854, values in CENTIMETRES)
    NDVI_TIF          -- adelaide_ndvi_2022.tif exported from get_ndvi_2022.py
                          (10m resolution, EPSG:4326)

WHAT IT DOES:
    1. Reads the canopy height raster at a downsampled resolution using its
       built-in pyramids (fast -- avoids loading the full 0.5m grid).
    2. Reprojects/resamples that height data onto the NDVI raster's grid
       (mean height per NDVI pixel), since NDVI is much coarser (10m vs 0.5m).
    3. Masks out NoData and non-canopy (height <= 0) pixels.
    4. Computes Pearson correlation and produces a scatter/density plot.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy import stats
import matplotlib.pyplot as plt

CANOPY_HEIGHT_TIF = "data/UrbanCanopyHeight2022/UrbanCanopyHeight2022.tif"   
NDVI_TIF = "data/earth_engine_exports/adelaide_ndvi_2022.tif"                
CANOPY_NODATA = 65535
HEIGHT_SCALE = 100.0  # raw values are centimetres; divide by this for metres

# ---------------------------------------------------------------------------
# 1. Open both rasters, read NDVI fully (it's small -- 10m resolution)
# ---------------------------------------------------------------------------
print("Opening rasters...")
with rasterio.open(NDVI_TIF) as ndvi_src:
    ndvi = ndvi_src.read(1)
    ndvi_transform = ndvi_src.transform
    ndvi_crs = ndvi_src.crs
    ndvi_shape = ndvi_src.shape
    ndvi_nodata = ndvi_src.nodata

print(f"NDVI grid: {ndvi_shape}, CRS: {ndvi_crs}")

# ---------------------------------------------------------------------------
# 2. Reproject/resample the canopy height raster onto the NDVI grid.
#    rasterio's reproject() with Resampling.average will use the file's
#    built-in pyramids automatically when downsampling this much, so this
#    does NOT require loading the full 0.5m-resolution array into memory.
# ---------------------------------------------------------------------------
print("Reprojecting canopy height onto NDVI grid (this may take a few minutes)...")
height_on_ndvi_grid = np.zeros(ndvi_shape, dtype=np.float32)

with rasterio.open(CANOPY_HEIGHT_TIF) as height_src:
    reproject(
        source=rasterio.band(height_src, 1),
        destination=height_on_ndvi_grid,
        src_transform=height_src.transform,
        src_crs=height_src.crs,
        src_nodata=CANOPY_NODATA,
        dst_transform=ndvi_transform,
        dst_crs=ndvi_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,  # mean height per NDVI pixel
    )

height_on_ndvi_grid = height_on_ndvi_grid / HEIGHT_SCALE  # cm -> metres

# ---------------------------------------------------------------------------
# 3. Mask: drop NoData, drop non-canopy (height <= 0), drop NDVI NoData
# ---------------------------------------------------------------------------
valid = (
    ~np.isnan(height_on_ndvi_grid)
    & (height_on_ndvi_grid > 0)
    & ~np.isnan(ndvi)
)
if ndvi_nodata is not None:
    valid &= (ndvi != ndvi_nodata)

heights_flat = height_on_ndvi_grid[valid]
ndvi_flat = ndvi[valid]

print(f"\nValid paired pixels: {len(heights_flat):,}")
print(f"Height range: {heights_flat.min():.2f}m to {heights_flat.max():.2f}m "
      f"(mean {heights_flat.mean():.2f}m)")
print(f"NDVI range: {ndvi_flat.min():.3f} to {ndvi_flat.max():.3f} "
      f"(mean {ndvi_flat.mean():.3f})")

# ---------------------------------------------------------------------------
# 4. Correlation
# ---------------------------------------------------------------------------
r, p_value = stats.pearsonr(ndvi_flat, heights_flat)
print(f"\nPearson correlation (NDVI vs canopy height): r = {r:.3f}, p = {p_value:.2e}")

rho, p_rho = stats.spearmanr(ndvi_flat, heights_flat)
print(f"Spearman correlation (rank-based, catches non-linear trends): "
      f"rho = {rho:.3f}, p = {p_rho:.2e}")

# ---------------------------------------------------------------------------
# 5. Visualize -- hexbin, since there will likely be millions of points
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(ndvi_flat, heights_flat, gridsize=60, cmap='viridis', mincnt=1, bins='log')
ax.set_xlabel('NDVI (2022)')
ax.set_ylabel('Canopy height (m, 2022 LiDAR)')
ax.set_title(f'NDVI vs. Canopy Height — Adelaide Metro 2022\n'
             f'Pearson r = {r:.3f}  |  Spearman \u03c1 = {rho:.3f}  |  n = {len(heights_flat):,}')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(pixel count)')
plt.tight_layout()
plt.savefig('ndvi_vs_canopy_height_2022.png', dpi=170, bbox_inches='tight')
print("\nSaved: ndvi_vs_canopy_height_2022.png")
