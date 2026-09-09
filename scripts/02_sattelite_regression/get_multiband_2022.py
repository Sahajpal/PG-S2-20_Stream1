"""
Pull a richer 2022 satellite feature stack for Adelaide metro: Sentinel-2
NDVI/EVI/NDMI plus Sentinel-1 SAR VV/VH, all as one multi-band GeoTIFF.

This extends get_ndvi_2022.py -- same setup, same AOI. Run this INSTEAD of
(or in addition to) get_ndvi_2022.py.

SETUP: same as before (pip install earthengine-api, ee.Authenticate())
"""

import ee

# Use the SAME AOI you already tested with get_ndvi_2022.py, so the new
# export lines up with the canopy height file you already have.
AOI_BOUNDS = [138.406583, -35.374270, 138.850815, -34.566007]

PROJECT_ID = "veg-risk-ndvi"

ee.Initialize(project=PROJECT_ID)
aoi = ee.Geometry.Rectangle(AOI_BOUNDS)

# --- Sentinel-2: NDVI, EVI, NDMI ---
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)

s2 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2022-01-01', '2022-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .median()
)

ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
ndmi = s2.normalizedDifference(['B8', 'B11']).rename('NDMI')  # moisture index
evi = s2.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
    {'NIR': s2.select('B8'), 'RED': s2.select('B4'), 'BLUE': s2.select('B2')}
).rename('EVI')

# --- Sentinel-1: VV, VH (already speckle-filtered/calibrated by the archive) ---
s1 = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate('2022-01-01', '2022-12-31')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH'])
    .median()
)

# Stack everything into one multi-band image.
# .toFloat() is required here -- EVI (built via .expression()) comes out as
# Float64 by default, while NDVI/NDMI (via normalizedDifference) and the
# Sentinel-1 VV/VH bands are Float32. Earth Engine's exporter rejects mixed
# band types, so every band must be cast to the same type before export.
stack = ee.Image.cat([ndvi, evi, ndmi, s1.select('VV'), s1.select('VH')]).toFloat()

print("Bands in export:", stack.bandNames().getInfo())

task = ee.batch.Export.image.toDrive(
    image=stack,
    description='adelaide_features_2022',
    folder='EarthEngineExports',
    fileNamePrefix='adelaide_features_2022',
    region=aoi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9,
)
task.start()
print("Export started -- 5 bands: NDVI, EVI, NDMI, VV, VH")
print("Check progress at: https://code.earthengine.google.com/tasks")