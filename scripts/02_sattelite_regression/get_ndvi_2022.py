"""
Pull a 2022 Sentinel-2 NDVI composite for the Adelaide metro area,
matching the extent of the 2022 canopy height LiDAR layer.

SETUP (one-time, see accompanying steps):
    pip install earthengine-api
    python -c "import ee; ee.Authenticate()"

RUN:
    python get_ndvi_2022.py

You MUST edit AOI_BOUNDS below to match the real extent of your
downloaded canopy height file (Step 1) before running this for real.
"""

import ee

# --- Adelaide metro extent, from UrbanCanopyHeight2022 metadata (WGS84 lat/lon) ---
# NOTE: this is the FULL metro area (~40km x 90km) -- for a first test run, consider
# shrinking this to a single suburb (e.g. Unley) to get a fast result before committing
# to the full export. Example small test box around Unley:
#   AOI_BOUNDS = [138.60, -34.97, 138.63, -34.94]
AOI_BOUNDS = [138.406583, -35.374270, 138.850815, -34.566007]

PROJECT_ID = "veg-risk-ndvi" 

ee.Initialize(project=PROJECT_ID)

aoi = ee.Geometry.Rectangle(AOI_BOUNDS)

# Sentinel-2 Surface Reflectance, cloud-masked, for 2022
def mask_clouds(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)

collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2022-01-01', '2022-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_clouds)
)

print(f"Number of images found for 2022: {collection.size().getInfo()}")

# Median composite across the year (reduces cloud/noise artifacts)
composite = collection.median()

# NDVI = (NIR - Red) / (NIR + Red) -> Sentinel-2 bands B8 (NIR), B4 (Red)
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Export to Google Drive as a GeoTIFF you can download
task = ee.batch.Export.image.toDrive(
    image=ndvi,
    description='adelaide_ndvi_2022',
    folder='EarthEngineExports',
    fileNamePrefix='adelaide_ndvi_2022',
    region=aoi,
    scale=10,  # Sentinel-2 native resolution in metres
    crs='EPSG:4326',
    maxPixels=1e9,
)
task.start()
print("Export started. Check Google Drive folder 'EarthEngineExports' in a few minutes.")
print("Check progress at: https://code.earthengine.google.com/tasks")
