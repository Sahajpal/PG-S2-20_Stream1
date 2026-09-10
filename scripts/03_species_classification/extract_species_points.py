"""
Stream 1: Species classification -- Step 1, build the training table
=========================================================================
Samples the satellite feature stack (NDVI/EVI/NDMI/VV/VH) at each real
Unley tree's exact coordinates, pairing satellite "fingerprints" with
known species -- ground truth for a classifier, the same way the LiDAR
raster was ground truth for height regression.

SETUP:
    pip install rasterio pandas numpy

INPUTS (edit paths below):
    UNLEY_CSV     -- Street_Trees.csv (has LAT, LNG, GENUS columns)
    FEATURES_TIF  -- adelaide_features_2022.tif from get_multiband_2022.py

OUTPUT:
    species_training_table.csv -- one row per tree: genus + 5 satellite bands
"""

import pandas as pd
import numpy as np
import rasterio

UNLEY_CSV = "C:/Users/sahaj/OneDrive/Desktop/Capstone Project/Stream1/data/raw/Street_Trees.csv"
FEATURES_TIF = "C:/Users/sahaj/OneDrive/Desktop/Capstone Project/Stream1/data/earth_engine_exports/adelaide_features_2022.tif"
BAND_NAMES = ['NDVI', 'EVI', 'NDMI', 'VV', 'VH']
TOP_N_GENERA = 10  # keep the N most common genera, bucket the rest as "Other"

# ---------------------------------------------------------------------------
# 1. Load Unley trees, keep only rows with valid coordinates and a real genus
# ---------------------------------------------------------------------------
print("Loading Unley tree records...")
df = pd.read_csv(UNLEY_CSV, low_memory=False)
df = df[df['STATUS'] == 'Tree'].copy()  # exclude vacant sites/stumps/removed
df = df.dropna(subset=['LAT', 'LNG', 'GENUS'])
df = df[df['GENUS'].str.strip() != '']

print(f"Trees with valid location + genus: {len(df):,}")

# Bucket rare genera (same approach as the risk classifier)
top_genera = df['GENUS'].value_counts().nlargest(TOP_N_GENERA).index
df['genus_bucket'] = df['GENUS'].where(df['GENUS'].isin(top_genera), 'Other')
print("\nGenus distribution:")
print(df['genus_bucket'].value_counts())

# ---------------------------------------------------------------------------
# 2. Sample the satellite feature stack at each tree's exact coordinates
# ---------------------------------------------------------------------------
print("\nSampling satellite features at tree locations...")
with rasterio.open(FEATURES_TIF) as src:
    coords = list(zip(df['LNG'], df['LAT']))  # rasterio wants (x, y) = (lon, lat)
    samples = list(src.sample(coords))
    nodata = src.nodata

sample_array = np.array(samples)  # shape: (n_trees, n_bands)
for i, band in enumerate(BAND_NAMES):
    df[band] = sample_array[:, i]

# Drop trees that fell outside the raster extent or hit NoData
before = len(df)
valid = np.ones(len(df), dtype=bool)
for band in BAND_NAMES:
    valid &= ~df[band].isna()
    if nodata is not None:
        valid &= (df[band] != nodata)
df = df[valid]
print(f"\nDropped {before - len(df):,} trees outside raster extent / NoData")
print(f"Final training table: {len(df):,} trees")

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
output_cols = ['ASSET_ID', 'genus_bucket', 'LAT', 'LNG'] + BAND_NAMES
df[output_cols].to_csv('species_training_table.csv', index=False)
print("\nSaved: species_training_table.csv")
