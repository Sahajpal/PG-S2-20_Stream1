"""
Stream 1: Trained Satellite Regression -- height model
=========================================================
Trains an XGBoost REGRESSOR to predict canopy height (metres) from
Sentinel-1/2 spectral features, using the real 2022 LiDAR-derived
canopy height layer as ground truth.

This is the core "Trained Satellite Regression" pipeline: train once
on real ground truth, then apply to fresh satellite imagery without
needing repeat LiDAR surveys.

Uses boosting trees (XGBoost), not Random Forest/SVM, per direct
supervisor guidance.

SETUP:
    pip install rasterio numpy pandas scikit-learn xgboost matplotlib scipy

INPUTS (edit paths below):
    CANOPY_HEIGHT_TIF -- UrbanCanopyHeight2022.tif (ground truth, cm, EPSG:7854)
    FEATURES_TIF      -- adelaide_features_2022.tif from get_multiband_2022.py
                         (5 bands: NDVI, EVI, NDMI, VV, VH; EPSG:4326)
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

CANOPY_HEIGHT_TIF = "C:/Users/sahaj/OneDrive/Desktop/Capstone Project/Stream1/data/raw/UrbanCanopyHeight2022/UrbanCanopyHeight2022.tif"
FEATURES_TIF = "C:/Users/sahaj/OneDrive/Desktop/Capstone Project/Stream1/data/earth_engine_exports/adelaide_features_2022.tif"
CANOPY_NODATA = 65535
HEIGHT_SCALE = 100.0
MIN_HEIGHT_M = 1.0  # only train on real vegetation, not bare ground (see note below)

BAND_NAMES = ['NDVI', 'EVI', 'NDMI', 'VV', 'VH']

# ---------------------------------------------------------------------------
# 1. Load the feature stack (small -- 10m resolution)
# ---------------------------------------------------------------------------
print("Loading satellite feature stack...")
with rasterio.open(FEATURES_TIF) as src:
    features = src.read()  # shape: (bands, rows, cols)
    feat_transform = src.transform
    feat_crs = src.crs
    feat_shape = src.shape
    feat_nodata = src.nodata

print(f"Feature grid: {feat_shape}, bands: {features.shape[0]}")

# ---------------------------------------------------------------------------
# 2. Reproject canopy height onto the feature grid (same approach as the
#    correlation script -- averages the 0.5m LiDAR up to 10m)
# ---------------------------------------------------------------------------
print("Reprojecting canopy height onto feature grid...")
height_on_grid = np.zeros(feat_shape, dtype=np.float32)
with rasterio.open(CANOPY_HEIGHT_TIF) as height_src:
    reproject(
        source=rasterio.band(height_src, 1),
        destination=height_on_grid,
        src_transform=height_src.transform,
        src_crs=height_src.crs,
        src_nodata=CANOPY_NODATA,
        dst_transform=feat_transform,
        dst_crs=feat_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
height_on_grid = height_on_grid / HEIGHT_SCALE

# ---------------------------------------------------------------------------
# 3. Build a flat feature table, masking invalid pixels
#    NOTE: we restrict training to pixels with real canopy (height > 1m) --
#    otherwise ~90% of pixels are bare ground/roads at height=0, and the
#    model would trivially "win" by predicting near-zero everywhere without
#    actually learning to estimate height where it matters.
# ---------------------------------------------------------------------------
valid = ~np.isnan(height_on_grid) & (height_on_grid > MIN_HEIGHT_M)
for b in range(features.shape[0]):
    band = features[b]
    if feat_nodata is not None:
        valid &= (band != feat_nodata)
    valid &= ~np.isnan(band)

print(f"Valid vegetation pixels for training: {valid.sum():,} "
      f"({valid.sum()/valid.size*100:.1f}% of all pixels)")

X = pd.DataFrame({name: features[i][valid] for i, name in enumerate(BAND_NAMES)})
y = height_on_grid[valid]

# ---------------------------------------------------------------------------
# 4. Train/test split and XGBoost regression
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\nTraining XGBoost regressor...")
model = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=4
)
model.fit(X_train, y_train)
model.save_model('height_regression_model.json')

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"\n=== Results (held-out test set, n={len(y_test):,}) ===")
print(f"R\u00b2:   {r2:.3f}  (fraction of height variance explained by satellite features)")
print(f"MAE:  {mae:.2f} m  (average absolute error)")
print(f"RMSE: {rmse:.2f} m")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# ---------------------------------------------------------------------------
# 6. Visualize: predicted vs actual, and feature importance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
hb = ax.hexbin(y_test, pred, gridsize=50, cmap='viridis', mincnt=1, bins='log')
lims = [0, max(y_test.max(), pred.max())]
ax.plot(lims, lims, '--', color='red', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual height (m, 2022 LiDAR)')
ax.set_ylabel('Predicted height (m)')
ax.set_title(f'Predicted vs Actual\nR\u00b2={r2:.3f}, MAE={mae:.2f}m, RMSE={rmse:.2f}m')
ax.legend(fontsize=9)
fig.colorbar(hb, ax=ax, label='log10(count)')

ax = axes[1]
importance.sort_values().plot(kind='barh', ax=ax, color='#3F6E58')
ax.set_title('Feature Importance')
ax.set_xlabel('Importance')

plt.suptitle('Stream 1: Trained Satellite Height Regression (XGBoost)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('height_regression_results.png', dpi=170, bbox_inches='tight')
print("\nSaved: height_regression_results.png, height_regression_model.json")
