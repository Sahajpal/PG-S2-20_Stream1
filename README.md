# Stream 1 — Vegetation–Network Risk Modelling
**Project pg-s2-20 | Vegetation Modelling for Risk Assessment and Outage Reduction | SA Power Networks**

Stream 1 builds a probabilistic, risk-ranking model of vegetation-related outage risk using statistical/ML techniques — vegetation proximity and condition, environmental covariates, and historic outage data. This README documents everything built and validated so far.

**Modelling approach:** gradient-boosted trees (XGBoost) throughout — not Random Forest or SVM — per direct supervisor guidance (Artur Sokolovsky), who described boosting trees as the more current generation of tree-based methods with better performance.

---

## 1. Boosting-Tree Risk Classifier (City of Unley baseline)

**Script:** `scripts/01_unley_risk_classifier/train_and_visualize.py`

**Why:** Before SAPN's outage data was released, we needed to validate the full modelling pipeline — data → features → boosting trees → evaluation → interpretable output — end to end on real data, so we're not starting from zero the moment SAPN's data lands.

**Data:** City of Unley's open street-tree dataset (opendata.unley.sa.gov.au) — 30,881 real trees, filtered to 30,040 with a valid council risk rating.

**Method:** Predicted "Elevated Risk" (Unley's Moderate + High risk categories combined) vs. "Low" from structural/species attributes (structural condition, health, canopy size, species genus, useful-life-expectancy, powerlines proximity flag). Only **1.18% of trees are labelled elevated risk** — severe class imbalance, handled explicitly via `scale_pos_weight` rather than ignored.

**Results:**
| Metric | Value |
|---|---|
| ROC-AUC | 0.903 |
| PR-AUC | 0.152 (12.8× lift over the 0.0119 random baseline at this class balance) |
| Recall (Elevated) | 0.78 |
| Precision (Elevated) | 0.07 |

Top features: structural condition, species genus (Eucalyptus, Lophostemon), health status, canopy size — all human-interpretable, not black-box signals.

**Honest caveat:** this predicts Unley council's own *arboricultural* risk rating (structural/health condition), **not** SAPN outage risk — that label isn't released yet. This validates the pipeline architecture; the target variable gets swapped once real outage data arrives.

---

## 2. Photo Feature-Extraction Proof of Concept

**Scripts:** `scripts/02_photo_feature_extraction/`

**Why:** Direct supervisor guidance (Artur): never ask an LLM/VLM to output a risk verdict directly from an image — it hallucinates unreliably in both directions. Instead, extract discrete structured features, then run those features through a tabular model.

**Method:** Manually extracted structured visual features (vegetation-to-conductor proximity, canopy density, height relative to pole, visible damage) from 2 sample poles Artur provided (each with an "overall" and "pole-top" photo), then mapped those features onto the Unley model's schema and scored them.

**Result:** the model correctly ranked the visually denser, damaged pole (Pole B) as **~29× higher predicted risk** than the healthier, more distant pole (Pole A) — demonstrating the same architecture generalizes across data sources, from council records to photo-derived features.

**Known blocker:** scaling this to the full ~3,000-image set requires a production VLM (via API batch processing — script ready in `extract_features_batch.py`). **This is currently blocked pending confirmation from Artur/SAPN on whether client-supplied photos can be sent to a third-party API under the current NDA terms.** Not yet resolved — do not run the batch script on real SAPN photos until confirmed.

---

## 3. NDVI vs. 2022 LiDAR Canopy Height Correlation

**Scripts:** `scripts/03_satellite_regression/get_ndvi_2022.py`, `correlate_ndvi_height.py`

**Why:** Direct supervisor suggestion (Artur) — check whether NDVI correlates with the 2022 LiDAR-derived canopy height data, since both are accessible without SAPN geolocation, making this fully unblocked ahead of any data release.

**Data:** Sentinel-2 NDVI (2022 cloud-masked median composite, via Google Earth Engine) vs. the Adelaide Metro 2022 canopy height layer (SA DEW/Green Adelaide, 0.5m resolution LiDAR-derived, GDA2020 MGA Zone 54).

**Result (full metro Adelaide, 9,007,164 valid paired pixels):**
| Metric | Value |
|---|---|
| Pearson r | 0.445 |
| Spearman ρ | 0.458 |

**Finding:** real, statistically robust correlation, but the relationship **saturates** at higher NDVI values — trees from ~15m to 50m tall show similar NDVI, because NDVI measures greenness/photosynthetic density, not physical structure. This is expected behaviour (well-documented in remote sensing) and directly motivated adding a structural signal (SAR) in the next stage.

---

## 4. Trained Satellite Height Regression

**Scripts:** `scripts/03_satellite_regression/get_multiband_2022.py`, `train_height_regression.py`

**Why:** This is Stream 1's core "Trained Satellite Regression" deliverable — train once on real 2022 LiDAR ground truth, then apply to fresh satellite imagery without needing repeat LiDAR surveys.

**Features:** NDVI, EVI, NDMI (Sentinel-2 optical) + VV, VH (Sentinel-1 SAR, added specifically to address NDVI's saturation at high canopy heights). Training restricted to pixels with height > 1m (real vegetation only — otherwise ~80% bare-ground/road pixels at height≈0 would let the model trivially "win" without learning anything useful).

**Results — validated at two scales, consistent between them:**
| | Pilot (small area) | Full metro Adelaide |
|---|---|---|
| Valid pixels | 19,113 (test) | 2,251,769 (test) |
| R² | 0.300 | 0.292 |
| MAE | 1.77 m | 2.08 m |
| RMSE | 2.52 m | 2.79 m |

**Feature importance (full metro):** NDVI 58%, NDMI 17%, EVI 12%, VV 7%, VH 5%.

**Key findings:**
- The model explains ~29% of height variance from satellite features alone — real signal, but the majority of what determines tree height isn't captured by these bands.
- **SAR contributes less than expected** (~12% combined) — a single yearly-median SAR composite doesn't appear to resolve the NDVI saturation problem it was added to address. Likely needs seasonal splitting or texture-based SAR features rather than a raw yearly average (documented as a next step, not yet attempted).
- The model **systematically under-predicts tall trees** (20m+ actual height predicted as ~10-20m) — the direct downstream consequence of NDVI saturation.
- For risk-banding purposes (short/medium/tall categories, rather than precise height in metres), ~2m average error is likely usable even with this limitation.

---

## What's NOT Done Yet

- **Species classification** — not started. Stream 1's design calls for height *and* species output; only height exists so far.
- **Combined risk-description output** — height regression is an input to this, not the final deliverable.
- **SAR improvement** (seasonal split / texture features) — identified, not yet attempted.
- **Batch photo pipeline at full scale (~3,000 images)** — blocked on NDA/API confirmation (see Section 2).
- **Validation against real SAPN outage data** — not possible until that data is released; all current risk-model work uses Unley council data as a proxy.

---

## Setup

```bash
pip install -r requirements.txt
```

Google Earth Engine additionally requires a one-time free account + Cloud project registration — see comments in `get_ndvi_2022.py` for the full setup sequence.

## Key Data Sources

- City of Unley Street Trees — opendata.unley.sa.gov.au/datasets/street-trees
- Adelaide Metro Urban Canopy Height 2022 — SA DEW / Green Adelaide, data.sa.gov.au
- Sentinel-1 (SAR) / Sentinel-2 (optical) — via Google Earth Engine, COPERNICUS/S1_GRD and COPERNICUS/S2_SR_HARMONIZED collections
- SAPN pole/tree photos — sample provided directly by Industry Supervisor; full ~3,000-image set pending NDA resolution

## Supervisor Guidance Incorporated

- Boosting trees (XGBoost), not Random Forest or SVM (Artur Sokolovsky)
- VLM feature extraction, never a direct LLM/VLM risk verdict (Artur Sokolovsky)
- NDVI/LiDAR correlation as an unblocked validation step (Artur Sokolovsky)
- Vegetation and network/asset data kept modular, not fused prematurely (Karamjit Kaur)