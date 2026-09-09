"""
Stream 1 baseline: gradient-boosted tree vegetation-risk classifier
=====================================================================
Trains an XGBoost model on the City of Unley street-tree dataset to
predict "elevated risk" trees from structural/species attributes.

Uses boosting trees specifically (not Random Forest or SVM), per
direct supervisor guidance (Artur Sokolovsky).

SETUP (run once):
    pip install pandas numpy scikit-learn xgboost matplotlib

RUN:
    python train_and_visualize.py

INPUT:
    Street_Trees.csv must be in the same folder as this script
    (City of Unley open data: opendata.unley.sa.gov.au/datasets/street-trees)

OUTPUT:
    - Console: dataset stats, ROC-AUC, PR-AUC, classification report, feature importance
    - stream1_model_results.png: 3-panel chart (feature importance, ROC curve, PR curve)
    - xgb_risk_model.json: the trained model, reloadable later
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve)
import xgboost as xgb

CSV_PATH = "Street_Trees.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and label
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH, low_memory=False)

# Keep only rows with a real RISK label; combine Moderate+High into "elevated"
df = df[df['RISK'].isin(['Low', 'Moderate', 'High'])].copy()
df['elevated_risk'] = df['RISK'].isin(['Moderate', 'High']).astype(int)

print(f"Total labelled rows: {len(df)}")
print(f"Elevated risk (Moderate+High): {df['elevated_risk'].sum()} "
      f"({df['elevated_risk'].mean()*100:.2f}%)\n")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
cat_cols = ['AGE', 'CANOPY', 'STRUCTURE', 'HEALTH', 'ULE', 'POWERLINES',
            'PARK_TREE', 'PRIVATE_TR', 'SIGNIFICAN']
for c in cat_cols:
    df[c] = df[c].fillna('Unknown').replace('', 'Unknown')

# Bucket rare genera to avoid excessive cardinality
top_genus = df['GENUS'].value_counts().nlargest(15).index
df['GENUS_bucket'] = df['GENUS'].where(df['GENUS'].isin(top_genus), 'Other')
cat_cols.append('GENUS_bucket')

df['DBH_VALUE'] = pd.to_numeric(df['DBH_VALUE'], errors='coerce').fillna(df['DBH_VALUE'].median())

X = pd.get_dummies(df[cat_cols + ['DBH_VALUE']], columns=cat_cols, drop_first=False)
X.columns = [c.replace('<', 'lt').replace('>', 'gt').replace('[', '').replace(']', '') for c in X.columns]
y = df['elevated_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Train: gradient-boosted trees, with class-imbalance handling
# ---------------------------------------------------------------------------
print("Training XGBoost classifier...")
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
    random_state=RANDOM_STATE, n_jobs=4
)
model.fit(X_train, y_train)
model.save_model('xgb_risk_model.json')

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)
baseline_pr = y_test.mean()

print("\n=== Results (held-out test set) ===")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}  (random baseline at this class balance = {baseline_pr:.4f}, "
      f"a {pr_auc/baseline_pr:.1f}x lift)")
print()
print(classification_report(y_test, preds, target_names=['Low', 'Elevated']))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 12 features:")
print(importance.head(12))

# ---------------------------------------------------------------------------
# 5. Visualize
# ---------------------------------------------------------------------------
print("\nBuilding visualization...")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
imp_sorted = importance.head(15).sort_values()
colors = ['#3F6E58' if v > 0.03 else '#9AA5A0' for v in imp_sorted.values]
ax.barh(imp_sorted.index.str.replace('_', ': '), imp_sorted.values, color=colors)
ax.set_title('Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
ax.tick_params(axis='y', labelsize=8)

ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, proba)
ax.plot(fpr, tpr, color='#17105F', lw=2, label=f'XGBoost (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#9AA5A0', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')

ax = axes[2]
prec, rec, _ = precision_recall_curve(y_test, proba)
ax.plot(rec, prec, color='#C97A1A', lw=2, label=f'XGBoost (PR-AUC = {pr_auc:.3f})')
ax.axhline(baseline_pr, ls='--', color='#9AA5A0', lw=1, label=f'Random baseline ({baseline_pr:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Stream 1 Baseline: Boosting-Tree Elevated-Risk Classifier — City of Unley Data',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('stream1_model_results.png', dpi=180, bbox_inches='tight', facecolor='white')
print("Saved: stream1_model_results.png, xgb_risk_model.json")
