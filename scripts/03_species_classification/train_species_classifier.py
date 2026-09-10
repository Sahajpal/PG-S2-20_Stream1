"""
Stream 1: Species classification -- Step 2, train the model
================================================================
Trains an XGBoost multiclass classifier to predict tree genus from
satellite spectral features, using real Unley tree records as
ground truth. Boosting trees per direct supervisor guidance.

SETUP:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

INPUT:
    species_training_table.csv -- from extract_species_points.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt

INPUT_CSV = "C:/Users/sahaj/OneDrive/Desktop/Capstone Project/Stream1/data/raw/species_training_table.csv"
BAND_NAMES = ['NDVI', 'EVI', 'NDMI', 'VV', 'VH']

# ---------------------------------------------------------------------------
# 1. Load and encode
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Total trees: {len(df):,}")
print(f"\nClass balance:\n{df['genus_bucket'].value_counts()}")

le = LabelEncoder()
y = le.fit_transform(df['genus_bucket'])
X = df[BAND_NAMES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
print("\nTraining XGBoost multiclass classifier...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    objective='multi:softprob', num_class=len(le.classes_),
    eval_metric='mlogloss', random_state=42, n_jobs=4
)
model.fit(X_train, y_train)
model.save_model('species_classifier_model.json')

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average='macro')
weighted_f1 = f1_score(y_test, pred, average='weighted')

print(f"\n=== Results (held-out test set, n={len(y_test):,}) ===")
print(f"Accuracy:      {acc:.3f}")
print(f"Macro F1:      {macro_f1:.3f}  (treats every genus equally, regardless of how common it is)")
print(f"Weighted F1:   {weighted_f1:.3f}  (weighted by how common each genus actually is)")
print(f"\nFor comparison, always guessing the most common genus would score: "
      f"{(df['genus_bucket'].value_counts().iloc[0] / len(df)):.3f} accuracy")

print("\nPer-class report:")
print(classification_report(y_test, pred, target_names=le.classes_))

importance = pd.Series(model.feature_importances_, index=BAND_NAMES).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# ---------------------------------------------------------------------------
# 4. Visualize: confusion matrix + feature importance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm = confusion_matrix(y_test, pred, normalize='true')
im = axes[0].imshow(cm, cmap='Greens', vmin=0, vmax=1)
axes[0].set_xticks(range(len(le.classes_)))
axes[0].set_yticks(range(len(le.classes_)))
axes[0].set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(le.classes_, fontsize=8)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix (row-normalized)\nAccuracy={acc:.3f}, Macro F1={macro_f1:.3f}')
fig.colorbar(im, ax=axes[0], label='Fraction of actual class')

importance.sort_values().plot(kind='barh', ax=axes[1], color='#3F6E58')
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance')

plt.suptitle('Stream 1: Species Classification from Satellite Bands (XGBoost)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('species_classification_results.png', dpi=170, bbox_inches='tight')
print("\nSaved: species_classification_results.png, species_classifier_model.json")
