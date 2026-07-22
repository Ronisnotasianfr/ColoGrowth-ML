"""demo_predict.py - Quick inference demo for ScienceMontgomery judging.
Loads a pre-trained model and runs predict_proba on 5 test samples in <5 seconds.
Usage:
    python scripts/demo_predict.py
"""

import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

model_path = BASE / "models" / "random_forest_geo_pan.joblib"
X_path = BASE / "data" / "processed" / "geo_pan_X_features.csv"
y_path = BASE / "data" / "processed" / "geo_pan_y_target.csv"

if not model_path.exists():
    print(f"Model not found at {model_path}")
    print("Run the full pipeline first: python src/train.py --dataset geo_pan")
    exit(1)

model = joblib.load(model_path)

X = pd.read_csv(X_path, index_col=0).iloc[:5]
y = pd.read_csv(y_path, index_col=0).iloc[:5]

probs = model.predict_proba(X)[:, 1]

print("=" * 60)
print("ColoGrowth-ML Inference Demo")
print(f"Model: Random Forest (GEO_PAN)")
print("=" * 60)
for i in range(5):
    true_label = y.iloc[i, 0]
    pred_prob = probs[i]
    pred_label = 1 if pred_prob > 0.5 else 0
    correct = pred_label == true_label
    mark = "CORRECT" if correct else "WRONG"
    print(f"  Sample {i+1}: True={true_label}, Predicted prob={pred_prob:.3f}, Class={pred_label} [{mark}]")
print("=" * 60)
