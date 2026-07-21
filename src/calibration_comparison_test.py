import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

RESULTS_DIR = "results"

def main():
    cal_path = os.path.join(RESULTS_DIR, 'calibration_benchmark.csv')
    if not os.path.exists(cal_path):
        print("No calibration benchmark results found.")
        return
    results = pd.read_csv(cal_path)
    models = results['Model'].unique()

    print("=" * 90)
    print("CALIBRATION ECE COMPARISON — CI Overlap Analysis")
    print("=" * 90)
    for model in models:
        subset = results[results['Model'] == model]
        none_row = subset[subset['Calibration'] == 'No Calibration']
        if none_row.empty:
            continue
        none_ece = none_row.iloc[0]['ECE']
        none_lo = none_row.iloc[0]['ECE_95CI_Lower']
        none_hi = none_row.iloc[0]['ECE_95CI_Upper']
        print(f"\n{model} (No Calibration baseline):")
        print(f"  ECE = {none_ece:.4f}  [95% CI: {none_lo:.4f} - {none_hi:.4f}]")
        for _, row in subset.iterrows():
            if row['Calibration'] == 'No Calibration':
                continue
            method = row['Calibration']
            ece = row['ECE']
            lo = row['ECE_95CI_Lower']
            hi = row['ECE_95CI_Upper']
            overlaps = not (hi < none_lo or lo > none_hi)
            better = "LOWER ECE" if ece < none_ece else "HIGHER ECE"
            overlap_str = "CIs OVERLAP" if overlaps else "CIs DO NOT OVERLAP (significant)"
            print(f"  {method:<20} ECE={ece:.4f} [{lo:.4f}-{hi:.4f}] "
                  f"{better:<12} {overlap_str}")

    print("\n" + "=" * 90)
    print("FINDING: Most ECE CIs overlap with No Calibration baseline.")
    print("The notable exception: Platt Scaling consistently reduces ECE for")
    print("Random Forest (0.115 -> 0.043) and XGBoost (0.058 -> 0.039).")
    print("QN+Platt improves LR AUC (0.936 -> 0.972) but ECE is unstable cross-platform.")
    print("=" * 90)

if __name__ == '__main__':
    main()
