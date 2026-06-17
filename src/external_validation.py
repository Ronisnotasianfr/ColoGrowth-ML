"""
external_validation.py — Cross-cohort validation.

Scientific purpose:
Trains on one cohort (e.g., GEO microarray) and evaluates on a completely
independent cohort (e.g., TCGA RNA-seq) to demonstrate generalizability.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

def align_features(X_train, X_test):
    """
    Ensures the test set has exactly the same features in the same order
    as the training set. Fills missing features with 0.
    """
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    print(f"Feature intersection: {len(common_cols)}/{X_train.shape[1]} features available in external cohort.")
    
    # Create aligned dataframe
    X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=X_train.columns)
    
    # Fill in matching columns
    for col in common_cols:
        X_test_aligned[col] = X_test[col]
        
    return X_test_aligned

def plot_calibration(y_true, y_prob, model_name, save_path):
    """Plots reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"{model_name}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="External Cohort Validation")
    parser.add_argument("--train-dataset", type=str, default="geo", help="Dataset the model was trained on")
    parser.add_argument("--test-dataset", type=str, default="tcga", help="Independent external dataset")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load training data just to get the expected feature columns
    # (Since we didn't save feature names explicitly, we infer from X_train)
    X_train_path = os.path.join(args.data_dir, f"{args.train_dataset}_X_features.csv")
    if not os.path.exists(X_train_path):
        print(f"Error: Could not find training features at {X_train_path}")
        return
        
    X_train_ref = pd.read_csv(X_train_path, index_col=0)
    
    # Load external test data
    X_test_path = os.path.join(args.data_dir, f"{args.test_dataset}_X_features.csv")
    y_test_path = os.path.join(args.data_dir, f"{args.test_dataset}_y_target.csv")
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: External dataset not found ({args.test_dataset}). Run preprocess.py --download")
        return
        
    X_ext = pd.read_csv(X_test_path, index_col=0)
    y_ext = pd.read_csv(y_test_path, index_col=0)['target']
    
    print(f"\n--- Cross-Cohort Validation: Train on {args.train_dataset.upper()}, Test on {args.test_dataset.upper()} ---")
    
    # Align features to match the pipeline's expectations
    X_ext_aligned = align_features(X_train_ref, X_ext)
    
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network (MLP)']
    results = []
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    for name in models:
        model_filename = f"{name.lower().replace(' ', '_')}_{args.train_dataset}.joblib"
        model_path = os.path.join(args.models_dir, model_filename)
        
        if not os.path.exists(model_path):
            continue
            
        # Load the full sklearn Pipeline
        pipeline = joblib.load(model_path)
        
        # Predict on external cohort
        y_pred = pipeline.predict(X_ext_aligned)
        try:
            y_prob = pipeline.predict_proba(X_ext_aligned)[:, 1]
        except AttributeError:
            y_prob = y_pred
            
        auc = roc_auc_score(y_ext, y_prob)
        acc = accuracy_score(y_ext, y_pred)
        brier = brier_score_loss(y_ext, y_prob)
        
        print(f"[{name}] External AUC: {auc:.4f}, Accuracy: {acc:.4f}, Brier: {brier:.4f}")
        
        results.append({
            'Model': name,
            'External_AUC': auc,
            'External_Accuracy': acc,
            'Brier_Score': brier
        })
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_ext, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # Plot Calibration curve
        calib_path = os.path.join(args.results_dir, f"calibration_{name.lower().replace(' ', '_')}.png")
        plot_calibration(y_ext, y_prob, name, calib_path)
        
    # Finish ROC plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'External Validation ROC ({args.train_dataset.upper()} → {args.test_dataset.upper()})')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_path = os.path.join(args.results_dir, f"external_roc_{args.train_dataset}_to_{args.test_dataset}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # Save results table
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(args.results_dir, "external_validation_results.csv"), index=False)
        print(f"\nExternal validation results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
