import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocess import generate_synthetic_data, preprocess_and_save_data
from src.model import build_logistic_regression, build_random_forest, build_xgboost, build_mlp

def train_and_eval_fold(model, X_train, y_train, X_val, y_val, scale=True):
    """
    Standard function to train on a fold, scale features, and return predictions/probabilities.
    """
    if scale:
        scaler = StandardScaler()
        # Scale clinical features and gene expression features differently or altogether
        # We can just scale everything
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        scaler = None
        
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_val_scaled)
    try:
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
    except AttributeError:
        # If classifier has no predict_proba
        y_prob = y_pred
        
    return y_pred, y_prob, scaler

def run_cross_validation(model_name, model_builder, X, y, cv=5, scale=True):
    """
    Performs stratified K-fold cross-validation and records metrics.
    """
    print(f"\n--- Running {cv}-Fold Cross-Validation for {model_name} ---")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    # Check if target is series
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
        
        # Instantiate model fresh for each fold
        clf = model_builder()
        y_pred, y_prob, _ = train_and_eval_fold(clf, X_tr, y_tr, X_va, y_va, scale=scale)
        
        metrics['accuracy'].append(accuracy_score(y_va, y_pred))
        metrics['precision'].append(precision_score(y_va, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_va, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_va, y_pred, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_va, y_prob))
        
        print(f"Fold {fold+1}: Accuracy={metrics['accuracy'][-1]:.4f}, AUC={metrics['auc'][-1]:.4f}")
        
    summary = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    print(f"CV Summary for {model_name}:")
    for k, (mean, std) in summary.items():
        print(f"  {k.capitalize()}: {mean:.4f} +/- {std:.4f}")
        
    return summary

def main():
    parser = argparse.ArgumentParser(description="Colon Cancer ML Project Training Pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Generate and train on synthetic data")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to save trained models")
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Load dataset
    if args.synthetic or not os.path.exists(os.path.join(args.data_dir, "X_features.csv")):
        print("Processed files not found or synthetic flag enabled. Generating synthetic datasets...")
        expr, clin = generate_synthetic_data()
        X, y = preprocess_and_save_data(expr, clin, output_dir=args.data_dir)
    else:
        print(f"Loading processed datasets from {args.data_dir}...")
        X = pd.read_csv(os.path.join(args.data_dir, "X_features.csv"), index_col=0)
        y = pd.read_csv(os.path.join(args.data_dir, "y_target.csv"), index_col=0)['target']
        
    # Split train/test (80% train for cross-validation, 20% test for final holdout evaluation)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Save the split dataset for evaluate.py
    os.makedirs(args.data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(args.data_dir, "X_train.csv"))
    X_test.to_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_train.to_frame(name="target").to_csv(os.path.join(args.data_dir, "y_train.csv"))
    y_test.to_frame(name="target").to_csv(os.path.join(args.data_dir, "y_test.csv"))
    
    model_builders = {
        'Logistic Regression': build_logistic_regression,
        'Random Forest': build_random_forest,
        'XGBoost': build_xgboost,
        'Neural Network (MLP)': build_mlp
    }
    
    cv_results = {}
    
    # Standardize scale
    scale = True
    
    for name, builder in model_builders.items():
        # Evaluate model with CV
        cv_summary = run_cross_validation(name, builder, X_train, y_train, cv=args.cv, scale=scale)
        cv_results[name] = cv_summary
        
        # Train final model on full train set
        print(f"Training final {name} model on all training data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = builder()
        model.fit(X_train_scaled, y_train)
        
        # Save model and its scaler
        joblib.dump(scaler, os.path.join(args.models_dir, f"{name.lower().replace(' ', '_')}_scaler.joblib"))
        if name == 'Neural Network (MLP)':
            # Custom wrapper pickle
            joblib.dump(model, os.path.join(args.models_dir, "neural_network_(mlp).joblib"))
        else:
            joblib.dump(model, os.path.join(args.models_dir, f"{name.lower().replace(' ', '_')}.joblib"))
            
    # Save cross-validation summary table
    summary_data = []
    for name, metrics in cv_results.items():
        row = {'Model': name}
        for metric, (mean, std) in metrics.items():
            row[f'{metric.upper()}'] = f"{mean:.4f} (+/- {std:.4f})"
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(args.models_dir, "cv_results_summary.csv"), index=False)
    print("\nTraining and cross-validation completed. Final models saved.")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
