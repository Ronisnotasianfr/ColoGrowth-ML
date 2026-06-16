import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes standard classification metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Plots and saves confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Proliferation', 'High Proliferation'],
                yticklabels=['Low Proliferation', 'High Proliferation'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(model_probs, y_test, save_path):
    """
    Plots ROC curves for all models in a single plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    
    for name, prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.title('ROC Curves Comparison', fontsize=14, pad=15)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_shap_importance(model, X_test_scaled, feature_names, model_name, save_path):
    """
    Computes and plots SHAP summary values for the model.
    """
    try:
        import shap
        print(f"Generating SHAP plots for {model_name}...")
        
        # Select background data to speed up if needed (or just use test data)
        # Use TreeExplainer for Tree models, LinearExplainer for Logistic Regression, Kernel/Deep for Neural Networks
        if 'xgboost' in model_name.lower() or 'forest' in model_name.lower():
            explainer = shap.TreeExplainer(model)
            # Some TreeExplainer versions return list of arrays for multiclass or single array for binary
            shap_values = explainer.shap_values(X_test_scaled)
            
            # Handle shape differences in shap versions
            if isinstance(shap_values, list):
                # For binary classification with scikit-learn RF, shap_values might be a list of length 2
                # representing negative and positive classes. We take the positive class (index 1).
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # XGBoost shap values might be shape (samples, features, classes) or similar
                shap_values = shap_values[:, :, 1]
        elif 'logistic' in model_name.lower():
            explainer = shap.LinearExplainer(model, X_test_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
        else:
            # For Neural Network or other models, fallback to KernelExplainer on a subset of data
            # Use a small background dataset to keep evaluation fast
            background = shap.kmeans(X_test_scaled, 5) if X_test_scaled.shape[0] > 5 else X_test_scaled
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
        # Generate summary plot
        plt.figure(figsize=(10, 6))
        # shap.summary_plot works with numpy arrays or DataFrame
        df_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
        shap.summary_plot(shap_values, df_scaled, show=False)
        plt.title(f"SHAP Feature Importance Summary - {model_name}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot saved to {save_path}")
        
    except Exception as e:
        print(f"Could not compute SHAP plot for {model_name}: {e}")
        # Simple feature importance fallback for Random Forest/XGBoost
        if hasattr(model, 'feature_importances_'):
            print("Falling back to standard Gini/Gain feature importance...")
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title(f'Top 20 Features - {model_name}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Colon Cancer ML Project Evaluation Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to save evaluation results")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load test sets
    X_test_path = os.path.join(args.data_dir, "X_test.csv")
    y_test_path = os.path.join(args.data_dir, "y_test.csv")
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print("Test datasets not found! Run training pipeline first to save train/test splits.")
        return
        
    X_test = pd.read_csv(X_test_path, index_col=0)
    y_test = pd.read_csv(y_test_path, index_col=0)['target']
    
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network (MLP)']
    
    model_probs = {}
    evaluation_results = []
    
    for name in models:
        model_key = name.lower().replace(' ', '_')
        model_path = os.path.join(args.models_dir, f"{model_key}.joblib")
        scaler_path = os.path.join(args.models_dir, f"{model_key}_scaler.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Skipping {name}: model or scaler checkpoint not found.")
            continue
            
        print(f"\nEvaluating {name}...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Scale test set
        X_test_scaled = scaler.transform(X_test)
        
        # Predictions & Probs
        y_pred = model.predict(X_test_scaled)
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        except AttributeError:
            y_prob = y_pred
            
        model_probs[name] = y_prob
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        evaluation_results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['auc']
        })
        
        # Plot Confusion Matrix
        cm_path = os.path.join(args.results_dir, f"confusion_matrix_{model_key}.png")
        plot_confusion_matrix(y_test, y_pred, name, cm_path)
        
        # Plot SHAP (only for trees and baseline)
        if name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
            shap_path = os.path.join(args.results_dir, f"shap_summary_{model_key}.png")
            plot_shap_importance(model, X_test_scaled, X_test.columns, name, shap_path)
            
    # Plot all ROC curves together
    if model_probs:
        roc_path = os.path.join(args.results_dir, "roc_curves_comparison.png")
        plot_roc_curves(model_probs, y_test, roc_path)
        print(f"ROC curves comparison saved to {roc_path}")
        
    # Create and save results summary table
    df_eval = pd.DataFrame(evaluation_results)
    df_eval.to_csv(os.path.join(args.results_dir, "evaluation_results_summary.csv"), index=False)
    
    print("\n=== Final Holdout Evaluation Results ===")
    print(df_eval.to_string(index=False))
    
    # Save a markdown version for reports/README
    with open(os.path.join(args.results_dir, "evaluation_results_table.md"), "w") as f:
        f.write(df_eval.to_markdown(index=False))
        
    print(f"\nAll evaluations complete. Visualizations saved to {args.results_dir}.")

if __name__ == "__main__":
    main()
