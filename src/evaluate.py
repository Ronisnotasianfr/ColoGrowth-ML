"""
evaluate.py - Holdout evaluation and visualization using unified sklearn Pipelines.

Loads saved pipeline models (scaler + feature selector + classifier) from models/
and evaluates on the same stratified holdout split used by train.py.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.train import MODEL_BUILDERS, model_type_slug, load_dataset

MODEL_NAMES = list(MODEL_BUILDERS.keys())


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
    }


def get_selected_feature_names(pipeline, feature_columns):
    """Map post-selection transformed columns back to original feature names."""
    var_support = pipeline.named_steps['var_thresh'].get_support()
    surviving = np.array(feature_columns)[var_support]
    kbest_support = pipeline.named_steps['feature_select'].get_support()
    return list(surviving[kbest_support])


def transform_for_interpretation(pipeline, X):
    """Apply all preprocessing steps except the final classifier."""
    Xt = pipeline.named_steps['scaler'].transform(X)
    Xt = pipeline.named_steps['var_thresh'].transform(Xt)
    Xt = pipeline.named_steps['feature_select'].transform(Xt)
    return Xt


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Low Proliferation', 'High Proliferation'],
        yticklabels=['Low Proliferation', 'High Proliferation'],
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(model_probs, y_test, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')

    for name, prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    plt.title('ROC Curves Comparison (Leakage-Free Holdout)', fontsize=14, pad=15)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_shap_importance(pipeline, X_test, model_name, save_path):
    """Generate SHAP or fallback feature-importance plots on pipeline-transformed data."""
    classifier = pipeline.named_steps['classifier']
    X_transformed = transform_for_interpretation(pipeline, X_test)
    feature_names = get_selected_feature_names(pipeline, X_test.columns)

    try:
        import shap
        print(f"Generating SHAP plots for {model_name}...")

        if 'xgboost' in model_name.lower() or 'forest' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        elif 'logistic' in model_name.lower():
            explainer = shap.LinearExplainer(classifier, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
        else:
            background = (
                shap.kmeans(X_transformed, 5)
                if X_transformed.shape[0] > 5
                else X_transformed
            )
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        plt.figure(figsize=(10, 6))
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        shap.summary_plot(shap_values, df_transformed, show=False)
        plt.title(f"SHAP Feature Importance Summary - {model_name}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot saved to {save_path}")

    except Exception as exc:
        print(f"Could not compute SHAP plot for {model_name}: {exc}")
        if hasattr(classifier, 'feature_importances_'):
            print("Falling back to standard feature importance...")
            plt.figure(figsize=(10, 6))
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[-20:]
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title(f'Top 20 Features - {model_name}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved pipeline models on the holdout test split"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["geo", "tcga", "synthetic"],
        help="Dataset used during training",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    try:
        X, y = load_dataset(args.dataset, args.data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(
        f"Evaluating holdout split: {X_test.shape[0]} test samples, "
        f"{X_test.shape[1]} leakage-free features"
    )

    model_probs = {}
    evaluation_results = []

    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(args.models_dir, f"{slug}_{args.dataset}.joblib")

        if not os.path.exists(model_path):
            print(f"Skipping {name}: pipeline not found at {model_path}")
            continue

        print(f"\nEvaluating {name} ...")
        pipeline = joblib.load(model_path)

        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred

        model_probs[name] = y_prob
        metrics = compute_metrics(y_test, y_pred, y_prob)
        evaluation_results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['auc'],
        })

        cm_path = os.path.join(args.results_dir, f"confusion_matrix_{slug}.png")
        plot_confusion_matrix(y_test, y_pred, name, cm_path)

        if name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
            shap_path = os.path.join(args.results_dir, f"shap_summary_{slug}.png")
            plot_shap_importance(pipeline, X_test, name, shap_path)

    if not evaluation_results:
        print(
            "No trained pipelines found. Run training first, e.g. "
            f"python -m src.train --dataset {args.dataset}"
        )
        return

    roc_path = os.path.join(args.results_dir, "roc_curves_comparison.png")
    plot_roc_curves(model_probs, y_test, roc_path)
    print(f"ROC curves comparison saved to {roc_path}")

    df_eval = pd.DataFrame(evaluation_results)
    summary_path = os.path.join(args.results_dir, "evaluation_results_summary.csv")
    df_eval.to_csv(summary_path, index=False)

    print("\n=== Leakage-Free Holdout Evaluation Results ===")
    print(df_eval.to_string(index=False))

    md_path = os.path.join(args.results_dir, "evaluation_results_table.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(df_eval.to_markdown(index=False))

    print(f"\nAll evaluations complete. Visualizations saved to {args.results_dir}.")


if __name__ == "__main__":
    main()
