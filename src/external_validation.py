"""
external_validation.py — Cross-cohort validation.

Trains on one cohort (e.g., GEO microarray) and evaluates on a completely
independent cohort (e.g., TCGA RNA-seq) to demonstrate generalizability.
"""

import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

from src.preprocess import remove_proliferation_genes, validate_no_leakage
from src.train import MODEL_BUILDERS, model_type_slug

MODEL_NAMES = list(MODEL_BUILDERS.keys())


def align_features(X_train, X_test):
    """
    Ensure the external cohort has the same feature columns and order as training.
    Missing features are zero-filled.
    """
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    print(
        f"Feature intersection: {len(common_cols)}/{X_train.shape[1]} "
        f"features available in external cohort."
    )

    X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=X_train.columns)
    for col in common_cols:
        X_test_aligned[col] = X_test[col]

    return X_test_aligned


def plot_calibration(y_true, y_prob, model_name, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=model_name)
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_cohort_features(data_dir, dataset):
    """Load and leakage-clean a cohort feature matrix."""
    prefix = f"{dataset}_" if dataset != "dataset" else ""
    x_path = os.path.join(data_dir, f"{prefix}X_features.csv")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Could not find features at {x_path}")

    X = pd.read_csv(x_path, index_col=0)
    X = remove_proliferation_genes(X)
    validate_no_leakage(X)
    return X


def main():
    parser = argparse.ArgumentParser(description="External Cohort Validation")
    parser.add_argument("--train-dataset", type=str, default="geo")
    parser.add_argument("--test-dataset", type=str, default="tcga")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    try:
        X_train_ref = load_cohort_features(args.data_dir, args.train_dataset)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    prefix = f"{args.test_dataset}_" if args.test_dataset != "dataset" else ""
    y_test_path = os.path.join(args.data_dir, f"{prefix}y_target.csv")

    try:
        X_ext = load_cohort_features(args.data_dir, args.test_dataset)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    if not os.path.exists(y_test_path):
        print(f"Error: External labels not found at {y_test_path}")
        return

    y_ext = pd.read_csv(y_test_path, index_col=0)['target']

    print(
        f"\n--- Cross-Cohort Validation: "
        f"Train on {args.train_dataset.upper()}, "
        f"Test on {args.test_dataset.upper()} ---"
    )

    X_ext_aligned = align_features(X_train_ref, X_ext)
    results = []

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(
            args.models_dir, f"{slug}_{args.train_dataset}.joblib"
        )

        if not os.path.exists(model_path):
            print(f"Skipping {name}: pipeline not found at {model_path}")
            continue

        pipeline = joblib.load(model_path)
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
            'Brier_Score': brier,
        })

        fpr, tpr, _ = roc_curve(y_ext, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

        calib_path = os.path.join(args.results_dir, f"calibration_{slug}.png")
        plot_calibration(y_ext, y_prob, name, calib_path)

    if not results:
        print(
            "No trained pipelines found. Run training first, e.g. "
            f"python -m src.train --dataset {args.train_dataset}"
        )
        return

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'External Validation ROC ({args.train_dataset.upper()} -> {args.test_dataset.upper()})'
    )
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_path = os.path.join(
        args.results_dir,
        f"external_roc_{args.train_dataset}_to_{args.test_dataset}.png",
    )
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(args.results_dir, "external_validation_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nExternal validation results saved to {out_csv}")


if __name__ == "__main__":
    main()
