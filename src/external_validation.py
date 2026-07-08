"""
external_validation.py — Cross-cohort validation with platform calibration.

Trains on one cohort (e.g., GEO microarray) and evaluates on a completely
independent cohort (e.g., TCGA RNA-seq) to demonstrate generalizability.

Implements:
  1. Feature alignment across platforms
  2. Platt scaling (manual sigmoid calibration via LogisticRegression on
     raw predicted probabilities) to fix cross-platform threshold shift
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def platt_scale(y_calib, prob_calib, prob_eval):
    """
    Manual Platt scaling: fit a logistic regression on raw probabilities
    from the calibration set, then transform evaluation probabilities.
    This re-maps the decision threshold to account for platform shift.
    """
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(prob_calib.reshape(-1, 1), y_calib)
    prob_calibrated = lr.predict_proba(prob_eval.reshape(-1, 1))[:, 1]
    pred_calibrated = lr.predict(prob_eval.reshape(-1, 1))
    return pred_calibrated, prob_calibrated


def plot_calibration_comparison(y_true, y_prob_raw, y_prob_cal, model_name, save_path):
    """Plot before vs. after calibration curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_prob, title in [
        (axes[0], y_prob_raw, f"{model_name} — Before Calibration"),
        (axes[1], y_prob_cal, f"{model_name} — After Platt Scaling"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.plot(prob_pred, prob_true, "s-", label=title.split("—")[-1].strip())
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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

    # --- Load cohorts (raw — pipeline has its own StandardScaler) ---
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
        f"\n{'='*65}\n"
        f"  Cross-Cohort Validation: "
        f"Train on {args.train_dataset.upper()}, "
        f"Test on {args.test_dataset.upper()}\n"
        f"  With Platt scaling probability calibration\n"
        f"{'='*65}"
    )

    X_ext_aligned = align_features(X_train_ref, X_ext)

    # --- Split TCGA 50/50: calibration set + evaluation set ---
    X_calib, X_eval, y_calib, y_eval = train_test_split(
        X_ext_aligned, y_ext, test_size=0.5, stratify=y_ext, random_state=42
    )
    print(
        f"TCGA split: {len(y_calib)} calibration samples, "
        f"{len(y_eval)} evaluation samples.\n"
    )

    raw_results = []
    calibrated_results = []

    # --- ROC plot setup ---
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(14, 6))
    for ax in axes_roc:
        ax.plot([0, 1], [0, 1], 'k--', label='Random')

    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(
            args.models_dir, f"{slug}_{args.train_dataset}.joblib"
        )

        if not os.path.exists(model_path):
            print(f"Skipping {name}: pipeline not found at {model_path}")
            continue

        pipeline = joblib.load(model_path)

        # ----- RAW (uncalibrated) evaluation on FULL TCGA -----
        y_pred_raw = pipeline.predict(X_ext_aligned)
        try:
            y_prob_raw_full = pipeline.predict_proba(X_ext_aligned)[:, 1]
        except AttributeError:
            y_prob_raw_full = y_pred_raw.astype(float)

        raw_auc = roc_auc_score(y_ext, y_prob_raw_full)
        raw_acc = accuracy_score(y_ext, y_pred_raw)
        raw_brier = brier_score_loss(y_ext, y_prob_raw_full)

        raw_results.append({
            'Model': name,
            'Raw_AUC': raw_auc,
            'Raw_Accuracy': raw_acc,
            'Raw_Brier': raw_brier,
        })

        fpr, tpr, _ = roc_curve(y_ext, y_prob_raw_full)
        axes_roc[0].plot(fpr, tpr, label=f'{name} (AUC={raw_auc:.3f})')

        # ----- PLATT SCALING CALIBRATION -----
        try:
            # Get raw probabilities for calibration and evaluation splits
            prob_calib = pipeline.predict_proba(X_calib)[:, 1]
            prob_eval_raw = pipeline.predict_proba(X_eval)[:, 1]

            # Fit sigmoid on calibration split, predict on evaluation split
            y_pred_cal, prob_eval_cal = platt_scale(
                y_calib.values, prob_calib, prob_eval_raw
            )

            cal_auc = roc_auc_score(y_eval, prob_eval_cal)
            cal_acc = accuracy_score(y_eval, y_pred_cal)
            cal_brier = brier_score_loss(y_eval, prob_eval_cal)

            calibrated_results.append({
                'Model': name,
                'Calibrated_AUC': cal_auc,
                'Calibrated_Accuracy': cal_acc,
                'Calibrated_Brier': cal_brier,
            })

            fpr_cal, tpr_cal, _ = roc_curve(y_eval, prob_eval_cal)
            axes_roc[1].plot(fpr_cal, tpr_cal, label=f'{name} (AUC={cal_auc:.3f})')

            print(
                f"[{name}]\n"
                f"  Raw:        AUC={raw_auc:.4f}  Acc={raw_acc:.4f}  Brier={raw_brier:.4f}\n"
                f"  Calibrated: AUC={cal_auc:.4f}  Acc={cal_acc:.4f}  Brier={cal_brier:.4f}"
            )

            # Calibration comparison plot
            calib_path = os.path.join(args.results_dir, f"calibration_comparison_{slug}.png")
            plot_calibration_comparison(y_eval, prob_eval_raw, prob_eval_cal, name, calib_path)

        except Exception as exc:
            print(f"[{name}] Calibration failed: {exc}. Using raw results only.")
            calibrated_results.append({
                'Model': name,
                'Calibrated_AUC': raw_auc,
                'Calibrated_Accuracy': raw_acc,
                'Calibrated_Brier': raw_brier,
            })

    if not raw_results:
        print(
            "No trained pipelines found. Run training first, e.g. "
            f"python -m src.train --dataset {args.train_dataset}"
        )
        return

    # --- Finalize ROC plots ---
    train_label = args.train_dataset.upper()
    test_label = args.test_dataset.upper()

    axes_roc[0].set_title(f'Raw External ROC ({train_label} → {test_label})', fontsize=12)
    axes_roc[0].set_xlabel('False Positive Rate')
    axes_roc[0].set_ylabel('True Positive Rate')
    axes_roc[0].legend(loc='lower right', fontsize=9)
    axes_roc[0].grid(True, linestyle=':', alpha=0.6)

    axes_roc[1].set_title(f'Calibrated External ROC ({train_label} → {test_label})', fontsize=12)
    axes_roc[1].set_xlabel('False Positive Rate')
    axes_roc[1].set_ylabel('True Positive Rate')
    axes_roc[1].legend(loc='lower right', fontsize=9)
    axes_roc[1].grid(True, linestyle=':', alpha=0.6)

    fig_roc.tight_layout()
    roc_path = os.path.join(
        args.results_dir,
        f"external_roc_{args.train_dataset}_to_{args.test_dataset}.png",
    )
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)

    # --- Save combined results CSV ---
    df_raw = pd.DataFrame(raw_results)
    df_cal = pd.DataFrame(calibrated_results)
    df_combined = pd.merge(df_raw, df_cal, on='Model')
    out_csv = os.path.join(args.results_dir, "external_validation_results.csv")
    df_combined.to_csv(out_csv, index=False)

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print("  EXTERNAL VALIDATION SUMMARY (with Platt Scaling Calibration)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'Raw Acc':>8} {'Cal Acc':>8} {'Raw AUC':>8} {'Cal AUC':>8} {'Raw Brier':>10} {'Cal Brier':>10}")
    print("-" * 85)
    for _, row in df_combined.iterrows():
        print(
            f"{row['Model']:<25} "
            f"{row['Raw_Accuracy']:>8.4f} {row['Calibrated_Accuracy']:>8.4f} "
            f"{row['Raw_AUC']:>8.4f} {row['Calibrated_AUC']:>8.4f} "
            f"{row['Raw_Brier']:>10.4f} {row['Calibrated_Brier']:>10.4f}"
        )

    print(f"\nResults saved to {out_csv}")
    print(f"ROC comparison saved to {roc_path}")


if __name__ == "__main__":
    main()
