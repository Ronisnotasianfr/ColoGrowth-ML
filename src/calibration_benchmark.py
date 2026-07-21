import os, argparse, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from src.preprocess import remove_proliferation_genes
from src.train import MODEL_BUILDERS, model_type_slug

def _ece_bootstrap_sample(y_true, y_prob, n_bins, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    idx = rng.integers(0, n, n)
    y_boot = y_true[idx]
    p_boot = y_prob[idx]
    p_true, p_pred = calibration_curve(y_boot, p_boot, n_bins=n_bins)
    bin_counts, _ = np.histogram(p_boot, bins=n_bins, range=(0, 1))
    return np.sum(bin_counts / n * np.abs(p_true - p_pred))

def _brier_bootstrap_sample(y_true, y_prob, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    idx = rng.integers(0, n, n)
    return brier_score_loss(y_true[idx], y_prob[idx])

def bootstrap_ece(y_true, y_prob, n_bins=10, n_bootstrap=1000, ci=95, n_jobs=-1):
    seeds = [42 + i for i in range(n_bootstrap)]
    eces = Parallel(n_jobs=n_jobs)(
        delayed(_ece_bootstrap_sample)(y_true, y_prob, n_bins, s) for s in seeds
    )
    lower = np.percentile(eces, (100 - ci) / 2)
    upper = np.percentile(eces, 100 - (100 - ci) / 2)
    return np.mean(eces), lower, upper

def bootstrap_brier(y_true, y_prob, n_bootstrap=1000, ci=95, n_jobs=-1):
    seeds = [42 + i for i in range(n_bootstrap)]
    briers = Parallel(n_jobs=n_jobs)(
        delayed(_brier_bootstrap_sample)(y_true, y_prob, s) for s in seeds
    )
    lower = np.percentile(briers, (100 - ci) / 2)
    upper = np.percentile(briers, 100 - (100 - ci) / 2)
    return np.mean(briers), lower, upper

def expected_calibration_error(y_true, y_prob, n_bins=10):
    p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    n = len(y_true)
    return np.sum(bin_counts / n * np.abs(p_true - p_pred))

MODEL_NAMES = list(MODEL_BUILDERS.keys())


def align_features(X_train, X_test):
    common = [c for c in X_train.columns if c in X_test.columns]
    X_aligned = pd.DataFrame(0, index=X_test.index, columns=X_train.columns)
    for col in common:
        X_aligned[col] = X_test[col]
    return X_aligned


def quantile_normalize_column(ref_col, target_col):
    ref_clean = ref_col.dropna().values
    if len(ref_clean) == 0:
        return target_col
    ref_sorted = np.sort(ref_clean)
    target_vals = target_col.values
    n = len(target_vals)
    if n <= 1:
        return target_col
    ranks = np.argsort(np.argsort(target_vals))
    pcts = np.linspace(0, 1, len(ref_sorted))
    f = interp1d(pcts, ref_sorted, kind='linear', fill_value='extrapolate')
    return pd.Series(f(ranks / (n - 1)), index=target_col.index)


def quantile_normalize_dataset(X_ref, X_target):
    X_mapped = X_target.copy()
    for col in X_ref.columns:
        if col in X_target.columns:
            X_mapped[col] = quantile_normalize_column(X_ref[col], X_target[col])
    return X_mapped


def calibrate_platt(y_calib, prob_calib, prob_eval):
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(prob_calib.reshape(-1, 1), y_calib)
    return lr.predict_proba(prob_eval.reshape(-1, 1))[:, 1]


def calibrate_isotonic(y_calib, prob_calib, prob_eval):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(prob_calib, y_calib)
    return ir.transform(prob_eval)


def calibrate_none(prob_eval):
    return prob_eval


def evaluate(y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
        'Brier': brier,
        'ECE': ece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', default='geo')
    parser.add_argument('--test-dataset', default='tcga')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load training features (for quantile norm reference)
    train_prefix = f'{args.train_dataset}_' if args.train_dataset != 'dataset' else ''
    X_train = remove_proliferation_genes(
        pd.read_csv(os.path.join(args.data_dir, f'{train_prefix}X_features.csv'), index_col=0))

    # Load test data
    test_prefix = f'{args.test_dataset}_' if args.test_dataset != 'dataset' else ''
    X_test = remove_proliferation_genes(
        pd.read_csv(os.path.join(args.data_dir, f'{test_prefix}X_features.csv'), index_col=0))
    y_test = pd.read_csv(os.path.join(args.data_dir, f'{test_prefix}y_target.csv'), index_col=0)['target']

    X_aligned = align_features(X_train, X_test)
    X_qnorm = quantile_normalize_dataset(X_train, X_aligned)
    X_cal, X_eval, y_cal, y_eval = train_test_split(X_qnorm, y_test, test_size=0.5,
                                                     stratify=y_test, random_state=42)

    CAL_METHODS = {
        'None': lambda p_cal, p_eval: calibrate_none(p_eval),
        'Platt Scaling': calibrate_platt,
        'Isotonic Regression': calibrate_isotonic,
    }

    all_rows = []
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    COLORS = {'None': '#2B3A67', 'Platt Scaling': '#E85D75',
              'Isotonic Regression': '#F4D35E'}

    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(args.models_dir, f'{slug}_{args.train_dataset}.joblib')
        if not os.path.exists(model_path):
            continue
        pipeline = joblib.load(model_path)

        prob_cal_raw = pipeline.predict_proba(X_cal)[:, 1]
        prob_eval_raw = pipeline.predict_proba(X_eval)[:, 1]
        prob_full_raw = pipeline.predict_proba(X_aligned)[:, 1]

        for method_name, method_fn in CAL_METHODS.items():
            if method_name == 'None':
                prob_calibrated = prob_eval_raw
            else:
                prob_calibrated = method_fn(y_cal.values, prob_cal_raw, prob_eval_raw)

            metrics = evaluate(y_eval, prob_calibrated)
            brier_mean, brier_lo, brier_hi = bootstrap_brier(y_eval.values, prob_calibrated)
            ece_mean, ece_lo, ece_hi = bootstrap_ece(y_eval.values, prob_calibrated)
            all_rows.append({
                'Model': name, 'Calibration': method_name,
                'AUC': metrics['AUC'], 'Accuracy': metrics['Accuracy'],
                'Brier': metrics['Brier'],
                'ECE': metrics['ECE'],
                'Brier_95CI_Lower': brier_lo, 'Brier_95CI_Upper': brier_hi,
                'ECE_95CI_Lower': ece_lo, 'ECE_95CI_Upper': ece_hi,
            })

        # Plot calibration curves for this model
        ax_idx = MODEL_NAMES.index(name)
        if ax_idx < 3:
            ax = axes[ax_idx]
            ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1.5, label='Perfect')
            for method_name in ['None', 'Platt Scaling', 'Isotonic Regression']:
                if method_name == 'None':
                    prob = prob_eval_raw
                else:
                    fn = CAL_METHODS[method_name]
                    prob = fn(y_cal.values, prob_cal_raw, prob_eval_raw)
                p_true, p_pred = calibration_curve(y_eval, prob, n_bins=10)
                ax.plot(p_pred, p_true, 'o-', color=COLORS[method_name], linewidth=2,
                        markersize=6, label=method_name)
            ax.set_title(name, fontsize=12, fontweight='bold', color='#2B3A67')
            ax.set_xlabel('Mean predicted probability', fontsize=10)
            ax.set_ylabel('Fraction of positives', fontsize=10)
            ax.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#CCCCCC')
            ax.tick_params(labelsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

    # Also benchmark quantile normalization as a standalone method
    # Re-load without QN to test raw feature alignment
    X_raw = align_features(X_train, X_test)
    X_cal_raw, X_eval_raw, _, _ = train_test_split(X_raw, y_test, test_size=0.5,
                                                     stratify=y_test, random_state=42)
    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(args.models_dir, f'{slug}_{args.train_dataset}.joblib')
        if not os.path.exists(model_path):
            continue
        pipeline = joblib.load(model_path)
        prob_eval_raw = pipeline.predict_proba(X_eval_raw)[:, 1]
        prob_cal_raw = pipeline.predict_proba(X_cal_raw)[:, 1]
        prob_qp = calibrate_platt(y_cal.values, prob_cal_raw, prob_eval_raw)
        metrics = evaluate(y_eval, prob_qp)
        brier_mean, brier_lo, brier_hi = bootstrap_brier(y_eval.values, prob_qp)
        ece_mean, ece_lo, ece_hi = bootstrap_ece(y_eval.values, prob_qp)
        all_rows.append({
            'Model': name, 'Calibration': 'QN+Platt',
            'AUC': metrics['AUC'], 'Accuracy': metrics['Accuracy'],
            'Brier': metrics['Brier'], 'ECE': metrics['ECE'],
            'Brier_95CI_Lower': brier_lo, 'Brier_95CI_Upper': brier_hi,
            'ECE_95CI_Lower': ece_lo, 'ECE_95CI_Upper': ece_hi,
        })

    # Also add QN-only (no Platt)
    for name in MODEL_NAMES:
        slug = model_type_slug(name)
        model_path = os.path.join(args.models_dir, f'{slug}_{args.train_dataset}.joblib')
        if not os.path.exists(model_path):
            continue
        pipeline = joblib.load(model_path)
        prob_qn = pipeline.predict_proba(X_eval)[:, 1]
        metrics = evaluate(y_eval, prob_qn)
        brier_mean, brier_lo, brier_hi = bootstrap_brier(y_eval.values, prob_qn)
        ece_mean, ece_lo, ece_hi = bootstrap_ece(y_eval.values, prob_qn)
        all_rows.append({
            'Model': name, 'Calibration': 'QN Only',
            'AUC': metrics['AUC'], 'Accuracy': metrics['Accuracy'],
            'Brier': metrics['Brier'], 'ECE': metrics['ECE'],
            'Brier_95CI_Lower': brier_lo, 'Brier_95CI_Upper': brier_hi,
            'ECE_95CI_Lower': ece_lo, 'ECE_95CI_Upper': ece_hi,
        })

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'calibration_benchmark.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(args.results_dir, 'calibration_benchmark.pdf'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.results_dir, 'calibration_benchmark.csv')
    df.to_csv(csv_path, index=False)

    # Print full calibration benchmark results
    print('Calibration Benchmark Results (with 95% CI)')
    print('=' * 130)
    print(f'{"Model":<22} {"Method":<16} {"AUC":>8} {"Acc":>8} {"Brier":>10} {"Brier CI":>16} {"ECE":>8} {"ECE CI":>14}')
    print('-' * 130)
    for name in MODEL_NAMES:
        subset = df[df['Model'] == name]
        for _, row in subset.iterrows():
            b_ci = f"({row['Brier_95CI_Lower']:.4f}-{row['Brier_95CI_Upper']:.4f})"
            e_ci = f"({row['ECE_95CI_Lower']:.4f}-{row['ECE_95CI_Upper']:.4f})"
            print(f'{name:<22} {row["Calibration"]:<16} {row["AUC"]:>8.4f} {row["Accuracy"]:>8.4f} '
                  f'{row["Brier"]:>10.4f} {b_ci:>16} {row["ECE"]:>8.4f} {e_ci:>14}')
    print('=' * 130)

    # Best per model summary
    print('\nBest calibration method per model (by AUC):')
    for name in MODEL_NAMES:
        subset = df[df['Model'] == name]
        best = subset.loc[subset['AUC'].idxmax()]
        print(f'  {name:<22} → {best["Calibration"]:<16} AUC={best["AUC"]:.4f} '
              f'ECE={best["ECE"]:.4f} [{best["ECE_95CI_Lower"]:.4f}-{best["ECE_95CI_Upper"]:.4f}]')
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
