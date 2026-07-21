import os, argparse, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from src.model import build_logistic_regression, build_random_forest, build_xgboost, build_mlp
from src.preprocess import remove_proliferation_genes, validate_no_leakage, PROLIF_GENES
from src.stability_selector import StabilitySelector
from src.train import MODEL_BUILDERS, load_dataset, model_type_slug

RESULTS_DIR = "results"
DATA_DIR = "data/processed"
N_CV_SPLITS = 3
FEATURE_SELECT_K = 500
STABILITY_BOOTSTRAP = 50
STABILITY_MIN_PCT = 0.3


def build_pipeline_selectk(model_builder):
    return Pipeline([
        ('var_thresh', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=FEATURE_SELECT_K)),
        ('classifier', model_builder()),
    ])


def build_pipeline_stability(model_builder):
    return Pipeline([
        ('var_thresh', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('selector', StabilitySelector(k=FEATURE_SELECT_K, n_bootstrap=STABILITY_BOOTSTRAP,
                                       min_pct=STABILITY_MIN_PCT, n_jobs=-1)),
        ('classifier', model_builder()),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='geo')
    parser.add_argument('--cv-splits', type=int, default=N_CV_SPLITS)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset(args.dataset, DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    methods = {
        'SelectKBest (k=500)': build_pipeline_selectk,
        'StabilitySelector (B=50, p=0.3)': build_pipeline_stability,
    }

    all_rows = []
    for model_name, model_builder in MODEL_BUILDERS.items():
        for method_label, pipeline_fn in methods.items():
            pipeline = pipeline_fn(model_builder)
            cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=42)
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv,
                                         scoring=['accuracy', 'roc_auc'], n_jobs=-1)
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)
            n_features_selected = X_train.shape[1]
            try:
                selector = pipeline.named_steps['selector']
                mask = selector.get_support()
                n_features_selected = int(mask.sum())
            except Exception:
                pass
            all_rows.append({
                'Model': model_name,
                'Selector': method_label,
                'CV_Accuracy': f"{cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}",
                'CV_AUC': f"{cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}",
                'Holdout_Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                'Holdout_AUC': f"{roc_auc_score(y_test, y_prob):.4f}",
                'Features_Selected': n_features_selected,
            })

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, 'ablation_study.csv')
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("ABLATION STUDY: StabilitySelector vs SelectKBest")
    print("=" * 100)
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} {row['Selector']:<35} "
              f"AUC={row['Holdout_AUC']:<12} Acc={row['Holdout_Accuracy']:<12} "
              f"Feat={row['Features_Selected']}")
    print("=" * 100)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()
