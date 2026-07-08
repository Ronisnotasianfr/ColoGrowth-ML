"""
train.py - Leakage-free model training, cross-validation, and evaluation.

Workflow (triggered via `python -m src.train --dataset geo`):
  1. Load processed features and strip proliferation signature genes.
  2. Stratified train/test split.
  3. Per model: 5-fold CV on the training pool (pipeline fits inside each fold).
  4. Hyperparameter tuning via GridSearchCV on the training pool.
  5. Final fit on full training pool and holdout evaluation on the test split.
  6. Print and export unbiased metrics to results/.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from src.model import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    build_mlp,
)
from src.preprocess import (
    PROLIF_GENES,
    remove_proliferation_genes,
    validate_no_leakage,
)

# Hyperparameter search spaces
PARAM_GRIDS = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_leaf': [2, 4],
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
    },
    'Neural Network (MLP)': {
        'classifier__hidden_layer_sizes': [(128, 64), (256, 128, 64)],
        'classifier__alpha': [0.0001, 0.001],
    },
}

MODEL_BUILDERS = {
    'Logistic Regression': build_logistic_regression,
    'Random Forest': build_random_forest,
    'XGBoost': build_xgboost,
    'Neural Network (MLP)': build_mlp,
}

OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
FEATURE_SELECT_K = 500


def model_type_slug(model_name: str) -> str:
    """Convert display name to a filesystem-safe slug."""
    return model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')


def create_pipeline(model_builder):
    """
    Build an sklearn Pipeline that chains scaling, variance filtering,
    feature selection, and classification. When used inside cross_validate
    or GridSearchCV, every preprocessing step is refit on training folds only.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('var_thresh', VarianceThreshold(threshold=0.01)),
        ('feature_select', SelectKBest(score_func=f_classif, k=FEATURE_SELECT_K)),
        ('classifier', model_builder()),
    ])


def configure_feature_selection(pipeline, n_features):
    """Set SelectKBest k dynamically when the feature count is small."""
    if n_features < FEATURE_SELECT_K:
        pipeline.set_params(feature_select__k='all')
    return pipeline


def load_dataset(dataset: str, data_dir: str):
    """
    Load X and y, strip proliferation signature genes, and validate no leakage.
    Gene removal happens before any train-test split.
    """
    prefix = f"{dataset}_" if dataset != "dataset" else ""
    x_path = os.path.join(data_dir, f"{prefix}X_features.csv")
    y_path = os.path.join(data_dir, f"{prefix}y_target.csv")

    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"Could not find training data at {x_path}. Run preprocess.py first."
        )

    print(f"Loading {dataset.upper()} dataset from {data_dir} ...")
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)['target']

    print(f"Raw feature matrix: {X.shape[0]} samples, {X.shape[1]} columns.")
    print(f"Stripping {len(PROLIF_GENES)} proliferation signature genes before splitting ...")
    X = remove_proliferation_genes(X)
    validate_no_leakage(X)

    return X, y


def run_unbiased_cross_validation(pipeline, X_train, y_train, n_splits=OUTER_CV_SPLITS):
    """
    Run outer StratifiedKFold CV on the training pool.
    The pipeline is cloned and fit independently inside each fold.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"  Running {n_splits}-fold StratifiedKFold cross-validation on training pool ...")
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=outer_cv,
        scoring=['accuracy', 'roc_auc'],
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
    }


def tune_hyperparameters(model_name, pipeline, X_train, y_train):
    """
    Tune hyperparameters with GridSearchCV on the training pool.
    Inner CV keeps preprocessing fold-local during the search.
    """
    param_grid = PARAM_GRIDS.get(model_name, {})
    inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)

    print(f"  Tuning hyperparameters with GridSearchCV ({INNER_CV_SPLITS}-fold inner CV) ...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best inner CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_holdout(model, X_test, y_test):
    """Evaluate a fitted pipeline on the untouched holdout split."""
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred

    return {
        'holdout_accuracy': accuracy_score(y_test, y_pred),
        'holdout_roc_auc': roc_auc_score(y_test, y_prob),
    }


def build_metrics_dataframe(model_name, cv_metrics, holdout_metrics):
    """Compile CV and holdout metrics into a single-row summary DataFrame."""
    return pd.DataFrame([{
        'Model': model_name,
        'CV_Accuracy_Mean': cv_metrics['cv_accuracy_mean'],
        'CV_Accuracy_Std': cv_metrics['cv_accuracy_std'],
        'CV_ROC_AUC_Mean': cv_metrics['cv_roc_auc_mean'],
        'CV_ROC_AUC_Std': cv_metrics['cv_roc_auc_std'],
        'Holdout_Accuracy': holdout_metrics['holdout_accuracy'],
        'Holdout_ROC_AUC': holdout_metrics['holdout_roc_auc'],
    }])


def train_single_model(model_name, model_builder, X_train, y_train, X_test, y_test,
                       models_dir, results_dir, dataset):
    """
    Full leakage-free workflow for one model:
      CV (training pool) -> tune -> final fit -> holdout eval -> export metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print('=' * 60)

    base_pipeline = configure_feature_selection(
        create_pipeline(model_builder),
        X_train.shape[1],
    )

    cv_metrics = run_unbiased_cross_validation(base_pipeline, X_train, y_train)

    tuned_pipeline = configure_feature_selection(
        create_pipeline(model_builder),
        X_train.shape[1],
    )
    final_model = tune_hyperparameters(model_name, tuned_pipeline, X_train, y_train)

    holdout_metrics = evaluate_holdout(final_model, X_test, y_test)

    metrics_df = build_metrics_dataframe(model_name, cv_metrics, holdout_metrics)

    slug = model_type_slug(model_name)
    metrics_path = os.path.join(results_dir, f"{slug}_leakage_fixed_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    model_filename = f"{slug}_{dataset}.joblib"
    joblib.dump(final_model, os.path.join(models_dir, model_filename))

    print(f"  CV Accuracy : {cv_metrics['cv_accuracy_mean']:.4f} "
          f"(+/- {cv_metrics['cv_accuracy_std']:.4f})")
    print(f"  CV ROC-AUC  : {cv_metrics['cv_roc_auc_mean']:.4f} "
          f"(+/- {cv_metrics['cv_roc_auc_std']:.4f})")
    print(f"  Holdout Accuracy : {holdout_metrics['holdout_accuracy']:.4f}")
    print(f"  Holdout ROC-AUC  : {holdout_metrics['holdout_roc_auc']:.4f}")
    print(f"  Metrics saved to {metrics_path}")
    print(f"  Model saved to {model_filename}")

    return metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="Train models with leakage-free CV, tuning, and holdout evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="geo",
        choices=["geo", "tcga", "synthetic"],
        help="Which dataset to use for training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to save trained models",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to save evaluation metrics",
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    try:
        X, y = load_dataset(args.dataset, args.data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain pool: {X_train.shape[0]} samples | Holdout test: {X_test.shape[0]} samples")
    print(f"Features after leakage removal: {X_train.shape[1]}")

    all_metrics = []
    for name, builder in MODEL_BUILDERS.items():
        metrics_df = train_single_model(
            model_name=name,
            model_builder=builder,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            dataset=args.dataset,
        )
        all_metrics.append(metrics_df)

    summary = pd.concat(all_metrics, ignore_index=True)
    summary_path = os.path.join(
        args.results_dir, f"all_models_{args.dataset}_leakage_fixed_metrics.csv"
    )
    summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"LEAKAGE-FREE EVALUATION SUMMARY - {args.dataset.upper()}")
    print('=' * 60)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nCombined summary exported to {summary_path}")


if __name__ == "__main__":
    main()
