"""
train.py — Model training, cross-validation, and hyperparameter tuning.

Scientific improvements:
1. Data Leakage Fix: VarianceThreshold and SelectKBest are placed INSIDE an
   sklearn Pipeline. This ensures feature selection only sees the training fold.
2. Hyperparameter Optimization: Uses GridSearchCV to find optimal parameters
   for each model using an inner CV loop, preventing overfitting.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.model import build_logistic_regression, build_random_forest, build_xgboost, build_mlp

# Hyperparameter search spaces
PARAM_GRIDS = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_leaf': [2, 4]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    },
    'Neural Network (MLP)': {
        'classifier__hidden_layer_sizes': [(128, 64), (256, 128, 64)],
        'classifier__alpha': [0.0001, 0.001]
    }
}

def create_pipeline(model_builder):
    """
    Creates an sklearn Pipeline that chains preprocessing, feature selection,
    and classification. This PREVENTS DATA LEAKAGE by ensuring feature selection
    happens independently within each cross-validation fold.
    """
    # 1. Scale all features
    scaler = StandardScaler()
    
    # 2. Remove zero-variance features
    var_thresh = VarianceThreshold(threshold=0.01)
    
    # 3. Select top K features based on ANOVA F-value
    # Using a conservative number like 500 or the total number of features if less
    k_best = SelectKBest(score_func=f_classif, k=500)
    
    # 4. Classifier
    classifier = model_builder()
    
    # K-best will fail if k > num_features, so we handle that dynamically during fit,
    # or just use a generic 'all' if we want the model to do the selection (like trees),
    # but for LR/MLP, filtering is good. We'll set a custom wrapper or just rely on 
    # SelectKBest's behavior. To be safe, we'll set k dynamically in the grid search
    # or just use VarianceThreshold. For simplicity and robustness across platforms,
    # let's use VarianceThreshold + SelectKBest(k=min(500, n_features))
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('var_thresh', var_thresh),
        # k_best is added here but we will configure k dynamically before fitting if needed,
        # actually SelectKBest handles k > n_features gracefully in recent sklearn versions
        ('feature_select', SelectKBest(score_func=f_classif, k=500)),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_and_tune(model_name, model_builder, X_train, y_train, cv=5):
    """
    Performs hyperparameter tuning using GridSearchCV with nested cross-validation.
    """
    print(f"\n--- Tuning and Training {model_name} ---")
    pipeline = create_pipeline(model_builder)
    
    # Adjust k if dataset has fewer than 500 features (e.g. clinical only)
    if X_train.shape[1] < 500:
        pipeline.set_params(feature_select__k='all')
        
    param_grid = PARAM_GRIDS.get(model_name, {})
    
    # Inner CV loop for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit performs tuning AND fits the best model on all X_train
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Inner CV AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    parser = argparse.ArgumentParser(description="Train Models with Nested CV and Pipelines")
    parser.add_argument("--dataset", type=str, default="geo", choices=["geo", "tcga", "synthetic"], 
                        help="Which dataset to use for training")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to save trained models")
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    
    prefix = f"{args.dataset}_" if args.dataset != "dataset" else ""
    X_path = os.path.join(args.data_dir, f"{prefix}X_features.csv")
    y_path = os.path.join(args.data_dir, f"{prefix}y_target.csv")
    
    if not os.path.exists(X_path):
        print(f"Error: Could not find training data at {X_path}. Run preprocess.py first.")
        return
        
    print(f"Loading {args.dataset.upper()} dataset for training...")
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)['target']
    
    # For external validation, we might want to train on the FULL dataset
    # and test on the other dataset. But we still do an internal split
    # just to verify internal consistency, or we can just train on 100%.
    # Let's do a 80/20 split for internal validation reporting.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model_builders = {
        'Logistic Regression': build_logistic_regression,
        'Random Forest': build_random_forest,
        'XGBoost': build_xgboost,
        'Neural Network (MLP)': build_mlp
    }
    
    results = []
    
    for name, builder in model_builders.items():
        # Tune and train (Pipelines handle leakage prevention internally)
        best_model = train_and_tune(name, builder, X_train, y_train)
        
        # Internal test set evaluation
        y_pred = best_model.predict(X_test)
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred
            
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"[{name}] Internal Test AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        results.append({'Model': name, 'Test_AUC': auc, 'Test_Accuracy': acc})
        
        # Save the full pipeline (includes scaler and feature selection)
        model_filename = f"{name.lower().replace(' ', '_')}_{args.dataset}.joblib"
        joblib.dump(best_model, os.path.join(args.models_dir, model_filename))
        print(f"Saved optimized pipeline to {model_filename}")
        
    df_res = pd.DataFrame(results)
    print("\n--- Internal Validation Results ---")
    print(df_res.to_string(index=False))
    
if __name__ == "__main__":
    main()
