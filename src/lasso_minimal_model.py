"""
lasso_minimal_model.py - L1-regularized (LASSO) minimal gene set experiment.

Direct response to external review: does a much smaller, sparser model
retain the performance of the full StabilitySelector pipelines?

Compares:
  1. Full pipeline (StabilitySelector k=500) holdout AUC [from train.py results]
  2. LASSO logistic regression (L1 penalty) tuned via GridSearchCV,
     evaluated on the SAME holdout split (re-binarized with training-only
     median to stay leakage-free)
  3. Bootstrap stability of the selected gene set (which genes survive
     across resamples of the training pool)

Usage:
    python -m src.lasso_minimal_model --dataset geo_pan
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import remove_proliferation_genes, validate_no_leakage

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_BOOTSTRAP = 30


def count_selected(pipeline):
    """Count genes with non-zero coefficients in the LASSO classifier."""
    coef = pipeline.named_steps["classifier"].coef_.ravel()
    return int(np.sum(coef != 0))


def selected_genes(pipeline, feature_columns):
    """Map non-zero coefficients back to original feature names."""
    var_support = pipeline.named_steps["var_thresh"].get_support()
    surviving = np.array(feature_columns)[var_support]
    coef = pipeline.named_steps["classifier"].coef_.ravel()
    return list(surviving[coef != 0])


def bootstrap_select(pipeline_best_C, X_train_pool, y_train_pool, feature_columns, seed_base=1000):
    """Refit the best-C LASSO on bootstrap resamples; report selection frequency."""
    seeds = [seed_base + i for i in range(N_BOOTSTRAP)]

    def one_run(seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_train_pool), len(X_train_pool))
        Xb = X_train_pool.iloc[idx]
        yb = y_train_pool[idx]
        pipe = Pipeline([
            ("var_thresh", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l1", solver="liblinear", C=pipeline_best_C,
                max_iter=5000, random_state=42,
            )),
        ])
        pipe.fit(Xb, yb)
        return set(selected_genes(pipe, feature_columns))

    selected_sets = Parallel(n_jobs=-1)(delayed(one_run)(s) for s in seeds)
    all_genes = sorted(set().union(*selected_sets))
    if not all_genes:
        return pd.DataFrame(), 0.0
    freq = pd.DataFrame({
        "gene": all_genes,
        "selection_frequency": [sum(g in s for s in selected_sets) / N_BOOTSTRAP
                                for g in all_genes],
    }).sort_values("selection_frequency", ascending=False)
    mean_jaccard = np.mean([
        len(s1 & s2) / len(s1 | s2)
        for i, s1 in enumerate(selected_sets)
        for s2 in selected_sets[i + 1:]
    ])
    return freq, mean_jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="geo_pan",
                        choices=["geo", "geo_pan", "synthetic"])
    args = parser.parse_args()

    prefix = f"{args.dataset}_"
    X = pd.read_csv(os.path.join("data/processed", f"{prefix}X_features.csv"), index_col=0)
    y = pd.read_csv(os.path.join("data/processed", f"{prefix}y_target.csv"), index_col=0)["target"]
    scores = pd.read_csv(os.path.join("data/processed", f"{prefix}proliferation_scores.csv"),
                         index_col=0)
    scores = scores.iloc[:, 0]
    print(f"[{args.dataset}] Raw feature matrix: {X.shape[0]} samples, {X.shape[1]} columns.")
    X = remove_proliferation_genes(X)
    validate_no_leakage(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scores_train = scores.loc[X_train.index]
    scores_test = scores.loc[X_test.index]
    train_threshold = scores_train.median()
    y_train = (scores_train >= train_threshold).astype(int).values
    y_test = (scores_test >= train_threshold).astype(int).values
    print(f"[{args.dataset}] Train={len(X_train)} Test={len(X_test)} "
          f"Features={X.shape[1]} | binarized at training-only median "
          f"{train_threshold:.4f}")

    base_pipe = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.01)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=5000, random_state=42,
        )),
    ])

    grid = GridSearchCV(
        base_pipe,
        {"classifier__C": C_GRID},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_c = grid.best_params_["classifier__C"]

    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = best.predict(X_test)
    n_sel = count_selected(best)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nBest C = {best_c} | genes selected = {n_sel}")
    print(f"Holdout AUC = {auc:.4f} | Accuracy = {acc:.4f}")

    genes = selected_genes(best, X.columns)
    gene_df = pd.DataFrame({"gene": genes})
    gene_df.to_csv(os.path.join(RESULTS_DIR, f"lasso_selected_genes_{args.dataset}.csv"),
                   index=False)

    freq_df, mean_jaccard = bootstrap_select(best_c, X_train, y_train, X.columns)
    summary = pd.DataFrame([{
        "dataset": args.dataset,
        "best_C": best_c,
        "n_genes_selected": n_sel,
        "holdout_auc": auc,
        "holdout_accuracy": acc,
        "bootstrap_mean_pairwise_jaccard": mean_jaccard,
        "bootstrap_genes_selected_ge_50pct": int(
            freq_df["selection_frequency"].ge(0.5).sum()) if len(freq_df) else 0,
    }])
    summary.to_csv(os.path.join(RESULTS_DIR,
                                f"lasso_minimal_model_{args.dataset}.csv"), index=False)

    if len(freq_df):
        freq_df.to_csv(os.path.join(RESULTS_DIR,
                                    f"lasso_gene_stability_{args.dataset}.csv"), index=False)
        top = freq_df.head(20)
        print("\nTop 20 most stable LASSO genes:")
        for _, r in top.iterrows():
            print(f"  {r['gene']:<12s} {r['selection_frequency']:.2f}")
    print(f"\nSaved: results/lasso_minimal_model_{args.dataset}.csv")
    if args.dataset == "geo_pan":
        print("Reference (full StabilitySelector pipeline, train.py):")
        print("  Logistic Regression holdout AUC = 0.9834 | ~500 features")


if __name__ == "__main__":
    main()