"""
synthetic_validation.py - Ground-truth validation of the full ML pipeline.

Generates synthetic expression data where the true predictive genes are KNOWN,
runs the full training pipeline, and measures:
  1. Feature recovery precision/recall (does the pipeline find the right genes?)
  2. ROC-AUC on a controlled problem
  3. False positive rate (are non-signal genes ever selected?)

Usage:
    python -m src.synthetic_validation [--n-genes 5000] [--n-samples 500]
                                       [--signal-genes 30]
"""

import argparse, sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Re-use the same model builders and pipeline from train.py
from src.model import build_logistic_regression, build_random_forest, build_xgboost, build_mlp
from src.stability_selector import StabilitySelector

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score


def generate_ground_truth_data(
    n_samples: int = 500,
    n_genes: int = 5000,
    n_signal: int = 30,
    signal_strength: float = 1.5,
    noise_genes: int = 100,
    noise_strength: float = 0.4,
    random_state: int = 42,
):
    """Generate synthetic expression data where exactly `n_signal` genes drive the target.

    Returns
    -------
    X : pd.DataFrame (n_samples x n_genes)
    y : pd.Series (n_samples,)  binary {0,1}
    true_signal_indices : list[int]  column indices of the truly predictive genes
    """
    rng = np.random.RandomState(random_state)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

    # Base expression: log-normal-ish
    X = rng.lognormal(mean=0.5, sigma=0.8, size=(n_samples, n_genes))

    # Randomly pick which genes carry the true signal
    true_idx = sorted(rng.choice(n_genes, size=n_signal, replace=False))
    noise_idx = sorted(
        rng.choice(
            [i for i in range(n_genes) if i not in true_idx],
            size=min(noise_genes, n_genes - n_signal),
            replace=False,
        )
    )

    # Latent proliferation variable
    latent = rng.normal(0, 1, size=n_samples)

    # Inject signal into the true genes
    for i in true_idx:
        X[:, i] += latent * signal_strength * rng.uniform(0.8, 1.2)

    # Inject weak signal into noise genes
    for i in noise_idx:
        X[:, i] += latent * noise_strength * rng.uniform(0.0, 0.6)

    # Target = binarized latent (median split)
    y = (latent > np.median(latent)).astype(int)

    X = pd.DataFrame(X, columns=gene_names, index=[f"S{i}" for i in range(n_samples)])
    y = pd.Series(y, name="target", index=[f"S{i}" for i in range(n_samples)])

    return X, y, true_idx, noise_idx


def create_pipeline(model_builder, k=500):
    """Same pipeline as train.py but with configurable k."""
    return Pipeline([
        ("vt", VarianceThreshold(threshold=0.01)),
        ("scaler", StandardScaler()),
        ("select", StabilitySelector(k=k, n_bootstrap=30, n_jobs=2)),
        ("clf", model_builder()),
    ])


def tune_and_evaluate(X_train, y_train, X_test, y_test, model_builder, param_grid):
    """Train, tune, and evaluate a single model, returning metrics + selected features."""
    pipe = create_pipeline(model_builder)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=2)
    gs.fit(X_train, y_train)

    y_prob = gs.predict_proba(X_test)[:, 1]
    y_pred = gs.predict(X_test)
    auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))

    # Extract selected feature indices from the pipeline
    selector = gs.best_estimator_.named_steps["select"]
    vt = gs.best_estimator_.named_steps["vt"]
    scaler = gs.best_estimator_.named_steps["scaler"]

    # Get which genes survived VT + scaling
    vt_mask = vt.get_support()
    kept_indices = np.where(vt_mask)[0]

    # Get which of those were selected by stability selection
    select_mask = selector.get_support()
    # select_mask aligns to post-VT features
    # Map back to original indices
    selected_original = kept_indices[select_mask].tolist()

    return auc, acc, selected_original, gs.best_params_


def compute_recovery_metrics(selected, true_idx, n_total):
    """Compute precision, recall, and enrichment of true signal genes in selected set."""
    selected_set = set(selected)
    true_set = set(true_idx)
    tp = len(selected_set & true_set)

    precision = tp / len(selected_set) if selected_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0

    # Expected precision by random chance
    expected = len(true_set) / n_total
    enrichment = precision / expected if expected > 0 else 0.0

    return {
        "true_signal_total": len(true_idx),
        "features_selected": len(selected),
        "true_positives": tp,
        "false_positives": len(selected) - tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "enrichment_vs_random": round(enrichment, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic ground-truth validation")
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--signal-genes", type=int, default=30)
    parser.add_argument("--signal-strength", type=float, default=1.5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("=" * 65)
    print("SYNTHETIC GROUND-TRUTH VALIDATION")
    print("=" * 65)
    print(f"  Samples: {args.n_samples}")
    print(f"  Genes:   {args.n_genes}")
    print(f"  True signal genes: {args.signal_genes}")
    print(f"  Signal strength:   {args.signal_strength}")
    print()

    # 1. Generate data with known ground truth
    X, y, true_idx, noise_idx = generate_ground_truth_data(
        n_samples=args.n_samples,
        n_genes=args.n_genes,
        n_signal=args.signal_genes,
        signal_strength=args.signal_strength,
    )

    print(f"  Class balance: {y.value_counts().to_dict()}")
    print()

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print()

    # 3. Hyperparameter grids (simplified from train.py)
    param_grids = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1.0]},
        "Random Forest": {"clf__n_estimators": [100], "clf__max_depth": [5, 10]},
        "XGBoost": {"clf__n_estimators": [100], "clf__max_depth": [3, 5]},
        "Neural Network (MLP)": {"clf__hidden_layer_sizes": [(128, 64)], "clf__alpha": [0.001, 0.01]},
    }

    builders = {
        "Logistic Regression": build_logistic_regression,
        "Random Forest": build_random_forest,
        "XGBoost": build_xgboost,
        "Neural Network (MLP)": build_mlp,
    }

    results = []
    row_data = []

    for name in ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network (MLP)"]:
        print(f"  Training {name}...")
        auc, acc, selected, best_params = tune_and_evaluate(
            X_train, y_train, X_test, y_test,
            builders[name], param_grids[name],
        )

        recovery = compute_recovery_metrics(selected, true_idx, args.n_genes)

        results.append({
            "model": name,
            "auc": auc,
            "accuracy": acc,
            **recovery,
        })

        row_data.append((name, f"{auc:.4f}", f"{acc:.4f}", str(recovery["features_selected"]),
                         str(recovery["true_positives"]), f"{recovery['precision']:.4f}",
                         f"{recovery['recall']:.4f}", f"{recovery['enrichment_vs_random']:.1f}x"))

        print(f"    AUC={auc:.4f}  Acc={acc:.4f}  "
              f"TP={recovery['true_positives']}/{recovery['true_signal_total']}  "
              f"Prec={recovery['precision']:.4f}  Rec={recovery['recall']:.4f}  "
              f"Enrich={recovery['enrichment_vs_random']:.1f}x")
        print()

    # 4. Print summary table
    print()
    print("=" * 110)
    print(f"{'Model':<24} {'AUC':>8} {'Acc':>8} {'Selected':>10} {'TP':>6} "
          f"{'Precision':>10} {'Recall':>8} {'Enrichment':>12}")
    print("-" * 110)
    for row in row_data:
        print(f"{row[0]:<24} {row[1]:>8} {row[2]:>8} {row[3]:>10} {row[4]:>6} "
              f"{row[5]:>10} {row[6]:>8} {row[7]:>12}")
    print("=" * 110)

    # 5. Save
    df = pd.DataFrame(results)
    out_path = RESULTS / "synthetic_validation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save a summary JSON with the best model
    best_idx = int(df["auc"].idxmax())
    summary = {
        "n_genes": args.n_genes,
        "n_samples": args.n_samples,
        "n_signal": args.signal_genes,
        "signal_strength": args.signal_strength,
        "best_model": df.iloc[best_idx]["model"],
        "best_auc": float(df.iloc[best_idx]["auc"]),
        "avg_precision": float(df["precision"].mean()),
        "avg_recall": float(df["recall"].mean()),
        "avg_enrichment": float(df["enrichment_vs_random"].mean()),
    }
    json_path = RESULTS / "synthetic_validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
