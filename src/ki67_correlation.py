"""
ki67_correlation.py - Correlate model predictions with MKI67 expression.

Shows that even though MKI67 was removed from training features,
the model's predicted proliferation probability correlates with
actual MKI67 expression across all cohorts.
"""

import os
import joblib
import gzip
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = "results"
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_geo_mki67():
    path = os.path.join(RAW_DIR, "GSE39582_series_matrix.txt.gz")
    with gzip.open(path, "rt", encoding="latin-1") as f:
        lines = f.readlines()
    in_table = False
    table_lines = []
    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_table:
            table_lines.append(line)
    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    expr = expr.T
    expr.index.name = None
    print(f"  GEO raw: {expr.shape}")

    # GEO uses probe IDs; map to gene symbols via GPL570 annotation
    annot_path = os.path.join(RAW_DIR, "GPL570.annot.gz")
    if os.path.exists(annot_path):
        # Skip header lines before the table
        with gzip.open(annot_path, "rt", encoding="latin-1") as f:
            all_lines = f.readlines()
        table_start = None
        for i, line in enumerate(all_lines):
            if line.startswith("!platform_table_begin"):
                table_start = i + 1
                break
        if table_start is None:
            table_start = 0
        table_str = "".join(all_lines[table_start:])
        annot = pd.read_csv(io.StringIO(table_str), sep="\t", low_memory=False)
        # Look for MKI67 in Gene symbol column
        if "Gene symbol" in annot.columns:
            mask = annot["Gene symbol"].astype(str).str.contains(r"\bMKI67\b", na=False)
            matched = annot[mask]
            if len(matched) > 0:
                probe_id = matched.iloc[0]["ID"]
                print(f"  MKI67 probe ID: {probe_id}")
                if probe_id in expr.columns:
                    return expr[probe_id]
    print("  WARNING: Could not map MKI67 in GEO data")
    return None


def load_tcga_mki67():
    path = os.path.join(RAW_DIR, "tcga_coad_expression.tsv.gz")
    expr = pd.read_csv(path, sep="\t", compression="gzip")
    expr = expr.set_index("sample").T
    expr.index.name = None
    print(f"  TCGA raw: {expr.shape}")
    if "MKI67" in expr.columns:
        return expr["MKI67"]
    print("  WARNING: MKI67 not found in TCGA raw data")
    return None


def load_cptac_mki67():
    path = os.path.join(RAW_DIR, "cptac_coad_rnaseq.txt")
    if not os.path.exists(path):
        print("  CPTAC raw data not found")
        return None
    expr = pd.read_csv(path, sep="\t", index_col=0)
    print(f"  CPTAC raw: {expr.shape}")
    if "MKI67" in expr.index:
        s = expr.loc["MKI67"]
        s.index = s.index.astype(str).str.strip()
        return s
    print("  WARNING: MKI67 not found in CPTAC raw data")
    return None


def align_features(X_ref, X_test):
    """Zero-fill missing features in test set to match training features."""
    common = [c for c in X_ref.columns if c in X_test.columns]
    print(f"  Aligned features: {len(common)}/{X_ref.shape[1]}")
    X_aligned = pd.DataFrame(0, index=X_test.index, columns=X_ref.columns)
    for col in common:
        X_aligned[col] = X_test[col]
    return X_aligned


def get_predictions(model_path, X, dataset_name):
    model = joblib.load(model_path)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        return pd.Series(y_prob, index=X.index, name="predicted_prob")
    except Exception as e:
        print(f"  Predict error: {e}")
        return None


def run_correlation(mki67, y_prob, scores, label):
    common = mki67.dropna().index.intersection(y_prob.index)
    if scores is not None:
        common = common.intersection(scores.index)
    if len(common) < 5:
        print(f"  Too few overlapping samples ({len(common)}), skipping")
        return None

    m = mki67.loc[common]
    p = y_prob.loc[common]
    s = scores.loc[common] if scores is not None else None

    # Pearson
    r_p, p_p = pearsonr(m, p)
    rho_s, p_s = spearmanr(m, p)
    print(f"  MKI67 vs Probability: Pearson r={r_p:.4f} (p={p_p:.4e}), Spearman rho={rho_s:.4f} (p={p_s:.4e})")

    result = {
        "dataset": label,
        "n": len(common),
        "pearson_r": round(r_p, 4),
        "pearson_p": f"{p_p:.4e}",
        "spearman_rho": round(rho_s, 4),
        "spearman_p": f"{p_s:.4e}",
    }

    # Score correlation
    if s is not None:
        r2_p, p2_p = pearsonr(s, p)
        r2_s, p2_s = spearmanr(s, p)
        print(f"  Score vs Probability: Pearson r={r2_p:.4f} (p={p2_p:.4e}), Spearman rho={r2_s:.4f} (p={p2_s:.4e})")
        result["score_pearson_r"] = round(r2_p, 4)
        result["score_spearman_rho"] = round(r2_s, 4)

    # Plot
    fig, axes = plt.subplots(1, 2 if s is not None else 1, figsize=(14, 5) if s is not None else (7, 5))

    if s is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes

    ax1.scatter(m, p, alpha=0.5, s=20, c="#2c3e6b", edgecolors="none")
    ax1.set_xlabel("MKI67 Expression (raw)", fontsize=11)
    ax1.set_ylabel("Predicted Proliferation Probability", fontsize=11)
    ax1.set_title(f"{label}\nMKI67 vs Predicted Probability", fontsize=12)
    z = np.polyfit(m, p, 1)
    poly = np.poly1d(z)
    xl = np.linspace(m.min(), m.max(), 100)
    ax1.plot(xl, poly(xl), "r--", lw=1.5, alpha=0.7)
    ax1.text(0.05, 0.92, f"Pearson r = {r_p:.3f}\nSpearman rho = {rho_s:.3f}",
             transform=ax1.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.grid(True, alpha=0.3)

    if s is not None:
        ax2.scatter(s, p, alpha=0.5, s=20, c="#1a7a3b", edgecolors="none")
        ax2.set_xlabel("Proliferation Score (10-gene mean z-score)", fontsize=11)
        ax2.set_ylabel("Predicted Proliferation Probability", fontsize=11)
        ax2.set_title(f"{label}\nProlif. Score vs Predicted Probability", fontsize=12)
        z2 = np.polyfit(s, p, 1)
        poly2 = np.poly1d(z2)
        xl2 = np.linspace(s.min(), s.max(), 100)
        ax2.plot(xl2, poly2(xl2), "r--", lw=1.5, alpha=0.7)
        ax2.text(0.05, 0.92, f"Pearson r = {r2_p:.3f}\nSpearman rho = {r2_s:.3f}",
                 transform=ax2.transAxes, fontsize=10, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    sl = label.lower().replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"ki67_correlation_{sl}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Plot: {path}")
    plt.close(fig)

    return result


print("=" * 60)
print("Ki-67 / MKI67 Correlation Analysis")
print("=" * 60)

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_geo.joblib")
X_geo_ref = pd.read_csv(os.path.join(DATA_DIR, "geo_X_features.csv"), index_col=0)
results = []

# ============================================================
# GEO (training cohort)
# ============================================================
print("\n--- GEO GSE39582 ---")
geo_mki67 = load_geo_mki67()
if geo_mki67 is not None:
    geo_probs = get_predictions(MODEL_PATH, X_geo_ref, "geo")
    geo_scores = pd.read_csv(os.path.join(DATA_DIR, "geo_proliferation_scores.csv"), index_col=0)["score"]
    if geo_probs is not None:
        r = run_correlation(geo_mki67, geo_probs, geo_scores, "GEO GSE39582")
        if r: results.append(r)

# ============================================================
# TCGA-COAD (external validation)
# ============================================================
print("\n--- TCGA-COAD ---")
tcga_mki67 = load_tcga_mki67()
if tcga_mki67 is not None:
    X_tcga = pd.read_csv(os.path.join(DATA_DIR, "tcga_X_features.csv"), index_col=0)
    X_tcga_aligned = align_features(X_geo_ref, X_tcga)
    tcga_probs = get_predictions(MODEL_PATH, X_tcga_aligned, "tcga")
    tcga_scores = pd.read_csv(os.path.join(DATA_DIR, "tcga_proliferation_scores.csv"), index_col=0)["score"]
    if tcga_probs is not None:
        r = run_correlation(tcga_mki67, tcga_probs, tcga_scores, "TCGA-COAD")
        if r: results.append(r)

# ============================================================
# CPTAC-COAD (third cohort)
# ============================================================
print("\n--- CPTAC-COAD ---")
cptac_mki67 = load_cptac_mki67()
if cptac_mki67 is not None:
    X_cptac = pd.read_csv(os.path.join(DATA_DIR, "cptac_X_features.csv"), index_col=0)
    X_cptac_aligned = align_features(X_geo_ref, X_cptac)
    cptac_probs = get_predictions(MODEL_PATH, X_cptac_aligned, "cptac")
    cptac_scores = pd.read_csv(os.path.join(DATA_DIR, "cptac_proliferation_scores.csv"), index_col=0)["score"]
    if cptac_probs is not None:
        r = run_correlation(cptac_mki67, cptac_probs, cptac_scores, "CPTAC-COAD")
        if r: results.append(r)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS_DIR, "ki67_correlation_summary.csv"), index=False)
print(f"\nSaved: results/ki67_correlation_summary.csv")
