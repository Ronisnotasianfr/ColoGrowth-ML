"""
complete_analysis.py - Advanced statistical, sensitivity, pathway, and clinical analyses.
Generates all figures and tables required for Phase 1-3 improvements.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import norm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set plot style for academic journals
sns.set_theme(style='ticks')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    ece = 0.0
    n_samples = len(y_true)
    
    # Reconstruct bin assignments
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get samples in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return ece


def compute_bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, seed=42):
    """Compute 95% Confidence Intervals using bootstrapping."""
    np.random.seed(seed)
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Handle case where bootstrap sample contains only one class
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_prob[indices])
        scores.append(score)
        
    scores = np.sort(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return lower, upper


def bootstrap_roc_comparison(y_true, prob_a, prob_b, n_bootstrap=1000, seed=42):
    """Test statistical significance of ROC-AUC difference between two models using bootstrapping."""
    np.random.seed(seed)
    diffs = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_a = roc_auc_score(y_true[indices], prob_a[indices])
        auc_b = roc_auc_score(y_true[indices], prob_b[indices])
        diffs.append(auc_a - auc_b)
        
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # Calculate two-tailed z-score and p-value
    z = mean_diff / (std_diff + 1e-10)
    p_val = 2 * (1 - norm.cdf(np.abs(z)))
    return mean_diff, p_val


def run_baselines(X_train, y_train, X_test, y_test):
    """Train and evaluate baseline models (prevalence class & simple LR without feature selection)."""
    print("\n--- Running Baseline Models ---")
    
    # 1. Prevalence classifier (majority class)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
    
    dummy_acc = accuracy_score(y_test, y_pred_dummy)
    dummy_auc = roc_auc_score(y_test, y_prob_dummy)
    
    # 2. Simple Logistic Regression (all features, standard scaling, VT, NO SelectKBest)
    simple_lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('var_thresh', VarianceThreshold(threshold=0.01)),
        ('classifier', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42, solver='liblinear'))
    ])
    
    simple_lr_pipe.fit(X_train, y_train)
    y_pred_simple = simple_lr_pipe.predict(X_test)
    y_prob_simple = simple_lr_pipe.predict_proba(X_test)[:, 1]
    
    simple_acc = accuracy_score(y_test, y_pred_simple)
    simple_auc = roc_auc_score(y_test, y_prob_simple)
    
    print(f"Prevalence baseline: Holdout Accuracy = {dummy_acc:.4f}, AUC = {dummy_auc:.4f}")
    print(f"Simple LR (no KBest): Holdout Accuracy = {simple_acc:.4f}, AUC = {simple_auc:.4f}")
    
    return {
        'dummy_accuracy': dummy_acc,
        'dummy_auc': dummy_auc,
        'simple_lr_accuracy': simple_acc,
        'simple_lr_auc': simple_auc
    }


def analyze_feature_selection_stability(X_train, y_train):
    """Analyze feature selection frequency across folds and top 20 genes."""
    print("\n--- Feature Selection Stability Analysis ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Exclude clinical features from gene stability mapping
    gene_cols = [c for c in X_train.columns if not c.startswith('clinical_')]
    X_train_genes = X_train[gene_cols]
    
    selected_counts = pd.Series(0, index=gene_cols)
    
    # Fold local preprocessing
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_genes, y_train)):
        X_fold_tr = X_train_genes.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fold_tr)
        
        # Variance filter
        vt = VarianceThreshold(threshold=0.01)
        X_vt = vt.fit_transform(X_scaled)
        vt_support = vt.get_support()
        surviving_genes = np.array(gene_cols)[vt_support]
        
        # SelectKBest (k=500)
        kbest = SelectKBest(score_func=f_classif, k=min(500, len(surviving_genes)))
        kbest.fit(X_vt, y_fold_tr)
        kbest_support = kbest.get_support()
        
        selected_genes = surviving_genes[kbest_support]
        selected_counts[selected_genes] += 1
        
    # Calculate ANOVA F-scores on the entire training pool
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X_train_genes)
    vt_full = VarianceThreshold(threshold=0.01)
    X_vt_full = vt_full.fit_transform(X_scaled_full)
    surviving_genes_full = np.array(gene_cols)[vt_full.get_support()]
    
    kbest_full = SelectKBest(score_func=f_classif, k='all')
    kbest_full.fit(X_vt_full, y_train)
    
    # Build results table
    stability_df = pd.DataFrame({
        'Gene': surviving_genes_full,
        'ANOVA_F_Score': kbest_full.scores_,
        'CV_Selection_Frequency': selected_counts[surviving_genes_full].values
    })
    
    stability_df = stability_df.sort_values(by='ANOVA_F_Score', ascending=False).reset_index(drop=True)
    top_20 = stability_df.head(20)
    top_20.to_csv(os.path.join(RESULTS_DIR, "feature_selection_stability_top20.csv"), index=False)
    
    print("Top 20 selected genes saved to results/feature_selection_stability_top20.csv.")
    
    # Plot feature stability distribution
    plt.figure(figsize=(7, 4.5))
    sns.countplot(x='CV_Selection_Frequency', data=stability_df, hue='CV_Selection_Frequency', palette='Blues', legend=False)
    plt.title("Feature Selection Stability Across 5 CV Folds")
    plt.xlabel("Number of Folds Feature Was Selected In")
    plt.ylabel("Gene Count")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_selection_stability.png"), dpi=300)
    plt.close()
    
    return top_20


def run_pathway_enrichment(top_genes):
    """Run pathway enrichment analysis on top genes and export results."""
    print("\n--- Pathway Enrichment Analysis ---")
    gene_list = list(top_genes['Gene'].values)
    
    try:
        import gseapy as gp
        print("Querying Enrichr via gseapy...")
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021', 'Reactome_2021'],
            organism='human',
            outdir=None
        )
        results = enr.results
        results = results[results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
        
        if len(results) == 0:
            print("No statistically significant pathways found (FDR < 0.05).")
            raise Exception("No significant pathways")
            
        results.to_csv(os.path.join(RESULTS_DIR, "pathway_enrichment_results.csv"), index=False)
        print("Enrichment results saved to results/pathway_enrichment_results.csv.")
        
        # Plot top 10 pathways
        plt.figure(figsize=(9, 5))
        top_10 = results.head(10)
        sns.barplot(
            x=-np.log10(top_10['Adjusted P-value']),
            y=top_10['Term'],
            hue=top_10['Term'],
            palette='crest_r',
            legend=False
        )
        plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='FDR = 0.05')
        plt.title("Top Enriched Pathways (g:Profiler/Enrichr)")
        plt.xlabel("-log10(Adjusted P-value)")
        plt.ylabel("Pathway Term")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "pathway_enrichment.png"), dpi=300)
        plt.savefig(os.path.join(RESULTS_DIR, "pathway_enrichment.pdf"), format='pdf')
        plt.close()
        
    except Exception as exc:
        print(f"Enrichment service unavailable: {exc}. Pathway enrichment results will not be generated.")
        print("Re-run with a working internet connection to get real enrichment data.")


def run_sensitivity_analysis(X, y, dataset="geo"):
    """Run sensitivity analysis for hyperparameters and feature pre-processing."""
    print("\n--- Running Sensitivity Analyses ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 1. Feature selection SelectKBest k sensitivity
    k_values = [100, 200, 300, 500, 1000]
    k_results = []
    
    for k in k_values:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('var_thresh', VarianceThreshold(threshold=0.01)),
            ('feature_select', SelectKBest(score_func=f_classif, k=k)),
            ('classifier', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42, solver='liblinear'))
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        k_results.append(auc)
        
    # 2. Variance Threshold sensitivity
    vt_values = [0.001, 0.005, 0.01, 0.05]
    vt_results = []
    vt_counts = []
    
    for vt in vt_values:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('var_thresh', VarianceThreshold(threshold=vt)),
            ('feature_select', SelectKBest(score_func=f_classif, k=500)),
            ('classifier', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42, solver='liblinear'))
        ])
        pipe.fit(X_train, y_train)
        
        # Get count of features passing VT
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        vt_obj = VarianceThreshold(threshold=vt)
        vt_obj.fit(X_scaled)
        passed_count = np.sum(vt_obj.get_support())
        
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        vt_results.append(auc)
        vt_counts.append(passed_count)
        
    # 3. Binarization threshold sensitivity
    # We load raw proliferation scores, binarize at different percentiles, split and compute holdout ROC-AUC
    scores_path = os.path.join(DATA_DIR, f"{dataset}_proliferation_scores.csv")
    if os.path.exists(scores_path):
        scores_df = pd.read_csv(scores_path, index_col=0)
        scores = scores_df['score']
        percentiles = [25, 33, 50, 67, 75]
        bin_results = []
        
        for p in percentiles:
            thresh = np.percentile(scores, p)
            y_bin = (scores >= thresh).astype(int)
            
            # Align
            common = X.index.intersection(y_bin.index)
            X_sub = X.loc[common]
            y_sub = y_bin.loc[common]
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42)
            
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('var_thresh', VarianceThreshold(threshold=0.01)),
                ('feature_select', SelectKBest(score_func=f_classif, k=500)),
                ('classifier', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42, solver='liblinear'))
            ])
            pipe.fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            bin_results.append(auc)
    else:
        print("Proliferation scores not found, skipping binarization threshold sensitivity.")
        percentiles = [25, 33, 50, 67, 75]
        bin_results = []
        
    # Save sensitivity results
    sensitivity_df = pd.DataFrame({
        'k_value': k_values,
        'k_AUC': k_results
    })
    sensitivity_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_kbest.csv"), index=False)
    
    # Save VT
    vt_df = pd.DataFrame({
        'VT_threshold': vt_values,
        'Features_Passed': vt_counts,
        'VT_AUC': vt_results
    })
    vt_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_variance.csv"), index=False)
    
    # Plot sensitivity analyses side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Plot k
    axes[0].plot(k_values, k_results, 'o-', color='#1F4E79', linewidth=2)
    axes[0].set_title("SelectKBest k Sensitivity")
    axes[0].set_xlabel("Number of Features (k)")
    axes[0].set_ylabel("Holdout ROC-AUC")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot VT
    axes[1].plot(vt_values, vt_results, 's-', color='#2D4F6C', linewidth=2)
    axes[1].set_title("Variance Threshold Sensitivity")
    axes[1].set_xlabel("Variance Threshold (VT)")
    axes[1].set_ylabel("Holdout ROC-AUC")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot binarization (only if we have data)
    if len(bin_results) > 0:
        axes[2].plot(percentiles, bin_results, '^-', color='#C55A11', linewidth=2)
        axes[2].set_title("Binarization Threshold Sensitivity")
        axes[2].set_xlabel("Binarization Percentile")
        axes[2].set_ylabel("Holdout ROC-AUC")
    else:
        axes[2].text(0.5, 0.5, 'No proliferation scores\navailable for this cohort',
                     ha='center', va='center', transform=axes[2].transAxes,
                     fontsize=10, color='gray')
        axes[2].set_title("Binarization Threshold Sensitivity")
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Model Robustness and Sensitivity Analyses", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_plots.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_plots.pdf"), format='pdf')
    plt.close()
    print("Sensitivity plots and data exported.")


def run_decision_curve_analysis(y_test, model_probs):
    """Generate Decision Curve Analysis (DCA) plot and calculate Net Benefit & NNT."""
    print("\n--- Clinical Decision Curve Analysis (DCA) ---")
    
    thresholds = np.linspace(0.01, 0.99, 100)
    n_samples = len(y_test)
    prevalence = np.mean(y_test)
    
    # Baseline: Treat All
    net_benefit_all = []
    for pt in thresholds:
        tp = np.sum(y_test == 1)
        fp = np.sum(y_test == 0)
        nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
        net_benefit_all.append(nb)
        
    # Baseline: Treat None
    net_benefit_none = np.zeros_like(thresholds)
    
    # Model net benefits
    model_nbs = {}
    nnt_records = []
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_all, '--', color='gray', alpha=0.7, label='Treat All (High Prolif)')
    plt.plot(thresholds, net_benefit_none, '-', color='black', linewidth=1.2, label='Treat None')
    
    colors_dca = {'Logistic Regression': '#1F4E79', 'XGBoost': '#C55A11', 'Random Forest': '#2E7D32', 'Neural Network (MLP)': '#AD1457'}
    
    for name, probs in model_probs.items():
        nb_vals = []
        for pt in thresholds:
            y_pred = (probs >= pt).astype(int)
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
            nb_vals.append(nb)
            
        model_nbs[name] = nb_vals
        plt.plot(thresholds, nb_vals, '-', color=colors_dca.get(name, 'blue'), linewidth=2, label=name)
        
        # Calculate NNT at 50% and 60% thresholds
        for cutoff in [0.50, 0.60]:
            y_pred_cut = (probs >= cutoff).astype(int)
            tp_cut = np.sum((y_test == 1) & (y_pred_cut == 1))
            fp_cut = np.sum((y_test == 0) & (y_pred_cut == 1))
            nb_cut = (tp_cut / n_samples) - (fp_cut / n_samples) * (cutoff / (1 - cutoff))
            nnt = 1.0 / nb_cut if nb_cut > 0 else np.nan
            
            # Sensitivity, Specificity, PPV, NPV
            tp_rate = np.sum((y_test == 1) & (y_pred_cut == 1))
            fn_rate = np.sum((y_test == 1) & (y_pred_cut == 0))
            fp_rate = np.sum((y_test == 0) & (y_pred_cut == 1))
            tn_rate = np.sum((y_test == 0) & (y_pred_cut == 0))
            
            sens = tp_rate / (tp_rate + fn_rate) if (tp_rate + fn_rate) > 0 else 0
            spec = tn_rate / (tn_rate + fp_rate) if (tn_rate + fp_rate) > 0 else 0
            ppv = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0
            npv = tn_rate / (tn_rate + fn_rate) if (tn_rate + fn_rate) > 0 else 0
            
            nnt_records.append({
                'Model': name,
                'Threshold': cutoff,
                'Sensitivity': sens,
                'Specificity': spec,
                'PPV': ppv,
                'NPV': npv,
                'Net_Benefit': nb_cut,
                'NNT': nnt
            })
            
    plt.title("Clinical Decision Curve Analysis (DCA)")
    plt.xlabel("Threshold Probability ($P_t$)")
    plt.ylabel("Net Benefit")
    plt.ylim(-0.1, 0.65)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "clinical_dca.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "clinical_dca.pdf"), format='pdf')
    plt.close()
    
    nnt_df = pd.DataFrame(nnt_records)
    nnt_df.to_csv(os.path.join(RESULTS_DIR, "clinical_utility_nnt.csv"), index=False)
    print("DCA plots and clinical utility table saved.")


def bootstrap_subgroup_interaction(y_test, y_prob, mask_a, mask_b, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    n_samples = len(y_test)
    
    # Original difference
    orig_auc_a = roc_auc_score(y_test[mask_a], y_prob[mask_a])
    orig_auc_b = roc_auc_score(y_test[mask_b], y_prob[mask_b])
    orig_diff = orig_auc_a - orig_auc_b
    
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_test[idx]
        prob_boot = y_prob[idx]
        mask_a_boot = mask_a[idx]
        mask_b_boot = mask_b[idx]
        
        # Verify both subgroups have both classes present in the bootstrap sample
        if (len(np.unique(y_boot[mask_a_boot])) < 2) or (len(np.unique(y_boot[mask_b_boot])) < 2):
            continue
            
        auc_a = roc_auc_score(y_boot[mask_a_boot], prob_boot[mask_a_boot])
        auc_b = roc_auc_score(y_boot[mask_b_boot], prob_boot[mask_b_boot])
        diffs.append(auc_a - auc_b)
        
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return orig_diff, orig_diff, orig_diff, 1.0
        
    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    
    # Center diffs around original difference to represent H0: diff = 0
    centered_diffs = diffs - orig_diff
    p_val = np.mean(np.abs(centered_diffs) >= np.abs(orig_diff))
    p_val = min(1.0, max(0.0, p_val))
    return orig_diff, lower, upper, p_val


def run_subgroup_analysis(X_test, y_test, model_names, dataset="geo_pan"):
    """Evaluate holdout test metrics stratified by demographics (Age, Sex, Stage) and test for interactions."""
    print("\n--- Subgroup Performance Analysis ---")
    
    # Extract clinical variables from X_test
    # age is in clinical_age, sex in clinical_is_male (1=Male, 0=Female), stage in clinical_stage
    age = X_test['clinical_age']
    sex = X_test['clinical_is_male']
    stage = X_test['clinical_stage']
    
    subgroups = {
        'All': np.ones(len(y_test), dtype=bool),
        'Age < 65': age < 65,
        'Age >= 65': age >= 65,
        'Male': sex == 1,
        'Female': sex == 0,
        'Stage I/II': stage.isin([1, 2]),
        'Stage III/IV': stage.isin([3, 4])
    }
    
    subgroup_records = []
    
    for name in model_names:
        slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODELS_DIR, f"{slug}_{dataset}.joblib")
        if not os.path.exists(model_path):
            continue
        pipeline = joblib.load(model_path)
        
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred.astype(float)
            
        # Compute interaction testing
        interaction_results = {}
        for pair_name, (g1, g2) in [
            ('Age', ('Age < 65', 'Age >= 65')),
            ('Sex', ('Male', 'Female')),
            ('Stage', ('Stage I/II', 'Stage III/IV'))
        ]:
            mask1 = subgroups[g1]
            mask2 = subgroups[g2]
            if len(np.unique(y_test[mask1])) >= 2 and len(np.unique(y_test[mask2])) >= 2:
                diff, lower, upper, p_val = bootstrap_subgroup_interaction(
                    y_test.values, y_prob, mask1.values, mask2.values, n_bootstrap=1000, seed=42
                )
                interaction_results[g1] = (diff, lower, upper, p_val)
                interaction_results[g2] = (diff, lower, upper, p_val)
            else:
                interaction_results[g1] = (0.0, 0.0, 0.0, 1.0)
                interaction_results[g2] = (0.0, 0.0, 0.0, 1.0)
                
        for group_name, mask in subgroups.items():
            n_sub = np.sum(mask)
            if n_sub < 5 or len(np.unique(y_test[mask])) < 2:
                # Skip subgroup if too small or lacks positive/negative classes
                continue
                
            acc = accuracy_score(y_test[mask], y_pred[mask])
            auc = roc_auc_score(y_test[mask], y_prob[mask])
            
            # CI for AUC via bootstrap
            auc_lower, auc_upper = compute_bootstrap_ci(
                y_test[mask].values, y_prob[mask], roc_auc_score, n_bootstrap=200
            )
            
            # Lookup interaction details
            if group_name in interaction_results:
                diff, lower, upper, p_val = interaction_results[group_name]
                int_p_str = f"{p_val:.4f}"
                int_ci_str = f"({lower:.4f} to {upper:.4f})"
            else:
                int_p_str = "-"
                int_ci_str = "-"
                
            subgroup_records.append({
                'Model': name,
                'Subgroup': group_name,
                'N': n_sub,
                'Accuracy': acc,
                'ROC_AUC': auc,
                'ROC_AUC_95_CI': f"({auc_lower:.4f}-{auc_upper:.4f})",
                'Interaction_p_value': int_p_str,
                'Interaction_95_CI': int_ci_str
            })
            
    subgroup_df = pd.DataFrame(subgroup_records)
    subgroup_df.to_csv(os.path.join(RESULTS_DIR, "subgroup_analysis_results.csv"), index=False)
    print("Subgroup analysis table exported to results/subgroup_analysis_results.csv.")
    
    # Plot stratified Kaplan-Meier curves
    # We load clinical and target label data
    clinical_path = os.path.join(DATA_DIR, "geo_clinical.csv")
    target_path = os.path.join(DATA_DIR, "geo_y_target.csv")
    
    if os.path.exists(clinical_path) and os.path.exists(target_path):
        clinical = pd.read_csv(clinical_path, index_col=0)
        y = pd.read_csv(target_path, index_col=0)['target']
        
        # Rename columns to standard
        rename_map = {
            'os.event': 'os_event',
            'os.delay_(months)': 'os_time',
            'os.delay': 'os_time',
            'tnm.stage': 'stage'
        }
        clinical = clinical.rename(columns=rename_map)
        if 'stage' not in clinical.columns:
            for col in clinical.columns:
                if 'stage' in col.lower():
                    clinical['stage'] = clinical[col]
                    break
        
        merged = pd.concat([clinical, y], axis=1, join='inner')
        merged['os_time'] = pd.to_numeric(merged['os_time'], errors='coerce')
        merged['os_event'] = pd.to_numeric(merged['os_event'], errors='coerce')
        
        if 'stage' not in merged.columns:
            merged['stage'] = 'Stage II'
            
        merged = merged.dropna(subset=['os_time', 'os_event', 'target'])
        
        # Plot Stage I/II vs Stage III/IV KM curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Stage I/II
        stage_mask = merged['stage'].astype(str).str.upper().str.contains('I|II')
        # Exclude Stage III and IV from Stage I/II (since 'III' contains 'II')
        stage_mask_12 = stage_mask & (~merged['stage'].astype(str).str.upper().str.contains('III|IV'))
        
        for idx, (mask, title) in enumerate([
            (stage_mask_12, "Stage I/II Patients"),
            (~stage_mask_12, "Stage III/IV Patients")
        ]):
            df_sub = merged[mask]
            if len(df_sub) == 0:
                continue
            
            kmf_high = KaplanMeierFitter()
            kmf_low = KaplanMeierFitter()
            
            high_mask = df_sub['target'] == 1
            low_mask = df_sub['target'] == 0
            
            t_high = df_sub.loc[high_mask, 'os_time']
            e_high = df_sub.loc[high_mask, 'os_event']
            t_low = df_sub.loc[low_mask, 'os_time']
            e_low = df_sub.loc[low_mask, 'os_event']
            
            kmf_high.fit(t_high, e_high, label="High Prolif")
            kmf_low.fit(t_low, e_low, label="Low Prolif")
            
            kmf_low.plot_survival_function(ax=axes[idx], color='blue', ci_show=True)
            kmf_high.plot_survival_function(ax=axes[idx], color='red', ci_show=True)
            
            lr_res = logrank_test(t_high, t_low, e_high, e_low)
            axes[idx].text(0.05, 0.1, f"Log-Rank p = {lr_res.p_value:.3e}", transform=axes[idx].transAxes,
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            axes[idx].set_title(f"Overall Survival: {title}")
            axes[idx].set_xlabel("Time (Months)")
            axes[idx].set_ylabel("Survival Probability")
            axes[idx].grid(True, linestyle=':', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "kaplan_meier_stage_stratified.png"), dpi=300)
        plt.savefig(os.path.join(RESULTS_DIR, "kaplan_meier_stage_stratified.pdf"), format='pdf')
        plt.close()
        print("Stage-stratified survival plots saved.")


def run_cox_proportional_hazards(dataset="geo_pan"):
    """Fit a multivariate Cox Proportional Hazards model to show independent prognostic value."""
    print(f"\n--- Cox Proportional Hazards Regression ({dataset}) ---")
    
    prefix = f"{dataset}_" if dataset != "dataset" else ""
    clinical_path = os.path.join(DATA_DIR, f"{prefix}clinical.csv")
    target_path = os.path.join(DATA_DIR, f"{prefix}y_target.csv")
    features_path = os.path.join(DATA_DIR, f"{prefix}X_features.csv")
    
    if not (os.path.exists(clinical_path) and os.path.exists(target_path) and os.path.exists(features_path)):
        print("Required files for Cox model missing. Skipping.")
        return
        
    clinical = pd.read_csv(clinical_path, index_col=0)
    y = pd.read_csv(target_path, index_col=0)['target']
    X = pd.read_csv(features_path, index_col=0)
    
    # Rename columns to standard
    rename_map = {}
    for col in clinical.columns:
        if col in ('os.event',):
            rename_map[col] = 'os_event'
        elif col in ('os.delay_(months)', 'os.delay', 'os_delay_months'):
            rename_map[col] = 'os_time'
        elif col == 'overall_event_(death_from_any_cause)':
            rename_map[col] = 'os_event'
        elif col == 'overall_survival_follow-up_time':
            rename_map[col] = 'os_time'
    clinical = clinical.rename(columns=rename_map)
    clinical = clinical.rename(columns=rename_map)
    
    # Extract clinical variables
    age = X['clinical_age']
    sex = X['clinical_is_male']
    stage = X['clinical_stage']
    
    # Combine into a single DataFrame for lifelines
    df_cox = pd.DataFrame({
        'os_time': pd.to_numeric(clinical['os_time'], errors='coerce'),
        'os_event': pd.to_numeric(clinical['os_event'], errors='coerce'),
        'High_Proliferation': y,
        'Age': age,
        'Is_Male': sex,
        'Stage': stage
    })
    
    # Clean missing values
    df_cox = df_cox.dropna()
    
    # Ensure some variance in stage and positive events
    if len(df_cox) > 0 and df_cox['os_event'].sum() > 0:
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col='os_time', event_col='os_event')
        
        # Print summary
        summary = cph.summary
        print(summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])
        
        # Save summary
        summary.to_csv(os.path.join(RESULTS_DIR, "cox_ph_model_summary.csv"))
        
        # Render a forest plot
        plt.figure(figsize=(7.5, 4))
        cph.plot()
        plt.title("Multivariate Cox PH Hazard Ratios")
        plt.axvline(0, color='red', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "cox_ph_forest_plot.png"), dpi=300)
        plt.savefig(os.path.join(RESULTS_DIR, "cox_ph_forest_plot.pdf"), format='pdf')
        plt.close()
        print("Cox model summary and forest plot exported.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Complete analysis pipeline")
    parser.add_argument("--dataset", default="geo_pan", choices=["geo", "geo_pan", "tcga", "tcga_pan", "synthetic"])
    args = parser.parse_args()
    dataset = args.dataset
    
    # Load dataset
    print(f"Loading {dataset} data...")
    prefix = f"{dataset}_"
    x_path = os.path.join(DATA_DIR, f"{prefix}X_features.csv")
    y_path = os.path.join(DATA_DIR, f"{prefix}y_target.csv")
    
    if not os.path.exists(x_path):
        print(f"Data not found at {x_path}. Run preprocess.py first.")
        return
        
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)['target']
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 1. Run baselines
    baselines = run_baselines(X_train, y_train, X_test, y_test)
    pd.DataFrame([baselines]).to_csv(os.path.join(RESULTS_DIR, "baseline_model_results.csv"), index=False)
    
    # 2. Run model-specific evaluations to collect holdout probs
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network (MLP)']
    model_probs = {}
    model_preds = {}
    
    # Table for calibration and ECE metrics
    calibration_records = []
    
    # Setup for bootstrap ROC comparisons
    metrics_records = []
    
    # We will generate confusion matrices too
    for name in model_names:
        slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODELS_DIR, f"{slug}_{dataset}.joblib")
        
        if not os.path.exists(model_path):
            print(f"Trained model not found for {name} at {model_path}.")
            continue
            
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred.astype(float)
            
        model_probs[name] = y_prob
        model_preds[name] = y_pred
        
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test.values, y_prob)
        
        # Bootstrap CIs
        acc_low, acc_high = compute_bootstrap_ci(y_test.values, y_pred, accuracy_score)
        auc_low, auc_high = compute_bootstrap_ci(y_test.values, y_prob, roc_auc_score)
        
        metrics_records.append({
            'Model': name,
            'Accuracy': acc,
            'Accuracy_95_CI': f"{acc:.4f} (95% CI: {acc_low:.4f}-{acc_high:.4f})",
            'ROC_AUC': auc,
            'ROC_AUC_95_CI': f"{auc:.4f} (95% CI: {auc_low:.4f}-{auc_high:.4f})",
            'F1_Score': f1,
            'Brier_Score': brier,
            'ECE': ece
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=['Pred_Low', 'Pred_High'], index=['True_Low', 'True_High'])
        cm_df.to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_values_{slug}.csv"))
        
        # Plot confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_plot_{slug}.png"), dpi=300)
        plt.close()
        
    metrics_summary = pd.DataFrame(metrics_records)
    metrics_summary.to_csv(os.path.join(RESULTS_DIR, "detailed_model_metrics_with_ci.csv"), index=False)
    
    # 3. Pairwise Bootstrap ROC Comparison
    pairwise_records = []
    loaded_models = list(model_probs.keys())
    for i in range(len(loaded_models)):
        for j in range(i + 1, len(loaded_models)):
            name_a = loaded_models[i]
            name_b = loaded_models[j]
            mean_diff, p_val = bootstrap_roc_comparison(
                y_test.values, model_probs[name_a], model_probs[name_b]
            )
            pairwise_records.append({
                'Model_A': name_a,
                'Model_B': name_b,
                'AUC_Difference': mean_diff,
                'Bootstrap_p_value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
    pairwise_df = pd.DataFrame(pairwise_records)
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "pairwise_roc_comparison_pvalues.csv"), index=False)
    
    # 4. Calibration plots
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    colors_cal = {'Logistic Regression': '#1F4E79', 'XGBoost': '#C55A11', 'Random Forest': '#2E7D32', 'Neural Network (MLP)': '#AD1457'}
    
    for name, probs in model_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", color=colors_cal.get(name, 'blue'), label=f"{name} (ECE={metrics_summary.loc[metrics_summary['Model'] == name, 'ECE'].values[0]:.4f})")
        
    plt.title("Probability Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_comparison_curves.png"), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_comparison_curves.pdf"), format='pdf')
    plt.close()
    
    # 5. Stability & Top 20 Genes
    top_20 = analyze_feature_selection_stability(X_train, y_train)
    
    # 6. Pathway Enrichment
    run_pathway_enrichment(top_20)
    
    # 7. Sensitivity analysis
    run_sensitivity_analysis(X, y, dataset=args.dataset)
    
    # 8. DCA & clinical utility
    run_decision_curve_analysis(y_test, model_probs)
    
    # 9. Subgroup Analysis
    run_subgroup_analysis(X_test, y_test, loaded_models, dataset)
    
    # 10. Cox proportional hazards
    run_cox_proportional_hazards()
    
    print("\n--- ALL ANALYSES COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()
