"""
survival.py — Survival analysis for clinical relevance.

Scientific purpose:
Links the machine learning predictions (proliferation status) to actual
patient outcomes (overall survival) using Kaplan-Meier curves and Log-Rank tests.
This demonstrates clinical utility beyond mere classification accuracy.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Warning: 'lifelines' package not found. Survival analysis requires 'pip install lifelines'.")

def run_survival_analysis(clinical_path, scores_path, results_dir, dataset_name):
    """
    Computes Kaplan-Meier survival curves and Log-Rank statistics comparing
    High Proliferation vs Low Proliferation predicted classes.
    """
    if not LIFELINES_AVAILABLE:
        print("Cannot run survival analysis without 'lifelines'. Exiting.")
        return
        
    print(f"\n--- Survival Analysis ({dataset_name}) ---")
    
    # Load clinical data (should contain os_time and os_event)
    clinical = pd.read_csv(clinical_path, index_col=0)
    
    # Normalize column names to handle different dataset formats
    # GEO uses: os.event, os.delay_(months), rfs.event, rfs.delay
    # GSE17538 uses: overall_event_(death_from_any_cause), overall_survival_follow-up_time
    # Expected: os_event, os_time
    rename_map = {}
    for col in clinical.columns:
        if col in ('os.event',):
            rename_map[col] = 'os_event'
        elif col in ('os.delay_(months)', 'os.delay', 'os_delay_months'):
            rename_map[col] = 'os_time'
        elif col in ('rfs.event',):
            rename_map[col] = 'rfs_event'
        elif col in ('rfs.delay',):
            rename_map[col] = 'rfs_time'
        elif col == 'overall_event_(death_from_any_cause)':
            rename_map[col] = 'os_event'
        elif col == 'overall_survival_follow-up_time':
            rename_map[col] = 'os_time'
    if rename_map:
        clinical = clinical.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")
    
    # Load target/predictions (can be ground truth proliferation or model predictions)
    # Using ground truth target here for simplicity, or we could load model predictions.
    # The requirement is just to link proliferation class to survival.
    scores_df = pd.read_csv(scores_path, index_col=0)
    
    # Determine the column name. If it's y_target.csv, column is 'target'. 
    # If proliferation_scores.csv, column is 'score'. Let's support both.
    col_name = scores_df.columns[0]
    
    # Merge on index
    merged = pd.concat([clinical, scores_df], axis=1, join='inner')
    
    if 'os_time' not in merged.columns or 'os_event' not in merged.columns:
        print(f"Dataset {dataset_name} lacks survival columns (os_time, os_event). Skipping.")
        return
    
    # Convert string-valued os_event (TCGA uses "DECEASED"/"LIVING", GSE17538 uses "death"/"no death") to numeric
    if merged['os_event'].dtype == object:
        event_map = {'DECEASED': 1, 'LIVING': 0, 'Dead': 1, 'Alive': 0, 'death': 1, 'no death': 0}
        merged['os_event'] = merged['os_event'].map(event_map)
        print(f"  Converted string os_event to numeric (mapped: {event_map})")
    
    # Ensure os_time is numeric
    merged['os_time'] = pd.to_numeric(merged['os_time'], errors='coerce')
    
    # For censored patients (event==0) with missing os_time, use days_to_last_followup
    if 'days_to_last_followup' in merged.columns:
        followup = pd.to_numeric(merged['days_to_last_followup'], errors='coerce')
        mask_missing_time = merged['os_time'].isna() & (merged['os_event'] == 0)
        merged.loc[mask_missing_time, 'os_time'] = followup[mask_missing_time]
        filled = mask_missing_time.sum()
        if filled > 0:
            print(f"  Filled {filled} missing os_time values from days_to_last_followup")
        
    # Drop rows with missing survival data
    merged = merged.dropna(subset=['os_time', 'os_event', col_name])
    
    # If the score is continuous, binarize it at median to get High/Low classes
    if merged[col_name].nunique() > 2:
        median_val = merged[col_name].median()
        merged['class'] = (merged[col_name] >= median_val).astype(int)
    else:
        merged['class'] = merged[col_name]
        
    mask_high = merged['class'] == 1
    mask_low = merged['class'] == 0
    
    time_high = merged.loc[mask_high, 'os_time']
    event_high = merged.loc[mask_high, 'os_event']
    
    time_low = merged.loc[mask_low, 'os_time']
    event_low = merged.loc[mask_low, 'os_event']
    
    print(f"High Proliferation Cohort: n={len(time_high)}, events={event_high.sum()}")
    print(f"Low Proliferation Cohort: n={len(time_low)}, events={event_low.sum()}")
    
    if len(time_high) == 0 or len(time_low) == 0:
        print("Not enough samples in groups to perform survival analysis.")
        return
        
    # Log-Rank Test
    results = logrank_test(time_high, time_low, event_observed_A=event_high, event_observed_B=event_low)
    p_value = results.p_value
    print(f"Log-Rank Test p-value: {p_value:.4e}")

    # Save log-rank p-values for paper generation
    logrank_path = os.path.join(results_dir, "logrank_results.csv")
    logrank_df = pd.DataFrame([{
        'dataset': dataset_name,
        'logrank_p': p_value,
    }])
    if os.path.exists(logrank_path):
        existing = pd.read_csv(logrank_path)
        if dataset_name in existing['dataset'].values:
            existing.loc[existing['dataset'] == dataset_name, 'logrank_p'] = p_value
            existing.to_csv(logrank_path, index=False)
        else:
            pd.concat([existing, logrank_df], ignore_index=True).to_csv(logrank_path, index=False)
    else:
        logrank_df.to_csv(logrank_path, index=False)
    print(f"  Saved log-rank result to {logrank_path}")
    
    # Kaplan-Meier Plot
    plt.figure(figsize=(8, 6))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(time_high, event_high, label="High Proliferation")
    kmf_low.fit(time_low, event_low, label="Low Proliferation")
    
    ax = kmf_low.plot_survival_function(color='blue')
    kmf_high.plot_survival_function(ax=ax, color='red')
    
    # Add p-value to plot
    plt.text(0.05, 0.1, f"Log-Rank p = {p_value:.3e}", transform=ax.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
             
    plt.title(f"Overall Survival by Proliferation Status ({dataset_name.upper()})")
    plt.xlabel("Time (Days/Months)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save plot
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"kaplan_meier_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Saved Kaplan-Meier plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Survival Analysis")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    
    # Check which datasets are available
    datasets = ["geo", "geo17538", "geo_pan", "tcga", "tcga_read", "tcga_pan", "cptac", "synthetic"]
    
    for ds in datasets:
        prefix = f"{ds}_" if ds != "dataset" else ""
        clin_path = os.path.join(args.data_dir, f"{prefix}clinical.csv")
        target_path = os.path.join(args.data_dir, f"{prefix}y_target.csv")
        
        if os.path.exists(clin_path) and os.path.exists(target_path):
            run_survival_analysis(clin_path, target_path, args.results_dir, ds)

if __name__ == "__main__":
    main()
